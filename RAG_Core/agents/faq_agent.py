# RAG_Core/agents/faq_agent.py (DIRECT ANSWER VERSION)

from typing import Dict, Any, List
from models.llm_model import llm_model
from tools.vector_search import search_faq, rerank_faq
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class FAQAgent:
    def __init__(self):
        self.name = "FAQ"

        # Ngưỡng cho các giai đoạn khác nhau
        self.vector_threshold = 0.5  # Ngưỡng thấp hơn cho vector search
        self.rerank_threshold = 0.6  # Ngưỡng cao hơn cho reranked results

        # Ngưỡng để trả lời trực tiếp (không cần LLM)
        self.direct_answer_threshold = 0.6  # Rất chắc chắn -> trả lời luôn

        # Có sử dụng LLM hay không (có thể tắt hoàn toàn)
        self.use_llm = True  # Set False để LUÔN trả lời trực tiếp

        # Prompt cho câu hỏi thông thường (chỉ dùng khi score thấp hơn)
        self.standard_prompt = """Bạn là một chuyên viên tư vấn khách hàng người Việt Nam thân thiện và chuyên nghiệp.

Câu hỏi người dùng: "{question}"

Kết quả tìm kiếm FAQ (đã được rerank):
{faq_results}

Hướng dẫn:
1. Kết quả đã được sắp xếp theo độ phù hợp (rerank_score)
2. Nếu FAQ đầu tiên có rerank_score > {rerank_threshold}, hãy trả lời dựa trên đó
3. Nếu không có FAQ phù hợp, trả về "NOT_FOUND"
4. Trả lời bằng tiếng Việt, thân thiện và chính xác
5. Có thể kết hợp thông tin từ nhiều FAQ nếu cần

Trả lời:"""

    def process(
            self,
            question: str,
            is_followup: bool = False,
            context: str = "",
            **kwargs
    ) -> Dict[str, Any]:
        """Xử lý câu hỏi FAQ với khả năng trả lời trực tiếp"""
        try:
            # BƯỚC 1: Vector search
            logger.info(f"Step 1: Vector search for FAQ with threshold={self.vector_threshold}")
            faq_results = search_faq.invoke({"query": question})

            if not faq_results or "error" in str(faq_results):
                logger.warning("FAQ vector search failed or returned error")
                return self._route_to_retriever("Vector search failed")

            # Lọc theo vector threshold
            filtered_faqs = [
                faq for faq in faq_results
                if faq.get("similarity_score", 0) >= self.vector_threshold
            ]

            if not filtered_faqs:
                logger.info(f"No FAQ passed vector threshold {self.vector_threshold}")
                return self._route_to_retriever("No FAQ above vector threshold")

            logger.info(f"Found {len(filtered_faqs)} FAQs above vector threshold")

            # BƯỚC 2: Rerank với cross-encoder
            logger.info("Step 2: Reranking FAQs with cross-encoder")
            reranked_faqs = rerank_faq.invoke({
                "query": question,
                "faq_results": filtered_faqs
            })

            if not reranked_faqs:
                logger.warning("Reranking returned empty results")
                return self._route_to_retriever("Reranking failed")

            # BƯỚC 3: Kiểm tra rerank score
            best_faq = reranked_faqs[0]
            rerank_score = best_faq.get("rerank_score", 0)
            similarity_score = best_faq.get("similarity_score", 0)

            logger.info(
                f"Best FAQ: rerank={rerank_score:.3f}, "
                f"similarity={similarity_score:.3f}"
            )

            # Quyết định dựa trên rerank_score
            if rerank_score < self.rerank_threshold:
                logger.info(
                    f"Rerank score {rerank_score:.3f} below threshold "
                    f"{self.rerank_threshold}, routing to RETRIEVER"
                )
                return self._route_to_retriever(
                    f"Best rerank score ({rerank_score:.3f}) too low"
                )

            # BƯỚC 4: Quyết định trả lời trực tiếp hay dùng LLM

            # TH1: Điểm số rất cao hoặc tắt LLM -> TRẢ LỜI TRỰC TIẾP
            if not self.use_llm or rerank_score >= self.direct_answer_threshold:
                logger.info(
                    f"✅ DIRECT ANSWER: rerank={rerank_score:.3f} "
                    f"(threshold={self.direct_answer_threshold})"
                )

                answer = self._format_direct_answer(best_faq, question)

                return {
                    "status": "SUCCESS",
                    "answer": answer,
                    "mode": "direct",  # Đánh dấu là trả lời trực tiếp
                    "references": [
                        {
                            "document_id": best_faq.get("faq_id"),
                            "type": "FAQ",
                            "rerank_score": round(rerank_score, 4),
                            "similarity_score": round(similarity_score, 4)
                        }
                    ],
                    "next_agent": "end"
                }

            # TH2: Điểm số trung bình -> DÙNG LLM ĐỂ LÀM MỊN
            else:
                logger.info(
                    f"🤖 LLM MODE: rerank={rerank_score:.3f} "
                    f"(below direct threshold {self.direct_answer_threshold})"
                )

                faq_text = self._format_reranked_faq(reranked_faqs[:3])

                prompt = self.standard_prompt.format(
                    question=question,
                    faq_results=faq_text,
                    rerank_threshold=self.rerank_threshold
                )

                response = llm_model.invoke(prompt)

                # Validate response
                if "NOT_FOUND" in response.upper():
                    logger.info("LLM determined FAQ not sufficient")
                    return self._route_to_retriever("LLM rejected FAQ")

                if not response or len(response.strip()) < 10:
                    logger.warning("Generated answer too short")
                    return self._route_to_retriever("Answer too short")

                logger.info(f"FAQ answer generated via LLM (rerank={rerank_score:.3f})")

                return {
                    "status": "SUCCESS",
                    "answer": response,
                    "mode": "llm",  # Đánh dấu là qua LLM
                    "references": [
                        {
                            "document_id": best_faq.get("faq_id"),
                            "type": "FAQ",
                            "rerank_score": round(rerank_score, 4),
                            "similarity_score": round(similarity_score, 4)
                        }
                    ],
                    "next_agent": "end"
                }

        except Exception as e:
            logger.error(f"Error in FAQ agent: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "answer": f"Lỗi xử lý FAQ: {str(e)}",
                "references": [],
                "next_agent": "RETRIEVER"
            }

    def _format_direct_answer(self, faq: Dict[str, Any], question: str) -> str:
        """
        Format câu trả lời trực tiếp từ FAQ (không qua LLM)
        Có thể custom thêm greeting, format đẹp hơn
        """
        answer = faq.get('answer', '')

        # Option 1: Trả lời ngắn gọn (chỉ answer)
        # return answer

        # Option 2: Thêm chút context (recommended)
        return f"{answer}"

        # Option 3: Format chi tiết hơn
        # return f"Dựa vào thông tin từ FAQ:\n\n{answer}\n\nNếu bạn cần thêm thông tin, vui lòng hỏi thêm nhé!"

    def _format_reranked_faq(self, faq_results: List[Dict[str, Any]]) -> str:
        """Format FAQ đã được rerank với điểm số (cho LLM)"""
        if not faq_results:
            return "Không tìm thấy FAQ phù hợp"

        formatted_lines = []
        for i, faq in enumerate(faq_results, 1):
            question = faq.get('question', '')
            answer = faq.get('answer', '')
            rerank_score = faq.get('rerank_score', 0)
            similarity_score = faq.get('similarity_score', 0)

            formatted_lines.append(
                f"FAQ {i} (Rerank: {rerank_score:.3f}, Similarity: {similarity_score:.3f}):\n"
                f"Q: {question}\n"
                f"A: {answer}\n"
            )

        return "\n".join(formatted_lines)

    def _route_to_retriever(self, reason: str) -> Dict[str, Any]:
        """Helper để route sang RETRIEVER"""
        logger.info(f"Routing to RETRIEVER: {reason}")
        return {
            "status": "NOT_FOUND",
            "answer": "",
            "references": [],
            "next_agent": "RETRIEVER"
        }

    def set_thresholds(
            self,
            vector_threshold: float = None,
            rerank_threshold: float = None,
            direct_answer_threshold: float = None,
            use_llm: bool = None
    ):
        """Điều chỉnh ngưỡng động"""
        if vector_threshold is not None:
            self.vector_threshold = vector_threshold
            logger.info(f"Vector threshold updated to {vector_threshold}")

        if rerank_threshold is not None:
            self.rerank_threshold = rerank_threshold
            logger.info(f"Rerank threshold updated to {rerank_threshold}")

        if direct_answer_threshold is not None:
            self.direct_answer_threshold = direct_answer_threshold
            logger.info(f"Direct answer threshold updated to {direct_answer_threshold}")

        if use_llm is not None:
            self.use_llm = use_llm
            logger.info(f"Use LLM mode: {use_llm}")