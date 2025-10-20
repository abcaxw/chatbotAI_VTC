# RAG_Core/agents/faq_agent.py (OPTIMIZED VERSION)

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
        self.vector_threshold = 0.5  # Ngưỡng thấp hơn cho vector search (cast net wide)
        self.rerank_threshold = 0.6  # Ngưỡng cao hơn cho reranked results

        # Prompt cho câu hỏi thông thường
        self.standard_prompt = """Bạn là một chuyên viên tư vấn khách hàng người Việt Nam thân thiện và chuyên nghiệp - chuyên gia trả lời các câu hỏi thường gặp và hỗ trợ khách hàng.

Nhiệm vụ: 
1. Chào hỏi thân thiện khi khách hàng bắt đầu cuộc trò chuyện
2. Tìm kiếm và trả lời câu hỏi từ cơ sở dữ liệu FAQ
3. Hướng dẫn khách hàng nếu cần hỗ trợ thêm

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

        # Prompt cho follow-up question
        self.followup_prompt = """Bạn là một chuyên viên tư vấn khách hàng người Việt Nam thân thiện và chuyên nghiệp - chuyên gia trả lời các câu hỏi thường gặp.

⚠️ ĐÂY LÀ CÂU HỎI FOLLOW-UP (khách hàng hỏi tiếp về chủ đề đang thảo luận)

Ngữ cảnh: {context}

Câu hỏi follow-up: "{question}"

Kết quả tìm kiếm FAQ (đã được rerank):
{faq_results}

Hướng dẫn đặc biệt cho follow-up:
1. Nhận biết đây là câu hỏi tiếp theo, không phải câu hỏi mới
2. Sử dụng FAQ có rerank_score > {rerank_threshold}
3. KHÔNG sử dụng markdown, bullet points hay định dạng đặc biệt
4. Nếu không tìm thấy FAQ phù hợp, trả về "NOT_FOUND"
5. Có thể kết hợp thông tin từ ngữ cảnh trước và FAQ mới

Trả lời:"""

    def process(
            self,
            question: str,
            is_followup: bool = False,
            context: str = "",
            **kwargs
    ) -> Dict[str, Any]:
        """Xử lý câu hỏi FAQ với reranking 2 tầng"""
        try:
            # BƯỚC 1: Vector search với threshold thấp (cast wide net)
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

            # BƯỚC 4: Format FAQ results cho LLM
            faq_text = self._format_reranked_faq(reranked_faqs[:3])  # Top 3

            # BƯỚC 5: Chọn prompt phù hợp
            if is_followup and context:
                prompt = self.followup_prompt.format(
                    question=question,
                    context=context,
                    faq_results=faq_text,
                    rerank_threshold=self.rerank_threshold
                )
                logger.info("Using follow-up FAQ prompt with reranking")
            else:
                prompt = self.standard_prompt.format(
                    question=question,
                    faq_results=faq_text,
                    rerank_threshold=self.rerank_threshold
                )
                logger.info("Using standard FAQ prompt with reranking")

            # BƯỚC 6: Tạo câu trả lời
            response = llm_model.invoke(prompt)

            # Validate response
            if "NOT_FOUND" in response.upper():
                logger.info("LLM determined FAQ not sufficient")
                return self._route_to_retriever("LLM rejected FAQ")

            if not response or len(response.strip()) < 10:
                logger.warning("Generated answer too short")
                return self._route_to_retriever("Answer too short")

            logger.info(f"FAQ answer generated successfully (rerank={rerank_score:.3f})")

            return {
                "status": "SUCCESS",
                "answer": response,
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

    def _format_reranked_faq(self, faq_results: List[Dict[str, Any]]) -> str:
        """Format FAQ đã được rerank với điểm số"""
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

    def set_thresholds(self, vector_threshold: float = None, rerank_threshold: float = None):
        """Điều chỉnh ngưỡng động (optional)"""
        if vector_threshold is not None:
            self.vector_threshold = vector_threshold
            logger.info(f"Vector threshold updated to {vector_threshold}")

        if rerank_threshold is not None:
            self.rerank_threshold = rerank_threshold
            logger.info(f"Rerank threshold updated to {rerank_threshold}")