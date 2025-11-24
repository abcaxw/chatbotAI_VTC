# RAG_Core/agents/generator_agent.py

from typing import Dict, Any, List
from models.llm_model import llm_model
import logging

logger = logging.getLogger(__name__)


class GeneratorAgent:
    def __init__(self):
        self.name = "GENERATOR"

        # Prompt cho câu hỏi thông thường
        self.standard_prompt = """Bạn là một chuyên viên tư vấn khách hàng người Việt Nam thân thiện và chuyên nghiệp.

Câu hỏi của khách hàng: "{question}"

Thông tin tham khảo từ tài liệu:
{documents}

Lịch sử trò chuyện gần đây:
{history}

Yêu cầu trả lời:
- Trả lời bằng giọng văn tự nhiên như người Việt Nam nói chuyện
- Trả lời thẳng vào vấn đề, ngắn gọn súc tích
- Dựa vào thông tin tài liệu nhưng diễn đạt theo cách hiểu của bạn
- Kết thúc bằng câu hỏi ngắn để tiếp tục hỗ trợ nếu cần

Hãy trả lời như đang nói chuyện trực tiếp với khách hàng:"""

        # Prompt cho follow-up question (có context)
        self.followup_prompt = """Bạn là một chuyên viên tư vấn khách hàng người Việt Nam thân thiện và chuyên nghiệp.

🔍 NGỮ CẢNH CUỘC TRÒ CHUYỆN:
{context_summary}

📝 LỊCH SỬ GẦN NHẤT:
{recent_history}

❓ CÂU HỎI FOLLOW-UP CỦA KHÁCH HÀNG: "{question}"

📚 THÔNG TIN TÀI LIỆU LIÊN QUAN:
{documents}

⚠️ YÊU CẦU ĐẶC BIỆT cho follow-up question:
1. Nhận biết rằng khách hàng đang hỏi tiếp về chủ đề đã thảo luận
2. Tham chiếu đến thông tin đã cung cấp trước đó một cách tự nhiên
3. Trả lời cụ thể vào phần mà khách hàng muốn biết thêm
4. KHÔNG lặp lại toàn bộ thông tin đã nói, chỉ tập trung vào phần được hỏi

📋 YÊU CẦU CHUNG:
- Trả lời bằng giọng văn tự nhiên như người Việt Nam nói chuyện
- Ngắn gọn, súc tích, đúng trọng tâm
- Kết thúc bằng câu hỏi để tiếp tục hỗ trợ nếu cần

Hãy trả lời:"""

    def _deduplicate_references(self, references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Loại bỏ các reference trùng lặp dựa trên document_id"""
        if not references:
            return []

        seen_doc_ids = set()
        unique_references = []

        for ref in references:
            doc_id = ref.get('document_id')
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                unique_references.append(ref)

        return unique_references

    def _format_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents thành text"""
        if not documents:
            return "Không có tài liệu tham khảo"

        doc_lines = []
        for i, doc in enumerate(documents[:5], 1):  # Chỉ lấy top 5
            description = doc.get('description', '')
            score = doc.get('similarity_score', 0)
            doc_lines.append(f"[Tài liệu {i}] (Độ liên quan: {score:.2f})\n{description}")

        return "\n\n".join(doc_lines)

    def _format_history(self, history: List, max_turns: int = 2) -> str:
        """Format lịch sử hội thoại, xử lý cả dict và ChatMessage objects"""
        if not history:
            return "Không có lịch sử"

        # Normalize history first
        normalized_history = []
        for msg in history:
            if isinstance(msg, dict):
                normalized_history.append({
                    "role": msg.get("role", ""),
                    "content": msg.get("content", "")
                })
            else:
                # ChatMessage object
                normalized_history.append({
                    "role": getattr(msg, "role", ""),
                    "content": getattr(msg, "content", "")
                })

        # Lấy N turn gần nhất (mỗi turn = 2 messages)
        recent_history = normalized_history[-(max_turns * 2):] if len(
            normalized_history) > max_turns * 2 else normalized_history

        history_lines = []
        for msg in recent_history:
            role = "👤 Khách hàng" if msg.get("role") == "user" else "🤖 Trợ lý"
            content = msg.get("content", "")
            if content:
                history_lines.append(f"{role}: {content}")

        return "\n".join(history_lines) if history_lines else "Không có lịch sử"

    def _extract_context_summary(self, history: List) -> str:
        """Trích xuất tóm tắt ngữ cảnh từ lịch sử, xử lý cả dict và ChatMessage"""
        if not history or len(history) < 2:
            return "Đây là câu hỏi đầu tiên"

        # Normalize history
        normalized_history = []
        for msg in history:
            if isinstance(msg, dict):
                normalized_history.append(msg)
            else:
                normalized_history.append({
                    "role": getattr(msg, "role", ""),
                    "content": getattr(msg, "content", "")
                })

        # Lấy câu hỏi trước và câu trả lời trước
        for i in range(len(normalized_history) - 1, -1, -1):
            if normalized_history[i].get("role") == "user":
                prev_question = normalized_history[i].get("content", "")

                # Tìm câu trả lời tương ứng
                for j in range(i + 1, len(normalized_history)):
                    if normalized_history[j].get("role") == "assistant":
                        prev_answer = normalized_history[j].get("content", "")
                        return f"Chủ đề đang thảo luận: {prev_question}\nĐã trả lời: {prev_answer[:200]}..."

                return f"Chủ đề đang thảo luận: {prev_question}"

        return "Đang trong cuộc trò chuyện"

    def process(
            self,
            question: str,
            documents: List[Dict[str, Any]],
            references: List[Dict[str, Any]] = None,
            history: List[Dict[str, str]] = None,
            is_followup: bool = False,
            context_summary: str = "",
            **kwargs
    ) -> Dict[str, Any]:
        """Tạo câu trả lời từ tài liệu đã được đánh giá"""
        try:
            if not documents:
                return {
                    "status": "ERROR",
                    "answer": "Không có tài liệu để tạo câu trả lời",
                    "references": [],
                    "next_agent": "end"
                }

            # Format documents
            doc_text = self._format_documents(documents)

            # Format history
            history_text = self._format_history(history or [], max_turns=2)

            # Chọn prompt phù hợp
            if is_followup:
                # Sử dụng prompt đặc biệt cho follow-up
                if not context_summary:
                    context_summary = self._extract_context_summary(history or [])

                prompt = self.followup_prompt.format(
                    question=question,
                    context_summary=context_summary,
                    recent_history=history_text,
                    documents=doc_text
                )

                logger.info(f"Using follow-up prompt for question: {question}")
                logger.info(f"Context: {context_summary}")
            else:
                # Sử dụng prompt thông thường
                prompt = self.standard_prompt.format(
                    question=question,
                    history=history_text,
                    documents=doc_text
                )

                logger.info(f"Using standard prompt for question: {question}")

            # Tạo câu trả lời
            answer = llm_model.invoke(prompt)

            # Validate answer
            if not answer or len(answer.strip()) < 10:
                answer = "Tôi đã tìm thấy thông tin liên quan nhưng gặp khó khăn trong việc tạo câu trả lời. Bạn có thể diễn đạt câu hỏi theo cách khác được không?"

            # Loại bỏ references trùng lặp
            unique_references = self._deduplicate_references(references or [])

            logger.info(f"Generated answer with {len(unique_references)} references")

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": unique_references,
                "next_agent": "end"
            }

        except Exception as e:
            logger.error(f"Error in generator agent: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "answer": f"Lỗi tạo câu trả lời: {str(e)}",
                "references": [],
                "next_agent": "end"
            }