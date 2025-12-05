# RAG_Core/utils/context_processor.py

from typing import List, Dict, Any, Optional
from models.llm_model import llm_model
import logging
import json
import re
import unicodedata
logger = logging.getLogger(__name__)


class ContextProcessor:
    """Xử lý context từ lịch sử hội thoại"""

    def __init__(self):
        self.context_analysis_prompt = """Bạn là chuyên gia phân tích ngữ cảnh hội thoại.

Nhiệm vụ: Phân tích lịch sử hội thoại để chuyển đổi câu hỏi hiện tại thành câu hỏi độc lập, đầy đủ ngữ cảnh.

Lịch sử hội thoại:
{history}

Câu hỏi hiện tại: "{current_question}"

Nguyên tắc xử lý:

1. **Nhận diện câu hỏi phụ thuộc ngữ cảnh:**
   - Chứa đại từ: "nó", "đó", "này", "cái đó", "điều này"
   - Chứa từ chỉ thứ tự: "thứ nhất", "thứ 2", "mục 3", "điểm 4", "cái cuối"
   - Chứa từ tham chiếu: "tiếp tục", "chi tiết hơn", "giải thích thêm", "còn...", "thế còn...", "OK", "Okay", "có", "co", "vâng",...
   - Chỉ có một từ/cụm từ ngắn: "chi tiết", "ví dụ", "tại sao", "như thế nào"

2. **Tìm ngữ cảnh từ lịch sử:**
   - Quét ngược từ tin nhắn gần nhất
   - Xác định chủ đề chính đang được thảo luận
   - Tìm danh sách, khái niệm, thuật ngữ được đề cập
   - Lưu ý các con số, tên riêng, địa danh cụ thể

3. **Chuyển đổi câu hỏi:**
   - Thay thế đại từ bằng danh từ cụ thể
   - Thay số thứ tự bằng tên đầy đủ của mục đó
   - Bổ sung chủ đề/ngữ cảnh nếu câu hỏi quá ngắn
   - Giữ nguyên ý định và giọng điệu của người hỏi
   - Đảm bảo câu hỏi mới có thể hiểu được mà không cần đọc lịch sử

4. **Trường hợp đặc biệt:**
   - Nếu câu hỏi đã đầy đủ ngữ cảnh → Giữ nguyên
   - Nếu không tìm thấy ngữ cảnh phù hợp → Giữ nguyên và thêm "[cần làm rõ]"
   - Nếu câu hỏi mơ hồ có nhiều cách hiểu → Chọn cách hiểu hợp lý nhất dựa trên ngữ cảnh gần nhất

5. **Quy tắc đầu ra:**
   - CHỈ trả về câu hỏi đã được làm rõ
   - KHÔNG thêm lời giải thích, mở đầu hay kết luận
   - KHÔNG thay đổi ngôn ngữ gốc (giữ nguyên tiếng Việt/tiếng Anh)
   - KHÔNG thêm thông tin mà người dùng không hỏi

Ví dụ minh họa:

Ví dụ 1:
Lịch sử: "Khung năng lực số có 6 nhóm kỹ năng..."
Câu hỏi: "Chi tiết kỹ năng số 3"
→ Câu hỏi làm rõ: "Chi tiết về nhóm kỹ năng thứ 3 'Giao tiếp và hợp tác trên môi trường số' trong khung năng lực số"

Ví dụ 2:
Lịch sử: "Python có 3 cách xử lý file: read(), write(), append()"
Câu hỏi: "Cái thứ 2 dùng như thế nào?"
→ Câu hỏi làm rõ: "Phương thức write() trong Python dùng như thế nào để xử lý file?"

Ví dụ 3:
Lịch sử: "React và Vue đều là framework frontend phổ biến"
Câu hỏi: "Thế còn Angular?"
→ Câu hỏi làm rõ: "Angular là framework frontend như thế nào so với React và Vue?"

Ví dụ 4:`
Lịch sử: [Không có]
Câu hỏi: "Giá iPhone 15 bao nhiêu?"
→ Câu hỏi làm rõ: "Giá iPhone 15 bao nhiêu?"

Bây giờ hãy xử lý câu hỏi trên:"""

    def extract_context_from_history(
            self,
            history: List,
            current_question: str
    ) -> Dict[str, Any]:
        """
        Trích xuất context từ history và phân tích câu hỏi hiện tại
        Xử lý cả dict và ChatMessage objects

        Returns:
            {
                "original_question": str,
                "contextualized_question": str,
                "is_followup": bool,
                "relevant_context": str
            }
        """
        try:
            # Convert history to dict format if needed
            normalized_history = self._normalize_history(history)

            # Kiểm tra nếu không có history
            if not normalized_history or len(normalized_history) == 0:
                return {
                    "original_question": current_question,
                    "contextualized_question": current_question,
                    "is_followup": False,
                    "relevant_context": ""
                }

            # Kiểm tra xem có phải follow-up question không
            is_followup = self._is_followup_question(current_question)

            if not is_followup:
                return {
                    "original_question": current_question,
                    "contextualized_question": current_question,
                    "is_followup": False,
                    "relevant_context": ""
                }

            # Format history cho LLM
            history_text = self._format_history(normalized_history)

            # Tạo câu hỏi đầy đủ context
            prompt = self.context_analysis_prompt.format(
                history=history_text,
                current_question=current_question
            )

            contextualized_question = llm_model.invoke(prompt).strip()

            # Fallback nếu LLM không hoạt động
            if not contextualized_question or len(contextualized_question) < 5:
                contextualized_question = self._manual_contextualize(
                    current_question,
                    normalized_history
                )

            return {
                "original_question": current_question,
                "contextualized_question": contextualized_question,
                "is_followup": True,
                "relevant_context": self._extract_relevant_context(normalized_history)
            }

        except Exception as e:
            logger.error(f"Error in context extraction: {e}")
            return {
                "original_question": current_question,
                "contextualized_question": current_question,
                "is_followup": False,
                "relevant_context": ""
            }

    def _normalize_history(self, history: List) -> List[Dict[str, str]]:
        """Convert history to standardized dict format"""
        if not history:
            return []

        normalized = []
        for msg in history:
            if isinstance(msg, dict):
                # Already dict format
                normalized.append({
                    "role": msg.get("role", ""),
                    "content": msg.get("content", "")
                })
            else:
                # ChatMessage object or similar
                normalized.append({
                    "role": getattr(msg, "role", ""),
                    "content": getattr(msg, "content", "")
                })

        return normalized

    def _is_followup_question(self, question: str) -> bool:
        """Kiểm tra xem câu hỏi có phải follow-up không (nâng cao)"""
        # Chuẩn hóa chuỗi để tránh lỗi ký tự đặc biệt
        question_norm = unicodedata.normalize("NFKC", question.lower().strip())

        # Keywords chỉ ra follow-up question
        followup_keywords = [
            "thứ", "cái đó", "nó", "điều đó", "phần đó",
            "tiếp theo", "còn", "thêm", "chi tiết",
            "giải thích", "phân tích", "làm rõ",
            "ví dụ", "cụ thể", "so với"
        ]
        pronouns = ["nó", "cái đó", "điều đó", "phần đó"]
        ordinals = ["thứ 1", "thứ 2", "thứ 3", "đầu tiên", "thứ hai", "thứ ba"]

        # Match bằng regex để tránh lỗi khoảng trắng đặc biệt
        def contains_kw(text, keywords):
            return any(re.search(rf"\b{re.escape(kw)}\b", text) for kw in keywords)

        has_kw = contains_kw(question_norm, followup_keywords)
        has_pronoun = contains_kw(question_norm, pronouns)
        has_ordinal = contains_kw(question_norm, ordinals)
        is_short = len(question_norm.split()) < 10  # tăng ngưỡng nhẹ

        return (has_kw or has_pronoun or has_ordinal or is_short)

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format history thành text dễ đọc"""
        history_lines = []

        # Chỉ lấy 3 turn gần nhất
        recent_history = history[-6:] if len(history) > 6 else history

        for msg in recent_history:
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            content = msg["content"]
            history_lines.append(f"{role}: {content}")

        return "\n".join(history_lines)

    def _extract_relevant_context(self, history: List[Dict[str, str]]) -> str:
        """Trích xuất context liên quan từ lần trả lời gần nhất"""
        if not history:
            return ""

        # Lấy câu trả lời gần nhất của assistant
        for msg in reversed(history):
            role = msg.get("role", "")
            if role == "assistant":
                content = msg.get("content", "")
                return content[:500]  # Lấy 500 ký tự đầu

        return ""

    def _manual_contextualize(self, question: str, history: List[Dict[str, str]]) -> str:
        """
        Phương pháp dự phòng: tự động thêm context bằng rule-based nâng cao
        """
        question_lower = unicodedata.normalize("NFKC", question.lower())
        last_assistant_msg = ""
        last_user_msg = ""

        # Lấy tin nhắn gần nhất
        for msg in reversed(history):
            if msg.get("role") == "assistant" and not last_assistant_msg:
                last_assistant_msg = msg.get("content", "")
            if msg.get("role") == "user" and not last_user_msg:
                last_user_msg = msg.get("content", "")
            if last_assistant_msg and last_user_msg:
                break

        # Nếu không có gì, trả nguyên
        if not last_assistant_msg:
            return question

        # Nếu có "thứ N" → cố gắng map sang phần tương ứng
        ordinal_match = re.search(r'thứ (\d+|hai|ba|tư|năm)', question_lower)
        if ordinal_match:
            ordinal = ordinal_match.group(1)
            number_map = {"1": 1, "2": 2, "3": 3, "hai": 2, "ba": 3, "tư": 4}
            idx = number_map.get(ordinal, 1) - 1

            items = self._extract_items_from_text(last_assistant_msg)
            if items and idx < len(items):
                referenced_item = items[idx]
                contextualized = f"Phân tích chi tiết về {referenced_item}, thuộc {last_user_msg}"
                return contextualized

        # Nếu không có pattern, nối ngữ cảnh cũ
        context = last_user_msg or last_assistant_msg[:80]
        return f"{question} (Liên quan đến: {context})"

    def _extract_items_from_text(self, text: str) -> List[str]:
        """Trích xuất danh sách items từ text"""
        items = []

        # Pattern 1: Numbered list (1., 2., 3.)
        numbered_pattern = r'\d+\.\s*([^.;]+)'
        numbered_items = re.findall(numbered_pattern, text)
        items.extend(numbered_items)

        # Pattern 2: Dash list (- item, • item)
        dash_pattern = r'[-•]\s*([^-•\n]+)'
        dash_items = re.findall(dash_pattern, text)
        items.extend(dash_items)

        # Pattern 3: Semicolon-separated
        if ';' in text:
            semicolon_items = text.split(';')
            items.extend([item.strip() for item in semicolon_items if item.strip()])

        # Cleanup items
        cleaned_items = []
        for item in items:
            cleaned = item.strip()
            if len(cleaned) > 5:  # Skip very short items
                cleaned_items.append(cleaned)

        return cleaned_items


# Global instance
context_processor = ContextProcessor()