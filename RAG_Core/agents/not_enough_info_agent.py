from typing import Dict, Any, List
from models.llm_model import llm_model
from config.settings import settings


class NotEnoughInfoAgent:
    def __init__(self):
        self.name = "NOT_ENOUGH_INFO"

        # Prompt mới: yêu cầu LLM trả lời dựa trên kiến thức của nó
        self.prompt_template = """Bạn là một chuyên viên tư vấn khách hàng người Việt Nam thân thiện và chuyên nghiệp - chuyên gia về chuyển đổi số và công nghệ.

TÌNH HUỐNG: Hệ thống không tìm thấy thông tin chính xác trong cơ sở dữ liệu để trả lời câu hỏi này.

Câu hỏi người dùng: "{question}"

NHIỆM VỤ CỦA BẠN:
1. Thừa nhận rằng bạn chưa có thông tin chính thức trong hệ thống
2. NHƯNG dựa trên kiến thức chuyên môn của bạn về chuyển đổi số, hãy cung cấp:
   - Câu trả lời hữu ích và mang tính tham khảo
   - Chia sẻ kiến thức chung về chủ đề (nếu có)
   - Gợi ý hướng tìm hiểu hoặc giải pháp thay thế
3. Cuối cùng, đề xuất khách hàng liên hệ hotline để được tư vấn chính xác hơn

YÊU CẦU:
- Trả lời bằng tiếng Việt tự nhiên, thân thiện
- Thể hiện sự chuyên nghiệp nhưng cũng khiêm tốn
- Luôn làm rõ đây là ý kiến tham khảo, không phải thông tin chính thức

Số điện thoại hỗ trợ: {support_phone}

Hãy trả lời:"""

    def process(self, question: str, **kwargs) -> Dict[str, Any]:
        """Xử lý trường hợp không đủ thông tin - nhưng vẫn cố gắng hỗ trợ"""
        try:
            prompt = self.prompt_template.format(
                question=question,
                support_phone=settings.SUPPORT_PHONE
            )

            answer = llm_model.invoke(prompt)

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": [
                    {
                        "document_id": "llm_knowledge",
                        "type": "GENERAL_KNOWLEDGE"
                    }
                ],
                "next_agent": "end"
            }

        except Exception as e:
            return {
                "status": "ERROR",
                "answer": f"Xin lỗi, hệ thống gặp lỗi: {str(e)}. Vui lòng liên hệ {settings.SUPPORT_PHONE} để được hỗ trợ.",
                "references": [],
                "next_agent": "end"
            }