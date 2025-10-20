from typing import Dict, Any, List
from models.llm_model import llm_model
from config.settings import settings


class NotEnoughInfoAgent:
    def __init__(self):
        self.name = "NOT_ENOUGH_INFO"
        self.prompt_template = """Bạn là một chuyên viên tư vấn khách hàng người Việt Nam thân thiện và chuyên nghiệp - chuyên gia xử lý trường hợp không đủ thông tin.

Nhiệm vụ: Thông báo lịch sự khi không tìm thấy thông tin phù hợp và hướng dẫn khách hàng.

Câu hỏi người dùng: "{question}"
Số điện thoại hỗ trợ: {support_phone}

Hướng dẫn:
1. Xin lỗi vì không tìm thấy thông tin phù hợp
2. Gợi ý khách hàng:
   - Thử diễn đạt câu hỏi khác
   - Liên hệ số hotline hỗ trợ
   - Cung cấp thêm thông tin chi tiết
3. Giữ thái độ thân thiện và chuyên nghiệp

Trả lời:"""

    def process(self, question: str, **kwargs) -> Dict[str, Any]:
        """Xử lý trường hợp không đủ thông tin"""
        try:
            prompt = self.prompt_template.format(
                question=question,
                support_phone=settings.SUPPORT_PHONE
            )

            answer = llm_model.invoke(prompt)

            # Fallback answer nếu LLM không hoạt động
            if not answer or len(answer.strip()) < 10:
                answer = f"""Xin lỗi, tôi không tìm thấy thông tin phù hợp để trả lời câu hỏi của bạn.

Bạn có thể:
• Thử diễn đạt câu hỏi theo cách khác
• Cung cấp thêm thông tin chi tiết
• Liên hệ hotline hỗ trợ: {settings.SUPPORT_PHONE} để được hỗ trợ tốt nhất

Cảm ơn bạn đã sử dụng dịch vụ!"""

            return {
                "status": "SUCCESS",
                "answer": answer,
                "references": [],
                "next_agent": "end"
            }

        except Exception as e:
            return {
                "status": "ERROR",
                "answer": f"Xin lỗi, hệ thống gặp sự cố. Vui lòng liên hệ {settings.SUPPORT_PHONE} để được hỗ trợ.",
                "references": [],
                "next_agent": "end"
            }