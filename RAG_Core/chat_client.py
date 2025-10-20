#!/usr/bin/env python3
"""
RAG Chatbot Client - Test API từ terminal
Chạy: python chat_client.py
"""

import requests
import json
import sys
from typing import List, Dict
import time


class RAGChatClient:
    def __init__(self, base_url: str = "http://localhost:8501"):
        self.base_url = base_url
        self.session = requests.Session()
        self.chat_history = []

    def check_health(self):
        """Kiểm tra tình trạng API"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"🟢 API Status: {health_data['status']}")
                print(f"📊 Message: {health_data['message']}")
                print(f"🔗 Database: {'Connected' if health_data['database_connected'] else 'Disconnected'}")
                return True
            else:
                print(f"🔴 API Error: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return False

    def list_agents(self):
        """Hiển thị danh sách agents"""
        try:
            response = self.session.get(f"{self.base_url}/agents", timeout=5)
            if response.status_code == 200:
                agents_data = response.json()
                print("\n🤖 Available Agents:")
                print("=" * 50)
                for agent, description in agents_data["agents"].items():
                    print(f"• {agent}: {description}")
                print(f"\n🔄 Workflow: {agents_data['workflow']}")
                return True
            else:
                print(f"🔴 Error getting agents: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def send_message(self, question: str) -> Dict:
        """Gửi câu hỏi tới API"""
        try:
            payload = {
                "question": question,
                "history": self.chat_history
            }

            print("⏳ Đang xử lý...")
            start_time = time.time()

            response = self.session.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=30
            )

            end_time = time.time()

            if response.status_code == 200:
                result = response.json()

                # Thêm vào history
                self.chat_history.append({"role": "user", "content": question})
                self.chat_history.append({"role": "assistant", "content": result["answer"]})

                # Hiển thị kết quả
                print("\n" + "=" * 60)
                print(f"❓ Câu hỏi: {question}")
                print(f"⏱️  Thời gian xử lý: {end_time - start_time:.2f}s")
                print(f"📊 Status: {result.get('status', 'UNKNOWN')}")
                print("-" * 60)
                print(f"💬 Trả lời:\n{result['answer']}")

                # Hiển thị references nếu có
                if result.get("references"):
                    print("\n📚 Tài liệu tham khảo:")
                    for i, ref in enumerate(result["references"], 1):
                        print(f"  {i}. {ref['type']}: {ref['document_id']}")

                print("=" * 60)
                return result
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                print(f"🔴 {error_msg}")
                return {"error": error_msg}

        except requests.exceptions.Timeout:
            print("⏰ Request timeout - API có thể đang xử lý câu hỏi phức tạp")
            return {"error": "Timeout"}
        except Exception as e:
            print(f"❌ Error: {e}")
            return {"error": str(e)}

    def show_history(self):
        """Hiển thị lịch sử chat"""
        if not self.chat_history:
            print("📝 Chưa có lịch sử chat")
            return

        print("\n📜 Lịch sử chat:")
        print("=" * 50)
        for i, msg in enumerate(self.chat_history):
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            print(f"{role_icon} {msg['role'].title()}: {msg['content'][:100]}...")
        print("=" * 50)

    def clear_history(self):
        """Xóa lịch sử chat"""
        self.chat_history.clear()
        print("🗑️ Đã xóa lịch sử chat")

    def interactive_mode(self):
        """Chế độ chat tương tác"""
        print("🚀 RAG Chatbot Client Started!")
        print("-" * 50)

        # Kiểm tra kết nối
        if not self.check_health():
            print("❌ Không thể kết nối tới API. Hãy đảm bảo server đang chạy!")
            return

        print("\n💡 Commands:")
        print("  /help    - Hiển thị help")
        print("  /agents  - Danh sách agents")
        print("  /history - Xem lịch sử")
        print("  /clear   - Xóa lịch sử")
        print("  /health  - Kiểm tra API")
        print("  /quit    - Thoát")
        print("\n" + "=" * 50)

        while True:
            try:
                question = input("\n❓ Nhập câu hỏi (hoặc /help): ").strip()

                if not question:
                    continue

                # Commands
                if question == "/quit":
                    print("👋 Tạm biệt!")
                    break
                elif question == "/help":
                    print("\n💡 Commands available:")
                    print("  /help    - Hiển thị help")
                    print("  /agents  - Danh sách agents")
                    print("  /history - Xem lịch sử")
                    print("  /clear   - Xóa lịch sử")
                    print("  /health  - Kiểm tra API")
                    print("  /quit    - Thoát")
                elif question == "/agents":
                    self.list_agents()
                elif question == "/history":
                    self.show_history()
                elif question == "/clear":
                    self.clear_history()
                elif question == "/health":
                    self.check_health()
                elif question.startswith("/"):
                    print("❌ Command không hợp lệ. Gõ /help để xem danh sách commands.")
                else:
                    # Gửi câu hỏi
                    self.send_message(question)

            except KeyboardInterrupt:
                print("\n👋 Tạm biệt!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")


def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Single question mode
        client = RAGChatClient()
        question = " ".join(sys.argv[1:])
        if client.check_health():
            client.send_message(question)
    else:
        # Interactive mode
        client = RAGChatClient()
        client.interactive_mode()


if __name__ == "__main__":
    main()