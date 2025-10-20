from typing import Dict, Any, List
from models.llm_model import llm_model
from tools.vector_search import search_documents
from config.settings import settings


class RetrieverAgent:
    def __init__(self):
        self.name = "RETRIEVER"
        self.tools = [search_documents]

    def process(self, question: str, **kwargs) -> Dict[str, Any]:
        """Tìm kiếm tài liệu liên quan đến câu hỏi"""
        try:
            # Tìm kiếm tài liệu
            search_results = search_documents.invoke({"query": question})

            if not search_results or "error" in str(search_results):
                return {
                    "status": "ERROR",
                    "documents": [],
                    "next_agent": "NOT_ENOUGH_INFO"
                }

            # Lọc kết quả theo similarity threshold
            relevant_docs = [
                doc for doc in search_results
                if doc.get("similarity_score", 0) > settings.SIMILARITY_THRESHOLD
            ]

            if not relevant_docs:
                return {
                    "status": "NOT_FOUND",
                    "documents": search_results,  # Trả về tất cả để GRADER đánh giá
                    "next_agent": "GRADER"
                }

            return {
                "status": "SUCCESS",
                "documents": relevant_docs,
                "next_agent": "GRADER"
            }

        except Exception as e:
            return {
                "status": "ERROR",
                "documents": [],
                "next_agent": "REPORTER"
            }