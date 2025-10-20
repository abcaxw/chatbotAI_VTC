# agents/grader_agent.py
from typing import Dict, Any, List
from tools.vector_search import rerank_documents
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class GraderAgent:
    def __init__(self):
        self.name = "GRADER"
        # Ngưỡng cho reranking score (thường cao hơn similarity)
        self.reranking_threshold = 0.6  # Cross-encoder score thường thấp hơn

    def process(self, question: str, documents: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Đánh giá chất lượng tài liệu bằng reranking model"""
        try:
            if not documents:
                logger.warning("No documents to grade")
                return {
                    "status": "INSUFFICIENT",
                    "qualified_documents": [],
                    "references": [],
                    "next_agent": "NOT_ENOUGH_INFO"
                }

            # Bước 1: Rerank documents
            logger.info(f"Reranking {len(documents)} documents for question: {question[:50]}...")
            reranked_docs = rerank_documents.invoke({
                "query": question,
                "documents": documents
            })

            if not reranked_docs or "error" in str(reranked_docs):
                logger.error("Reranking failed, using fallback grading")
                return self._fallback_grading(documents)

            # Bước 2: Lọc documents theo 2 tiêu chí
            qualified_docs = []
            for doc in reranked_docs:
                rerank_score = doc.get("rerank_score", 0)
                similarity_score = doc.get("similarity_score", 0)

                # Kiểm tra CẢ HAI điểm số
                if (rerank_score >= self.reranking_threshold and
                        similarity_score >= settings.SIMILARITY_THRESHOLD):
                    qualified_docs.append(doc)
                    logger.debug(
                        f"Doc {doc.get('document_id')}: "
                        f"similarity={similarity_score:.3f}, rerank={rerank_score:.3f} ✓"
                    )
                else:
                    logger.debug(
                        f"Doc {doc.get('document_id')}: "
                        f"similarity={similarity_score:.3f}, rerank={rerank_score:.3f} ✗"
                    )

            # Bước 3: Quyết định
            if qualified_docs:
                logger.info(f"Found {len(qualified_docs)} qualified documents")
                return {
                    "status": "SUFFICIENT",
                    "qualified_documents": qualified_docs,
                    "references": [
                        {
                            "document_id": doc.get("document_id"),
                            "type": "DOCUMENT",
                            "rerank_score": round(doc.get("rerank_score", 0), 4),
                            "similarity_score": round(doc.get("similarity_score", 0), 4)
                        }
                        for doc in qualified_docs
                    ],
                    "next_agent": "GENERATOR"
                }
            else:
                logger.warning("No documents passed grading thresholds")
                return {
                    "status": "INSUFFICIENT",
                    "qualified_documents": [],
                    "references": [],
                    "next_agent": "NOT_ENOUGH_INFO"
                }

        except Exception as e:
            logger.error(f"Error in grader agent: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "qualified_documents": [],
                "references": [],
                "next_agent": "NOT_ENOUGH_INFO"
            }

    def _fallback_grading(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Phương pháp dự phòng khi reranking thất bại.
        Chỉ dùng similarity score từ vector search.
        """
        logger.info("Using fallback grading with similarity scores only")

        qualified_docs = [
            doc for doc in documents
            if doc.get("similarity_score", 0) >= settings.SIMILARITY_THRESHOLD
        ]

        if qualified_docs:
            return {
                "status": "SUFFICIENT",
                "qualified_documents": qualified_docs,
                "references": [
                    {
                        "document_id": doc.get("document_id"),
                        "type": "DOCUMENT",
                        "similarity_score": round(doc.get("similarity_score", 0), 4)
                    }
                    for doc in qualified_docs
                ],
                "next_agent": "GENERATOR"
            }
        else:
            return {
                "status": "INSUFFICIENT",
                "qualified_documents": [],
                "references": [],
                "next_agent": "NOT_ENOUGH_INFO"
            }