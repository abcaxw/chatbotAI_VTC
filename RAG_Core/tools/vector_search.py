# RAG_Core/tools/vector_search.py (NO FALLBACK VERSION)

from langchain_core.tools import tool
from typing import List, Dict, Any
import numpy as np
from models.embedding_model import embedding_model
from database.milvus_client import milvus_client
from sentence_transformers import CrossEncoder
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# RERANKING MODEL CONFIGURATION - NO FALLBACK
# ============================================================================

def load_reranker_model():
    """
    Load reranker model - KHÔNG CÓ FALLBACK
    Nếu model config fail → raise Exception
    """
    model_name = getattr(settings, 'RERANKER_MODEL', 'BAAI/bge-reranker-base')
    max_length = getattr(settings, 'RERANKER_MAX_LENGTH', 512)

    logger.info(f"Loading reranker model: {model_name}")
    logger.info(f"Max length: {max_length}")

    try:
        model = CrossEncoder(model_name, max_length=max_length)
        logger.info(f"✅ Reranker model loaded successfully: {model_name}")
        return model, model_name

    except Exception as e:
        error_msg = f"❌ CRITICAL: Failed to load reranker model '{model_name}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


# Load model at startup - fail fast if error
try:
    reranker_model, reranker_model_name = load_reranker_model()
    logger.info(f"🎯 Active reranker: {reranker_model_name}")
except Exception as e:
    logger.critical(f"❌ Cannot start without reranker model: {e}")
    reranker_model = None
    reranker_model_name = None
    # Re-raise để stop service nếu trong production mode
    if getattr(settings, 'FAIL_FAST_ON_MODEL_ERROR', True):
        raise


# ============================================================================
# FAQ RERANKING - NO FALLBACK
# ============================================================================

@tool
def rerank_faq(query: str, faq_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rerank FAQ results - KHÔNG CÓ FALLBACK
    Nếu reranker fail → raise Exception
    """
    if not faq_results:
        logger.warning("No FAQ to rerank")
        return []

    if reranker_model is None:
        error_msg = "❌ Reranker model not available - cannot proceed"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        # Prepare pairs với nhiều variants
        pairs = []
        faq_variants = []

        for idx, faq in enumerate(faq_results):
            question = faq.get('question', '').strip()
            answer = faq.get('answer', '').strip()

            if not question:
                continue

            # Variant 1: Query vs Question only
            pairs.append([query, question])
            faq_variants.append(('question_only', idx))

            # Variant 2: Query vs Question+Answer (truncate nếu quá dài)
            combined = f"{question} {answer}"
            if len(combined) > 500:
                combined = combined[:500]
            pairs.append([query, combined])
            faq_variants.append(('question_answer', idx))

            # Variant 3: Query vs Answer only
            if len(answer) > 400:
                answer = answer[:400]
            pairs.append([query, answer])
            faq_variants.append(('answer_only', idx))

        if not pairs:
            logger.warning("No valid FAQ pairs created")
            return []

        # Predict scores với batch processing
        logger.info(f"Reranking {len(pairs)} FAQ variants using {reranker_model_name}")

        batch_size = getattr(settings, 'RERANKER_BATCH_SIZE', 32)
        scores = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_scores = reranker_model.predict(batch)
            scores.extend(batch_scores)

        # Aggregate scores
        faq_scores = {}
        for i, (variant_type, faq_idx) in enumerate(faq_variants):
            if faq_idx not in faq_scores:
                faq_scores[faq_idx] = {}
            faq_scores[faq_idx][variant_type] = float(scores[i])

        # Calculate final scores với weighted average
        weights = {
            'question_only': getattr(settings, 'FAQ_QUESTION_WEIGHT', 0.5),
            'question_answer': getattr(settings, 'FAQ_QA_WEIGHT', 0.3),
            'answer_only': getattr(settings, 'FAQ_ANSWER_WEIGHT', 0.2)
        }

        reranked_faq = []
        for faq_idx, faq in enumerate(faq_results):
            if faq_idx not in faq_scores:
                continue

            variant_scores = faq_scores[faq_idx]

            final_score = sum(
                variant_scores.get(variant, 0) * weight
                for variant, weight in weights.items()
            )

            # Bonus for consistent high scores
            consistency_threshold = getattr(settings, 'FAQ_CONSISTENCY_THRESHOLD', 0.6)
            if all(variant_scores.get(v, 0) > consistency_threshold for v in weights.keys()):
                bonus = getattr(settings, 'FAQ_CONSISTENCY_BONUS', 1.1)
                final_score *= bonus
                logger.debug(f"FAQ {faq_idx} received consistency bonus")

            faq_copy = faq.copy()
            faq_copy['rerank_score'] = final_score
            faq_copy['rerank_details'] = variant_scores
            faq_copy['reranker_model'] = reranker_model_name
            reranked_faq.append(faq_copy)

        # Sort by final score
        reranked_faq.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

        logger.info(f"✅ Reranked {len(reranked_faq)} FAQs. Best: {reranked_faq[0].get('rerank_score', 0):.3f}")

        return reranked_faq

    except Exception as e:
        error_msg = f"❌ FAQ reranking failed: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


# ============================================================================
# DOCUMENT RERANKING - NO FALLBACK
# ============================================================================

@tool
def rerank_documents(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rerank documents - KHÔNG CÓ FALLBACK
    Nếu reranker fail → raise Exception
    """
    if not documents:
        logger.warning("No documents to rerank")
        return []

    if reranker_model is None:
        error_msg = "❌ Reranker model not available - cannot proceed"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        # Prepare pairs với text truncation
        pairs = []
        for doc in documents:
            doc_text = doc.get('description', '') or doc.get('answer', '') or ''

            # Truncate để model xử lý tốt hơn
            if len(doc_text) > 500:
                doc_text = doc_text[:500]

            pairs.append([query, doc_text])

        # Batch prediction
        logger.info(f"Reranking {len(pairs)} documents using {reranker_model_name}")
        batch_size = getattr(settings, 'RERANKER_BATCH_SIZE', 32)

        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_scores = reranker_model.predict(batch)
            scores.extend(batch_scores)

        # Add rerank_score
        reranked_docs = []
        for i, doc in enumerate(documents):
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = float(scores[i])
            doc_copy['reranker_model'] = reranker_model_name
            reranked_docs.append(doc_copy)

        # Sort by rerank_score
        reranked_docs.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

        logger.info(f"✅ Reranked {len(reranked_docs)} documents. Best: {reranked_docs[0].get('rerank_score', 0):.3f}")
        return reranked_docs

    except Exception as e:
        error_msg = f"❌ Document reranking failed: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


# ============================================================================
# SEARCH FUNCTIONS
# ============================================================================

def pad_vector_to_dimension(vector: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad vector with zeros to reach target dimension"""
    current_dim = vector.shape[0] if vector.ndim == 1 else vector.shape[1]

    if current_dim >= target_dim:
        return vector[:target_dim] if vector.ndim == 1 else vector[:, :target_dim]

    if vector.ndim == 1:
        padding = np.zeros(target_dim - current_dim, dtype=vector.dtype)
        return np.concatenate([vector, padding])
    else:
        padding = np.zeros((vector.shape[0], target_dim - current_dim), dtype=vector.dtype)
        return np.concatenate([vector, padding], axis=1)


def safe_encode_and_fix_dimension(query: str, target_collection: str, target_field: str) -> np.ndarray:
    """Encode query and automatically fix dimension if needed"""
    try:
        query_vector = embedding_model.encode_single(query)
        expected_dim = milvus_client._get_collection_dimension(target_collection, target_field)

        if expected_dim > 0 and query_vector.shape[0] != expected_dim:
            logger.warning(
                f"Dimension mismatch. Expected: {expected_dim}, Got: {query_vector.shape[0]}. Auto-fixing..."
            )
            query_vector = pad_vector_to_dimension(query_vector, expected_dim)
            logger.info(f"Vector dimension fixed to {expected_dim}")

        return query_vector

    except Exception as e:
        logger.error(f"Error encoding query: {str(e)}")
        raise


@tool
def search_documents(query: str) -> List[Dict[str, Any]]:
    """Tìm kiếm tài liệu liên quan đến câu hỏi"""
    try:
        query_vector = safe_encode_and_fix_dimension(
            query,
            settings.DOCUMENT_COLLECTION,
            "description_vector"
        )

        results = milvus_client.search_documents(query_vector, settings.TOP_K)
        return results

    except Exception as e:
        logger.error(f"Error in search_documents: {str(e)}")
        raise


@tool
def search_faq(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """Tìm kiếm FAQ với top_k cao hơn để reranking có nhiều lựa chọn"""
    try:
        if top_k is None:
            top_k = getattr(settings, 'FAQ_TOP_K', 10)

        query_vector = safe_encode_and_fix_dimension(
            query,
            settings.FAQ_COLLECTION,
            "question_vector"
        )

        results = milvus_client.search_faq(query_vector, top_k)
        logger.info(f"Retrieved {len(results)} FAQ candidates for reranking")

        return results

    except Exception as e:
        logger.error(f"Error in search_faq: {str(e)}")
        raise


# ============================================================================
# DATABASE CONNECTION CHECK
# ============================================================================

@tool
def check_database_connection() -> Dict[str, Any]:
    """Kiểm tra kết nối cơ sở dữ liệu"""
    try:
        is_connected = milvus_client.check_connection()

        result = {
            "connected": is_connected,
            "message": "Kết nối bình thường" if is_connected else "Mất kết nối cơ sở dữ liệu",
            "reranker_model": reranker_model_name,
            "reranker_status": "loaded" if reranker_model else "not_loaded"
        }

        if is_connected:
            try:
                test_vector = embedding_model.encode_single("test")
                embedding_dim = test_vector.shape[0]

                doc_dim = milvus_client._get_collection_dimension(
                    settings.DOCUMENT_COLLECTION, "description_vector"
                )
                faq_dim = milvus_client._get_collection_dimension(
                    settings.FAQ_COLLECTION, "question_vector"
                )

                result["dimension_info"] = {
                    "embedding_model_dimension": embedding_dim,
                    "document_collection_dimension": doc_dim,
                    "faq_collection_dimension": faq_dim,
                    "dimension_match": {
                        "documents": embedding_dim == doc_dim,
                        "faq": embedding_dim == faq_dim
                    }
                }

                if embedding_dim != doc_dim or embedding_dim != faq_dim:
                    result["warning"] = "Dimension mismatch detected - using auto-fix with zero padding"

            except Exception as dim_error:
                result["dimension_check_error"] = str(dim_error)

        return result

    except Exception as e:
        return {
            "connected": False,
            "message": f"Lỗi kiểm tra kết nối: {str(e)}"
        }


# ============================================================================
# DIAGNOSTIC TOOLS
# ============================================================================

@tool
def diagnose_vector_dimensions() -> Dict[str, Any]:
    """Công cụ chẩn đoán chi tiết về dimension mismatch"""
    try:
        diagnosis = {
            "embedding_model": {},
            "collections": {},
            "reranker_info": {
                "model_name": reranker_model_name,
                "status": "loaded" if reranker_model else "FAILED",
                "fail_fast_mode": getattr(settings, 'FAIL_FAST_ON_MODEL_ERROR', True)
            },
            "recommendations": []
        }

        # Check embedding model
        try:
            test_vector = embedding_model.encode_single("test query")
            diagnosis["embedding_model"] = {
                "dimension": test_vector.shape[0],
                "dtype": str(test_vector.dtype),
                "sample_values": test_vector[:5].tolist()
            }
        except Exception as e:
            diagnosis["embedding_model"]["error"] = str(e)

        # Check collections
        collections = [
            (settings.DOCUMENT_COLLECTION, "description_vector"),
            (settings.FAQ_COLLECTION, "question_vector")
        ]

        for collection_name, vector_field in collections:
            try:
                info = milvus_client.get_collection_info(collection_name)
                dim = milvus_client._get_collection_dimension(collection_name, vector_field)

                diagnosis["collections"][collection_name] = {
                    "vector_field": vector_field,
                    "expected_dimension": dim,
                    "schema_info": info
                }
            except Exception as e:
                diagnosis["collections"][collection_name] = {"error": str(e)}

        # Generate recommendations
        embedding_dim = diagnosis["embedding_model"].get("dimension", 0)

        for collection_name, info in diagnosis["collections"].items():
            expected_dim = info.get("expected_dimension", 0)

            if embedding_dim > 0 and expected_dim > 0 and embedding_dim != expected_dim:
                diagnosis["recommendations"].append(
                    f"Collection '{collection_name}': embedding={embedding_dim}D, expected={expected_dim}D - "
                    f"Currently using zero padding/truncation"
                )

        if not diagnosis["recommendations"]:
            diagnosis["recommendations"].append("All dimensions match correctly!")

        return diagnosis

    except Exception as e:
        return {"error": f"Lỗi chẩn đoán: {str(e)}"}