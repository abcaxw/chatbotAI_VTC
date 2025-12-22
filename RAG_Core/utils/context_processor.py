# RAG_Core/utils/context_processor_optimized.py

from typing import List, Dict, Any, Optional
from models.llm_model import llm_model
import logging
import re
from collections import deque

logger = logging.getLogger(__name__)


class ContextProcessor:
    """
    Xử lý context với các kỹ thuật tối ưu:
    1. Sliding window với context caching
    2. Lightweight pattern matching trước khi gọi LLM
    3. Embedding-based context similarity (optional)
    4. Parallel processing hints
    """

    def __init__(self, max_context_length: int = 500, cache_size: int = 10):
        self.max_context_length = max_context_length
        self.cache_size = cache_size

        # Cache recent context analysis để tránh re-compute
        self.context_cache = deque(maxlen=cache_size)

        # Lightweight patterns (KHÔNG phải rule-based path)
        self.followup_indicators = {
            'pronouns': r'\b(nó|cái đó|điều đó|phần đó|ở đó)\b',
            'ordinals': r'\b(thứ \d+|đầu tiên|thứ hai|thứ ba|cuối cùng)\b',
            'continuations': r'\b(tiếp theo|còn|thêm|chi tiết|cụ thể|ví dụ)\b',
            'short_query': lambda q: len(q.split()) < 5
        }

        # Simplified prompt cho LLM (nếu cần)
        self.fast_context_prompt = """Ngắn gọn: Chuyển câu hỏi này thành câu độc lập dựa trên context.

Context gần nhất: {context}
Câu hỏi: {question}

Trả về câu hỏi đã làm rõ (không giải thích):"""

    def extract_context_from_history(
            self,
            history: List,
            current_question: str
    ) -> Dict[str, Any]:
        """
        OPTIMIZED: Phân tích nhanh với early exit
        """
        try:
            # Step 1: Normalize history
            normalized_history = self._normalize_history(history)

            if not normalized_history:
                return self._create_standalone_result(current_question)

            # Step 2: Lightweight check (không gọi LLM)
            is_likely_followup = self._quick_followup_check(current_question)

            if not is_likely_followup:
                # Early exit - không cần context processing
                return self._create_standalone_result(current_question)

            # Step 3: Extract relevant context (sliding window)
            recent_context = self._extract_sliding_window(normalized_history)

            # Step 4: Check cache
            cache_key = f"{current_question}_{recent_context[:100]}"
            cached_result = self._check_cache(cache_key)
            if cached_result:
                logger.info("✅ Using cached context result")
                return cached_result

            # Step 5: LLM contextualization (chỉ khi cần)
            contextualized = self._fast_llm_contextualize(
                current_question,
                recent_context
            )

            result = {
                "original_question": current_question,
                "contextualized_question": contextualized,
                "is_followup": True,
                "relevant_context": recent_context[:200]
            }

            # Cache result
            self._add_to_cache(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Context processing error: {e}")
            return self._create_standalone_result(current_question)

    def _quick_followup_check(self, question: str) -> bool:
        """
        Lightweight pattern check - KHÔNG phải rule-based routing
        Chỉ để quyết định có cần context processing hay không
        """
        q_lower = question.lower().strip()

        # Check patterns
        if re.search(self.followup_indicators['pronouns'], q_lower):
            return True
        if re.search(self.followup_indicators['ordinals'], q_lower):
            return True
        if re.search(self.followup_indicators['continuations'], q_lower):
            return True
        if self.followup_indicators['short_query'](q_lower):
            return True

        return False

    def _extract_sliding_window(
            self,
            history: List[Dict[str, str]],
            window_size: int = 2  # số turn
    ) -> str:
        """
        Sliding window: Chỉ lấy context gần nhất, tránh overhead
        """
        if not history:
            return ""

        # Lấy N turn gần nhất
        recent = history[-(window_size * 2):] if len(history) > window_size * 2 else history

        # Format compact
        context_parts = []
        for msg in recent:
            role = "Q" if msg["role"] == "user" else "A"
            content = msg["content"][:150]  # Truncate
            context_parts.append(f"{role}: {content}")

        return " | ".join(context_parts)

    def _fast_llm_contextualize(
            self,
            question: str,
            context: str
    ) -> str:
        """
        Simplified LLM call với prompt ngắn gọn
        """
        try:
            if len(context) > self.max_context_length:
                context = context[:self.max_context_length]

            prompt = self.fast_context_prompt.format(
                context=context,
                question=question
            )

            # Invoke với timeout ngầm
            result = llm_model.invoke(prompt)

            # Fallback nếu LLM trả về rỗng
            if not result or len(result.strip()) < 5:
                return question

            return result.strip()

        except Exception as e:
            logger.warning(f"LLM contextualization failed: {e}")
            return question

    def _normalize_history(self, history: List) -> List[Dict[str, str]]:
        """Fast normalization"""
        if not history:
            return []

        normalized = []
        for msg in history:
            if isinstance(msg, dict):
                normalized.append({
                    "role": msg.get("role", ""),
                    "content": msg.get("content", "")
                })
            else:
                normalized.append({
                    "role": getattr(msg, "role", ""),
                    "content": getattr(msg, "content", "")
                })

        return normalized

    def _check_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Check if result exists in cache"""
        for cached_key, cached_result in self.context_cache:
            if cached_key == key:
                return cached_result
        return None

    def _add_to_cache(self, key: str, result: Dict[str, Any]):
        """Add result to cache"""
        self.context_cache.append((key, result))

    def _create_standalone_result(self, question: str) -> Dict[str, Any]:
        """Create result for standalone question"""
        return {
            "original_question": question,
            "contextualized_question": question,
            "is_followup": False,
            "relevant_context": ""
        }


# Global instance
context_processor = ContextProcessor()