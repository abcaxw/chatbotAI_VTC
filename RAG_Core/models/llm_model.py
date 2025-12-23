# RAG_Core/models/llm_model.py - STREAMING VERSION

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import settings
import traceback
from typing import Iterator, Optional
import logging

logger = logging.getLogger(__name__)


class LLMModel:
    def __init__(self):
        logger.info(f"[LLMModel] Using model={settings.LLM_MODEL} base_url={getattr(settings, 'OLLAMA_URL', None)}")

        # Initialize Ollama LLM
        self.llm = OllamaLLM(
            model=settings.LLM_MODEL,
            base_url=getattr(settings, "OLLAMA_URL", "http://ollama:11434"),
            temperature=0.1,
        )
        self.output_parser = StrOutputParser()

    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Non-streaming invoke - giữ nguyên cho compatibility
        """
        try:
            resp = self.llm.invoke(prompt, **kwargs)
            return self.output_parser.parse(resp)
        except Exception as e:
            traceback.print_exc()
            return f"Lỗi xử lý: {str(e)}"

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Streaming invoke - trả về generator để stream từng chunk

        Usage:
            for chunk in llm_model.stream(prompt):
                print(chunk, end='', flush=True)
        """
        try:
            logger.info(f"Starting stream generation for prompt: {prompt[:100]}...")

            # Use Ollama's streaming capability
            for chunk in self.llm.stream(prompt, **kwargs):
                yield chunk

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            traceback.print_exc()
            yield f"\n\n[Lỗi streaming: {str(e)}]"

    async def astream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Async streaming invoke - cho FastAPI async endpoints

        Usage:
            async for chunk in llm_model.astream(prompt):
                yield chunk
        """
        try:
            logger.info(f"Starting async stream generation for prompt: {prompt[:100]}...")

            # Use async streaming
            async for chunk in self.llm.astream(prompt, **kwargs):
                yield chunk

        except Exception as e:
            logger.error(f"Async streaming error: {e}")
            traceback.print_exc()
            yield f"\n\n[Lỗi streaming: {str(e)}]"

    def create_chain(self, template: str):
        """Create a non-streaming chain"""
        prompt = PromptTemplate.from_template(template)
        return prompt | self.llm | self.output_parser


# Global instance
llm_model = LLMModel()