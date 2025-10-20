from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import settings
import traceback

class LLMModel:
    def __init__(self):
        print(f"[LLMModel] Using model={settings.LLM_MODEL} base_url={getattr(settings, 'OLLAMA_URL', None)}")
        self.llm = OllamaLLM(
            model=settings.LLM_MODEL,
            base_url=getattr(settings, "OLLAMA_URL", "http://ollama:11434"),
            temperature=0.1,
        )
        self.output_parser = StrOutputParser()

    def invoke(self, prompt: str, **kwargs) -> str:
        try:
            resp = self.llm.invoke(prompt, **kwargs)
            return self.output_parser.parse(resp)
        except Exception as e:
            traceback.print_exc()
            return f"Lỗi xử lý: {str(e)}"

    def create_chain(self, template: str):
        prompt = PromptTemplate.from_template(template)
        return prompt | self.llm | self.output_parser

llm_model = LLMModel()
