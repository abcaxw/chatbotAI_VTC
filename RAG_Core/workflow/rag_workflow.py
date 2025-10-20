# RAG_Core/workflow/rag_workflow.py - PHẦN ĐẦU FILE

from typing import Dict, Any, List
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
import logging

from agents.supervisor import SupervisorAgent
from agents.faq_agent import FAQAgent
from agents.retriever_agent import RetrieverAgent
from agents.grader_agent import GraderAgent
from agents.generator_agent import GeneratorAgent
from agents.not_enough_info_agent import NotEnoughInfoAgent
from agents.chatter_agent import ChatterAgent
from agents.reporter_agent import ReporterAgent
from agents.other_agent import OtherAgent

logger = logging.getLogger(__name__)


# UPDATED STATE: Thêm các trường mới cho context processing
class ChatbotState(TypedDict):
    # Câu hỏi
    question: str  # Câu hỏi hiện tại (đã được contextualize)
    original_question: str  # Câu hỏi gốc từ user

    # Lịch sử
    history: List[Dict[str, str]]  # Lịch sử hội thoại

    # Context processing
    is_followup: bool  # Có phải follow-up question không
    context_summary: str  # Tóm tắt ngữ cảnh
    relevant_context: str  # Context liên quan từ lịch sử

    # Agent routing
    current_agent: str  # Agent hiện tại

    # Documents và references
    documents: List[Dict[str, Any]]  # Tài liệu từ retriever
    qualified_documents: List[Dict[str, Any]]  # Tài liệu đã được grader đánh giá
    references: List[Dict[str, Any]]  # References cho câu trả lời

    # Kết quả
    answer: str  # Câu trả lời
    status: str  # Trạng thái xử lý

    # Tracking
    iteration_count: int  # Số lần lặp trong workflow


class RAGWorkflow:
    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.faq_agent = FAQAgent()
        self.retriever_agent = RetrieverAgent()
        self.grader_agent = GraderAgent()
        self.generator_agent = GeneratorAgent()
        self.not_enough_info_agent = NotEnoughInfoAgent()
        self.chatter_agent = ChatterAgent()
        self.reporter_agent = ReporterAgent()
        self.other_agent = OtherAgent()

        self.workflow = self._create_workflow()

    def _create_workflow(self):
        """Tạo workflow graph"""
        workflow = StateGraph(ChatbotState)

        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("faq", self._faq_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("grader", self._grader_node)
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("not_enough_info", self._not_enough_info_node)
        workflow.add_node("chatter", self._chatter_node)
        workflow.add_node("reporter", self._reporter_node)
        workflow.add_node("other", self._other_node)

        # Set entry point
        workflow.set_entry_point("supervisor")

        # Add edges
        workflow.add_conditional_edges(
            "supervisor",
            self._route_supervisor,
            {
                "FAQ": "faq",
                "CHATTER": "chatter",
                "REPORTER": "reporter",
                "OTHER": "other"
            }
        )

        workflow.add_conditional_edges(
            "faq",
            self._route_next_agent,
            {
                "RETRIEVER": "retriever",
                "end": "__end__"
            }
        )

        workflow.add_conditional_edges(
            "retriever",
            self._route_next_agent,
            {
                "GRADER": "grader",
                "NOT_ENOUGH_INFO": "not_enough_info"
            }
        )

        workflow.add_conditional_edges(
            "grader",
            self._route_next_agent,
            {
                "GENERATOR": "generator",
                "NOT_ENOUGH_INFO": "not_enough_info"
            }
        )

        # Terminal nodes
        workflow.add_edge("generator", "__end__")
        workflow.add_edge("not_enough_info", "__end__")
        workflow.add_edge("chatter", "__end__")
        workflow.add_edge("reporter", "__end__")
        workflow.add_edge("other", "__end__")

        return workflow.compile()

    def _supervisor_node(self, state: ChatbotState) -> ChatbotState:
        """Node xử lý supervisor với context processing"""
        try:
            # Phân loại và xử lý context
            classification = self.supervisor.classify_request(
                state["question"],
                state.get("history", [])
            )

            # Cập nhật state với thông tin đầy đủ
            state["current_agent"] = classification["agent"]
            state["question"] = classification["contextualized_question"]  # Cập nhật câu hỏi
            state["is_followup"] = classification.get("is_followup", False)
            state["context_summary"] = classification.get("context_summary", "")
            state["iteration_count"] = state.get("iteration_count", 0) + 1

            logger.info(f"Supervisor chose agent: {classification['agent']}")
            logger.info(f"Contextualized question: {state['question']}")
            logger.info(f"Is follow-up: {state.get('is_followup', False)}")

            return state

        except Exception as e:
            logger.error(f"Error in supervisor node: {e}", exc_info=True)
            state["current_agent"] = "OTHER"
            return state

    # ... (các node khác giữ nguyên, chỉ cần đảm bảo sử dụng state["question"])

    def run(self, question: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Chạy workflow với context processing"""
        try:
            initial_state = ChatbotState(
                question=question,
                original_question=question,  # Lưu câu hỏi gốc
                history=history or [],
                is_followup=False,
                context_summary="",
                relevant_context="",
                current_agent="supervisor",
                documents=[],
                qualified_documents=[],
                references=[],
                answer="",
                status="",
                iteration_count=0
            )

            final_state = self.workflow.invoke(initial_state)

            return {
                "answer": final_state.get("answer", "Lỗi xử lý"),
                "references": final_state.get("references", []),
                "status": final_state.get("status", "ERROR"),
                "original_question": final_state.get("original_question", question),
                "processed_question": final_state.get("question", question),
                "is_followup": final_state.get("is_followup", False),
                "context_summary": final_state.get("context_summary", "")
            }

        except Exception as e:
            logger.error(f"Error in workflow: {e}", exc_info=True)
            return {
                "answer": "Xin lỗi, hệ thống gặp sự cố. Vui lòng thử lại sau.",
                "references": [],
                "status": "ERROR",
                "original_question": question,
                "processed_question": question,
                "is_followup": False,
                "context_summary": ""
            }

    def _faq_node(self, state: ChatbotState) -> ChatbotState:
        """Node xử lý FAQ - sử dụng contextualized question"""
        try:
            result = self.faq_agent.process(
                state["question"],  # Sử dụng câu hỏi đã được contextualize
                is_followup=state.get("is_followup", False),
                context=state.get("context_summary", "")
            )

            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = result.get("next_agent", "end")

            return state

        except Exception as e:
            logger.error(f"Error in FAQ node: {e}", exc_info=True)
            state["current_agent"] = "RETRIEVER"
            return state

    def _retriever_node(self, state: ChatbotState) -> ChatbotState:
        """Node xử lý retriever - sử dụng contextualized question"""
        try:
            result = self.retriever_agent.process(
                state["question"]  # Sử dụng câu hỏi đã được contextualize
            )

            state["status"] = result["status"]
            state["documents"] = result.get("documents", [])
            state["current_agent"] = result.get("next_agent", "GRADER")

            return state

        except Exception as e:
            logger.error(f"Error in retriever node: {e}", exc_info=True)
            state["current_agent"] = "REPORTER"
            return state

    def _grader_node(self, state: ChatbotState) -> ChatbotState:
        """Node xử lý grader"""
        try:
            result = self.grader_agent.process(
                state["question"],  # Contextualized question
                state.get("documents", [])
            )

            state["status"] = result["status"]
            state["qualified_documents"] = result.get("qualified_documents", [])
            state["references"] = result.get("references", [])
            state["current_agent"] = result.get("next_agent", "GENERATOR")

            return state

        except Exception as e:
            logger.error(f"Error in grader node: {e}", exc_info=True)
            state["current_agent"] = "NOT_ENOUGH_INFO"
            return state

    def _generator_node(self, state: ChatbotState) -> ChatbotState:
        """Node xử lý generator - truyền context vào"""
        try:
            result = self.generator_agent.process(
                question=state["question"],
                documents=state.get("qualified_documents", []),
                references=state.get("references", []),
                history=state.get("history", []),
                is_followup=state.get("is_followup", False),
                context_summary=state.get("context_summary", "")
            )

            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"

            return state

        except Exception as e:
            logger.error(f"Error in generator node: {e}", exc_info=True)
            state["answer"] = "Lỗi tạo câu trả lời"
            state["current_agent"] = "end"
            return state

    def _not_enough_info_node(self, state: ChatbotState) -> ChatbotState:
        """Node xử lý not enough info"""
        try:
            result = self.not_enough_info_agent.process(
                state["question"],
                is_followup=state.get("is_followup", False)
            )

            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"

            return state

        except Exception as e:
            logger.error(f"Error in not_enough_info node: {e}", exc_info=True)
            state["answer"] = "Không tìm thấy thông tin phù hợp"
            return state

    def _chatter_node(self, state: ChatbotState) -> ChatbotState:
        """Node xử lý chatter"""
        try:
            result = self.chatter_agent.process(
                state["question"],
                state.get("history", [])
            )

            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"

            return state

        except Exception as e:
            logger.error(f"Error in chatter node: {e}", exc_info=True)
            state["answer"] = "Tôi hiểu cảm xúc của bạn. Vui lòng liên hệ hotline để được hỗ trợ."
            return state

    def _reporter_node(self, state: ChatbotState) -> ChatbotState:
        """Node xử lý reporter"""
        try:
            result = self.reporter_agent.process(state["question"])

            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"

            return state

        except Exception as e:
            logger.error(f"Error in reporter node: {e}", exc_info=True)
            state["answer"] = "Hệ thống đang bảo trì"
            return state

    def _other_node(self, state: ChatbotState) -> ChatbotState:
        """Node xử lý other"""
        try:
            result = self.other_agent.process(state["question"])

            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"

            return state

        except Exception as e:
            logger.error(f"Error in other node: {e}", exc_info=True)
            state["answer"] = "Đây không phải là tác vụ của tôi"
            return state

    def _route_supervisor(self, state: ChatbotState) -> str:
        """Route từ supervisor"""
        return state["current_agent"]

    def _route_next_agent(self, state: ChatbotState) -> str:
        """Route đến agent tiếp theo"""
        return state.get("current_agent", "end")