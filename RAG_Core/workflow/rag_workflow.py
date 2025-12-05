# RAG_Core/workflow/rag_workflow.py - REFACTORED VERSION

from typing import Dict, Any, List
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

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


class ChatbotState(TypedDict):
    question: str
    original_question: str
    history: List[Dict[str, str]]
    is_followup: bool
    context_summary: str
    relevant_context: str
    current_agent: str
    documents: List[Dict[str, Any]]
    qualified_documents: List[Dict[str, Any]]
    references: List[Dict[str, Any]]
    answer: str
    status: str
    iteration_count: int
    supervisor_classification: Dict[str, Any]
    faq_result: Dict[str, Any]
    retriever_result: Dict[str, Any]
    parallel_mode: bool


class RAGWorkflow:
    def __init__(self):
        # Initialize all agents
        self.supervisor = SupervisorAgent()
        self.faq_agent = FAQAgent()
        self.retriever_agent = RetrieverAgent()
        self.grader_agent = GraderAgent()
        self.generator_agent = GeneratorAgent()
        self.not_enough_info_agent = NotEnoughInfoAgent()
        self.chatter_agent = ChatterAgent()
        self.reporter_agent = ReporterAgent()
        self.other_agent = OtherAgent()

        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="RAG-Worker")

        self.workflow = self._create_workflow()

    def _create_workflow(self):
        """Tạo workflow graph với parallel execution"""
        workflow = StateGraph(ChatbotState)

        # Add nodes
        workflow.add_node("parallel_execution", self._parallel_execution_node)
        workflow.add_node("decision_router", self._decision_router_node)
        workflow.add_node("grader", self._grader_node)
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("not_enough_info", self._not_enough_info_node)
        workflow.add_node("chatter", self._chatter_node)
        workflow.add_node("reporter", self._reporter_node)
        workflow.add_node("other", self._other_node)

        # Set entry point
        workflow.set_entry_point("parallel_execution")

        # Flow: parallel → decision
        workflow.add_edge("parallel_execution", "decision_router")

        # Decision router routes to appropriate agent
        workflow.add_conditional_edges(
            "decision_router",
            self._route_after_decision,
            {
                "GRADER": "grader",
                "CHATTER": "chatter",
                "REPORTER": "reporter",
                "OTHER": "other",
                "end": "__end__"
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

    def _parallel_execution_node(self, state: ChatbotState) -> ChatbotState:
        """Chạy song song: Supervisor + FAQ + RETRIEVER"""
        question = state["question"]
        history = state.get("history", [])

        logger.info("🚀 Starting parallel execution: Supervisor + FAQ + RETRIEVER")

        # Submit all tasks
        future_supervisor = self.executor.submit(
            self._safe_execute_supervisor,
            question,
            history
        )

        future_faq = self.executor.submit(
            self._safe_execute_faq,
            question,
            history
        )

        future_retriever = self.executor.submit(
            self._safe_execute_retriever,
            question
        )

        # Collect results with timeout handling
        supervisor_result = self._get_result_with_timeout(
            future_supervisor,
            timeout=20,  # Tăng timeout cho SupervisorAgent (có LLM + context processing)
            default={
                "agent": "FAQ",  # Safe fallback
                "contextualized_question": question,
                "is_followup": False,
                "context_summary": ""
            },
            name="Supervisor"
        )

        faq_result = self._get_result_with_timeout(
            future_faq,
            timeout=10,
            default={"status": "ERROR", "answer": "", "references": []},
            name="FAQ"
        )

        retriever_result = self._get_result_with_timeout(
            future_retriever,
            timeout=10,
            default={"status": "ERROR", "documents": []},
            name="RETRIEVER"
        )

        logger.info("✅ All parallel tasks completed")

        # Update state với kết quả từ Supervisor
        state["supervisor_classification"] = supervisor_result
        state["question"] = supervisor_result.get("contextualized_question", question)
        state["is_followup"] = supervisor_result.get("is_followup", False)
        state["context_summary"] = supervisor_result.get("context_summary", "")

        # Store FAQ và Retriever results
        state["faq_result"] = faq_result
        state["retriever_result"] = retriever_result
        state["parallel_mode"] = True
        state["iteration_count"] = state.get("iteration_count", 0) + 1

        logger.info(f"📝 Contextualized question: {state['question'][:100]}")
        logger.info(f"🔄 Is follow-up: {state['is_followup']}")

        return state

    def _get_result_with_timeout(self, future, timeout: float, default: Dict, name: str) -> Dict:
        """Get result with timeout and fallback"""
        try:
            result = future.result(timeout=timeout)
            logger.info(f"✅ {name} completed successfully")
            return result
        except FutureTimeoutError:
            logger.warning(f"⏱️ {name} timeout after {timeout}s, using fallback")
            return default
        except Exception as e:
            logger.error(f"❌ {name} error: {e}")
            return default

    def _safe_execute_supervisor(self, question: str, history: List) -> Dict[str, Any]:
        """
        Execute SupervisorAgent.classify_request() với error handling
        ✅ SỬ DỤNG ĐÚNG SupervisorAgent CLASS
        """
        try:
            result = self.supervisor.classify_request(question, history)
            logger.info(f"✅ Supervisor classification: {result.get('agent', 'UNKNOWN')}")
            return result
        except Exception as e:
            logger.error(f"❌ Supervisor error: {e}", exc_info=True)
            # Fallback: Rule-based classification
            return self._fallback_supervisor_classification(question)

    def _fallback_supervisor_classification(self, question: str) -> Dict[str, Any]:
        """
        Fallback classification khi Supervisor agent fail
        Sử dụng logic đơn giản từ SupervisorAgent._fallback_classify
        """
        try:
            # Sử dụng method có sẵn trong SupervisorAgent
            agent = self.supervisor._fallback_classify(question)
            logger.info(f"⚠️ Using fallback classification: {agent}")

            return {
                "agent": agent,
                "contextualized_question": question,
                "context_summary": "Fallback mode - no context",
                "is_followup": False,
                "reasoning": "Fallback due to error"
            }
        except Exception as e:
            logger.error(f"❌ Fallback classification error: {e}")
            return {
                "agent": "FAQ",
                "contextualized_question": question,
                "context_summary": "",
                "is_followup": False,
                "reasoning": "Emergency fallback"
            }

    def _safe_execute_faq(self, question: str, history: List) -> Dict[str, Any]:
        """
        Execute FAQ với context info
        LƯU Ý: Chưa có is_followup và context ở đây vì chúng đến từ Supervisor
        """
        try:
            # FAQ chạy song song nên chưa có context từ Supervisor
            # Chỉ truyền question gốc
            result = self.faq_agent.process(
                question,
                is_followup=False,  # Sẽ được update sau
                context=""
            )
            return result
        except Exception as e:
            logger.error(f"FAQ error: {e}")
            return {
                "status": "ERROR",
                "answer": "",
                "references": [],
                "next_agent": "RETRIEVER"
            }

    def _safe_execute_retriever(self, question: str) -> Dict[str, Any]:
        """Execute RETRIEVER with error handling"""
        try:
            result = self.retriever_agent.process(question)
            return result
        except Exception as e:
            logger.error(f"RETRIEVER error: {e}")
            return {
                "status": "ERROR",
                "documents": [],
                "next_agent": "NOT_ENOUGH_INFO"
            }

    def _decision_router_node(self, state: ChatbotState) -> ChatbotState:
        """Router thông minh dựa trên kết quả parallel"""
        supervisor_agent = state.get("supervisor_classification", {}).get("agent", "FAQ")
        faq_result = state.get("faq_result", {})
        retriever_result = state.get("retriever_result", {})

        logger.info(f"🤔 Decision Router: Supervisor={supervisor_agent}")

        # Priority 1: Special cases (CHATTER, REPORTER, OTHER)
        if supervisor_agent in ["CHATTER", "REPORTER", "OTHER"]:
            logger.info(f"→ Route to {supervisor_agent}")
            state["current_agent"] = supervisor_agent
            return state

        # Priority 2: FAQ success
        if faq_result.get("status") == "SUCCESS":
            logger.info("→ FAQ has answer")
            state["status"] = faq_result["status"]
            state["answer"] = faq_result.get("answer", "")
            state["references"] = faq_result.get("references", [])
            state["current_agent"] = "end"
            return state

        # Priority 3: RETRIEVER has documents
        if retriever_result.get("documents"):
            logger.info("→ RETRIEVER → GRADER")
            state["documents"] = retriever_result.get("documents", [])
            state["status"] = retriever_result.get("status", "SUCCESS")
            state["current_agent"] = "GRADER"
            return state

        # No results
        logger.info("→ NOT_ENOUGH_INFO")
        state["current_agent"] = "NOT_ENOUGH_INFO"
        return state

    def _grader_node(self, state: ChatbotState) -> ChatbotState:
        """Grader agent"""
        try:
            result = self.grader_agent.process(
                state["question"],
                state.get("documents", [])
            )
            state["status"] = result["status"]
            state["qualified_documents"] = result.get("qualified_documents", [])
            state["references"] = result.get("references", [])
            state["current_agent"] = result.get("next_agent", "GENERATOR")
            return state
        except Exception as e:
            logger.error(f"Grader error: {e}")
            state["current_agent"] = "NOT_ENOUGH_INFO"
            return state

    def _generator_node(self, state: ChatbotState) -> ChatbotState:
        """Generator agent với full context"""
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
            logger.error(f"Generator error: {e}")
            state["answer"] = "Lỗi tạo câu trả lời"
            state["current_agent"] = "end"
            return state

    def _not_enough_info_node(self, state: ChatbotState) -> ChatbotState:
        """Not enough info agent"""
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
            logger.error(f"Not enough info error: {e}")
            state["answer"] = "Không tìm thấy thông tin"
            return state

    def _chatter_node(self, state: ChatbotState) -> ChatbotState:
        """Chatter agent"""
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
            logger.error(f"Chatter error: {e}")
            state["answer"] = "Tôi hiểu cảm xúc của bạn"
            return state

    def _reporter_node(self, state: ChatbotState) -> ChatbotState:
        """Reporter agent"""
        try:
            result = self.reporter_agent.process(state["question"])
            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"
            return state
        except Exception as e:
            logger.error(f"Reporter error: {e}")
            state["answer"] = "Hệ thống đang bảo trì"
            return state

    def _other_node(self, state: ChatbotState) -> ChatbotState:
        """Other agent"""
        try:
            result = self.other_agent.process(state["question"])
            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"
            return state
        except Exception as e:
            logger.error(f"Other error: {e}")
            state["answer"] = "Đây không phải tác vụ của tôi"
            return state

    def _route_after_decision(self, state: ChatbotState) -> str:
        """Route after decision"""
        return state.get("current_agent", "end")

    def _route_next_agent(self, state: ChatbotState) -> str:
        """Route to next agent"""
        return state.get("current_agent", "end")

    def run(self, question: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Run workflow"""
        try:
            initial_state = ChatbotState(
                question=question,
                original_question=question,
                history=history or [],
                is_followup=False,
                context_summary="",
                relevant_context="",
                current_agent="parallel_execution",
                documents=[],
                qualified_documents=[],
                references=[],
                answer="",
                status="",
                iteration_count=0,
                supervisor_classification={},
                faq_result={},
                retriever_result={},
                parallel_mode=False
            )

            logger.info(f"🚀 Workflow start: {question[:100]}")
            final_state = self.workflow.invoke(initial_state)

            return {
                "answer": final_state.get("answer", "Lỗi xử lý"),
                "references": final_state.get("references", []),
                "status": final_state.get("status", "ERROR"),
                "original_question": final_state.get("original_question", question),
                "processed_question": final_state.get("question", question),
                "is_followup": final_state.get("is_followup", False),
                "context_summary": final_state.get("context_summary", ""),
                "parallel_mode": final_state.get("parallel_mode", False),
                "supervisor_agent": final_state.get("supervisor_classification", {}).get("agent", "UNKNOWN")
            }

        except Exception as e:
            logger.error(f"❌ Workflow error: {e}", exc_info=True)
            return {
                "answer": "Xin lỗi, hệ thống gặp sự cố. Vui lòng thử lại.",
                "references": [],
                "status": "ERROR",
                "original_question": question,
                "processed_question": question,
                "is_followup": False,
                "context_summary": "",
                "parallel_mode": False,
                "supervisor_agent": "ERROR"
            }

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True, timeout=5)  # Graceful shutdown