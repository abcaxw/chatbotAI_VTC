from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from typing import List

from .schemas import ChatRequest, ChatResponse, HealthResponse, DocumentReference
from workflow.rag_workflow import RAGWorkflow
from database.milvus_client import milvus_client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Multi-Agent Chatbot API",
    description="API cho hệ thống chatbot RAG với multi-agent architecture",
    version="1.0.0"
)

# CORS middleware - cho phép tất cả origins trong Docker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workflow
rag_workflow = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG Workflow on startup"""
    global rag_workflow
    try:
        rag_workflow = RAGWorkflow()
        logger.info("✅ RAG Workflow initialized successfully")
        logger.info(f"✅ Ollama URL: {os.getenv('OLLAMA_URL', 'http://ollama:11434')}")
        logger.info(f"✅ Milvus: {os.getenv('MILVUS_HOST', 'milvus')}:{os.getenv('MILVUS_PORT', '19530')}")
    except Exception as e:
        logger.error(f"⚠️  Failed to initialize RAG Workflow: {e}")
        logger.warning("⚠️  API will start but /chat endpoint may not work")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "RAG Multi-Agent Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "agents": "/agents"
        },
        "environment": {
            "ollama_url": os.getenv('OLLAMA_URL', 'http://ollama:11434'),
            "milvus_host": os.getenv('MILVUS_HOST', 'milvus'),
            "milvus_port": os.getenv('MILVUS_PORT', '19530')
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chatbot endpoint"""
    try:
        if not rag_workflow:
            raise HTTPException(
                status_code=503,
                detail="Workflow not initialized. Please check server logs."
            )

        logger.info(f"Processing question: {request.question[:100]}...")

        # Run the workflow
        result = rag_workflow.run(request.question, request.history)

        # Convert references to proper format with description
        references = []
        for ref in result.get("references", []):
            references.append(DocumentReference(
                document_id=ref.get("document_id", "unknown"),
                type=ref.get("type", "DOCUMENT"),
                description=ref.get("description", None)  # Thêm description
            ))

        logger.info(f"Response generated with {len(references)} references")

        return ChatResponse(
            answer=result.get("answer", "Lỗi xử lý câu hỏi"),
            references=references,
            status=result.get("status", "ERROR")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db_connected = False
        try:
            db_connected = milvus_client.check_connection()
        except Exception as db_error:
            logger.warning(f"Database connection check failed: {db_error}")

        # Check workflow status
        workflow_ready = rag_workflow is not None

        if db_connected and workflow_ready:
            return HealthResponse(
                status="healthy",
                message="Hệ thống hoạt động bình thường",
                database_connected=True
            )
        elif workflow_ready and not db_connected:
            return HealthResponse(
                status="degraded",
                message="Mất kết nối cơ sở dữ liệu",
                database_connected=False
            )
        elif not workflow_ready:
            return HealthResponse(
                status="degraded",
                message="Workflow chưa được khởi tạo",
                database_connected=db_connected
            )
        else:
            return HealthResponse(
                status="unhealthy",
                message="Hệ thống gặp sự cố",
                database_connected=False
            )

    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Lỗi hệ thống: {str(e)}",
            database_connected=False
        )


@app.get("/agents", response_model=dict)
async def list_agents():
    """List all available agents and their descriptions"""
    return {
        "agents": {
            "SUPERVISOR": "Điều phối chính, phân loại yêu cầu và chọn agent phù hợp",
            "FAQ": "Tìm kiếm và trả lời câu hỏi thường gặp",
            "RETRIEVER": "Tìm kiếm thông tin từ cơ sở dữ liệu tài liệu",
            "GRADER": "Đánh giá chất lượng thông tin tìm được",
            "GENERATOR": "Tạo câu trả lời từ thông tin đã được đánh giá",
            "NOT_ENOUGH_INFO": "Xử lý trường hợp không đủ thông tin",
            "CHATTER": "An ủi và xử lý cảm xúc tiêu cực của khách hàng",
            "REPORTER": "Thông báo trạng thái hệ thống và bảo trì",
            "OTHER": "Xử lý yêu cầu ngoài phạm vi hỗ trợ"
        },
        "workflow": "supervisor -> (faq|retriever|chatter|reporter|other) -> grader -> generator -> end",
        "status": "ready" if rag_workflow else "not_initialized"
    }


@app.get("/status")
async def system_status():
    """Detailed system status endpoint"""
    return {
        "service": "rag-api",
        "port": 8501,
        "workflow_initialized": rag_workflow is not None,
        "environment": {
            "ollama_url": os.getenv('OLLAMA_URL', 'http://ollama:11434'),
            "milvus_host": os.getenv('MILVUS_HOST', 'milvus'),
            "milvus_port": os.getenv('MILVUS_PORT', '19530')
        },
        "components": {
            "fastapi": "running",
            "rag_workflow": "ready" if rag_workflow else "not_initialized",
            "milvus_client": "connected" if milvus_client else "disconnected"
        }
    }


if __name__ == "__main__":
    import uvicorn

    # Chạy với host 0.0.0.0 để Docker có thể truy cập
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8501,
        log_level="info"
    )