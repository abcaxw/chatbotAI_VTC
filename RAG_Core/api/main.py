# RAG_Core/api/main.py - ADD STREAMING ENDPOINT

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logging
import os
import json
import asyncio
from typing import List, AsyncGenerator

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

# CORS middleware
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
            "stream_chat": "/stream-chat",
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
    """Main chatbot endpoint (non-streaming)"""
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
                description=ref.get("description", None)
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


async def generate_stream_response(
        question: str,
        history: List
) -> AsyncGenerator[str, None]:
    """
    Generator function for streaming responses
    Yields Server-Sent Events (SSE) format
    """
    try:
        # Send initial event
        yield f"data: {json.dumps({'type': 'start', 'message': 'Processing your question...'})}\n\n"

        # Simulate step-by-step processing
        steps = [
            ("supervisor", "Analyzing question..."),
            ("retrieval", "Searching for relevant information..."),
            ("grading", "Evaluating document quality..."),
            ("generation", "Generating answer...")
        ]

        for step_name, step_message in steps:
            await asyncio.sleep(0.5)  # Simulate processing time
            yield f"data: {json.dumps({'type': 'step', 'step': step_name, 'message': step_message})}\n\n"

        # Run the actual workflow
        result = rag_workflow.run(question, history)

        # Stream the answer word by word (simulate streaming)
        answer = result.get("answer", "Lỗi xử lý câu hỏi")
        words = answer.split()

        current_text = ""
        for i, word in enumerate(words):
            current_text += word + " "

            # Send chunk every few words
            if (i + 1) % 5 == 0 or i == len(words) - 1:
                await asyncio.sleep(0.1)  # Simulate typing delay
                yield f"data: {json.dumps({'type': 'content', 'content': current_text.strip()})}\n\n"

        # Send references
        references = []
        for ref in result.get("references", []):
            references.append({
                "document_id": ref.get("document_id", "unknown"),
                "type": ref.get("type", "DOCUMENT"),
                "description": ref.get("description", None)
            })

        yield f"data: {json.dumps({'type': 'references', 'references': references})}\n\n"

        # Send completion event
        yield f"data: {json.dumps({'type': 'done', 'status': result.get('status', 'SUCCESS')})}\n\n"

    except Exception as e:
        logger.error(f"Error in stream generation: {e}", exc_info=True)
        error_msg = json.dumps({
            'type': 'error',
            'message': f"Error: {str(e)}"
        })
        yield f"data: {error_msg}\n\n"


@app.post("/stream-chat")
async def stream_chat(request: ChatRequest):
    """
    Streaming chatbot endpoint
    Returns Server-Sent Events (SSE) stream
    """
    try:
        if not rag_workflow:
            raise HTTPException(
                status_code=503,
                detail="Workflow not initialized. Please check server logs."
            )

        logger.info(f"Starting stream for question: {request.question[:100]}...")

        return StreamingResponse(
            generate_stream_response(request.question, request.history),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in stream chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        db_connected = False
        try:
            db_connected = milvus_client.check_connection()
        except Exception as db_error:
            logger.warning(f"Database connection check failed: {db_error}")

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8501,
        log_level="info"
    )
