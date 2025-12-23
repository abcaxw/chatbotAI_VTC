# RAG_Core/api/main.py - FIXED STREAMING VERSION

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logging
import os
from typing import List, AsyncIterator
import json
import asyncio

from .schemas import (
    ChatRequest, ChatResponse, StreamChunk,
    HealthResponse, DocumentReference
)
from workflow.rag_workflow import RAGWorkflow
from database.milvus_client import milvus_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Multi-Agent Chatbot API with Streaming",
    description="API cho hệ thống chatbot RAG với streaming support",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_workflow = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG Workflow on startup"""
    global rag_workflow
    try:
        rag_workflow = RAGWorkflow()
        logger.info("✅ RAG Workflow initialized successfully")
    except Exception as e:
        logger.error(f"⚠️  Failed to initialize RAG Workflow: {e}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "RAG Multi-Agent Chatbot API",
        "version": "2.0.0",
        "features": ["streaming", "multi-agent", "context-aware"],
        "endpoints": {
            "chat": "/chat",
            "health": "/health"
        }
    }


async def generate_streaming_response(
        question: str,
        history: List
) -> AsyncIterator[str]:
    """
    FIXED: Async generator với proper error handling
    """
    try:
        logger.info(f"🚀 Starting streaming for: {question[:50]}...")

        # Send start chunk
        start_chunk = {
            "type": "start",
            "content": None,
            "references": None,
            "status": "processing"
        }
        yield f"data: {json.dumps(start_chunk)}\n\n"
        await asyncio.sleep(0.01)  # Small delay

        # Run workflow
        result = await rag_workflow.run_with_streaming(question, history)

        # Get answer stream
        answer_stream = result.get("answer_stream")
        references = result.get("references", [])

        logger.info(f"📝 Got answer_stream: {answer_stream is not None}")
        logger.info(f"📚 References count: {len(references)}")

        # Stream chunks
        if answer_stream:
            chunk_count = 0
            async for chunk in answer_stream:
                if chunk:  # Only send non-empty chunks
                    chunk_count += 1
                    chunk_data = {
                        "type": "chunk",
                        "content": chunk,
                        "references": None,
                        "status": None
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    await asyncio.sleep(0.001)  # Tiny delay for smooth streaming

            logger.info(f"✅ Streamed {chunk_count} chunks")
        else:
            logger.warning("⚠️  No answer_stream available")
            error_chunk = {
                "type": "chunk",
                "content": "Không thể tạo câu trả lời.",
                "references": None,
                "status": None
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

        # Send references
        if references:
            # Convert references to serializable format
            serializable_refs = []
            for ref in references:
                serializable_refs.append({
                    "document_id": ref.get("document_id", ""),
                    "type": ref.get("type", "DOCUMENT"),
                    "description": ref.get("description", "")
                })

            ref_chunk = {
                "type": "references",
                "content": None,
                "references": serializable_refs,
                "status": None
            }
            yield f"data: {json.dumps(ref_chunk)}\n\n"
            logger.info(f"📚 Sent {len(serializable_refs)} references")

        # Send end chunk
        end_chunk = {
            "type": "end",
            "content": None,
            "references": None,
            "status": result.get("status", "SUCCESS")
        }
        yield f"data: {json.dumps(end_chunk)}\n\n"
        logger.info("✅ Streaming completed")

    except Exception as e:
        logger.error(f"❌ Streaming error: {e}", exc_info=True)
        error_chunk = {
            "type": "error",
            "content": f"Lỗi: {str(e)}",
            "references": None,
            "status": "ERROR"
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint - supports streaming and non-streaming
    """
    try:
        if not rag_workflow:
            raise HTTPException(
                status_code=503,
                detail="Workflow not initialized"
            )

        logger.info(f"📨 Question: {request.question[:100]}... (stream={request.stream})")

        # STREAMING MODE
        if request.stream:
            logger.info("🔄 Using streaming mode")
            return StreamingResponse(
                generate_streaming_response(request.question, request.history),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )

        # NON-STREAMING MODE
        logger.info("📋 Using non-streaming mode")
        result = rag_workflow.run(request.question, request.history)

        references = []
        for ref in result.get("references", []):
            references.append(DocumentReference(
                document_id=ref.get("document_id", "unknown"),
                type=ref.get("type", "DOCUMENT"),
                description=ref.get("description", None)
            ))

        return ChatResponse(
            answer=result.get("answer", "Lỗi xử lý câu hỏi"),
            references=references,
            status=result.get("status", "ERROR")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}", exc_info=True)
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
            logger.warning(f"Database check failed: {db_error}")

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
        else:
            return HealthResponse(
                status="unhealthy",
                message="Hệ thống gặp sự cố",
                database_connected=False
            )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Lỗi: {str(e)}",
            database_connected=False
        )


@app.get("/agents")
async def list_agents():
    """List available agents"""
    return {
        "agents": {
            "SUPERVISOR": "Điều phối chính",
            "FAQ": "Câu hỏi thường gặp",
            "RETRIEVER": "Tìm kiếm tài liệu",
            "GRADER": "Đánh giá chất lượng",
            "GENERATOR": "Tạo câu trả lời (streaming)",
            "NOT_ENOUGH_INFO": "Xử lý thiếu thông tin",
            "CHATTER": "Xử lý cảm xúc",
            "REPORTER": "Báo cáo hệ thống",
            "OTHER": "Yêu cầu ngoài phạm vi"
        },
        "features": {
            "streaming": "enabled",
            "context_aware": "enabled"
        },
        "status": "ready" if rag_workflow else "not_initialized"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8501, log_level="info")