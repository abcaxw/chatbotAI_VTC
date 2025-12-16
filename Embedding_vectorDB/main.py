from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import uuid
import re
import random
import time
from typing import List, Dict, Any
import json

# Import processing modules
from document_processor import DocumentProcessor
from embedding_service import EmbeddingService
from milvus_client import MilvusManager

app = FastAPI(
    title="Document Processing API",
    version="1.0.0",
    description="API for document processing, embedding, and FAQ management"
)

# CORS middleware - cho phép tất cả origins trong Docker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
milvus_host = os.getenv('MILVUS_HOST', 'localhost')
milvus_port = os.getenv('MILVUS_PORT', '19530')
milvus_manager = MilvusManager(host=milvus_host, port=milvus_port)
# Initialize services
doc_processor = DocumentProcessor()
embedding_service = EmbeddingService()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to be safe for file operations
    Remove special characters and spaces that might cause issues
    """
    if not filename:
        return "unknown_file"

    # Get file extension
    name, ext = os.path.splitext(filename)

    # Remove or replace problematic characters
    # Keep only alphanumeric, dots, hyphens, underscores
    safe_name = re.sub(r'[^\w\-_.]', '_', name)

    # Remove multiple consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)

    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')

    # If name is empty after sanitization, use a default
    if not safe_name:
        safe_name = "document"

    return safe_name + ext.lower()


def get_safe_temp_filename(original_filename: str) -> str:
    """
    Generate a safe temporary filename with unique identifier
    """
    # Get extension
    _, ext = os.path.splitext(original_filename)

    # Create unique filename to avoid conflicts
    unique_id = str(uuid.uuid4())[:8]
    safe_name = f"temp_doc_{unique_id}{ext.lower()}"

    return safe_name


@app.on_event("startup")
async def startup_event():
    """Initialize Milvus connection and create collections"""
    try:
        await milvus_manager.initialize()
        print("✅ Document API started successfully")
        print(f"✅ Milvus connected: {os.getenv('MILVUS_HOST', 'milvus')}:{os.getenv('MILVUS_PORT', '19530')}")
    except Exception as e:
        print(f"⚠️  Warning during startup: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Document Processing API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "process_document": "/api/v1/process-document",
            "embed_markdown": "/api/v1/embed-markdown",
            "add_faq": "/api/v1/faq/add",
            "delete_faq": "/api/v1/faq/delete/{faq_id}",
            "delete_document": "/api/v1/document/delete/{document_id}",
            "health": "/api/v1/health"
        }
    }


@app.post("/api/v1/process-document")
async def process_document(file: UploadFile = File(...)):
    """
    API 1: Convert various document formats to structured markdown
    Accepts: PDF, DOC, DOCX, XLS, XLSX, TXT files
    Returns: Processed markdown content
    """
    try:
        # Validate file exists and has content
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Validate file type
        allowed_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.txt']
        original_filename = file.filename
        file_extension = os.path.splitext(original_filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed: {allowed_extensions}"
            )

        # Generate safe temporary filename
        safe_temp_name = get_safe_temp_filename(original_filename)

        # Create temporary file with safe name
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, safe_temp_name)

        try:
            # Read and save file content
            content = await file.read()

            # Validate file has content
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            # Write to temporary file
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(content)

            # Process document based on type
            if file_extension == '.pdf':
                markdown_content = doc_processor.process_pdf(temp_file_path)
            elif file_extension in ['.doc', '.docx']:
                markdown_content = doc_processor.process_word(temp_file_path)
            elif file_extension in ['.xls', '.xlsx']:
                markdown_content = doc_processor.process_excel(temp_file_path)
            elif file_extension == '.txt':
                # For text files, try different encodings
                text_content = None
                for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                    try:
                        with open(temp_file_path, 'r', encoding=encoding) as f:
                            text_content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue

                if text_content is None:
                    raise HTTPException(status_code=400, detail="Could not decode text file")

                markdown_content = doc_processor.process_text(text_content)

            # Validate processing result
            if not markdown_content or len(markdown_content.strip()) == 0:
                raise HTTPException(
                    status_code=422,
                    detail="Could not extract content from file. The file might be empty or corrupted."
                )

            # Sanitize original filename for response
            safe_original_name = sanitize_filename(original_filename)

            return {
                "status": "success",
                "filename": safe_original_name,
                "original_filename": original_filename,
                "markdown_content": markdown_content,
                "processing_info": {
                    "file_type": file_extension,
                    "content_length": len(markdown_content),
                    "file_size_bytes": len(content)
                }
            }

        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except Exception as cleanup_error:
                # Log cleanup error but don't fail the request
                print(f"Warning: Could not clean up temp file {temp_file_path}: {cleanup_error}")

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )


@app.post("/api/v1/embed-markdown")
async def embed_markdown(request: dict):
    """
    API 2: Convert markdown content to embeddings and store in Milvus (with sentence-level chunking)
    Input: {"markdown_content": "...", "document_id": "...", "chunk_mode": "sentence|paragraph"}
    Returns: List of embeddings with metadata
    """
    try:
        markdown_content = request.get("markdown_content")
        document_id = request.get("document_id")
        chunk_mode = request.get("chunk_mode", "sentence").lower()  # Default to sentence-level

        if not markdown_content:
            raise HTTPException(status_code=400, detail="markdown_content is required")

        # Validate chunk_mode
        if chunk_mode not in ["sentence", "paragraph"]:
            raise HTTPException(
                status_code=400,
                detail="chunk_mode must be either 'sentence' or 'paragraph'"
            )

        # Sanitize document_id if provided, otherwise generate one
        if document_id:
            # Sanitize document_id to be safe for storage
            document_id = re.sub(r'[^\w\-_.]', '_', str(document_id))
            document_id = re.sub(r'_+', '_', document_id).strip('_')

        if not document_id:
            document_id = f"doc_{str(uuid.uuid4())[:8]}"

        # Validate markdown content
        if len(markdown_content.strip()) == 0:
            raise HTTPException(status_code=400, detail="markdown_content cannot be empty")

        # Parse markdown based on chunk mode
        if chunk_mode == "sentence":
            chunks = doc_processor.parse_markdown_to_sentences(markdown_content)
        else:
            chunks = doc_processor.parse_markdown_to_chunks(markdown_content)

        if not chunks:
            raise HTTPException(status_code=422, detail="Could not parse markdown into chunks")

        # Generate embeddings for each chunk
        embeddings_data = []
        successful_embeddings = 0

        for i, chunk in enumerate(chunks):
            try:
                embedding = embedding_service.get_embedding(chunk['content'])

                # Prepare metadata based on chunk structure
                metadata = {
                    "section_title": chunk['section_title'],
                    "chunk_index": i,
                    "content_length": len(chunk['content']),
                    "chunk_mode": chunk_mode
                }

                # Add sentence-specific metadata if available
                if chunk_mode == "sentence" and 'sentence' in chunk:
                    metadata.update({
                        "sentence": chunk['sentence'],
                        "sentence_length": len(chunk['sentence'])
                    })

                embedding_data = {
                    "id": f"{document_id}_{chunk_mode}_{i}",
                    "document_id": document_id,
                    "description": chunk['content'],
                    "description_vector": embedding,
                    "metadata": metadata
                }
                embeddings_data.append(embedding_data)
                successful_embeddings += 1

            except Exception as embedding_error:
                print(f"Error creating embedding for chunk {i}: {embedding_error}")
                continue

        if not embeddings_data:
            raise HTTPException(status_code=422, detail="Could not create embeddings for any chunks")

        # Store in Milvus
        stored_count = await milvus_manager.insert_embeddings(embeddings_data)

        # Prepare response chunks info
        chunks_info = []
        for item in embeddings_data[:10]:  # Limit preview to first 10 items
            chunk_info = {
                "id": item["id"],
                "section_title": item["metadata"]["section_title"],
                "content_preview": item["description"][:200] + "..." if len(item["description"]) > 200 else item[
                    "description"]
            }

            # Add sentence info if available
            if chunk_mode == "sentence" and 'sentence' in item["metadata"]:
                chunk_info["sentence_preview"] = item["metadata"]["sentence"][:100] + "..." if len(
                    item["metadata"]["sentence"]) > 100 else item["metadata"]["sentence"]

            chunks_info.append(chunk_info)

        return {
            "status": "success",
            "document_id": document_id,
            "chunk_mode": chunk_mode,
            "total_chunks": len(chunks),
            "successful_embeddings": successful_embeddings,
            "stored_count": stored_count,
            "processing_stats": {
                "original_content_length": len(markdown_content),
                "average_chunk_length": sum(len(chunk['content']) for chunk in chunks) / len(chunks) if chunks else 0,
                "success_rate": f"{(successful_embeddings / len(chunks) * 100):.1f}%" if chunks else "0%"
            },
            "chunks_preview": chunks_info
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Embedding error: {str(e)}"
        )


@app.post("/api/v1/faq/add")
async def add_faq(request: dict):
    """
    API 4: Add FAQ - Thêm câu hỏi và câu trả lời FAQ
    Input: {"question": "Câu hỏi", "answer": "Câu trả lời", "faq_id": "optional_id"}
    Returns: Success response with FAQ ID
    """
    try:
        question = request.get("question", "").strip()
        answer = request.get("answer", "").strip()
        faq_id = request.get("faq_id", "").strip()

        # Validate inputs
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        if not answer:
            raise HTTPException(status_code=400, detail="Answer is required")

        # Generate FAQ ID if not provided
        if not faq_id:
            faq_id = f"faq_{str(uuid.uuid4())[:8]}"
        else:
            # Sanitize FAQ ID
            faq_id = re.sub(r'[^\w\-_.]', '_', str(faq_id))
            faq_id = re.sub(r'_+', '_', faq_id).strip('_')

        # Generate embedding for the question
        question_embedding = embedding_service.get_embedding(question)

        # Insert FAQ into Milvus
        success = await milvus_manager.insert_faq(faq_id, question, answer, question_embedding)

        return {
            "status": "success",
            "faq_id": faq_id,
            "question": question,
            "answer": answer,
            "message": "FAQ added successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Add FAQ error: {str(e)}"
        )


@app.delete("/api/v1/faq/delete/{faq_id}")
async def delete_faq(faq_id: str):
    """
    API 5: Delete FAQ - Xóa FAQ theo ID
    Input: faq_id as path parameter
    Returns: Success response
    """
    try:
        if not faq_id or not faq_id.strip():
            raise HTTPException(status_code=400, detail="FAQ ID is required")

        faq_id = faq_id.strip()

        # Delete FAQ from Milvus
        success = await milvus_manager.delete_faq(faq_id)

        return {
            "status": "success",
            "faq_id": faq_id,
            "message": "FAQ deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Delete FAQ error: {str(e)}"
        )


@app.delete("/api/v1/document/delete/{document_id}")
async def delete_document(document_id: str):
    """
    API 6: Delete Document - Xóa tất cả embeddings của một document_id
    Input: document_id as path parameter
    Returns: Success response
    """
    try:
        if not document_id or not document_id.strip():
            raise HTTPException(status_code=400, detail="Document ID is required")

        document_id = document_id.strip()

        # Delete document embeddings from Milvus
        success = await milvus_manager.delete_document(document_id)

        return {
            "status": "success",
            "document_id": document_id,
            "message": "Document and all its embeddings deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Delete document error: {str(e)}"
        )


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    try:
        milvus_status = await milvus_manager.health_check()
        embedding_status = embedding_service.is_ready()

        return {
            "status": "healthy",
            "service": "document-api",
            "port": 8000,
            "services": {
                "milvus": milvus_status,
                "embedding_model": embedding_status,
                "document_processor": "ready"
            },
            "environment": {
                "milvus_host": os.getenv('MILVUS_HOST', 'milvus'),
                "milvus_port": os.getenv('MILVUS_PORT', '19530')
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "document-api",
            "port": 8000,
            "error": str(e)
        }


if __name__ == "__main__":
    # Chạy với host 0.0.0.0 để Docker có thể truy cập
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )