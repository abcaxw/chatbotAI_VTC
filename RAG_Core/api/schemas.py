# RAG_Core/api/schemas.py - STREAMING VERSION

from pydantic import BaseModel
from typing import List, Optional, Union, Literal

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: Optional[Union[List[str], List[ChatMessage]]] = []
    stream: Optional[bool] = False  # NEW: Enable streaming


class StreamChunk(BaseModel):
    """Single chunk in streaming response"""
    type: Literal["start", "chunk", "references", "end", "error"]
    content: Optional[str] = None
    references: Optional[List['DocumentReference']] = None
    status: Optional[str] = None


class DocumentReference(BaseModel):
    document_id: str
    type: str  # FAQ, DOCUMENT, SUPPORT, SYSTEM
    description: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    references: List[DocumentReference]
    status: str = "SUCCESS"


class HealthResponse(BaseModel):
    status: str
    message: str
    database_connected: bool