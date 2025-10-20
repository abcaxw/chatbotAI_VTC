from pydantic import BaseModel
from typing import List, Optional, Union

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: Optional[Union[List[str], List[ChatMessage]]] = []


class DocumentReference(BaseModel):
    document_id: str
    type: str  # FAQ, DOCUMENT, SUPPORT, SYSTEM


class ChatResponse(BaseModel):
    answer: str
    references: List[DocumentReference]
    status: str = "SUCCESS"


class MockChatResponse(BaseModel):
    answer: str
    references: List[str]  # document_ids for mock


class HealthResponse(BaseModel):
    status: str
    message: str
    database_connected: bool