from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentUpload(BaseModel):
    content: str = Field(..., description="Text content of the document")
    metadata: Optional[Dict[str, str]] = Field(default_factory=dict, description="Document metadata")
    
class DocumentResponse(BaseModel):
    document_id: str
    chunks_created: int
    message: str
    
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    
class RetrievalResult(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]
    
class QueryResponse(BaseModel):
    answer: str
    retrieval_results: List[RetrievalResult]
    confidence: float
    source: str
    processing_time: float
    
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    vector_db_status: str
    documents_count: int