from fastapi import APIRouter, HTTPException, UploadFile, File
import time
from datetime import datetime

from app.models import DocumentUpload, DocumentResponse, QueryRequest, QueryResponse, HealthResponse
from app.core.vector_store import vector_store
from app.core.agent import agent
from app.utils.text_processing import text_processor
from app.config import settings

router = APIRouter()

@router.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }

@router.get("/health", response_model=HealthResponse)
async def health_check():
    stats = vector_store.get_stats()
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow(),
        vector_db_status="operational",
        documents_count=stats["total_vectors"]
    )

@router.post("/documents", response_model=DocumentResponse)
async def upload_document(document: DocumentUpload):
    try:
        cleaned_text = text_processor.clean_text(document.content)
        chunks = text_processor.chunk_text(cleaned_text, document.metadata)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks created")
        
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        num_added = vector_store.add_documents(texts, metadatas)
        doc_id = chunks[0]["metadata"]["document_id"]
        
        return DocumentResponse(
            document_id=doc_id,
            chunks_created=num_added,
            message=f"Successfully processed {num_added} chunks"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/query", response_model=QueryResponse)
async def query_rag(q: str, top_k: int = 5):
    if not q or len(q.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    start_time = time.time()
    try:
        answer, retrieval_results, confidence, source = agent.decide_and_answer(q, top_k)
        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=answer,
            retrieval_results=retrieval_results,
            confidence=confidence,
            source=source,
            processing_time=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_statistics():
    return vector_store.get_stats()