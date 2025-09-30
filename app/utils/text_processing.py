from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import settings
import hashlib
from datetime import datetime

class TextProcessor:
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        chunks = self.text_splitter.split_text(text)
        
        result = []
        doc_id = self._generate_doc_id(text)
        
        for idx, chunk in enumerate(chunks):
            chunk_metadata = {
                "document_id": doc_id,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "created_at": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            
            result.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
        
        return result
    
    def _generate_doc_id(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]
    
    @staticmethod
    def clean_text(text: str) -> str:
        text = " ".join(text.split())
        return text.strip()

text_processor = TextProcessor()