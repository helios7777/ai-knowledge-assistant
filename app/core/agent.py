from typing import Tuple, List
from app.config import settings
from app.core.rag_pipeline import rag_pipeline
from app.models import RetrievalResult

class Agent:
    
    def __init__(self):
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
    
    def decide_and_answer(self, query: str, top_k: int = None) -> Tuple[str, List[RetrievalResult], float, str]:
        answer, retrieval_results, confidence = rag_pipeline.query(query, top_k)
        
        if confidence >= self.confidence_threshold:
            return answer, retrieval_results, confidence, "rag"
        else:
            answer = f"[Low confidence: {confidence:.2f}] {answer}"
            return answer, retrieval_results, confidence, "rag"

agent = Agent()