from typing import List, Tuple
from app.config import settings
from app.core.vector_store import vector_store
from app.models import RetrievalResult
from transformers import pipeline

class RAGPipeline:
    
    def __init__(self):
        print("Loading HuggingFace model for text generation...")
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512
        )
    
    def retrieve(self, query: str, top_k: int = None) -> Tuple[List[RetrievalResult], float]:
        top_k = top_k or settings.TOP_K_RESULTS
        results, scores = vector_store.search(query, top_k)
        avg_confidence = sum(scores) / len(scores) if scores else 0.0
        
        retrieval_results = [
            RetrievalResult(
                text=result["text"],
                score=score,
                metadata=result["metadata"]
            )
            for result, score in zip(results, scores)
        ]
        
        return retrieval_results, avg_confidence
    
    def generate_answer(self, query: str, context_docs: List[RetrievalResult]) -> str:
        if not context_docs:
            return "I don't have enough information in my knowledge base to answer this question."
        
        context = "\n\n".join([doc.text for doc in context_docs[:3]])
        
        prompt = f"""Answer the question based on the context below.

Context: {context}

Question: {query}

Answer:"""
        
        try:
            result = self.generator(prompt, max_length=200, do_sample=False)
            return result[0]['generated_text']
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def query(self, query: str, top_k: int = None) -> Tuple[str, List[RetrievalResult], float]:
        retrieval_results, confidence = self.retrieve(query, top_k)
        answer = self.generate_answer(query, retrieval_results)
        return answer, retrieval_results, confidence

rag_pipeline = RAGPipeline()