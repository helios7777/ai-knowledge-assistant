from typing import Dict, Any, List
from app.core.rag_pipeline import rag_pipeline
from transformers import pipeline
import time
import requests

class A2AClient:
    def __init__(self):
        self.sentiment_api = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    
    def get_sentiment(self, text: str) -> Dict:
        try:
            response = requests.post(
                self.sentiment_api,
                json={"inputs": text[:512]},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return {
                        "sentiment": result[0][0]["label"],
                        "score": result[0][0]["score"],
                        "source": "huggingface-api"
                    }
        except Exception as e:
            print(f"Sentiment API error: {e}")
        
        return {"sentiment": "UNKNOWN", "score": 0.0, "source": "error"}

class Orchestrator:
    
    def __init__(self):
        self.rag = rag_pipeline
        print("Loading summarizer model...")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        print("Loading translator model...")
        self.translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
        self.a2a_client = A2AClient()
        self.chains = {}
        print("Orchestrator initialized with all tools!")
        
    def rag_chain(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        answer, retrieval_results, confidence = self.rag.query(query)
        latency = time.time() - start_time
        
        tokens_estimate = len(answer.split())
        
        return {
            "answer": answer,
            "confidence": confidence,
            "latency": latency,
            "tokens": tokens_estimate,
            "tool": "rag"
        }
    
    def summarize_chain(self, text: str) -> Dict[str, Any]:
        start_time = time.time()
        
        if len(text.split()) > 1024:
            text = " ".join(text.split()[:1024])
        
        if len(text.split()) < 30:
            return {
                "summary": text,
                "latency": 0.0,
                "tokens": len(text.split()),
                "tool": "summarizer",
                "note": "Text too short to summarize"
            }
        
        summary = self.summarizer(text, max_length=150, min_length=50, do_sample=False)
        latency = time.time() - start_time
        
        tokens_estimate = len(summary[0]['summary_text'].split())
        
        return {
            "summary": summary[0]['summary_text'],
            "latency": latency,
            "tokens": tokens_estimate,
            "tool": "summarizer"
        }
    
    def translate_chain(self, text: str) -> Dict[str, Any]:
        start_time = time.time()
        
        if len(text.split()) > 512:
            text = " ".join(text.split()[:512])
        
        translation = self.translator(text)
        latency = time.time() - start_time
        
        tokens_estimate = len(translation[0]['translation_text'].split())
        
        return {
            "translation": translation[0]['translation_text'],
            "latency": latency,
            "tokens": tokens_estimate,
            "tool": "translator"
        }
    
    def sentiment_chain(self, text: str) -> Dict[str, Any]:
        start_time = time.time()
        sentiment = self.a2a_client.get_sentiment(text)
        latency = time.time() - start_time
        
        return {
            "sentiment": sentiment["sentiment"],
            "score": sentiment["score"],
            "source": sentiment["source"],
            "latency": latency,
            "tokens": 0,
            "tool": "sentiment_a2a"
        }
    
    def orchestrate(self, query: str, tools: List[str]) -> Dict[str, Any]:
        results = {}
        total_tokens = 0
        
        if "rag" in tools:
            results["rag"] = self.rag_chain(query)
            total_tokens += results["rag"].get("tokens", 0)
        
        if "summarize" in tools and "rag" in results:
            text = results["rag"]["answer"]
            results["summarize"] = self.summarize_chain(text)
            total_tokens += results["summarize"].get("tokens", 0)
        
        if "translate" in tools and "rag" in results:
            text = results["rag"]["answer"]
            results["translate"] = self.translate_chain(text)
            total_tokens += results["translate"].get("tokens", 0)
        
        if "sentiment" in tools and "rag" in results:
            text = results["rag"]["answer"]
            results["sentiment"] = self.sentiment_chain(text)
        
        results["total_tokens"] = total_tokens
        
        return results
    
def rag_chain_finetuned(self, query: str) -> Dict[str, Any]:
    """RAG with fine-tuned model"""
    from app.finetuning.trainer import finetuner
    
    start_time = time.time()
    
    # Get context from RAG
    retrieval_results, confidence = self.rag.retrieve(query)
    
    if retrieval_results:
        context = retrieval_results[0].text
        prompt = f"Question: {query} Context: {context}"
        
        # Use fine-tuned model
        if finetuner.model is None:
            finetuner.load_finetuned_model()
        
        answer = finetuner.generate(prompt) if finetuner.model else "Fine-tuned model not available"
    else:
        answer = "No relevant context found"
    
    latency = time.time() - start_time
    
    return {
        "answer": answer,
        "confidence": confidence,
        "latency": latency,
        "tokens": len(answer.split()),
        "tool": "rag_finetuned"
    }

orchestrator = Orchestrator()