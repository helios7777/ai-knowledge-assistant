import requests
from typing import Dict

class A2AClient:
    """API-to-API Protocol for external AI services"""
    
    def __init__(self):
        # Using HuggingFace Inference API (free)
        self.sentiment_api = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    
    def get_sentiment(self, text: str) -> Dict:
        """Call external sentiment analysis API"""
        try:
            response = requests.post(
                self.sentiment_api,
                json={"inputs": text},
                headers={"Authorization": "Bearer hf_xxx"}  # Free API, no key needed for demo
            )
            
            if response.status_code == 200:
                result = response.json()[0]
                return {
                    "sentiment": result[0]["label"],
                    "score": result[0]["score"],
                    "source": "huggingface-api"
                }
        except:
            return {"sentiment": "UNKNOWN", "score": 0.0, "source": "error"}

a2a_client = A2AClient()