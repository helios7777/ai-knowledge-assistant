import requests
from typing import Dict, List, Optional
import json

class AIKnowledgeAssistantClient:
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.api_prefix = "/api/v1"
        self.session = requests.Session()
    
    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}{self.api_prefix}{endpoint}"
    
    def health_check(self) -> Dict:
        response = self.session.get(self._url("/health"))
        response.raise_for_status()
        return response.json()
    
    def upload_document(self, content: str, metadata: Optional[Dict] = None) -> Dict:
        payload = {
            "content": content,
            "metadata": metadata or {}
        }
        
        response = self.session.post(
            self._url("/documents"),
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def upload_file(self, file_path: str) -> Dict:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(
                self._url("/documents/file"),
                files=files
            )
        response.raise_for_status()
        return response.json()
    
    def query(self, query: str, top_k: int = 5) -> Dict:
        params = {
            "q": query,
            "top_k": top_k
        }
        
        response = self.session.get(
            self._url("/query"),
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def query_post(self, query: str, top_k: int = 5) -> Dict:
        payload = {
            "query": query,
            "top_k": top_k
        }
        
        response = self.session.post(
            self._url("/query"),
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict:
        response = self.session.get(self._url("/stats"))
        response.raise_for_status()
        return response.json()
    
    def clear_documents(self) -> Dict:
        response = self.session.delete(self._url("/documents"))
        response.raise_for_status()
        return response.json()
    
    def pretty_print_query_result(self, result: Dict):
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(result["answer"])
        print("\n" + "="*80)
        print(f"METADATA: Source={result['source']}, Confidence={result['confidence']:.3f}, Time={result['processing_time']:.3f}s")
        print("="*80)
        print("\nRETRIEVED DOCUMENTS:")
        for i, doc in enumerate(result["retrieval_results"], 1):
            print(f"\n[{i}] Score: {doc['score']:.3f}")
            print(f"Text: {doc['text'][:200]}...")
            print(f"Metadata: {doc['metadata']}")
        print("="*80 + "\n")


if __name__ == "__main__":
    client = AIKnowledgeAssistantClient()
    
    print("Checking API health...")
    health = client.health_check()
    print(f"Status: {health['status']}")
    
    print("\nUploading document...")
    doc_response = client.upload_document(
        content="Python is a high-level programming language. It was created by Guido van Rossum and first released in 1991.",
        metadata={"topic": "programming", "language": "python"}
    )
    print(f"Document uploaded: {doc_response}")
    
    print("\nQuerying...")
    result = client.query("Who created Python?")
    client.pretty_print_query_result(result)