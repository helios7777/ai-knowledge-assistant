import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Dict
from app.config import settings
from app.core.embeddings import embedding_manager

class VectorStore:
    
    def __init__(self, dimension: int = None, index_path: str = None):
        self.dimension = dimension or embedding_manager.get_dimension()
        self.index_path = index_path or settings.VECTOR_DB_PATH
        self.index_file = os.path.join(self.index_path, "faiss_index.bin")
        self.metadata_file = os.path.join(self.index_path, "metadata.pkl")
        
        self.index = None
        self.metadata_store = []
        self.load_or_create_index()
    
    def load_or_create_index(self):
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, 'rb') as f:
                self.metadata_store = pickle.load(f)
            print(f"Loaded index with {self.index.ntotal} vectors")
        else:
            print("Creating new FAISS index...")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata_store = []
            print("New index created")
    
    def add_documents(self, texts: List[str], metadatas: List[Dict]) -> int:
        if len(texts) != len(metadatas):
            raise ValueError("Number of texts must match number of metadata entries")
        
        embeddings = embedding_manager.embed_texts(texts)
        
        self.index.add(embeddings.astype('float32'))
        
        for text, metadata in zip(texts, metadatas):
            self.metadata_store.append({
                "text": text,
                "metadata": metadata
            })
        
        self.save_index()
        
        return len(texts)
    
    def search(self, query: str, top_k: int = 5) -> Tuple[List[Dict], List[float]]:
        if self.index.ntotal == 0:
            return [], []
        
        query_embedding = embedding_manager.embed_text(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        scores = []
        
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata_store):
                similarity_score = 1 / (1 + distance)
                
                results.append(self.metadata_store[idx])
                scores.append(float(similarity_score))
        
        return results, scores
    
    def save_index(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata_store, f)
    
    def clear_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_store = []
        self.save_index()
    
    def get_stats(self) -> Dict:
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__
        }

vector_store = VectorStore()