"""
Encrypted local semantic search index for meeting context and local docs.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

class LocalRAGIndex:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None

    def add_documents(self, docs: list[str]) -> None:
        new_embeddings = self.model.encode(docs, normalize_embeddings=True)
        self.documents.extend(docs)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def query(self, text: str, top_k: int = 2) -> list[str]:
        if not self.documents or self.embeddings is None:
            return []
        query_vec = self.model.encode([text], normalize_embeddings=True)
        scores = np.dot(self.embeddings, query_vec.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]
