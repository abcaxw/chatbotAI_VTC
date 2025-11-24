from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from config.settings import settings


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        # Model sẽ tự động output 768D
        print(f"Embedding model loaded: {settings.EMBEDDING_MODEL}")
        print(f"Expected dimension: 768D")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings (768D)"""
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        # Verify dimension
        if embeddings.shape[1] != 768:
            print(f"Warning: Model output {embeddings.shape[1]}D instead of 768D")

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding (768D)"""
        embedding = self.model.encode([text], normalize_embeddings=True)[0]

        # Verify dimension
        if embedding.shape[0] != 768:
            print(f"Warning: Model output {embedding.shape[0]}D instead of 768D")

        return embedding


# Global embedding model instance
embedding_model = EmbeddingModel()