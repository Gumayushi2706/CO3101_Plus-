"""
Embedding Engine module.
Embeds text chunks using BGE-M3 (D=1024) with batch processing (batch_size=32)
and L2-normalization.
"""

from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    BATCH_SIZE,
    NORMALIZE_L2,
)


class EmbeddingEngine:
    r"""
    Embedding Engine wrapping BGE-M3 model for sentence/chunk representation.
    Guarantees L2-normalization: \hat{v}_i = \frac{v_i}{||v_i||_2}
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: str = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.model = None
        self._dim = EMBEDDING_DIM

    def _load_model(self):
        """Loads the embedding model (BGE-M3). Prioritizes local cache for instant offline startup."""
        if self.model is not None:
            return

        try:
            self.model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
        except Exception:
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {e}")

        if hasattr(self.model, "get_embedding_dimension"):
            self._dim = self.model.get_embedding_dimension()
        else:
            self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Returns the actual embedding dimension of the loaded model."""
        return self._dim

    def embed_texts(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Embeds a list of text strings in batches with L2 normalization.
        Returns:
            np.ndarray of shape (N, D), dtype=float32
        """
        self._load_model()
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # SentenceTransformer encodes and applies L2 normalization
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=NORMALIZE_L2,
            convert_to_numpy=True,
        )

        # Ensure float32 contiguous array for FAISS
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string with L2 normalization.
        Returns:
            np.ndarray of shape (1, D), dtype=float32
        """
        return self.embed_texts([query], batch_size=1)
