"""
Lightweight Cross-Encoder Reranker for CO3101 RAG.
Uses BAAI/bge-reranker-v2-m3 for precise token-level relevance scoring.
"""

import math
from typing import List, Tuple
import torch
from sentence_transformers import CrossEncoder
from src.schemas import Chunk
from src.config import RERANKER_MODEL_NAME


class RerankerEngine:
    """Cross-encoder reranking engine."""

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    local_files_only=True,
                )
            except Exception:
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                )

    def rerank(self, query: str, candidates: List[Chunk], top_k: int = 4) -> List[Tuple[Chunk, float]]:
        """
        Reranks candidate chunks with respect to the query using CrossEncoder.
        Normalizes scores to [0.0, 1.0] via sigmoid function.
        Returns top_k (Chunk, score) pairs sorted by relevance.
        """
        if not candidates:
            return []
        self._load_model()
        pairs = [(query, c.content) for c in candidates]
        raw_scores = self._model.predict(pairs)

        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()

        # Sigmoid normalization: 1 / (1 + exp(-s))
        norm_scores = []
        for s in raw_scores:
            val = float(s)
            sig = 1.0 / (1.0 + math.exp(-val)) if -50 <= val <= 50 else (1.0 if val > 50 else 0.0)
            norm_scores.append(sig)

        scored = list(zip(candidates, norm_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

