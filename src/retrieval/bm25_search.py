"""
Lightweight In-Memory BM25 Lexical Retriever for CO3101 RAG.
Provides sparse lexical matching over document chunks using BM25Plus.
"""

import re
from typing import List, Tuple, Set, Optional
from rank_bm25 import BM25Plus
from src.schemas import Chunk


def tokenize_text(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric tokens."""
    return re.findall(r"\w+", (text or "").lower())


class BM25Retriever:
    """In-memory BM25 lexical retriever for exact keyword matching."""

    def __init__(self, chunks: List[Chunk] = None):
        self.chunks: List[Chunk] = []
        self.token_sets: List[Set[str]] = []
        self.bm25: BM25Plus = None
        if chunks:
            self.fit(chunks)

    def fit(self, chunks: List[Chunk]):
        """Builds BM25Plus index over a list of chunks."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.bm25 = None
            self.token_sets = []
            return
        token_lists = [tokenize_text(c.content) for c in self.chunks]
        self.token_sets = [set(tl) for tl in token_lists]
        self.bm25 = BM25Plus(token_lists)

    @property
    def count(self) -> int:
        return len(self.chunks)

    def search(self, query: str, top_k: int = 20, partition: Optional[str] = None) -> List[Tuple[Chunk, float]]:
        """Searches top_k matching chunks containing query terms using BM25, with optional partition filtering."""
        if not self.bm25 or not self.chunks:
            return []
        tokens = tokenize_text(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        query_set = set(tokens)

        target_part = partition.lower() if partition and partition.lower() not in ("all", "*", "tất cả") else None

        # Retain chunks that share at least one keyword token with the query and match partition
        matching_pairs = []
        for idx, score in enumerate(scores):
            if query_set.intersection(self.token_sets[idx]):
                chunk = self.chunks[idx]
                chunk_part = str(getattr(chunk, "partition", "default") or "default").lower()
                if target_part and chunk_part != target_part:
                    continue
                matching_pairs.append((chunk, float(score)))

        matching_pairs.sort(key=lambda x: x[1], reverse=True)
        return matching_pairs[:top_k]
