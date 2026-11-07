"""
Post-processing module: Relevance Filtering & Reciprocal Rank Fusion (RRF).
Clean and streamlined implementation preserving core RAG mapping F : (D, q) -> (a, E).
"""

from typing import List
from src.schemas import Chunk
from src.config import SIMILARITY_THRESHOLD


def filter_and_reorder_contexts(
    retrieved_chunks: List[Chunk],
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[Chunk]:
    """
    1. Lọc ngưỡng tương quan tau >= 0.50 để loại bỏ các chunk rác / nhiễu.
    2. Sắp xếp giảm dần theo điểm tương quan (Top-1 có điểm cao nhất).
    """
    if not retrieved_chunks:
        return []

    # 1. Lọc ngưỡng tương quan tau >= threshold
    valid_chunks = [c for c in retrieved_chunks if (c.score is not None and c.score >= threshold)]
    if not valid_chunks:
        return []

    # 2. Sắp xếp giảm dần theo điểm tương đồng cosine (cao nhất đứng đầu)
    valid_chunks.sort(key=lambda x: (x.score if x.score is not None else 0.0), reverse=True)
    return valid_chunks


def reciprocal_rank_fusion(
    dense_results: List[Chunk],
    sparse_results: List[Chunk],
    k_rrf: int = 60,
    top_n: int = 10,
) -> List[Chunk]:
    """
    Combines dense HNSW ranking and sparse BM25 ranking via Reciprocal Rank Fusion:
    RRF(d) = sum(1 / (k_rrf + rank_m(d)))
    Takes lists of Chunk objects (pre-sorted by their respective retrieval scores).
    Returns top_n unique chunks sorted by fused RRF score.
    """
    rrf_scores = {}
    chunk_map = {}

    for rank, chunk in enumerate(dense_results, start=1):
        chunk_map[chunk.id] = chunk
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (k_rrf + rank))

    for rank, chunk in enumerate(sparse_results, start=1):
        chunk_map[chunk.id] = chunk
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (k_rrf + rank))

    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids[:top_n]]
