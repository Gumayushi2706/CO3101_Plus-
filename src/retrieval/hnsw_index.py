"""
HNSW Vector Index module using Faiss.
Implements faiss.IndexHNSWFlat with METRIC_INNER_PRODUCT for L2-normalized vectors.
Manages disk persistence (storage/faiss_index/).
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import faiss
import numpy as np

from src.schemas import Chunk
from src.config import (
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    INDEX_FILE_PATH,
    METADATA_FILE_PATH,
)


def create_hnsw_index(
    dim: int = 1024,
    M: int = 16,
    ef_c: int = 200,
    ef_s: int = 50,
) -> faiss.IndexHNSWFlat:
    """
    Creates an HNSW index configured as specified in Section 5.1 of project design:
    - METRIC_INNER_PRODUCT because vectors are L2-normalized (equivalent to cosine similarity).
    - efConstruction: size of dynamic candidate list during construction.
    - efSearch: size of dynamic candidate list during search.
    """
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_c
    index.hnsw.efSearch = ef_s
    return index


class HNSWVectorStore:
    """
    Manages building, persisting, loading, and querying the HNSW index and its chunk metadata.
    """

    def __init__(
        self,
        dim: int = 1024,
        M: int = HNSW_M,
        ef_c: int = HNSW_EF_CONSTRUCTION,
        ef_s: int = HNSW_EF_SEARCH,
        index_path: Path = INDEX_FILE_PATH,
        metadata_path: Path = METADATA_FILE_PATH,
    ):
        self.dim = dim
        self.M = M
        self.ef_c = ef_c
        self.ef_s = ef_s
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.index: Optional[faiss.IndexHNSWFlat] = None
        self.chunks_metadata: List[Dict[str, Any]] = []

    def initialize_new(self, dim: Optional[int] = None):
        """Initializes a new empty HNSW index."""
        if dim:
            self.dim = dim
        self.index = create_hnsw_index(
            dim=self.dim,
            M=self.M,
            ef_c=self.ef_c,
            ef_s=self.ef_s,
        )
        self.chunks_metadata = []

    @property
    def count(self) -> int:
        """Returns the number of indexed vectors."""
        if self.index is None:
            return 0
        return self.index.ntotal

    @property
    def chunks(self) -> List[Chunk]:
        """Returns all indexed chunks as Chunk objects."""
        return [Chunk(**m) for m in self.chunks_metadata]

    def get_indexed_sources(self, partition: Optional[str] = None) -> set:
        """Returns the set of unique document filenames currently stored in the index."""
        sources = set()
        target_part = partition.lower() if partition and partition.lower() not in ("all", "*", "tất cả") else None
        for meta in self.chunks_metadata:
            part = str(meta.get("partition", "default")).lower()
            if target_part and part != target_part:
                continue
            src = meta.get("source")
            if src:
                sources.add(Path(src).name)
                sources.add(str(src))
                # Also track combination with partition
                sources.add(f"{part}/{Path(src).name}")
        return sources

    def _rebuild_from_indices(self, keep_indices: List[int]):
        """Reconstructs FAISS vectors for kept indices without recomputing embeddings."""
        if len(keep_indices) == 0:
            self.initialize_new(dim=self.dim)
        else:
            all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
            kept_vectors = np.ascontiguousarray(all_vectors[keep_indices], dtype=np.float32)
            kept_metadata = [self.chunks_metadata[i] for i in keep_indices]

            self.initialize_new(dim=self.dim)
            self.index.add(kept_vectors)
            self.chunks_metadata = kept_metadata
        self.save()

    def remove_by_source(self, filename: str, partition: Optional[str] = None) -> int:
        """
        Removes all chunks and vectors belonging to the specified file name/source.
        Uses FAISS vector reconstruction to preserve remaining vectors instantaneously.
        Returns the number of removed chunks.
        """
        if self.index is None or self.index.ntotal == 0 or not self.chunks_metadata:
            return 0

        target_name = Path(filename).name.lower()
        target_part = partition.lower() if partition and partition.lower() not in ("all", "*", "tất cả") else None
        keep_indices = []
        removed_count = 0

        for i, meta in enumerate(self.chunks_metadata):
            src = meta.get("source", "")
            src_name = Path(src).name.lower() if src else ""
            part = str(meta.get("partition", "default")).lower()

            name_matches = (src_name == target_name or src == filename)
            part_matches = (target_part is None or part == target_part)

            if name_matches and part_matches:
                removed_count += 1
            else:
                keep_indices.append(i)

        if removed_count > 0:
            self._rebuild_from_indices(keep_indices)
        return removed_count

    def remove_by_partition(self, partition: str) -> int:
        """
        Removes all chunks and vectors belonging to a partition.
        Uses FAISS vector reconstruction without recomputing embeddings.
        Returns the number of removed chunks.
        """
        if self.index is None or self.index.ntotal == 0 or not self.chunks_metadata:
            return 0

        target_part = partition.lower()
        keep_indices = []
        removed_count = 0

        for i, meta in enumerate(self.chunks_metadata):
            part = str(meta.get("partition", "default")).lower()
            if part == target_part:
                removed_count += 1
            else:
                keep_indices.append(i)

        if removed_count > 0:
            self._rebuild_from_indices(keep_indices)
        return removed_count

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        """
        Adds vectors and corresponding chunk metadata into the HNSW index.
        """
        if len(chunks) == 0:
            return

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) does not match chunks count ({len(chunks)})"
            )

        actual_dim = embeddings.shape[1]
        if self.index is None or self.index.d != actual_dim:
            self.initialize_new(dim=actual_dim)

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings)

        for chunk in chunks:
            self.chunks_metadata.append(chunk.model_dump())

    @staticmethod
    def _make_chunk_from_meta(meta: Dict[str, Any], score: float) -> Chunk:
        """Instantiates a Chunk object from dictionary metadata with retrieved score."""
        return Chunk(
            id=meta["id"],
            content=meta["content"],
            source=meta["source"],
            partition=meta.get("partition", "default"),
            page=meta.get("page"),
            char_start=meta.get("char_start"),
            char_end=meta.get("char_end"),
            score=score,
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 4, partition: Optional[str] = None) -> List[Chunk]:
        """
        Searches the HNSW index for the top-k most similar chunks.
        Optionally filters results by partition to prevent cross-domain noise.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        is_filtered = bool(partition and partition.lower() not in ("all", "*", "tất cả"))
        target_part = partition.lower() if is_filtered else None
        candidate_k = min(self.index.ntotal, max(top_k * 5, 50)) if is_filtered else min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_vec, candidate_k)
        retrieved_chunks: List[Chunk] = []

        for rank in range(candidate_k):
            idx = indices[0][rank]
            if idx == -1 or idx >= len(self.chunks_metadata):
                continue

            meta = self.chunks_metadata[idx]
            if target_part and str(meta.get("partition", "default")).lower() != target_part:
                continue

            retrieved_chunks.append(self._make_chunk_from_meta(meta, float(scores[0][rank])))
            if len(retrieved_chunks) >= top_k:
                break

        return retrieved_chunks

    def save(self):
        """Persists the FAISS index (.bin) and metadata (.json) to disk."""
        if self.index is None:
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        # Save FAISS binary index
        faiss.write_index(self.index, str(self.index_path))

        # Save metadata as JSON
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks_metadata, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        """
        Loads the FAISS index and metadata from disk if they exist.
        Returns True if successful, False otherwise.
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False

        try:
            self.index = faiss.read_index(str(self.index_path))
            self.index.hnsw.efSearch = self.ef_s

            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.chunks_metadata = json.load(f)
            return True
        except Exception:
            return False
