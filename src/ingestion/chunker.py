"""
Sliding Window Chunker module.
Implements text chunking with configurable sliding window size (800-1000 chars)
and overlap (100-120 chars) while preserving metadata traceability.
"""

from typing import List, Dict, Any, Optional
from src.schemas import Chunk
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class SlidingWindowChunker:
    """
    Sliding window chunking algorithm for text documents.
    Preserves document source, page number, and character offsets.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and strictly smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        source: str = "unknown",
        page: Optional[int] = None,
        prefix_id: Optional[str] = None,
        partition: Optional[str] = "default",
    ) -> List[Chunk]:
        """
        Splits a single text string into sliding window Chunks.
        """
        text = text.strip()
        if not text:
            return []

        chunks: List[Chunk] = []
        text_length = len(text)
        part = partition or "default"
        doc_id_prefix = prefix_id or f"{part}_{source}_p{page or 1}"

        # If text is smaller than chunk size, return single chunk
        if text_length <= self.chunk_size:
            chunk_id = f"{doc_id_prefix}_chunk_0"
            return [
                Chunk(
                    id=chunk_id,
                    content=text,
                    source=source,
                    partition=part,
                    page=page,
                    char_start=0,
                    char_end=text_length,
                    score=None,
                )
            ]

        start = 0
        chunk_idx = 0

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Optional: try to find a natural break point (newline or period) near the end
            # if we are not at the very end of the text
            cut_end = end
            if cut_end < text_length:
                # Look back up to 60 characters for sentence boundary or whitespace
                search_back_range = min(60, cut_end - start - 50)
                if search_back_range > 0:
                    sub = text[cut_end - search_back_range : cut_end]
                    # Find last punctuation or newline
                    break_pos = -1
                    for delim in ["\n\n", ".\n", ". ", "? ", "! ", "\n"]:
                        pos = sub.rfind(delim)
                        if pos != -1:
                            # Advance past punctuation
                            break_pos = cut_end - search_back_range + pos + len(delim)
                            break
                    if break_pos != -1 and break_pos > start:
                        cut_end = break_pos

            chunk_content = text[start:cut_end].strip()

            if chunk_content:
                chunk_id = f"{doc_id_prefix}_chunk_{chunk_idx}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        content=chunk_content,
                        source=source,
                        partition=part,
                        page=page,
                        char_start=start,
                        char_end=cut_end,
                        score=None,
                    )
                )
                chunk_idx += 1

            if cut_end >= text_length:
                break

            # Slide window: next window starts at (cut_end - chunk_overlap) to preserve continuous context
            next_start = cut_end - self.chunk_overlap
            if chunks and next_start <= chunks[-1].char_start:
                next_start = chunks[-1].char_start + 1
            start = max(start + 1, next_start)

        return chunks

    def chunk_documents(self, loaded_pages: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunks an entire collection of loaded pages/sections.
        """
        all_chunks: List[Chunk] = []

        for item_idx, item in enumerate(loaded_pages):
            content = item.get("content", "")
            source = item.get("source", f"doc_{item_idx}")
            page = item.get("page")
            partition = item.get("partition", "default")
            prefix = f"{partition}_{source}_p{page or item_idx}"

            page_chunks = self.chunk_text(
                text=content,
                source=source,
                page=page,
                prefix_id=prefix,
                partition=partition,
            )
            all_chunks.extend(page_chunks)

        return all_chunks
