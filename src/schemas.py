"""
Pydantic Schemas for RAG Question-Answering System.
Defines data structures for Chunks, Evidence, and System Responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A text chunk extracted from a document with full metadata traceability."""
    id: str = Field(..., description="Unique identifier of the chunk (e.g., doc_0_chunk_1)")
    content: str = Field(..., description="Raw text content of the chunk")
    source: str = Field(..., description="Origin source document file name")
    partition: Optional[str] = Field("default", description="Partition/Folder name to which the chunk belongs")
    page: Optional[int] = Field(None, description="Page number where the chunk originates (1-indexed)")
    char_start: Optional[int] = Field(None, description="Start character offset in page/document")
    char_end: Optional[int] = Field(None, description="End character offset in page/document")
    score: Optional[float] = Field(None, description="Cosine similarity score (tau) from vector retrieval")


class Evidence(BaseModel):
    """
    Transparent citation evidence item:
    e_i = (citation_id, chunk_i, source_i, page_i, score_i, partition_i)
    """
    citation_id: int = Field(1, description="Sequential citation index [1], [2], ...")
    chunk: str = Field(..., description="Text content of the cited evidence")
    source: str = Field(..., description="Origin file name")
    partition: Optional[str] = Field("default", description="Partition name")
    page: Optional[int] = Field(None, description="Page number (1-indexed)")
    score: float = Field(..., description="Cosine similarity score (tau)")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "chunk": self.chunk,
            "source": self.source,
            "partition": self.partition,
            "page": self.page,
            "score": round(self.score, 4),
        }


class ChatMessage(BaseModel):
    """Single turn message in conversational memory."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class RagResponse(BaseModel):
    """
    Mathematical mapping output: F: (D, q) -> (a, E)
    Contains the answer, transparent evidence set, and operational telemetry.
    """
    question: str = Field(..., description="Original user query q")
    rewritten_query: Optional[str] = Field(None, description="Contextual standalone rewritten query (if multi-turn)")
    answer: str = Field(..., description="Generated natural language answer a")
    evidences: List[Evidence] = Field(default_factory=list, description="Transparent citation evidence set E")
    latency_seconds: float = Field(0.0, description="End-to-end execution latency in seconds")
    model_used: str = Field("unknown", description="LLM provider and model used to generate answer")
    dense_count: int = Field(15, description="Number of candidate chunks from dense HNSW search (top 15)")
    sparse_count: int = Field(15, description="Number of candidate chunks from sparse BM25 search (top 15)")
    fused_count: int = Field(10, description="Total chunks selected after fusion (top 10)")
    filtered_count: int = Field(0, description="Number of chunks retained after tau thresholding")
    partition_used: Optional[str] = Field("all", description="Target partition used for search filtering")
