"""
Ingestion module for loading documents and sliding window chunking.
"""
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import SlidingWindowChunker

__all__ = ["DocumentLoader", "SlidingWindowChunker"]
