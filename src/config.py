"""
Configuration module for RAG Question-Answering System.
CO3101 - Hệ thống hỏi - đáp trên tài liệu dựa trên RAG - ĐH Bách Khoa ĐHQG-HCM.
"""

from pathlib import Path
import os
import sys
import logging
import warnings
from dotenv import load_dotenv

# Suppress progress bars and noisy library logs
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Ensure UTF-8 console output on Windows
def setup_utf8_console():
    for stream in (sys.stdout, sys.stderr):
        if stream and hasattr(stream, "reconfigure"):
            try:
                if stream.encoding and stream.encoding.lower() != "utf-8":
                    stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

setup_utf8_console()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
STORAGE_DIR = BASE_DIR / "storage"
FAISS_STORAGE_DIR = STORAGE_DIR / "faiss_index"

# Load environment variables from .env in project root
load_dotenv(BASE_DIR / ".env")

# Ensure essential directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
FAISS_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# File paths for FAISS index persistence
INDEX_FILE_PATH = FAISS_STORAGE_DIR / "index.bin"
METADATA_FILE_PATH = FAISS_STORAGE_DIR / "metadata.json"

# ==========================================
# 1. Text Chunking (Sliding Window)
# ==========================================
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

# ==========================================
# 2. Vector Embedding (BGE-M3)
# ==========================================
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1024"))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
NORMALIZE_L2: bool = True

# ==========================================
# 3. HNSW Vector Indexing (faiss.IndexHNSWFlat)
# ==========================================
HNSW_M: int = int(os.getenv("HNSW_M", "16"))
HNSW_EF_CONSTRUCTION: int = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
HNSW_EF_SEARCH: int = int(os.getenv("HNSW_EF_SEARCH", "50"))
TOP_K: int = int(os.getenv("TOP_K", "3"))

# ==========================================
# 4. Retrieval & Reranking Candidate Pools
# ==========================================
VECTOR_SEARCH_TOP_K: int = int(os.getenv("VECTOR_SEARCH_TOP_K", "15"))
BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "15"))
RRF_TOP_N: int = int(os.getenv("RRF_TOP_N", "10"))
RRF_K: int = int(os.getenv("RRF_K", "60"))
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.50"))
RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

# ==========================================
# 5. LLM Generation
# ==========================================
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemini-3.5-flash-lite")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()

TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))
