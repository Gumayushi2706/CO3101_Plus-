"""
FastAPI High-Performance Web Server for RAG Question-Answering System.
Đồ án tổng hợp CO3101 - Hướng Trí tuệ Nhân tạo - ĐH Bách Khoa ĐHQG-HCM.
Kiến trúc Single-Page Application (SPA) bất đồng bộ, tốc độ tải trang < 50ms.
"""

import os
import sys
import time
import shutil
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(BASE_DIR / ".env")

from src.config import (
    RAW_DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    TOP_K,
    SIMILARITY_THRESHOLD,
    VECTOR_SEARCH_TOP_K,
    BM25_TOP_K,
    RRF_TOP_N,
    GEMINI_API_KEY,
    LLM_MODEL_NAME,
)
from src.pipeline import RagPipeline
from src.retrieval.embedding import EmbeddingEngine
from src.generation.llm_client import LLMClient

# Global pipeline instance initialized on startup
rag_pipeline: Optional[RagPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline
    embedding_engine = EmbeddingEngine()
    llm_client = LLMClient(
        gemini_api_key=GEMINI_API_KEY,
        model_name=LLM_MODEL_NAME,
    )
    rag_pipeline = RagPipeline(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
        embedding_engine=embedding_engine,
        llm_client=llm_client,
    )
    print(f"[Server] Sẵn sàng tại http://127.0.0.1:8000 | Kho vector HNSW: {rag_pipeline.vector_store.count} chunks")
    yield


app = FastAPI(
    title="CO3101 RAG QA Studio",
    description="Hệ thống Hỏi - Đáp Tài liệu RAG tốc độ cao - ĐH Bách Khoa ĐHQG-HCM",
    version="2.0.0",
    lifespan=lifespan,
)

# Static files directory
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Pydantic Schemas for Requests
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    model_name: Optional[str] = None
    gemini_key: Optional[str] = None
    session_id: Optional[str] = None
    partition: Optional[str] = "all"


class CreatePartitionRequest(BaseModel):
    name: str


class ConfigUpdateRequest(BaseModel):
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    vector_top_k: Optional[int] = None
    bm25_top_k: Optional[int] = None
    rrf_top_n: Optional[int] = None
    model_name: Optional[str] = None
    gemini_key: Optional[str] = None


# Routes
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/stats")
async def get_stats():
    global rag_pipeline
    total_files = 0
    if RAW_DATA_DIR.exists():
        total_files += len([f for f in RAW_DATA_DIR.glob("*.*") if f.is_file() and not f.name.startswith(".")])
        for d in RAW_DATA_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                total_files += len([f for f in d.glob("*.*") if f.is_file() and not f.name.startswith(".")])
    vector_count = rag_pipeline.vector_store.count if rag_pipeline else 0
    bm25_count = rag_pipeline.bm25_retriever.count if rag_pipeline else 0

    return {
        "raw_files_count": total_files,
        "vector_count": vector_count,
        "bm25_count": bm25_count,
        "chunk_size": rag_pipeline.chunk_size if rag_pipeline else CHUNK_SIZE,
        "chunk_overlap": rag_pipeline.chunk_overlap if rag_pipeline else CHUNK_OVERLAP,
        "top_k": rag_pipeline.top_k if rag_pipeline else TOP_K,
        "similarity_threshold": rag_pipeline.similarity_threshold if rag_pipeline else SIMILARITY_THRESHOLD,
        "vector_top_k": rag_pipeline.vector_top_k if rag_pipeline else VECTOR_SEARCH_TOP_K,
        "bm25_top_k": rag_pipeline.bm25_top_k if rag_pipeline else BM25_TOP_K,
        "rrf_top_n": rag_pipeline.rrf_top_n if rag_pipeline else RRF_TOP_N,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "hnsw_m": HNSW_M,
        "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
        "hnsw_ef_search": HNSW_EF_SEARCH,
        "current_model": rag_pipeline.llm_client.model_name if rag_pipeline else LLM_MODEL_NAME,
    }


@app.get("/api/partitions")
async def list_partitions():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    default_files = [f for f in RAW_DATA_DIR.glob("*.*") if f.is_file() and not f.name.startswith(".")]
    subfolders = sorted([d for d in RAW_DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")], key=lambda x: x.name)

    part_chunk_counts = {}
    if rag_pipeline and rag_pipeline.vector_store.chunks_metadata:
        for m in rag_pipeline.vector_store.chunks_metadata:
            p = str(m.get("partition", "default"))
            part_chunk_counts[p] = part_chunk_counts.get(p, 0) + 1

    total_files = len(default_files)
    sub_parts = []
    for d in subfolders:
        sub_files = [f for f in d.glob("*.*") if f.is_file() and not f.name.startswith(".")]
        total_files += len(sub_files)
        sub_parts.append({
            "id": d.name,
            "name": d.name,
            "is_default": False,
            "is_all": False,
            "file_count": len(sub_files),
            "vector_count": part_chunk_counts.get(d.name, 0),
        })

    partitions = [
        {"id": "all", "name": "Tất cả tài liệu", "is_default": False, "is_all": True, "file_count": total_files, "vector_count": rag_pipeline.vector_store.count if rag_pipeline else 0},
        {"id": "default", "name": "Default", "is_default": True, "is_all": False, "file_count": len(default_files), "vector_count": part_chunk_counts.get("default", 0)},
        *sub_parts,
    ]
    return {"partitions": partitions}


@app.post("/api/partitions")
async def create_partition(req: CreatePartitionRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Tên phân vùng không được để trống")
    
    # Check invalid chars for directory name
    invalid_chars = set(r'\/:*?"<>|')
    if any(ch in invalid_chars for ch in name) or name.startswith("."):
        raise HTTPException(status_code=400, detail="Tên phân vùng chứa ký tự không hợp lệ")

    if name.lower() in ("default", "all", "chung", "tất cả"):
        raise HTTPException(status_code=400, detail="Tên phân vùng trùng với phân vùng hệ thống")

    part_dir = RAW_DATA_DIR / name
    if part_dir.exists():
        raise HTTPException(status_code=400, detail="Phân vùng này đã tồn tại")

    part_dir.mkdir(parents=True, exist_ok=True)
    return {"status": "success", "message": f"Đã tạo phân vùng '{name}' thành công"}


@app.delete("/api/partitions/{partition_name}")
async def delete_partition(partition_name: str):
    global rag_pipeline
    clean_name = partition_name.strip()
    if clean_name.lower() in ("default", "all", "chung", "tất cả"):
        raise HTTPException(status_code=400, detail="Không thể xóa phân vùng mặc định")

    part_dir = RAW_DATA_DIR / clean_name
    if not part_dir.exists() or not part_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy phân vùng '{clean_name}'")

    # Remove all physical files in the partition folder
    try:
        shutil.rmtree(part_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa thư mục phân vùng: {e}")

    # Remove all vectors belonging to this partition
    removed_chunks = 0
    if rag_pipeline:
        removed_chunks = rag_pipeline.remove_partition(clean_name)

    return {
        "status": "success",
        "message": f"Đã xóa thành công phân vùng '{clean_name}'",
        "removed_chunks": removed_chunks,
        "total_vectors": rag_pipeline.vector_store.count if rag_pipeline else 0,
    }


@app.get("/api/documents")
async def list_documents(partition: Optional[str] = Query(None)):
    files_data = []
    indexed_sources = rag_pipeline.vector_store.get_indexed_sources() if rag_pipeline else set()
    target_part = partition.lower() if partition and partition.lower() not in ("all", "*", "tất cả") else None

    def process_file(f: Path, part: str):
        if f.is_file() and not f.name.startswith("."):
            stat = f.stat()
            source_key = f"{part.lower()}/{f.name}"
            is_indexed = (f.name in indexed_sources or str(f) in indexed_sources or source_key in indexed_sources)
            files_data.append({
                "name": f.name,
                "partition": part,
                "extension": f.suffix.lower(),
                "size_kb": round(stat.st_size / 1024, 2),
                "modified_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                "is_indexed": is_indexed,
            })

    if RAW_DATA_DIR.exists():
        if target_part is None or target_part == "default":
            for f in RAW_DATA_DIR.glob("*.*"):
                process_file(f, "default")

        for d in RAW_DATA_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                if target_part is None or d.name.lower() == target_part:
                    for f in d.glob("*.*"):
                        process_file(f, d.name)

    return {"documents": sorted(files_data, key=lambda x: (x["partition"], x["name"]))}


@app.post("/api/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    partition: str = Form("default")
):
    target_part = partition.strip()
    if not target_part or target_part.lower() in ("default", "all", "chung"):
        dest_dir = RAW_DATA_DIR
        actual_part = "default"
    else:
        dest_dir = RAW_DATA_DIR / target_part
        actual_part = target_part

    dest_dir.mkdir(parents=True, exist_ok=True)
    uploaded = []
    for file in files:
        if file.filename.startswith("."):
            continue
        file_path = dest_dir / file.filename
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        uploaded.append(file.filename)
    
    # Mặc định tự động phân đoạn và lập chỉ mục ngay khi tải tệp lên
    new_chunks = 0
    if rag_pipeline and uploaded:
        res = rag_pipeline.ingest_directory(RAW_DATA_DIR, incremental=True)
        new_chunks = res.get("new_chunks", 0)

    return {
        "status": "success",
        "partition": actual_part,
        "uploaded_files": uploaded,
        "new_chunks": new_chunks,
        "total_vectors": rag_pipeline.vector_store.count if rag_pipeline else 0,
    }


async def _do_delete_document(filename: str, partition: Optional[str] = None):
    global rag_pipeline
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="Pipeline chưa sẵn sàng")
    
    safe_name = Path(filename).name
    
    # Try finding file in partition folder or root
    candidate_paths = []
    if partition and partition.lower() not in ("default", "all"):
        candidate_paths.append(RAW_DATA_DIR / partition / safe_name)
    candidate_paths.append(RAW_DATA_DIR / safe_name)
    # Also check other subfolders
    if RAW_DATA_DIR.exists():
        for d in RAW_DATA_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                candidate_paths.append(d / safe_name)

    file_deleted = False
    for p in candidate_paths:
        if p.exists() and p.is_file():
            try:
                p.unlink()
                file_deleted = True
                break
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Không thể xóa tệp: {e}")

    # Xóa sạch vector và đồng bộ lại FAISS HNSW + BM25 ngay lập tức (< 10ms)
    removed_chunks = rag_pipeline.remove_document(safe_name, partition=partition)

    if not file_deleted and removed_chunks == 0:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy tài liệu hoặc chỉ mục của '{safe_name}'")

    return {
        "status": "success",
        "message": f"Đã xóa thành công tài liệu '{safe_name}'",
        "removed_chunks": removed_chunks,
        "total_vectors": rag_pipeline.vector_store.count,
    }


@app.delete("/api/documents/{filename:path}")
async def delete_document_path(filename: str, partition: Optional[str] = Query(None)):
    return await _do_delete_document(filename, partition=partition)


@app.delete("/api/documents")
async def delete_document_query(filename: str = Query(None), partition: Optional[str] = Query(None)):
    if not filename:
        raise HTTPException(status_code=400, detail="Thiếu tên tài liệu cần xóa")
    return await _do_delete_document(filename, partition=partition)


@app.post("/api/index/incremental")
async def incremental_index():
    global rag_pipeline
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="Pipeline chưa sẵn sàng")
    
    t0 = time.time()
    res = rag_pipeline.ingest_directory(RAW_DATA_DIR, incremental=True)
    duration = time.time() - t0
    
    return {
        "status": "success",
        "new_files": res["new_files"],
        "new_chunks": res["new_chunks"],
        "duration_seconds": round(duration, 2),
        "total_vectors": res["total_chunks"],
    }


@app.post("/api/index/rebuild")
async def rebuild_index():
    global rag_pipeline
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="Pipeline chưa sẵn sàng")
    
    t0 = time.time()
    res = rag_pipeline.ingest_directory(RAW_DATA_DIR, incremental=False)
    duration = time.time() - t0
    
    return {
        "status": "success",
        "chunks_indexed": res["new_chunks"],
        "duration_seconds": round(duration, 2),
        "total_vectors": res["total_chunks"],
    }



@app.post("/api/config")
async def update_config(config: ConfigUpdateRequest):
    global rag_pipeline
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="Pipeline chưa khởi tạo")
    
    if config.chunk_size is not None:
        rag_pipeline.chunk_size = config.chunk_size
        rag_pipeline.chunker.chunk_size = config.chunk_size
    if config.chunk_overlap is not None:
        rag_pipeline.chunk_overlap = config.chunk_overlap
        rag_pipeline.chunker.chunk_overlap = config.chunk_overlap
    if config.top_k is not None:
        rag_pipeline.top_k = config.top_k
    if config.similarity_threshold is not None:
        rag_pipeline.similarity_threshold = config.similarity_threshold
    if config.vector_top_k is not None:
        rag_pipeline.vector_top_k = config.vector_top_k
    if config.bm25_top_k is not None:
        rag_pipeline.bm25_top_k = config.bm25_top_k
    if config.rrf_top_n is not None:
        rag_pipeline.rrf_top_n = config.rrf_top_n
    if config.model_name is not None and config.model_name.strip():
        rag_pipeline.llm_client.model_name = config.model_name.strip()
    if config.gemini_key is not None and config.gemini_key.strip():
        rag_pipeline.llm_client.gemini_key = config.gemini_key.strip()

    return {"status": "success", "message": "Đã cập nhật cấu hình pipeline thành công"}


@app.delete("/api/chat/history/{session_id}")
async def clear_session_history(session_id: str):
    global rag_pipeline
    if rag_pipeline:
        rag_pipeline.clear_session(session_id)
    return {"status": "success", "message": f"Đã xóa lịch sử phiên {session_id}"}


@app.post("/api/query")
async def query_endpoint(req: QueryRequest):
    global rag_pipeline
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="Pipeline chưa sẵn sàng")
    
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")

    res = rag_pipeline.query(
        question=req.question,
        top_k=req.top_k or rag_pipeline.top_k,
        similarity_threshold=req.similarity_threshold or rag_pipeline.similarity_threshold,
        model_name=req.model_name,
        gemini_key=req.gemini_key,
        session_id=req.session_id,
        partition=req.partition,
    )

    return {
        "question": res.question,
        "rewritten_query": res.rewritten_query,
        "answer": res.answer,
        "latency_seconds": res.latency_seconds,
        "model_used": res.model_used,
        "dense_count": res.dense_count,
        "sparse_count": res.sparse_count,
        "fused_count": res.fused_count,
        "filtered_count": res.filtered_count,
        "partition_used": res.partition_used,
        "evidences": [e.to_dict() for e in res.evidences],
    }
