"""
Unified Pipeline Orchestrator for RAG Question-Answering System.
Đồ án CO3101: Triển khai ánh xạ toán học cốt lõi F : (D, q) ↦ (a, E).
Kết nối 5 bước: Ingestion -> BGE-M3 Embedding -> HNSW Indexing -> Noise Filtering & Positioning -> Generation & Fact-Checking.
"""

import time
import unicodedata
import re
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

# Base imports
from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    SIMILARITY_THRESHOLD,
    RAW_DATA_DIR,
    VECTOR_SEARCH_TOP_K,
    BM25_TOP_K,
    RRF_TOP_N,
    RRF_K,
)
from src.schemas import Chunk, Evidence, RagResponse, ChatMessage
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import SlidingWindowChunker
from src.retrieval import (
    EmbeddingEngine,
    HNSWVectorStore,
    BM25Retriever,
    RerankerEngine,
    filter_and_reorder_contexts,
    reciprocal_rank_fusion,
)
from src.generation.llm_client import LLMClient, build_rag_prompt, SYSTEM_PROMPT_ANTI_HALLUCINATION


def clean_and_normalize_query(query: str) -> str:
    """
    Chuẩn hóa làm sạch câu hỏi đầu vào trước khi nhúng vector:
    1. Chuẩn hóa Unicode NFC (tương thích toàn diện tiếng Việt).
    2. Dọn dẹp khoảng trắng thừa, tab, xuống dòng.
    3. Loại bỏ dấu câu vô nghĩa ở đầu câu.
    4. Rút gọn dấu câu lặp lại liên tiếp ở cuối câu (ví dụ '???' -> '?').
    """
    if not query:
        return ""

    # 1. Unicode NFC
    q = unicodedata.normalize("NFC", query)

    # 2. Xóa khoảng trắng thừa và ký tự điều khiển
    q = re.sub(r"[\r\n\t]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    # 3. Xóa dấu câu vô nghĩa ở đầu câu
    q = re.sub(r"^[\.\,\;\:\?\!\-\_\=\+\*\/\\\s]+", "", q).strip()

    # 4. Rút gọn dấu câu lặp lại liên tiếp ở cuối câu
    q = re.sub(r"\?+", "?", q)
    q = re.sub(r"\!+", "!", q)
    q = re.sub(r"\.+", ".", q)

    return q


class RagPipeline:
    """
    End-to-end RAG QA System Pipeline.
    Implements mathematical mapping: F: (D, q) -> (a, E)
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        embedding_engine: Optional[EmbeddingEngine] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.vector_top_k = VECTOR_SEARCH_TOP_K
        self.bm25_top_k = BM25_TOP_K
        self.rrf_top_n = RRF_TOP_N

        self.loader = DocumentLoader()
        self.chunker = SlidingWindowChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        # Core engines: BGE-M3 Dense + HNSW + BM25 Sparse + Cross-Encoder Reranker
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.vector_store = HNSWVectorStore(dim=self.embedding_engine.dimension)
        self.bm25_retriever = BM25Retriever()
        self.reranker = RerankerEngine()
        self.llm_client = llm_client or LLMClient()

        # In-memory session history for conversational multi-turn understanding
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

        # Load existing index on disk and sync BM25
        self.vector_store.load()
        if self.vector_store.count > 0:
            self.bm25_retriever.fit(self.vector_store.chunks)

    def ingest_files(self, file_paths: List[Union[str, Path]], append: bool = False) -> int:
        """
        Ingests a list of document files, computes embeddings, and updates the HNSW index.
        If append=True: appends new chunks incrementally.
        If append=False: re-initializes a fresh index from scratch (Full Rebuild).
        Returns the number of new indexed chunks.
        """
        all_pages = []
        for fp in file_paths:
            pages = self.loader.load_file(fp)
            all_pages.extend(pages)

        if not all_pages:
            print("[RagPipeline] No valid content found in provided files.")
            return 0

        chunks = self.chunker.chunk_documents(all_pages)
        if not chunks:
            print("[RagPipeline] No chunks produced.")
            return 0

        texts = [c.content for c in chunks]
        embeddings = self.embedding_engine.embed_texts(texts)

        # Re-initialize only if not appending or index is missing
        if not append or self.vector_store.index is None:
            self.vector_store.initialize_new(dim=self.embedding_engine.dimension)

        self.vector_store.add_chunks(chunks, embeddings)
        self.vector_store.save()
        self.bm25_retriever.fit(self.vector_store.chunks)

        mode_str = "gia tăng" if append else "toàn bộ"
        print(f"[Pipeline] Đã nạp {len(chunks)} chunks [{mode_str}] (Tổng kho: {self.vector_store.count}).")
        return len(chunks)

    def ingest_directory(
        self, dir_path: Union[str, Path] = RAW_DATA_DIR, incremental: bool = False
    ) -> Dict[str, Any]:
        """
        Ingests documents in a directory and any partition subdirectories.
        If incremental=True: only ingests files that have NOT been indexed yet.
        If incremental=False: full rebuild of all documents.
        """
        path = Path(dir_path)
        all_files = [
            f for f in path.rglob("*")
            if f.is_file() and f.suffix.lower() in self.loader.SUPPORTED_EXTENSIONS
            and not f.name.startswith(".") and not f.parent.name.startswith(".")
        ]

        if not all_files:
            print(f"[RagPipeline] No supported files found in {dir_path}")
            if not incremental:
                self.vector_store.initialize_new(dim=self.embedding_engine.dimension)
                self.vector_store.save()
                self.bm25_retriever.fit([])
            return {"new_files": [], "new_chunks": 0, "total_chunks": self.vector_store.count}

        if incremental:
            indexed_sources = self.vector_store.get_indexed_sources()
            new_files = []
            for f in all_files:
                parent_name = f.parent.name
                part = parent_name if parent_name != path.name and not parent_name.startswith(".") else "default"
                source_key = f"{part.lower()}/{f.name}"
                if (f.name not in indexed_sources and str(f) not in indexed_sources and source_key not in indexed_sources):
                    new_files.append(f)

            if not new_files:
                print("[RagPipeline] All documents are already indexed. No new files to ingest.")
                return {"new_files": [], "new_chunks": 0, "total_chunks": self.vector_store.count}

            new_chunks_count = self.ingest_files(new_files, append=True)
            return {
                "new_files": [f.name for f in new_files],
                "new_chunks": new_chunks_count,
                "total_chunks": self.vector_store.count,
            }
        else:
            total_chunks = self.ingest_files(all_files, append=False)
            return {
                "new_files": [f.name for f in all_files],
                "new_chunks": total_chunks,
                "total_chunks": self.vector_store.count,
            }

    def remove_document(self, filename: str, partition: Optional[str] = None) -> int:
        """
        Removes a document and its chunks from the RAG pipeline:
        1. Removes vectors from HNSW Vector Store instantaneously.
        2. Re-synchronizes the BM25 lexical retriever with remaining chunks.
        Returns the number of removed chunks.
        """
        removed_count = self.vector_store.remove_by_source(filename, partition=partition)
        self.bm25_retriever.fit(self.vector_store.chunks)
        print(f"[Pipeline] Đã xóa '{filename}' ({removed_count} chunks).")
        return removed_count

    def remove_partition(self, partition: str) -> int:
        """Removes all documents and chunks associated with an entire partition."""
        removed_count = self.vector_store.remove_by_partition(partition)
        self.bm25_retriever.fit(self.vector_store.chunks)
        print(f"[Pipeline] Đã xóa phân vùng '{partition}' ({removed_count} chunks).")
        return removed_count

    def clear_session(self, session_id: str):
        """Clears memory for a specific chat session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        model_name: Optional[str] = None,
        gemini_key: Optional[str] = None,
        session_id: Optional[str] = None,
        partition: Optional[str] = None,
    ) -> RagResponse:
        """
        Executes the 5-step RAG pipeline for a given user query:
        F: (D, q) -> (a, E)
        Thread-safe: allows per-query parameter overrides without mutating global instance state.
        Supports multi-turn conversation memory when session_id is provided.
        Supports targeted partition/collection retrieval filtering.
        """
        start_time = time.time()
        raw_question = question

        # Lấy lịch sử hội thoại nếu có phiên làm việc
        history = self.sessions.get(session_id, []) if session_id else []

        # Step 0: Chuẩn hóa câu hỏi
        normalized_q = clean_and_normalize_query(question)

        # Active LLM Client (cho phép ghi đè API key / model theo từng truy vấn)
        active_llm = self.llm_client
        if (gemini_key is not None) or (model_name and model_name.strip()):
            active_llm = LLMClient(
                gemini_api_key=gemini_key if gemini_key is not None else self.llm_client.gemini_key,
                model_name=model_name or self.llm_client.model_name,
            )

        # Step 0.5: Contextual Query Rewriting cho hội thoại đa lượt
        search_query = normalized_q
        rewritten_q_result = None
        if history and len(history) > 0:
            try:
                candidate_rewritten = active_llm.rewrite_query(normalized_q, history)
                if candidate_rewritten and candidate_rewritten.strip():
                    cleaned_rewritten = clean_and_normalize_query(candidate_rewritten)
                    if cleaned_rewritten and cleaned_rewritten != normalized_q:
                        search_query = cleaned_rewritten
                        rewritten_q_result = search_query
                        print(f"[QueryRewriter] Tối ưu: '{normalized_q}' -> '{search_query}'")
            except Exception:
                pass

        raw_k = top_k if (top_k is not None and top_k > 0) else self.top_k
        k = min(5, max(1, raw_k))
        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold

        if not normalized_q:
            return RagResponse(
                question=raw_question,
                rewritten_query=None,
                answer="Vui lòng nhập câu hỏi hợp lệ.",
                evidences=[],
                latency_seconds=0.0,
                model_used="none",
                dense_count=0,
                sparse_count=0,
                fused_count=0,
                filtered_count=0,
                partition_used=partition or "all",
            )

        # Step 1: BGE-M3 Dense Search qua FAISS HNSW (Mỗi bên lấy 15 chunks)
        query_embedding = self.embedding_engine.embed_query(search_query)
        dense_chunks = self.vector_store.search(query_embedding, top_k=self.vector_top_k, partition=partition)
        dense_count = len(dense_chunks)

        # Step 2: BM25 Sparse Lexical Search (Mỗi bên lấy 15 chunks)
        sparse_pairs = self.bm25_retriever.search(search_query, top_k=self.bm25_top_k, partition=partition)
        sparse_chunks = [c for c, _ in sparse_pairs]
        sparse_count = len(sparse_chunks)

        # Step 3: RRF (Reciprocal Rank Fusion) dung hợp cả hai lấy tổng 10 chunks
        fused_candidates = reciprocal_rank_fusion(
            dense_results=dense_chunks,
            sparse_results=sparse_chunks,
            k_rrf=RRF_K,
            top_n=self.rrf_top_n,  # Tổng lấy 10 chunks
        )
        fused_count = len(fused_candidates)

        # Step 4: Rerank với Cross-Encoder BAAI/bge-reranker-v2-m3 trên Top 10 ứng viên
        if fused_candidates:
            reranked_pairs = self.reranker.rerank(
                query=search_query,
                candidates=fused_candidates,
                top_k=len(fused_candidates),
            )
            for chunk, score in reranked_pairs:
                chunk.score = score

            ranked_chunks = [c for c, _ in reranked_pairs]
            valid_chunks = filter_and_reorder_contexts(ranked_chunks, threshold=threshold)[:k]
            filtered_count = len(valid_chunks)

            # Tập bằng chứng minh bạch E
            evidences = [
                Evidence(
                    citation_id=idx,
                    chunk=chunk.content,
                    source=chunk.source,
                    partition=getattr(chunk, "partition", "default"),
                    page=chunk.page,
                    score=round(chunk.score, 4),
                )
                for idx, chunk in enumerate(valid_chunks, start=1)
            ]
        else:
            valid_chunks = []
            evidences = []
            filtered_count = 0

        # Step 5: Gọi mô hình sinh phản hồi (LLM Generation)
        if not valid_chunks:
            final_answer = (
                f"Dựa trên tài liệu được cung cấp, không tìm thấy thông tin nào phù hợp hoặc "
                f"các đoạn trích không đạt ngưỡng tương quan (τ ≥ {threshold:.2f}) để trả lời câu hỏi."
            )
            model_used = "Relevance-Threshold-Filter"
        else:
            # Chuyển đổi lịch sử sang đối tượng ChatMessage cho prompt
            chat_history_objs = [
                ChatMessage(role=m["role"], content=m["content"]) for m in history
            ] if history else None

            user_prompt = build_rag_prompt(
                question=search_query,
                context_chunks=valid_chunks,
                conversation_history=chat_history_objs,
            )

            raw_answer, model_used = active_llm.generate_response(
                prompt=user_prompt,
                system_instruction=SYSTEM_PROMPT_ANTI_HALLUCINATION,
            )
            final_answer = raw_answer

        # Ghi nhận lượt trao đổi vào bộ nhớ phiên
        if session_id:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            self.sessions[session_id].append({"role": "user", "content": raw_question})
            self.sessions[session_id].append({"role": "assistant", "content": final_answer})
            # Giữ tối đa 10 lượt gần nhất để tiết kiệm bộ nhớ
            if len(self.sessions[session_id]) > 20:
                self.sessions[session_id] = self.sessions[session_id][-20:]

        latency = time.time() - start_time

        return RagResponse(
            question=question,
            rewritten_query=rewritten_q_result,
            answer=final_answer,
            evidences=evidences,
            latency_seconds=round(latency, 3),
            model_used=model_used,
            dense_count=dense_count,
            sparse_count=sparse_count,
            fused_count=fused_count,
            filtered_count=filtered_count,
            partition_used=partition or "all",
        )
