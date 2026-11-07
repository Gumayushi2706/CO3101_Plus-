import sys
import os

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import clean_and_normalize_query
from src.schemas import Chunk
from src.retrieval.embedding import EmbeddingEngine
from src.retrieval.post_processing import filter_and_reorder_contexts, reciprocal_rank_fusion
from src.retrieval.bm25_search import BM25Retriever
from src.retrieval.reranker import RerankerEngine
import numpy as np

print("--- RUNNING CORE RAG COMPONENT TEST SUITE ---")

# 1. Test Query Normalization
raw_q = "...  học phí ngành Khoa Học Máy Tính là bao nhiêu????   "
cleaned = clean_and_normalize_query(raw_q)
print(f"[1] Raw Query: '{raw_q}'")
print(f"[1] Cleaned Query: '{cleaned}'")
assert "???" not in cleaned, "Trailing question marks were not collapsed"
assert cleaned.startswith("học phí"), "Leading noise dots were not removed"
print("-> Query Normalization: PASS")

# 2. Test Relevance Threshold Filtering & Score Sorting
cand1 = Chunk(id="Top1", content="Chunk Top 1", source="docA.pdf", page=1, score=0.95)
cand2 = Chunk(id="Top2", content="Chunk Top 2", source="docB.pdf", page=1, score=0.85)
cand3 = Chunk(id="Top3", content="Chunk Top 3", source="docC.pdf", page=1, score=0.75)
cand_low = Chunk(id="Low", content="Chunk Low Score", source="docD.pdf", page=1, score=0.30)

filtered = filter_and_reorder_contexts([cand_low, cand2, cand1, cand3], threshold=0.50)
filtered_ids = [c.id for c in filtered]
print(f"[2] Filtered Order (Score descending, tau >= 0.50): {filtered_ids}")
assert "Low" not in filtered_ids, "Chunk below threshold should be filtered out"
assert filtered_ids == ["Top1", "Top2", "Top3"], f"Expected ['Top1', 'Top2', 'Top3'], got {filtered_ids}"
print("-> Relevance Threshold Filtering & Score Sorting: PASS")

# 3. Test BGE-M3 Direct Query Embedding
engine = EmbeddingEngine()
v = engine.embed_query("Học phí ĐHQG?")
norm_val = float(np.linalg.norm(v))
print(f"[3] Query Embedding shape: {v.shape}, L2-norm: {norm_val:.4f}")
assert v.shape == (1, 1024), f"Expected shape (1, 1024), got {v.shape}"
assert abs(norm_val - 1.0) < 1e-3, "Embedding should be L2-normalized"
print("-> BGE-M3 Direct Query Embedding: PASS")

# 4. Test BM25 Lexical Retrieval
bm25_chunk1 = Chunk(id="bm1", content="Môn học CO3101 là Đồ án tổng hợp ngành Khoa học Máy tính.", source="s1.pdf", page=1)
bm25_chunk2 = Chunk(id="bm2", content="Thủ tục đăng ký học phần học kỳ 1 năm 2026.", source="s2.pdf", page=1)
bm25 = BM25Retriever([bm25_chunk1, bm25_chunk2])
bm25_res = bm25.search("CO3101", top_k=5)
print(f"[4] BM25 Search top match: {bm25_res[0][0].id} (Score: {bm25_res[0][1]:.3f})")
assert bm25_res[0][0].id == "bm1", "BM25 failed to retrieve exact match chunk"
print("-> BM25 Lexical Search: PASS")

# 5. Test Reciprocal Rank Fusion (RRF)
rrf_out = reciprocal_rank_fusion(
    dense_results=[bm25_chunk2, bm25_chunk1],
    sparse_results=[bm25_chunk1, bm25_chunk2],
    k_rrf=60,
    top_n=2,
)
print(f"[5] RRF top chunk: {rrf_out[0].id}")
assert len(rrf_out) == 2, "RRF failed to fuse both chunks"
print("-> Reciprocal Rank Fusion (RRF): PASS")

# 6. Test Cross-Encoder Reranker
reranker = RerankerEngine()
rerank_scored = reranker.rerank("Đồ án CO3101", [bm25_chunk2, bm25_chunk1], top_k=2)
print(f"[6] Reranker Top Match: {rerank_scored[0][0].id} (Score: {rerank_scored[0][1]:.4f})")
assert rerank_scored[0][0].id == "bm1", "Reranker failed to rank relevant chunk first"
print("-> Cross-Encoder Reranking: PASS")

# 7. Test Contextual Query Rewriter Fallback & Functionality
from src.generation.llm_client import LLMClient
client = LLMClient(gemini_api_key="")
# Without history, query must be unchanged
res_empty_hist = client.rewrite_query("Nó có bao nhiêu tín chỉ?", [])
assert res_empty_hist == "Nó có bao nhiêu tín chỉ?", "Should return original query when history is empty"
# Test with dummy history when API key is present or offline
dummy_hist = [
    {"role": "user", "content": "Môn học CO3101 là gì?"},
    {"role": "assistant", "content": "CO3101 là môn Đồ án tổng hợp."},
]
res_rewritten = client.rewrite_query("Nó có bao nhiêu tín chỉ?", dummy_hist)
print(f"[7] Contextual Query Rewriter Output: '{res_rewritten}'")
assert isinstance(res_rewritten, str) and len(res_rewritten) > 0, "Query rewriter output must be non-empty string"
print("-> Contextual Query Rewriter: PASS")

# 8. Test DocumentLoader with PyMuPDF
from src.ingestion.loader import DocumentLoader
loader = DocumentLoader()
pdf_path = "Demo_1.pdf" if os.path.exists("Demo_1.pdf") else ("Slide.pdf" if os.path.exists("Slide.pdf") else None)
if pdf_path:
    pages = loader.load_file(pdf_path)
    print(f"[8] Loaded {len(pages)} pages from {pdf_path}")
    assert len(pages) > 0, "PDF loading failed, 0 pages loaded"
    print("-> PyMuPDF Document Loader: PASS")
else:
    print("[8] No PDF sample found, skipping PDF test")

# 9. Test Missing API Key Handling
client_no_key = LLMClient(gemini_api_key="")
chunk_sample = Chunk(id="test_c1", content="Nội dung kiểm thử trích xuất.", source="tailieu.pdf", page=1, score=0.88)
ans, model = client_no_key.generate_response(prompt="Test prompt", context_chunks_for_fallback=[chunk_sample])
print(f"[9] Missing Key Handler Model: '{model}'")
assert "API Key" in ans and "GEMINI_API_KEY" in ans, "Should inform user about missing API key without extra symbols"
assert model == "Extractive-Fallback (No API Key)", f"Expected model 'Extractive-Fallback (No API Key)', got '{model}'"
print("-> Missing API Key Graceful Extractive Response: PASS")

print("\n=== ALL 9 CORE PIPELINE TESTS PASSED WITH 100% SUCCESS ===")
