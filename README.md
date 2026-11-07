# Hệ thống Hỏi - Đáp trên Tài liệu dựa trên RAG (CO3101)
**Khoa Khoa học và Kỹ thuật Máy tính - Trường Đại học Bách Khoa, ĐHQG TP.HCM**  
**Hướng đề tài:** Trí tuệ Nhân tạo (Artificial Intelligence) | **Lớp:** L06 - HK261  
**Giảng viên hướng dẫn:** ThS. Vương Bá Thịnh  

---

## 🎯 1. Phát biểu bài toán
Hệ thống hiện thực hóa ánh xạ toán học từ tập tài liệu nguồn và câu hỏi người dùng đến câu trả lời tự nhiên kèm tập bằng chứng đối soát:

$$F : (D, q) \mapsto (a, E)$$

- **$D$**: Tập tài liệu nguồn (**PDF, DOCX, TXT**) được tổ chức theo các phân vùng (*Partitions*).
- **$q$**: Câu hỏi của người dùng (hỗ trợ ngữ cảnh hội thoại đa lượt).
- **$a$**: Câu trả lời tự nhiên từ mô hình ngôn ngữ lớn (LLM), trực diện, tuân thủ nguyên tắc chống ảo giác (*Anti-hallucination*).
- **$E$**: Tập bằng chứng minh bạch: $E = \{e_1, e_2, \dots, e_k\}$ với $e_i = (\text{id}, \text{chunk}, \text{source}, \text{partition}, \text{page}, \tau)$.

---

## ⚙️ 2. Kiến trúc luồng xử lý (5-Step Pipeline)

```
[Tài liệu D] ──> [Loader & Chunker] ──> [Kho Vector HNSW & Chỉ mục BM25]
                                                       │
[Câu hỏi q] ──> [Chuẩn hóa & Query Rewriter (Đa lượt)] │
      │                                                │
      ├──> Dense Search (BGE-M3 + FAISS HNSW) ─────────┼──> Top 15 Dense
      └──> Sparse Search (BM25Plus Lexical) ───────────┼──> Top 15 Sparse
                                                       │
                                                       ▼
      [Reciprocal Rank Fusion - RRF] ───────────────> Top 10 Fused Chunks
                                                       │
                                                       ▼
      [Cross-Encoder BGE-Reranker-v2-m3] ───────────> Sigmoid Reranking
      [Lọc ngưỡng tương quan tau >= 0.50] ───────────> Top-k Chunks (E)
                                                       │
                                                       ▼
      [LLM Generation: Google Gemini 3.5 Flash-Lite] ─> Câu trả lời (a) + Bằng chứng (E)
```

### Chi tiết các bước thực thi:
1. **Tiền xử lý & Phân đoạn (Ingestion & Chunking):**
   - Đọc tệp PDF (PyMuPDF tự động chuyển bảng biểu sang Markdown), DOCX, TXT với chuẩn hóa Unicode NFC.
   - Phân đoạn bằng thuật toán Sliding Window ($900$ ký tự, overlap $100$ ký tự), tự động dò tìm ranh giới câu.
2. **Truy xuất kết hợp (Hybrid Retrieval):**
   - *Contextual Query Rewriting:* Tự động viết lại câu hỏi đa lượt thành câu hỏi độc lập trước khi truy vấn.
   - *Dense Retrieval:* Nhúng vector 1024 chiều bằng **BGE-M3**, truy vấn qua chỉ mục **FAISS HNSW** ($M=16, efC=200, efS=50$) lấy Top 15 chunks.
   - *Sparse Retrieval:* Khớp từ khóa và mã học phần bằng thuật toán **BM25Plus** lấy Top 15 chunks.
3. **Dung hợp thứ hạng tương hỗ (RRF):**
   - Hợp nhất kết quả Dense và Sparse theo công thức: $\text{RRF}(d) = \sum \frac{1}{60 + \text{rank}(d)}$, chọn lọc Top 10 ứng viên.
4. **Tái xếp hạng Cross-Encoder & Lọc chống ảo giác:**
   - Đánh giá tương quan ngữ nghĩa sâu bằng **BGE-Reranker-v2-m3**, chuẩn hóa qua hàm Sigmoid.
   - Lọc ngưỡng tương quan tối thiểu $\tau \ge 0.50$, chọn Top-$k$ ($k=3$) bằng chứng đạt chuẩn làm tập $E$.
5. **Sinh câu trả lời (LLM Generation):**
   - Đưa câu hỏi và ngữ cảnh bằng chứng vào **Google Gemini 3.5 Flash-Lite** với System Prompt nghiêm ngặt cấm suy diễn, gắn nhãn trích dẫn `[i]` đối soát trực diện và minh bạch.

---

## 🗂️ 3. Phân vùng Dữ liệu (Partitions)
- Cho phép tạo các phân vùng độc lập (ví dụ: `Default`, `Luat_ATTP`, `DoAn_CO3101`) nhằm giới hạn phạm vi tìm kiếm, triệt tiêu nhiễu chéo giữa các lĩnh vực kiến thức.
- **Xóa tài liệu tức thời:** Tái cấu trúc vector trực tiếp từ bộ nhớ đệm FAISS (`IndexHNSWFlat.reconstruct_n`) và đồng bộ lại metadata cùng chỉ mục BM25 ngay lập tức mà không cần tính toán lại embedding.

---

## 📂 4. Cấu trúc thư mục

```text
CO3101/
├── data/raw/              # Tài liệu nguồn (PDF, DOCX, TXT) và các thư mục phân vùng con
├── src/
│   ├── config.py          # Cấu hình tập trung siêu tham số & môi trường UTF-8
│   ├── schemas.py         # Pydantic Models (Chunk, Evidence, RagResponse, ChatMessage)
│   ├── pipeline.py        # Pipeline kết nối 5 bước (Ingestion -> Hybrid Search -> Rerank -> LLM)
│   ├── ingestion/         # Module xử lý tài liệu (loader.py, chunker.py)
│   ├── retrieval/         # Module truy xuất (embedding.py, hnsw_index.py, bm25_search.py, reranker.py, post_processing.py)
│   ├── generation/        # Module sinh phản hồi (llm_client.py)
│   └── ui/                # Giao diện Web SPA (FastAPI server.py, static HTML/CSS/JS)
├── storage/faiss_index/   # Lưu trữ nhị phân chỉ mục FAISS (index.bin, metadata.json)
├── tests/                 # Bộ kiểm thử tự động (test_optimizations.py)
├── main.py                # CLI Runner thử nghiệm trên Terminal
├── requirements.txt       # Danh mục thư viện phụ thuộc tinh gọn
├── .env.example           # Tệp mẫu cấu hình biến môi trường
└── README.md              # Tài liệu kỹ thuật
```

---

## 🛠️ 5. Cài đặt & Khởi chạy

### Bước 1: Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10 - 3.13:
```bash
pip install -r requirements.txt
```

### Bước 2: Cấu hình biến môi trường
Tạo tệp `.env` từ `.env.example`:
```bash
# Windows:
copy .env.example .env

# Linux / macOS:
cp .env.example .env
```
Mở `.env` và điền khóa API Google Gemini:
```env
GEMINI_API_KEY=AIzaSy...
```

### Bước 3: Vận hành hệ thống

> **Mẹo trên Windows:** Nếu máy có nhiều phiên bản Python, hãy dùng lệnh `py -3.13` (ví dụ `py -3.13 main.py ...` hoặc `py -3.13 -m uvicorn ...`).

#### 1. Chạy trên Terminal (CLI Runner):
- **Nạp toàn bộ tài liệu:**
  ```bash
  python main.py --ingest
  # hoặc trên Windows: py -3.13 main.py --ingest
  ```
- **Nạp gia tăng (chỉ quét tệp mới):**
  ```bash
  python main.py --incremental
  ```
- **Truy vấn nhanh 1 câu:**
  ```bash
  python main.py --query "Môn học CO3101 là gì?"
  ```
- **Hội thoại đa lượt tương tác:**
  ```bash
  python main.py --interactive
  ```

#### 2. Khởi chạy Giao diện Web SPA (FastAPI):
```bash
uvicorn src.ui.server:app --host 127.0.0.1 --port 8000
# hoặc trên Windows: py -3.13 -m uvicorn src.ui.server:app --host 127.0.0.1 --port 8000
```
Truy cập: **`http://127.0.0.1:8000`**
- **Tab Trò chuyện:** Hỏi đáp trực quan, hiển thị câu trả lời Markdown trực diện với nhãn đối soát `[i]`, cửa sổ nguồn thông tin liên quan chi tiết (điểm tương quan $\tau$, phân vùng, số trang, trích đoạn), bộ lọc phân vùng và quản lý lịch sử hội thoại.
- **Tab Quản lý & Cấu hình:** Quản lý phân vùng dữ liệu, kéo thả tải tài liệu, điều chỉnh Chunk Size / Overlap và bảng danh sách tài liệu chi tiết.

#### 3. Chạy kiểm thử tự động:
```bash
python tests/test_optimizations.py
```

---

## 📊 6. Siêu tham số chuẩn (Hyperparameters)

| Tham số | Giá trị | Ý nghĩa |
|:---|:---:|:---|
| **Chunk Size / Overlap** | `900` / `100` | Kích thước đoạn trích và độ chồng lấp bảo toàn ngữ nghĩa |
| **Embedding Model** | `BAAI/bge-m3` | Mô hình nhúng đa ngôn ngữ 1024 chiều, chuẩn hóa L2 |
| **FAISS HNSW** | $M=16, efC=200, efS=50$ | Cấu trúc đồ thị HNSW cân bằng độ chính xác (>98%) và tốc độ (<5ms) |
| **Dense / Sparse Top-k** | `15` / `15` | Số lượng ứng viên từ HNSW và BM25Plus |
| **RRF Top-N / k** | `10` / `60` | Số lượng ứng viên sau dung hợp thứ hạng tương hỗ |
| **Reranker Model** | `BAAI/bge-reranker-v2-m3` | Cross-Encoder đánh giá tương quan ngữ nghĩa sâu |
| **Threshold ($\tau$)** | $\ge 0.50$ | Ngưỡng tương quan tối thiểu sau Sigmoid lọc sạch nhiễu |
| **LLM Top-k** | `3` | Số lượng bằng chứng tinh lọc tốt nhất đưa vào ngữ cảnh |
| **LLM Model** | `gemini-3.5-flash-lite` | Google Gemini 3.5 Flash-Lite |
