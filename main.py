"""
Command Line Interface (CLI) for RAG Question-Answering System.
Đồ án CO3101: Cho phép chạy thử nghiệm trọn vẹn end-to-end trên Terminal.
Ánh xạ toán học: F : (D, q) ↦ (a, E)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import RAW_DATA_DIR


def format_evidence_terminal(evidences):
    """Định dạng danh sách bằng chứng đối soát gọn gàng, rõ ràng."""
    if not evidences:
        return "  (Không có bằng chứng nào đạt ngưỡng tương quan τ ≥ 0.50)"
    
    lines = []
    for ev in evidences:
        page_str = f"Trang {ev.page}" if ev.page else "N/A"
        lines.append(f"  [{ev.citation_id}] {ev.source} | {page_str} | τ = {ev.score:.4f}")
        content_lines = [l.strip() for l in ev.chunk.splitlines() if l.strip()]
        for l in content_lines[:5]:
            lines.append(f"      {l}")
        if len(content_lines) > 5:
            lines.append("      ...")
    return "\n".join(lines)


def print_response(res):
    """In câu trả lời và tập bằng chứng đối soát minh bạch."""
    print(f"\n[Trả lời ({res.model_used})]:\n{res.answer}")
    print(f"\n[Bằng chứng ({len(res.evidences)} chunks | {res.latency_seconds}s)]:")
    print(format_evidence_terminal(res.evidences))
    print()


def main():
    parser = argparse.ArgumentParser(description="CLI Runner cho hệ thống RAG CO3101")
    parser.add_argument("--ingest", action="store_true", help="Nạp và lập chỉ mục lại toàn bộ tài liệu (Rebuild)")
    parser.add_argument("--incremental", action="store_true", help="Nạp gia tăng chỉ những tệp mới")
    parser.add_argument("--query", "-q", type=str, help="Truy vấn câu hỏi trực tiếp")
    parser.add_argument("--interactive", "-i", action="store_true", help="Chế độ đối thoại đa lượt trên Terminal")

    args = parser.parse_args()

    if not (args.ingest or args.incremental or args.query or args.interactive):
        print("\n=== CO3101 RAG QA SYSTEM (ĐH Bách Khoa ĐHQG-HCM) ===")
        parser.print_help()
        return

    from src.pipeline import RagPipeline
    pipeline = RagPipeline()

    if args.ingest or args.incremental:
        is_inc = args.incremental and not args.ingest
        mode_label = "gia tăng" if is_inc else "toàn bộ"
        print(f"[*] Đang nạp tài liệu [{mode_label}]...")
        res = pipeline.ingest_directory(RAW_DATA_DIR, incremental=is_inc)
        print(f"[+] Hoàn tất: {res['new_chunks']} chunks mới | Tổng kho: {res['total_chunks']} vectors | BM25: {pipeline.bm25_retriever.count}")
        if res['new_files']:
            print(f"[+] Tệp: {', '.join(res['new_files'])}")
        return

    if args.query:
        print(f"[*] Câu hỏi: {args.query}")
        res = pipeline.query(args.query)
        print_response(res)
        return

    if args.interactive:
        session_id = "cli_interactive_session"
        print("\n=== CHẾ ĐỘ ĐỐI THOẠI ĐA LƯỢT (Gõ 'exit' để thoát, 'clear' để làm mới) ===")
        while True:
            try:
                user_q = input("\n[Bạn]: ").strip()
                if user_q.lower() in ["exit", "quit", "q"]:
                    print("Đã thoát.")
                    break
                if user_q.lower() in ["clear", "cls", "reset"]:
                    pipeline.clear_session(session_id)
                    print("[Hệ thống]: Đã xóa ngữ cảnh hội thoại.")
                    continue
                if not user_q:
                    continue

                res = pipeline.query(user_q, session_id=session_id)
                print_response(res)
            except KeyboardInterrupt:
                print("\nĐã thoát.")
                break


if __name__ == "__main__":
    main()
