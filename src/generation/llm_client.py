"""
LLM Client module for Google Gemini Generation.
CO3101 - Hệ thống Hỏi - Đáp trên tài liệu dựa trên RAG - ĐH Bách Khoa ĐHQG-HCM.
Sử dụng trực tiếp mô hình Google Gemini 3.5 Flash-Lite qua Google GenAI SDK.
"""

import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("google_genai").setLevel(logging.ERROR)
try:
    from google.genai.models import Models
    Models._logged_afc_warning = True
except Exception:
    pass

from typing import Tuple, Optional, List
from google import genai
from google.genai import types

from src.config import (
    GEMINI_API_KEY,
    LLM_MODEL_NAME,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
)
from src.schemas import Chunk, ChatMessage


SYSTEM_PROMPT_ANTI_HALLUCINATION = """BẠN LÀ MỘT TRỢ LÝ TRÍ TUỆ NHÂN TẠO CHUYÊN TRẢ LỜI CÂU HỎI DỰA TRÊN TÀI LIỆU (RAG ASSISTANT).

HÃY TUÂN THỦ NGHIÊM NGẶT CÁC QUY TẮC SAU:
1. NGUYÊN TẮC TRUNG THỰC TUYỆT ĐỐI: Chỉ được sử dụng duy nhất các thông tin có trong phần [NGỮ CẢNH TÀI LIỆU] được cung cấp dưới đây để trả lời câu hỏi.
2. CHỐNG ẢO GIÁC (ANTI-HALLUCINATION): TUYỆT ĐỐI KHÔNG tự suy diễn, không phỏng đoán, không thêm thắt kiến thức ngoài tài liệu, không sử dụng kiến thức bên ngoài ngữ cảnh.
3. KHÔNG ĐỦ THÔNG TIN: Nếu phần ngữ cảnh không chứa đủ thông tin để trả lời chính xác câu hỏi, bạn PHẢI trả lời rõ ràng: "Dựa trên tài liệu được cung cấp, không có đủ thông tin để trả lời câu hỏi này." và không suy đoán thêm.
4. TRẢ LỜI TRỰC DIỆN, TỰ NHIÊN: Trả lời trực diện, tự nhiên, rõ ràng, mạch lạc vào trọng tâm. Tuyệt đối KHÔNG tự ghi xuất xứ nguồn, không ghi số trang, không thêm dòng chú thích dạng '(Nguồn: ...)' hay '(Theo tài liệu...)' ở cuối câu trả lời.
5. ĐỐI SOÁT MINH BẠCH: Khi khẳng định thông tin từ [ĐOẠN TRÍCH #i], hãy gắn nhãn [i] (ví dụ: [1], [2], hoặc [1, 3] nếu từ nhiều đoạn trích) ở cuối câu hoặc mệnh đề tương ứng để người dùng đối soát.
"""

SYSTEM_PROMPT_QUERY_REWRITER = """BẠN LÀ MỘT TRỢ LÝ TỐI ƯU CÂU HỎI TRUY VẤN (CONTEXTUAL QUERY REWRITER).
NHIỆM VỤ: Dựa vào [LỊCH SỬ HỘI THOẠI] và [CÂU HỎI MỚI NHẤT], hãy diễn đạt lại câu hỏi thành một CÂU HỎI ĐỘC LẬP (STANDALONE QUESTION) có đầy đủ chủ ngữ, tân ngữ và ngữ cảnh rõ ràng để phục vụ việc tra cứu tài liệu.

QUY TẮC BẮT BUỘC:
1. Giải quyết toàn bộ đại từ thay thế hoặc tham chiếu ẩn (như: 'nó', 'chúng', 'cái này', 'môn đó', 'thầy ấy', 'còn điều kiện thì sao?', 'ưu nhược điểm là gì?').
2. Nếu câu hỏi đã đầy đủ ngữ cảnh hoặc không liên quan đến lịch sử trước đó, HÃY GIỮ NGUYÊN câu hỏi gốc.
3. TUYỆT ĐỐI KHÔNG trả lời câu hỏi, KHÔNG thêm lời giải thích hay chào hỏi. CHỈ TRẢ VỀ DUY NHẤT câu hỏi độc lập đã viết lại.
"""


def format_context_chunks(chunks: List[Chunk]) -> str:
    """Định dạng danh sách các chunk kèm metadata rõ ràng để nhúng vào prompt."""
    if not chunks:
        return "[Không tìm thấy đoạn trích nào đạt ngưỡng tương đồng]"

    formatted_parts = []
    for idx, c in enumerate(chunks):
        page_str = f"Trang {c.page}" if c.page is not None else "Không xác định"
        score_val = c.score if c.score is not None else 0.0
        header = f"--- [ĐOẠN TRÍCH #{idx + 1}] | Nguồn: {c.source} | {page_str} | Độ phù hợp: {score_val:.4f} ---"
        formatted_parts.append(f"{header}\n{c.content}\n")

    return "\n".join(formatted_parts)


def build_rag_prompt(
    question: str,
    context_chunks: List[Chunk],
    conversation_history: Optional[List[ChatMessage]] = None,
) -> str:
    """Xây dựng prompt chuẩn mực cho Gemini 3.5 Flash-Lite."""
    context_str = format_context_chunks(context_chunks)

    history_str = ""
    if conversation_history:
        history_lines = [
            f"{'Người dùng' if m.role == 'user' else 'Trợ lý AI'}: {m.content}"
            for m in conversation_history[-4:]
        ]
        if history_lines:
            history_str = "[LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ]:\n" + "\n".join(history_lines) + "\n\n----------------------------------------\n"

    return f"""[NGỮ CẢNH TÀI LIỆU]:
{context_str}

----------------------------------------
{history_str}[CÂU HỎI CỦA NGƯỜI DÙNG]:
{question}

----------------------------------------
[YÊU CẦU]:
Hãy đọc kỹ [NGỮ CẢNH TÀI LIỆU] ở trên và trả lời [CÂU HỎI CỦA NGƯỜI DÙNG].
Nhớ tuân thủ nguyên tắc: Không suy diễn ngoài tài liệu, trả lời trung thực và trực diện. Gắn nhãn [i] (ví dụ: [1], [2], [1, 3]) tương ứng với [ĐOẠN TRÍCH #i] khi khẳng định thông tin để đối soát. Tuyệt đối không tự ghi dòng chú thích nguồn hay số trang ở cuối bài.
"""


class LLMClient:
    """Client kết nối trực tiếp mô hình Google Gemini 3.5 Flash-Lite."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.gemini_key = (gemini_api_key if gemini_api_key is not None else (GEMINI_API_KEY or "")).strip()
        self.model_name = (model_name or LLM_MODEL_NAME or "gemini-3.5-flash-lite").strip()

    def generate_response(
        self,
        prompt: str,
        system_instruction: str = SYSTEM_PROMPT_ANTI_HALLUCINATION,
        **kwargs,
    ) -> Tuple[str, str]:
        """Gọi trực tiếp Gemini 3.5 Flash-Lite để sinh câu trả lời tự nhiên."""
        if not self.gemini_key:
            return (
                "Chưa cấu hình API Key (GEMINI_API_KEY) trong tệp .env để kích hoạt mô hình.",
                "Extractive-Fallback (No API Key)",
            )

        try:
            client = genai.Client(api_key=self.gemini_key, http_options={"timeout": 60000})
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            if response and response.text:
                return response.text.strip(), self.model_name
            return "Không nhận được phản hồi từ mô hình.", self.model_name
        except Exception as err:
            return f"Lỗi kết nối mô hình ({self.model_name}): {err}", self.model_name

    def rewrite_query(
        self,
        query: str,
        conversation_history: List[dict],
    ) -> str:
        """Contextual Query Rewriter viết lại câu hỏi độc lập từ lịch sử hội thoại."""
        if not query or not query.strip() or not conversation_history or not self.gemini_key:
            return query

        try:
            history_lines = [
                f"{'Người dùng' if m.get('role') == 'user' else 'Trợ lý AI'}: {m.get('content', '').strip()}"
                for m in conversation_history[-4:] if m.get("content", "").strip()
            ]
            if not history_lines:
                return query

            history_text = "\n".join(history_lines)
            prompt = f"[LỊCH SỬ HỘI THOẠI]:\n{history_text}\n\n[CÂU HỎI MỚI NHẤT]:\n{query}\n\nCâu hỏi độc lập viết lại:"

            client = genai.Client(api_key=self.gemini_key, http_options={"timeout": 20000})
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_QUERY_REWRITER,
                temperature=0.0,
                max_output_tokens=150,
            )
            res = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            if res and res.text:
                rewritten = res.text.strip().replace("\n", " ").strip("\"'")
                if len(rewritten) >= 3:
                    return rewritten
        except Exception:
            pass

        return query
