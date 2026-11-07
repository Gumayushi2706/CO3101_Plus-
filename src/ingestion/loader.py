"""
Document Loader module.
Supports PDF, DOCX, TXT files with Vietnamese text normalization
and metadata preservation (source, page).
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import unicodedata
import re
import pymupdf
from docx import Document as DocxDocument


def normalize_vietnamese_text(text: str) -> str:
    """
    Normalizes Vietnamese text:
    - Normalizes Unicode to NFC form (canonical decomposition followed by canonical composition).
    - Cleans up excessive newlines, tabs, and spaces.
    - Strips leading and trailing whitespaces while preserving paragraph structures.
    """
    if not text:
        return ""
    # NFC normalization for consistent Vietnamese characters
    text = unicodedata.normalize("NFC", text)
    # Replace non-breaking spaces and exotic whitespace chars
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    # Replace sequences of 3+ newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace sequences of spaces/tabs with single space
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class DocumentLoader:
    """
    Unified Document Loader for PDF, DOCX, and TXT files.
    Extracts text per page/section with full metadata tracking.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    def __init__(self):
        pass

    def load_file(self, file_path: Union[str, Path], partition: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Loads a document file and returns a list of page/block dictionaries.
        Output format per item:
        {
            "content": str,
            "source": str,
            "partition": str,
            "page": int,
            "char_count": int
        }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Resolve partition: default to folder name if inside a subfolder under data/raw
        if not partition:
            parent_name = path.parent.name
            if parent_name and parent_name.lower() not in ("raw", "data", ""):
                partition = parent_name
            else:
                partition = "default"

        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._load_pdf(path, partition=partition)
        elif ext == ".docx":
            return self._load_docx(path, partition=partition)
        elif ext == ".txt":
            return self._load_txt(path, partition=partition)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: {self.SUPPORTED_EXTENSIONS}")

    def _load_pdf(self, path: Path, partition: str = "default") -> List[Dict[str, Any]]:
        """
        Reads PDF pages directly using PyMuPDF (high performance C-engine with structured table extraction).
        Converts tabular layouts to clean Markdown representations.
        """
        pages_data = []
        source_name = path.name

        try:
            doc = pymupdf.open(str(path))
            try:
                for page_idx, page in enumerate(doc):
                    raw_text = page.get_text("text") or ""

                    # Extract tabular structures if present and convert to markdown
                    try:
                        tabs = page.find_tables()
                        if tabs and tabs.tables:
                            table_mds = []
                            for tab in tabs.tables:
                                md = tab.to_markdown()
                                if md and md.strip():
                                    table_mds.append(md.strip())
                            if table_mds:
                                raw_text = raw_text.rstrip() + "\n\n[BẢNG BIỂU DỮ LIỆU]:\n" + "\n\n".join(table_mds)
                    except Exception:
                        pass

                    cleaned_text = normalize_vietnamese_text(raw_text)
                    if cleaned_text:
                        pages_data.append({
                            "content": cleaned_text,
                            "source": source_name,
                            "partition": partition,
                            "page": page_idx + 1,  # 1-indexed
                            "char_count": len(cleaned_text),
                        })
            finally:
                doc.close()
        except Exception as e:
            raise RuntimeError(f"Error parsing PDF file {path.name} with PyMuPDF: {e}")

        return pages_data

    def _load_docx(self, path: Path, partition: str = "default") -> List[Dict[str, Any]]:
        """Reads DOCX paragraphs and tables."""
        source_name = path.name
        try:
            doc = DocxDocument(str(path))
            full_text_parts = []

            for para in doc.paragraphs:
                p_text = para.text.strip()
                if p_text:
                    full_text_parts.append(p_text)

            for table in doc.tables:
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        full_text_parts.append(" | ".join(row_cells))

            combined_text = "\n\n".join(full_text_parts)
            cleaned = normalize_vietnamese_text(combined_text)

            if not cleaned:
                return []

            return [{
                "content": cleaned,
                "source": source_name,
                "partition": partition,
                "page": 1,
                "char_count": len(cleaned)
            }]
        except Exception as e:
            raise RuntimeError(f"Error parsing DOCX file {path.name}: {e}")

    def _load_txt(self, path: Path, partition: str = "default") -> List[Dict[str, Any]]:
        """Reads TXT files with auto fallback encoding detection (utf-8, cp1258, utf-16)."""
        source_name = path.name
        encodings = ["utf-8", "utf-8-sig", "cp1258", "latin-1"]
        content = ""

        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue

        cleaned = normalize_vietnamese_text(content)
        if not cleaned:
            return []

        return [{
            "content": cleaned,
            "source": source_name,
            "partition": partition,
            "page": 1,
            "char_count": len(cleaned)
        }]
