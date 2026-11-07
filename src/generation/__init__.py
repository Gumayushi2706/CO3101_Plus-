"""
Generation module for Prompting and Gemini 3.5 Flash-Lite LLM Client.
"""
from src.generation.llm_client import (
    LLMClient,
    build_rag_prompt,
    format_context_chunks,
    SYSTEM_PROMPT_ANTI_HALLUCINATION,
    SYSTEM_PROMPT_QUERY_REWRITER,
)

__all__ = [
    "build_rag_prompt",
    "format_context_chunks",
    "SYSTEM_PROMPT_ANTI_HALLUCINATION",
    "SYSTEM_PROMPT_QUERY_REWRITER",
    "LLMClient",
]
