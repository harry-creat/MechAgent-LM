"""
问答编排：检索 → 工程级 Prompt → LLM。
"""

from __future__ import annotations

from typing import Any

from deepseek_api import call_deepseek
from prompts import build_mechanical_rag_prompt
from retriever import retrieve


def ask_ai(
    question: str,
    *,
    use_llm_router: bool = False,
    top_k: int = 5,
    retrieval_pack: dict[str, Any] | None = None,
) -> str:
    """retrieval_pack 由 UI 传入时可避免重复向量检索。"""
    pack = retrieval_pack or retrieve(question, top_k=top_k, use_llm_router=use_llm_router)
    prompt = build_mechanical_rag_prompt(
        question,
        pack["hits"],
        pack["decision"],
        aux_hits=pack.get("aux_hits"),
    )
    return call_deepseek(prompt)
