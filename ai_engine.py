"""
问答编排：检索 → 工程级 Prompt → LLM。
支持参数计算模式：识别计算意图 → 调用计算函数 → 计算结果注入 Prompt。
"""

from __future__ import annotations

from typing import Any

from calculator import dispatch_calculation
from deepseek_api import call_deepseek
from prompts import build_calculation_rag_prompt, build_mechanical_rag_prompt
from retriever import retrieve


def ask_ai(
    question: str,
    *,
    use_llm_router: bool = False,
    top_k: int = 5,
    retrieval_pack: dict[str, Any] | None = None,
    calc_params: dict[str, Any] | None = None,
) -> str:
    """retrieval_pack 由 UI 传入时可避免重复向量检索。

    calc_params 为可选的计算参数字典，键名见 calculator.py 中各函数的文档。
    若未提供且问题被识别为计算意图，dispatch_calculation 会返回 matched=False，
    系统自动降级为纯 RAG 流程。
    """
    pack = retrieval_pack or retrieve(question, top_k=top_k, use_llm_router=use_llm_router)
    decision = pack["decision"]

    # 参数计算模式
    if getattr(decision, "calc_mode", False):
        params = calc_params or {}
        calc_result = dispatch_calculation(question, params)
        if calc_result.get("matched"):
            prompt = build_calculation_rag_prompt(
                question,
                calc_result,
                pack["hits"],
                decision,
            )
            return call_deepseek(prompt)

    # 通用 / 仿真模式（含计算降级回退）
    prompt = build_mechanical_rag_prompt(
        question,
        pack["hits"],
        decision,
        aux_hits=pack.get("aux_hits"),
    )
    return call_deepseek(prompt)
