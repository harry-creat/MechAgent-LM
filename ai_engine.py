"""
问答编排：检索 → 工程级 Prompt → LLM。
支持三种增强模式：
- calc_mode: 参数计算 → 计算结果注入五段式 Prompt
- recommend_mode: 结构推荐 → 方案推荐注入六段式 Prompt
- simulation_mode: 仿真分析 → 六段式 CAE Prompt
"""

from __future__ import annotations

from typing import Any

from calculator import dispatch_calculation
from deepseek_api import call_deepseek
from prompts import (
    build_calculation_rag_prompt,
    build_mechanical_rag_prompt,
    build_recommendation_prompt,
)
from recommender import format_recommendation, recommend_structure
from retriever import retrieve


def ask_ai(
    question: str,
    *,
    use_llm_router: bool = False,
    top_k: int = 5,
    retrieval_pack: dict[str, Any] | None = None,
    calc_params: dict[str, Any] | None = None,
    calc_result: dict[str, Any] | None = None,
) -> str:
    """retrieval_pack 由 UI 传入时可避免重复向量检索。

    calc_params: 可选计算参数字典，键名见 calculator.py 各函数文档。
    calc_result: 可选，来自前序计算的结果，供推荐模块排序使用。
    """
    pack = retrieval_pack or retrieve(question, top_k=top_k, use_llm_router=use_llm_router)
    decision = pack["decision"]

    # --- 参数计算模式 ---
    if getattr(decision, "calc_mode", False):
        params = calc_params or {}
        result = dispatch_calculation(question, params)
        if result.get("matched"):
            prompt = build_calculation_rag_prompt(
                question, result, pack["hits"], decision,
            )
            return call_deepseek(prompt)

    # --- 结构推荐模式 ---
    if getattr(decision, "recommend_mode", False):
        recs = recommend_structure(question, calc_result=calc_result)
        if recs:
            rec_text = format_recommendation(recs)
            prompt = build_recommendation_prompt(
                question, recs, rec_text, pack["hits"], decision,
            )
            return call_deepseek(prompt)

    # --- 通用 / 仿真模式（含计算/推荐降级回退）---
    prompt = build_mechanical_rag_prompt(
        question,
        pack["hits"],
        decision,
        aux_hits=pack.get("aux_hits"),
    )
    return call_deepseek(prompt)
