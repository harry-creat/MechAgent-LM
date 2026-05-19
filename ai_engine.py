"""
问答编排：检索 → 工程级 Prompt → LLM。
支持三种增强模式：
- calc_mode: 参数计算 → 计算结果注入五段式 Prompt
- recommend_mode: 结构推荐 → 方案推荐注入六段式 Prompt
- simulation_mode: 仿真分析 → 六段式 CAE Prompt
"""

from __future__ import annotations

import time
from typing import Any

from calculator import dispatch_calculation
from deepseek_api import call_deepseek
from logger import log_error, log_interaction
from prompts import (
    build_calculation_rag_prompt,
    build_mechanical_rag_prompt,
    build_recommendation_prompt,
)
from recommender import format_recommendation, recommend_structure
from retriever import retrieve


def _route_label(decision: Any) -> str:
    """生成路由结果的可读标签。"""
    if getattr(decision, "calc_mode", False):
        return "参数计算模式"
    if getattr(decision, "recommend_mode", False):
        return "结构推荐模式"
    if getattr(decision, "simulation_mode", False):
        return "仿真分析模式"
    return f"通用问答 ({decision.kb_id})"


def ask_ai(
    question: str,
    *,
    use_llm_router: bool = False,
    top_k: int = 5,
    retrieval_pack: dict[str, Any] | None = None,
    calc_params: dict[str, Any] | None = None,
    calc_result: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """retrieval_pack 由 UI 传入时可避免重复向量检索。

    calc_params: 可选计算参数字典，键名见 calculator.py 各函数文档。
    calc_result: 可选，来自前序计算的结果，供推荐模块排序使用。

    Returns:
        (answer_text, meta_dict)
        meta_dict 包含: route_label, calc_result, recommendations, has_hits
    """
    t0 = time.perf_counter()
    meta: dict[str, Any] = {
        "route_label": "通用问答",
        "calc_result": None,
        "recommendations": None,
        "has_hits": True,
    }

    try:
        pack = retrieval_pack or retrieve(question, top_k=top_k, use_llm_router=use_llm_router)
    except Exception as e:
        log_error("检索失败", str(e), context=question[:100])
        meta["has_hits"] = False
        # 降级：空检索结果 + 默认路由
        from router import RouteDecision
        from config_kb import KB_STRUCTURE
        pack = {
            "decision": RouteDecision(KB_STRUCTURE, "检索异常回退", "default"),
            "hits": [],
            "aux_hits": [],
        }

    decision = pack["decision"]
    hits = pack.get("hits", [])
    meta["route_label"] = _route_label(decision)

    if not hits:
        meta["has_hits"] = False

    # --- 参数计算模式 ---
    if getattr(decision, "calc_mode", False):
        params = calc_params or {}
        try:
            result = dispatch_calculation(question, params)
        except Exception as e:
            log_error("计算异常", str(e), context=question[:100])
            result = {"matched": False, "message": f"计算模块异常: {e}"}

        if result.get("matched"):
            meta["calc_result"] = result
            prompt = build_calculation_rag_prompt(
                question, result, hits, decision,
            )
            answer = call_deepseek(prompt)
            elapsed = time.perf_counter() - t0
            log_interaction(question, answer, meta["route_label"], elapsed,
                            calc_result=result)
            return answer, meta

    # --- 结构推荐模式 ---
    if getattr(decision, "recommend_mode", False):
        try:
            recs = recommend_structure(question, calc_result=calc_result)
        except Exception as e:
            log_error("推荐异常", str(e), context=question[:100])
            recs = []

        if recs:
            meta["recommendations"] = recs
            rec_text = format_recommendation(recs)
            prompt = build_recommendation_prompt(
                question, recs, rec_text, hits, decision,
            )
            answer = call_deepseek(prompt)
            elapsed = time.perf_counter() - t0
            log_interaction(question, answer, meta["route_label"], elapsed,
                            recommendations=recs)
            return answer, meta

    # --- 通用 / 仿真模式（含计算/推荐降级回退）---
    prompt = build_mechanical_rag_prompt(
        question,
        hits,
        decision,
        aux_hits=pack.get("aux_hits"),
    )
    answer = call_deepseek(prompt)
    elapsed = time.perf_counter() - t0
    log_interaction(question, answer, meta["route_label"], elapsed)
    return answer, meta
