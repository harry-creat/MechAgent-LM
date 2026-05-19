"""
问答编排：检索 → 工程级 Prompt → LLM。
支持三种增强模式：
- calc_mode: 参数计算 → 计算结果注入五段式 Prompt
- recommend_mode: 结构推荐 → 方案推荐注入六段式 Prompt
- simulation_mode: 仿真分析 → 六段式 CAE Prompt
"""

from __future__ import annotations

import time
from typing import Any, Generator

from calculator import dispatch_calculation
from deepseek_api import chat_messages, stream_chat_messages
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


_SYSTEM_PROMPT = (
    "你是一名专业的机械工程设计助手，擅长结构设计、材料选择、参数计算和仿真分析。"
    "请基于检索到的专业知识回答问题，保持回答的专业性和准确性。"
    "当用户使用代词（如「它」「这个」「上面提到的」）时，请结合对话历史理解指代对象。"
)


def _build_messages(rag_prompt: str, history: list[dict] | None) -> list[dict[str, str]]:
    """组装完整的多轮 messages 列表：系统设定 + 历史对话 + 当前 RAG prompt。"""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
    ]
    if history:
        for msg in history[-6:]:  # 最多保留最近 3 轮（6 条）
            messages.append({
                "role": msg["role"],
                "content": msg["content"][:500],  # 截断超长历史消息
            })
    messages.append({"role": "user", "content": rag_prompt})
    return messages


def ask_ai(
    question: str,
    *,
    use_llm_router: bool = False,
    top_k: int = 5,
    retrieval_pack: dict[str, Any] | None = None,
    calc_params: dict[str, Any] | None = None,
    calc_result: dict[str, Any] | None = None,
    history: list[dict] | None = None,
) -> tuple[str, dict[str, Any]]:
    """retrieval_pack 由 UI 传入时可避免重复向量检索。

    calc_params: 可选计算参数字典，键名见 calculator.py 各函数文档。
    calc_result: 可选，来自前序计算的结果，供推荐模块排序使用。
    history: 可选，多轮对话历史 [{"role":"user"/"assistant","content":"..."}]

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
        from param_extractor import (
            extract_params, identify_calc_type,
            has_sufficient_params, format_extracted_params,
        )
        auto_params = extract_params(question)
        merged_params = {**auto_params, **(calc_params or {})}
        calc_type = identify_calc_type(question, merged_params)
        sufficient, missing = has_sufficient_params(calc_type, merged_params)

        if sufficient:
            try:
                result = dispatch_calculation(question, merged_params)
            except Exception as e:
                log_error("计算异常", str(e), context=question[:100])
                result = {"matched": False, "message": f"计算模块异常: {e}"}
        else:
            result = {
                "matched": False,
                "missing_params": missing,
                "calc_type": calc_type,
                "extracted_params": merged_params,
                "param_summary": format_extracted_params(merged_params, calc_type),
            }

        if result.get("matched"):
            meta["calc_result"] = result
            prompt = build_calculation_rag_prompt(
                question, result, hits, decision,
            )
            messages = _build_messages(prompt, history)
            answer = chat_messages(messages)
            elapsed = time.perf_counter() - t0
            log_interaction(question, answer, meta["route_label"], elapsed,
                            calc_result=result)
            return answer, meta

        # 参数不足：将引导信息注入通用 prompt 降级处理
        if not sufficient and "missing_params" in result:
            meta["calc_result"] = result
            prompt = build_calculation_rag_prompt(
                question, result, hits, decision,
            )
            messages = _build_messages(prompt, history)
            answer = chat_messages(messages)
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
            messages = _build_messages(prompt, history)
            answer = chat_messages(messages)
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
    messages = _build_messages(prompt, history)
    answer = chat_messages(messages)
    elapsed = time.perf_counter() - t0
    log_interaction(question, answer, meta["route_label"], elapsed)
    return answer, meta


def ask_ai_stream(
    question: str,
    *,
    use_llm_router: bool = False,
    top_k: int = 5,
    retrieval_pack: dict[str, Any] | None = None,
    calc_params: dict[str, Any] | None = None,
    calc_result: dict[str, Any] | None = None,
    history: list[dict] | None = None,
) -> tuple["Generator[str, None, None]", dict[str, Any]]:
    """流式版本的 ask_ai，路由/检索/Prompt 逻辑与 ask_ai() 完全相同，
    仅最后使用 stream_chat_messages() 替代 chat_messages() 以逐块返回文本。

    history: 可选，多轮对话历史 [{"role":"user"/"assistant","content":"..."}]

    Returns:
        (chunk_generator, meta_dict)
        - chunk_generator: 逐块 yield 文本增量的生成器
        - meta_dict: 路由标签、计算结果、推荐结果等元数据（在流式之前已确定）
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

    # 确定最终 Prompt（与 ask_ai 相同的路由逻辑）
    final_prompt: str | None = None

    if getattr(decision, "calc_mode", False):
        from param_extractor import (
            extract_params, identify_calc_type,
            has_sufficient_params,
        )
        auto_params = extract_params(question)
        merged_params = {**auto_params, **(calc_params or {})}
        calc_type = identify_calc_type(question, merged_params)
        sufficient, missing = has_sufficient_params(calc_type, merged_params)

        if sufficient:
            try:
                result = dispatch_calculation(question, merged_params)
            except Exception as e:
                log_error("计算异常", str(e), context=question[:100])
                result = {"matched": False, "message": f"计算模块异常: {e}"}
        else:
            result = {
                "matched": False,
                "missing_params": missing,
                "calc_type": calc_type,
                "extracted_params": merged_params,
            }
        if result.get("matched"):
            meta["calc_result"] = result
            final_prompt = build_calculation_rag_prompt(question, result, hits, decision)
        elif "missing_params" in result:
            meta["calc_result"] = result
            final_prompt = build_calculation_rag_prompt(question, result, hits, decision)

    if final_prompt is None and getattr(decision, "recommend_mode", False):
        try:
            recs = recommend_structure(question, calc_result=calc_result)
        except Exception as e:
            log_error("推荐异常", str(e), context=question[:100])
            recs = []
        if recs:
            meta["recommendations"] = recs
            rec_text = format_recommendation(recs)
            final_prompt = build_recommendation_prompt(question, recs, rec_text, hits, decision)

    if final_prompt is None:
        final_prompt = build_mechanical_rag_prompt(
            question, hits, decision, aux_hits=pack.get("aux_hits"),
        )

    # 构建流式生成器
    messages = _build_messages(final_prompt, history)

    def _stream():
        full = ""
        for chunk in stream_chat_messages(messages):
            full += chunk
            yield chunk
        elapsed = time.perf_counter() - t0
        log_interaction(question, full, meta["route_label"], elapsed,
                        calc_result=meta.get("calc_result"),
                        recommendations=meta.get("recommendations"))

    return _stream(), meta
