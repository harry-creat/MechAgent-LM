"""
对话日志模块。

功能:
- 每次问答完成后追加 JSONL 日志到 logs/chat_history.jsonl
- 导出本次会话为 Markdown 报告
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any


LOG_DIR = "logs"
_HISTORY_FILE = os.path.join(LOG_DIR, "chat_history.jsonl")


def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def log_interaction(
    question: str,
    answer: str,
    route_result: str,
    elapsed_seconds: float,
    *,
    calc_result: dict[str, Any] | None = None,
    recommendations: list[dict[str, Any]] | None = None,
) -> None:
    """追加一条问答记录到日志文件。

    Args:
        question: 用户问题
        answer: AI 回复全文
        route_result: 路由结果描述字符串
        elapsed_seconds: 耗时（秒）
        calc_result: 可选，参数计算结果
        recommendations: 可选，结构推荐结果列表
    """
    _ensure_log_dir()
    record: dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "route": route_result,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "answer": answer,
    }
    if calc_result:
        # 只保留关键数值，去掉冗长的 process 字段
        slim = {
            "calc_type": calc_result.get("calc_type", ""),
            "matched": calc_result.get("matched"),
        }
        for k in ("max_shear_stress_MPa", "max_bending_stress_MPa", "safety_factor",
                   "is_qualified", "preload_N", "suggested_module_mm", "L10_hours",
                   "max_deflection_mm"):
            if k in calc_result:
                slim[k] = calc_result[k]
        record["calc_summary"] = slim

    if recommendations:
        record["recommendations"] = [
            {"name": r["name"], "scene": r.get("scene", ""), "match_score": r.get("match_score", 0)}
            for r in recommendations
        ]

    with open(_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_session_history() -> list[dict[str, Any]]:
    """读取所有历史日志记录。"""
    _ensure_log_dir()
    if not os.path.isfile(_HISTORY_FILE):
        return []
    records: list[dict[str, Any]] = []
    with open(_HISTORY_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def export_session_report(session_messages: list[dict[str, str]] | None = None) -> str:
    """将日志记录导出为 Markdown 格式报告。

    Args:
        session_messages: 可选，当前 session 的消息列表（role/content）
                          若为 None 则从日志文件读取全部历史

    Returns:
        Markdown 格式的报告文本
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# 机械AI助手 — 对话报告",
        f"导出时间: {now}",
        "",
        "---",
        "",
    ]

    if session_messages:
        lines.append("## 本次会话问答记录")
        lines.append("")
        for i, msg in enumerate(session_messages, 1):
            if msg["role"] == "user":
                lines.append(f"### Q{i // 2 + 1}: {msg['content'][:80]}{'...' if len(msg['content']) > 80 else ''}")
            else:
                lines.append(f"{msg['content']}")
                lines.append("")
                lines.append("---")
                lines.append("")
    else:
        records = load_session_history()
        if not records:
            lines.append("暂无对话记录。")
        else:
            lines.append(f"## 全部对话历史（共 {len(records)} 条）")
            lines.append("")
            for i, rec in enumerate(records, 1):
                ts = rec.get("timestamp", "未知")
                q = rec.get("question", "")
                route = rec.get("route", "")
                t = rec.get("elapsed_seconds", 0)
                lines.append(f"### {i}. [{ts}] {q[:60]}{'...' if len(q) > 60 else ''}")
                lines.append(f"- 路由: {route}")
                lines.append(f"- 耗时: {t}s")
                if "calc_summary" in rec:
                    lines.append(f"- 计算结果: {json.dumps(rec['calc_summary'], ensure_ascii=False)}")
                lines.append("")
                lines.append("<details>")
                lines.append("<summary>展开 AI 回复</summary>")
                lines.append("")
                lines.append(rec.get("answer", ""))
                lines.append("")
                lines.append("</details>")
                lines.append("")
                lines.append("---")
                lines.append("")

    return "\n".join(lines)


def save_feedback(
    question: str,
    answer: str,
    rating: int,
    route: str,
    use_hybrid: bool,
    session_id: str,
) -> None:
    """保存用户反馈到 logs/feedback.jsonl。

    Args:
        question: 用户问题
        answer: AI 回答
        rating: 1=👍  0=👎
        route: 路由结果标签
        use_hybrid: 是否使用了混合检索
        session_id: 会话 ID
    """
    _ensure_log_dir()
    feedback_path = os.path.join(LOG_DIR, "feedback.jsonl")
    record = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "question": question,
        "answer": answer[:500],
        "rating": rating,
        "route": route,
        "use_hybrid": use_hybrid,
        "answer_length": len(answer),
    }
    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_feedback_stats() -> dict[str, Any]:
    """读取反馈日志并返回统计信息。

    Returns:
        {"total": int, "positive": int, "negative": int,
         "satisfaction_rate": float, "by_route": {kb_id: {"pos":int,"neg":int}}}
    """
    _ensure_log_dir()
    feedback_path = os.path.join(LOG_DIR, "feedback.jsonl")
    if not os.path.isfile(feedback_path):
        return {"total": 0, "positive": 0, "negative": 0, "satisfaction_rate": 0.0, "by_route": {}}

    total = 0
    positive = 0
    negative = 0
    by_route: dict[str, dict[str, int]] = {}

    with open(feedback_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            r = rec.get("rating", 0)
            if r > 0:
                positive += 1
            else:
                negative += 1

            route = rec.get("route", "unknown")
            if route not in by_route:
                by_route[route] = {"pos": 0, "neg": 0}
            if r > 0:
                by_route[route]["pos"] += 1
            else:
                by_route[route]["neg"] += 1

    rate = (positive / total * 100) if total > 0 else 0.0
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "satisfaction_rate": round(rate, 1),
        "by_route": by_route,
    }


def log_error(error_type: str, message: str, context: str = "") -> None:
    """记录错误到 logs/error.log。

    Args:
        error_type: 错误类型标签（如 API超时、检索失败）
        message: 错误描述
        context: 额外的上下文字符串
    """
    _ensure_log_dir()
    error_path = os.path.join(LOG_DIR, "error.log")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(error_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] [{error_type}] {message}\n")
        if context:
            f.write(f"  上下文: {context}\n")


# ---------------------------------------------------------------------------
# 测试块
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== 日志模块测试 ===")

    # 测试日志写入
    log_interaction(
        "计算轴的扭转切应力",
        "扭转切应力为 39.789 MPa...",
        "参数计算模式 (structure_db)",
        1.23,
        calc_result={"calc_type": "轴扭转切应力", "max_shear_stress_MPa": 39.789, "matched": True},
    )
    print("已写入一条日志记录")

    # 测试读取
    records = load_session_history()
    print(f"当前日志共 {len(records)} 条")

    # 测试报告导出
    report = export_session_report()
    print(f"报告长度: {len(report)} 字符")

    # 测试错误日志
    log_error("API超时", "DeepSeek API 请求超时（120s）", context="question=测试问题")
    print("已写入错误日志")

    print("日志模块测试完成")
