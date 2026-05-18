"""
知识库自动路由：规则版（默认、低延迟） + 可选 LLM 版（语义更细，适合复杂问句）。

工程分类逻辑（与 config_kb 四库一致）：
- material_db：材料性能、选材、热处理、力学性能指标等
- structure_db：结构方案、连接、传动、刚度强度设计、设计手册条文等
- failure_db：失效模式、疲劳、断裂、磨损、腐蚀、可靠性等
- ansys_db：有限元 / ANSYS / 仿真建模、边界条件、网格、后处理、案例参数等
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from config_kb import KB_ANSYS, KB_FAILURE, KB_MATERIAL, KB_STRUCTURE
from simulation_mode import detect_simulation_mode

# 规则命中优先级：从上到下，先匹配更“专”的域，避免泛词误判
# 注：CAE/FEA 相关词由 simulation_mode 优先拦截并进入 ansys_db + 仿真专用 Prompt。
_RULE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        KB_ANSYS,
        (
            "ansys",
            "abaqus",
            "comsol",
            "fluent",
            "workbench",
            "有限元",
            "fea",
            "fem",
            "仿真",
            "simulation",
            "网格",
            "边界条件",
            "后处理",
            "求解器",
            "应变",
            "位移",
            "结构优化",
            "拓扑优化",
        ),
    ),
    (KB_FAILURE, ("失效", "疲劳", "断裂", "裂纹", "磨损", "腐蚀", "屈曲", "蠕变", "可靠性", "安全系数不足")),
    (KB_MATERIAL, ("材料", "钢材", "合金", "热处理", "硬度", "强度", "韧性", "弹性模量", "屈服", "抗拉")),
    (KB_STRUCTURE, ("结构", "轴承", "齿轮", "轴", "螺栓", "焊接", "键连接", "刚度", "挠度", "模态")),
)


@dataclass
class RouteDecision:
    """路由决策结果（答辩可展示：可解释 + 可切换 LLM）。"""

    kb_id: str
    reason: str
    method: Literal["rules", "llm", "default"]
    # 仿真分析模式：为 True 时强制走 ansys_db 主检索 + 六段式仿真 Prompt（可与 method 并存）
    simulation_mode: bool = False
    # 参数计算模式：为 True 时触发 calculator 计算 + 五段式计算 Prompt
    calc_mode: bool = False


def is_calculation_query(query: str) -> bool:
    """检测是否为机械参数计算意图。

    识别关键词：计算、校核、多大、能承受、强度、寿命、挠度、
    安全系数、扭矩、应力、预紧力、模数。
    """
    q = (query or "").strip()
    if not q:
        return False
    keywords = (
        "计算", "校核", "多大", "能承受",
        "强度", "寿命", "挠度", "安全系数",
        "扭矩", "应力", "预紧力", "模数",
        "切应力", "正应力", "弯曲应力", "扭转",
        "螺栓预紧", "轴承寿命", "L10",
        "挠跨比", "分度圆", "接触强度",
    )
    return any(kw in q for kw in keywords)


def route_query_rules(query: str) -> RouteDecision:
    q = query.strip().lower()
    if not q:
        return RouteDecision(KB_STRUCTURE, "空查询回退 structure_db", "default", simulation_mode=False)

    for kb_id, kws in _RULE_GROUPS:
        for kw in kws:
            if kw.lower() in q:
                return RouteDecision(kb_id, f"规则命中关键词: {kw}", "rules", simulation_mode=False)

    # 中英混合：再扫一遍原始大小写（部分英文关键词已 lower）
    for kb_id, kws in _RULE_GROUPS:
        for kw in kws:
            if len(kw) > 1 and kw in query:
                return RouteDecision(kb_id, f"规则命中关键词: {kw}", "rules", simulation_mode=False)

    return RouteDecision(
        KB_STRUCTURE,
        "未命中专域关键词，默认 structure_db（广义机械设计）",
        "default",
        simulation_mode=False,
    )


def route_query_llm(query: str) -> RouteDecision | None:
    """
    可选：调用 LLM 输出严格 JSON 选择 kb_id。
    失败时返回 None，由上层回退到规则路由。
    """
    try:
        from deepseek_api import chat_messages
    except ImportError:
        return None

    system = (
        "你是机械工程知识库路由器。根据用户问题，仅从以下四个值中选一个作为 kb_id：\n"
        f'"{KB_MATERIAL}", "{KB_STRUCTURE}", "{KB_FAILURE}", "{KB_ANSYS}"\n'
        "只输出一行 JSON，不要其它文字，格式："
        '{"kb_id":"<值>","reason":"<不超过40字的中文理由>"}'
    )
    try:
        text = chat_messages(
            [{"role": "system", "content": system}, {"role": "user", "content": query}],
            temperature=0.0,
        )
    except RuntimeError:
        return None

    m = re.search(r"\{[^{}]*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        kb = obj.get("kb_id", "")
        reason = obj.get("reason", "LLM 路由")
        if kb in (KB_MATERIAL, KB_STRUCTURE, KB_FAILURE, KB_ANSYS):
            return RouteDecision(kb, reason, "llm", simulation_mode=False)
    except json.JSONDecodeError:
        return None
    return None


def route_query(query: str, *, use_llm: bool = False) -> RouteDecision:
    # 参数计算意图优先判断：进入计算模式 + 五段式计算 Prompt
    calc_on = is_calculation_query(query)
    if calc_on:
        decision = route_query_rules(query)
        decision.calc_mode = True
        decision.reason = f"参数计算模式（原路由: {decision.reason}）"
        return decision

    # 仿真关键词优先于 LLM 路由：保证进入 ansys_db + 仿真专用 Prompt
    sim_on, sim_hits = detect_simulation_mode(query)
    if sim_on:
        return RouteDecision(
            KB_ANSYS,
            f"仿真分析模式（关键词: {sim_hits}）",
            "rules",
            simulation_mode=True,
        )
    if use_llm:
        d = route_query_llm(query)
        if d is not None:
            return d
    return route_query_rules(query)
