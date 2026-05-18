"""
仿真（CAE/FEA）分析模式识别。

当问句涉及主流 CAE 工具、有限元语境或典型仿真输入输出时，
系统进入「仿真分析模式」：强制检索 ansys_db，并使用六段式仿真工程 Prompt。
"""

from __future__ import annotations

import re
from typing import Final

# 命中即进入仿真分析模式（中英混合；大小写不敏感对 ASCII 部分）
_SIMULATION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"ansys", re.I),
    re.compile(r"abaqus", re.I),
    re.compile(r"comsol", re.I),
    re.compile(r"workbench", re.I),
    re.compile(r"fluent", re.I),
    re.compile(r"仿真"),
    re.compile(r"simulation", re.I),
    re.compile(r"有限元"),
    re.compile(r"\bfem\b", re.I),
    re.compile(r"\bfea\b", re.I),
    re.compile(r"应力"),
    re.compile(r"应变"),
    re.compile(r"位移|总变形|变形场"),
    re.compile(r"网格划分|网格加密|单元质量|网格收敛|网格"),
    re.compile(r"边界条件"),
    re.compile(r"boundary\s+condition", re.I),
    re.compile(r"载荷步|载荷子步|子步收敛|载荷谱|载荷"),
    re.compile(r"结构优化|拓扑优化|形状优化|参数优化"),
    re.compile(r"非线性|大变形|接触分析|摩擦接触"),
    re.compile(r"求解器|后处理|云图|等效应力|von\s*mises", re.I),
)


def detect_simulation_mode(query: str) -> tuple[bool, str]:
    """
    返回 (是否仿真模式, 简要命中说明)。
    未命中返回 (False, "").
    """
    q = (query or "").strip()
    if not q:
        return False, ""
    matched: list[str] = []
    for pat in _SIMULATION_PATTERNS:
        m = pat.search(q)
        if m:
            matched.append(m.group(0))
    if not matched:
        return False, ""
    # 去重保序
    seen: set[str] = set()
    uniq = []
    for x in matched:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return True, "、".join(uniq[:6]) + ("…" if len(matched) > 6 else "")
