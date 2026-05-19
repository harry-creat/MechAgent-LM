"""
自然语言参数自动提取模块。

从用户自然语言输入中识别数值 + 单位，转换为标准单位参数字典，
供 calculator.py 的计算函数直接调用。
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# 单位换算常量（→ 标准单位）
# ---------------------------------------------------------------------------
UNIT_MAP: dict[str, float] = {
    # 长度 → mm
    "mm": 1, "cm": 10, "m": 1000,
    # 力 → N
    "n": 1, "kn": 1000,
    # 力矩 → N·m
    "nm": 1, "n·m": 1, "kn·m": 1000, "knm": 1000,
    # 应力 → MPa
    "mpa": 1, "gpa": 1000, "kpa": 0.001,
    # 转速 → rpm
    "rpm": 1, "r/min": 1, "转/分": 1, "rad/s": 9.549,
    # 轴承载荷 → kN
    "kn": 1,
}

# 计算类型 → 必需参数列表
_REQUIRED_PARAMS: dict[str, list[str]] = {
    "shaft_torsion":  ["torque_Nm", "diameter_mm"],
    "shaft_bending":  ["moment_Nm", "diameter_mm"],
    "safety_factor":  ["allowable_stress_MPa", "actual_stress_MPa"],
    "bolt_preload":   ["diameter_mm", "yield_strength_MPa"],
    "gear_module":    ["torque_Nm", "z_teeth", "width_mm", "sigma_hlim_MPa"],
    "bearing_life":   ["C_kN", "P_kN", "speed_rpm"],
    "beam_deflection":["force_N", "length_mm", "E_GPa", "I_mm4"],
}

# 参数 → 中文名
_PARAM_LABELS: dict[str, str] = {
    "torque_Nm": "扭矩", "diameter_mm": "直径/轴径", "moment_Nm": "弯矩",
    "allowable_stress_MPa": "许用应力", "actual_stress_MPa": "实际应力",
    "yield_strength_MPa": "屈服强度", "z_teeth": "齿数", "width_mm": "齿宽",
    "sigma_hlim_MPa": "接触疲劳极限", "C_kN": "额定动载荷C", "P_kN": "当量动载荷P",
    "speed_rpm": "转速", "force_N": "载荷/力", "length_mm": "长度/跨度",
    "E_GPa": "弹性模量", "I_mm4": "截面惯性矩",
}


# ---------------------------------------------------------------------------
# 提取正则模式（顺序影响优先级）
# ---------------------------------------------------------------------------
def _compile_patterns() -> list[tuple[str, str, str]]:
    """返回 [(参数名, 正则, 默认单位), ...] 列表。"""
    return [
        # 直径/轴径
        ("diameter_mm", r"(?:直径|轴径|轴直径|d\s*[=＝]\s*)\s*(\d+\.?\d*)\s*(mm|cm|m)?", "mm"),
        # 扭矩（同时匹配"扭矩200Nm"和"200Nm扭矩"）
        ("torque_Nm", r"(?:扭矩|转矩|T\s*[=＝]\s*)\s*(\d+\.?\d*)\s*(N·?m|kN·?m|Nm|kNm)?", "Nm"),
        ("torque_Nm", r"(\d+\.?\d*)\s*(N·?m|Nm|kN·?m)\s*(?:的)?(?:扭矩|转矩)", "Nm"),
        # 弯矩
        ("moment_Nm", r"(?:弯矩|M\s*[=＝]\s*)\s*(\d+\.?\d*)\s*(N·?m|kN·?m|Nm)?", "Nm"),
        # 转速
        ("speed_rpm", r"(\d+\.?\d*)\s*(?:rpm|r/min|转/分|转每分)", "rpm"),
        ("speed_rpm", r"(?:转速|n\s*[=＝]\s*)\s*(\d+\.?\d*)\s*(?:rpm|r/min|转/分)?", "rpm"),
        # 载荷/力（避免匹配"应力"中的"力"）
        ("force_N", r"(?:载荷|载力|集中力|F\s*[=＝]\s*)\s*(\d+\.?\d*)\s*(N|kN|n|kn)?", "N"),
        # 长度/跨度
        ("length_mm", r"(?:长度|跨度|跨距|L\s*[=＝]\s*)\s*(\d+\.?\d*)\s*(mm|cm|m)?", "mm"),
        # 许用应力
        ("allowable_stress_MPa", r"(?:许用应力|\\[σ\\]|[σ])\s*[=＝]?\s*(\d+\.?\d*)\s*(MPa|GPa|mpa|gpa)?", "MPa"),
        # 实际应力
        ("actual_stress_MPa", r"(?:实际应力|工作应力|σ\s*[=＝])\s*(\d+\.?\d*)\s*(MPa|mpa)?", "MPa"),
        # 屈服强度
        ("yield_strength_MPa", r"(?:屈服强度|屈服极限|σs|σ_s)\s*[=＝]?\s*(\d+\.?\d*)\s*(MPa|mpa)?", "MPa"),
        # 齿数
        ("z_teeth", r"(?:齿数|z\s*[=＝])\s*(\d+)", ""),
        ("z_teeth", r"(\d+)\s*(?:齿|个齿)", ""),
        # 齿宽
        ("width_mm", r"(?:齿宽|b\s*[=＝])\s*(\d+\.?\d*)\s*(mm)?", "mm"),
        # 接触疲劳极限
        ("sigma_hlim_MPa", r"(?:接触疲劳|σ[Hh]|σ_H)\s*[=＝]?\s*(\d+\.?\d*)\s*(MPa|mpa)?", "MPa"),
        # 额定动载荷 C
        ("C_kN", r"[Cc]\s*[=＝]\s*(\d+\.?\d*)\s*(?:kN|kn)?", "kN"),
        ("C_kN", r"(?:额定动载荷|基本额定动载荷)\s*[=＝]?\s*(\d+\.?\d*)\s*(?:kN|kn)?", "kN"),
        # 当量动载荷 P
        ("P_kN", r"[Pp]\s*[=＝]\s*(\d+\.?\d*)\s*(?:kN|kn)?", "kN"),
        ("P_kN", r"(?:当量动载荷|当量载荷)\s*[=＝]?\s*(\d+\.?\d*)\s*(?:kN|kn)?", "kN"),
        # 弹性模量
        ("E_GPa", r"(?:弹性模量|E\s*[=＝])\s*(\d+\.?\d*)\s*(?:GPa|gpa)?", "GPa"),
        # 截面惯性矩（支持科学计数法如 1.5e7）
        ("I_mm4", r"(?:惯性矩|截面惯性矩|I\s*[=＝])\s*(\d+\.?\d*(?:e\d+)?)\s*(?:mm4|mm⁴)?", "mm4"),
    ]


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------
def extract_params(query: str) -> dict[str, Any]:
    """从自然语言查询中提取标准化参数。

    Args:
        query: 用户自然语言输入

    Returns:
        {"diameter_mm": 40, "torque_Nm": 200, ...}  # 值已转换为标准单位
    """
    params: dict[str, Any] = {}
    seen_positions: set[int] = set()  # 防止同一数字被多次匹配

    for param_name, pattern, default_unit in _compile_patterns():
        for m in re.finditer(pattern, query, re.IGNORECASE):
            start = m.start()
            if start in seen_positions:
                continue

            if param_name in ("z_teeth",):
                value = float(m.group(1))
                params[param_name] = int(value)
                seen_positions.add(start)
                break

            # E_GPa / I_mm4 不换算，保留原始值
            if param_name in ("E_GPa", "I_mm4"):
                value = float(m.group(1))
                params[param_name] = value
                seen_positions.add(start)
                break

            value = float(m.group(1))
            raw_unit = m.group(2) if m.lastindex and m.lastindex >= 2 else default_unit
            unit = (raw_unit or default_unit).strip().lower().replace(" ", "")

            # 单位换算
            factor = UNIT_MAP.get(unit, 1)
            std_value = value * factor
            params[param_name] = std_value
            seen_positions.add(start)
            break  # 每种参数只取第一次匹配

    return params


def identify_calc_type(query: str, params: dict[str, Any]) -> str:
    """根据查询关键词和已提取参数判断计算类型。

    Args:
        query: 用户查询
        params: extract_params() 返回的参数字典

    Returns:
        计算类型字符串（如 "shaft_torsion"），无法判断时返回 "unknown"
    """
    q = query.lower()
    has_d = "diameter_mm" in params

    # 按优先级判断（齿轮/轴承/梁 关键字优先于 扭矩/弯曲 泛词）
    if any(kw in q for kw in ("齿轮", "模数", "分度圆", "接触强度")) and "z_teeth" in params:
        return "gear_module"
    if any(kw in q for kw in ("轴承", "寿命", "L10", "额定寿命")) and "speed_rpm" in params:
        return "bearing_life"
    if any(kw in q for kw in ("挠度", "挠曲", "梁", "挠跨比")) and "length_mm" in params:
        return "beam_deflection"
    if any(kw in q for kw in ("安全系数", "安全裕度", "强度校核")):
        return "safety_factor"
    if any(kw in q for kw in ("螺栓", "预紧力", "拧紧")) and has_d:
        return "bolt_preload"
    if any(kw in q for kw in ("扭转", "扭矩", "切应力", "剪应力", "τ")):
        if has_d:
            return "shaft_torsion"
        if "torque_Nm" in params:
            return "shaft_torsion"
    if any(kw in q for kw in ("弯曲", "弯矩", "正应力", "抗弯", "σ")):
        if has_d:
            return "shaft_bending"

    # 关键词匹配不上时，用参数推断
    if has_d and "torque_Nm" in params:
        return "shaft_torsion"
    if has_d and "moment_Nm" in params:
        return "shaft_bending"
    if "C_kN" in params and "P_kN" in params:
        return "bearing_life"
    if "force_N" in params and "length_mm" in params:
        return "beam_deflection"

    return "unknown"


def has_sufficient_params(calc_type: str, params: dict[str, Any]) -> tuple[bool, list[str]]:
    """检查给定计算类型的参数是否充足。

    Args:
        calc_type: identify_calc_type() 的返回值
        params: 参数字典

    Returns:
        (是否充足, 缺少的参数中文名列表)
    """
    if calc_type == "unknown" or calc_type not in _REQUIRED_PARAMS:
        return False, ["无法确定计算类型"]

    required = _REQUIRED_PARAMS[calc_type]
    missing = [k for k in required if k not in params]
    missing_labels = [_PARAM_LABELS.get(k, k) for k in missing]
    return len(missing) == 0, missing_labels


def format_extracted_params(params: dict[str, Any], calc_type: str) -> str:
    """将提取到的参数格式化为中文说明。

    Args:
        params: 参数字典
        calc_type: 计算类型

    Returns:
        格式化中文描述字符串
    """
    type_names = {
        "shaft_torsion": "轴扭转切应力", "shaft_bending": "轴弯曲正应力",
        "safety_factor": "安全系数校核", "bolt_preload": "螺栓预紧力",
        "gear_module": "齿轮模数估算", "bearing_life": "轴承额定寿命",
        "beam_deflection": "简支梁挠度", "unknown": "未知计算",
    }
    type_label = type_names.get(calc_type, calc_type)

    param_strs = []
    for key, value in params.items():
        label = _PARAM_LABELS.get(key, key)
        # 添加单位
        unit = ""
        if key.endswith("_mm"):
            unit = " mm"
        elif key.endswith("_MPa"):
            unit = " MPa"
        elif key.endswith("_GPa"):
            unit = " GPa"
        elif key.endswith("_Nm"):
            unit = " N·m"
        elif key.endswith("_kN"):
            unit = " kN"
        elif key.endswith("_N"):
            unit = " N"
        elif key.endswith("_rpm"):
            unit = " rpm"
        elif key.endswith("_mm4"):
            unit = " mm⁴"
        param_strs.append(f"{label} {value}{unit}")

    params_text = "，".join(param_strs) if param_strs else "未提取到参数"
    return f"已识别参数：{params_text}。计算类型：{type_label}"


# ---------------------------------------------------------------------------
# 测试块
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        "直径40mm的轴能承受多大扭矩",
        "直径40mm承受200Nm扭矩的轴切应力是多少",
        "C=15kN P=5kN 转速1500rpm 轴承寿命",
        "怎么选轴的材料",
        "轴径50mm 弯矩300Nm 的弯曲应力",
        "螺栓直径16mm 屈服强度640MPa 预紧力",
        "齿轮模数 齿数20 齿宽50mm 扭矩200Nm 接触疲劳1200MPa",
        "简支梁跨度2000mm 载荷5000N 弹性模量206GPa 惯性矩1.5e7mm4求挠度",
        "许用应力250MPa 实际应力180MPa 校核安全系数",
    ]

    for q in test_cases:
        params = extract_params(q)
        ctype = identify_calc_type(q, params)
        sufficient, missing = has_sufficient_params(ctype, params)
        summary = format_extracted_params(params, ctype)
        print(f"输入: {q}")
        print(f"  参数: {params}")
        print(f"  类型: {ctype}, 充足: {sufficient}, 缺少: {missing}")
        print(f"  说明: {summary}")
        print()
