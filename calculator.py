"""
机械参数计算模块。

提供常用机械设计计算公式，每个函数包含完整的参数说明、单位注释、
输入校验和结构化的计算结果返回。
"""

from __future__ import annotations

import math
import re
from typing import Any


# ---------------------------------------------------------------------------
# 1. 实心圆轴扭转切应力
# ---------------------------------------------------------------------------
def calc_shaft_torsion(torque_Nm: float, diameter_mm: float) -> dict[str, Any]:
    """计算实心圆轴扭转切应力。

    公式: τ = 16·T / (π·d³)

    Args:
        torque_Nm: 轴传递的扭矩，单位 N·m
        diameter_mm: 轴的直径，单位 mm

    Returns:
        max_shear_stress_MPa: 最大扭转切应力，单位 MPa
        Wp_mm3: 抗扭截面系数，单位 mm³
        torque_Nm: 输入扭矩
        diameter_mm: 输入直径
        process: 计算过程说明
    """
    if torque_Nm <= 0:
        return {"error": "扭矩必须大于 0", "matched": False}
    if diameter_mm <= 0:
        return {"error": "直径必须大于 0", "matched": False}

    T = torque_Nm * 1000  # N·m → N·mm
    d = diameter_mm
    Wp = math.pi * d**3 / 16  # mm³
    tau_MPa = T / Wp  # N·mm / mm³ = N/mm² = MPa

    return {
        "matched": True,
        "calc_type": "轴扭转切应力",
        "max_shear_stress_MPa": round(tau_MPa, 3),
        "Wp_mm3": round(Wp, 2),
        "torque_Nm": torque_Nm,
        "diameter_mm": diameter_mm,
        "process": (
            f"① 扭矩转换: T = {torque_Nm} N·m = {T:.1f} N·mm\n"
            f"② 抗扭截面系数: Wp = π·d³/16 = π×{d}³/16 = {Wp:.2f} mm³\n"
            f"③ 扭转切应力: τ = T/Wp = {T:.1f}/{Wp:.2f} = {tau_MPa:.3f} MPa"
        ),
    }


# ---------------------------------------------------------------------------
# 2. 实心圆轴弯曲正应力
# ---------------------------------------------------------------------------
def calc_shaft_bending(moment_Nm: float, diameter_mm: float) -> dict[str, Any]:
    """计算实心圆轴弯曲正应力。

    公式: σ = 32·M / (π·d³)

    Args:
        moment_Nm: 弯矩，单位 N·m
        diameter_mm: 轴的直径，单位 mm

    Returns:
        max_bending_stress_MPa: 最大弯曲正应力，单位 MPa
        W_mm3: 抗弯截面系数，单位 mm³
        process: 计算过程说明
    """
    if moment_Nm <= 0:
        return {"error": "弯矩必须大于 0", "matched": False}
    if diameter_mm <= 0:
        return {"error": "直径必须大于 0", "matched": False}

    M = moment_Nm * 1000  # N·m → N·mm
    d = diameter_mm
    W = math.pi * d**3 / 32  # mm³
    sigma_MPa = M / W  # MPa

    return {
        "matched": True,
        "calc_type": "轴弯曲正应力",
        "max_bending_stress_MPa": round(sigma_MPa, 3),
        "W_mm3": round(W, 2),
        "moment_Nm": moment_Nm,
        "diameter_mm": diameter_mm,
        "process": (
            f"① 弯矩转换: M = {moment_Nm} N·m = {M:.1f} N·mm\n"
            f"② 抗弯截面系数: W = π·d³/32 = π×{d}³/32 = {W:.2f} mm³\n"
            f"③ 弯曲正应力: σ = M/W = {M:.1f}/{W:.2f} = {sigma_MPa:.3f} MPa"
        ),
    }


# ---------------------------------------------------------------------------
# 3. 安全系数校核
# ---------------------------------------------------------------------------
def calc_safety_factor(
    allowable_stress_MPa: float, actual_stress_MPa: float
) -> dict[str, Any]:
    """计算安全系数并判断是否合格。

    公式: n = [σ] / σ

    Args:
        allowable_stress_MPa: 许用应力 [σ]，单位 MPa
        actual_stress_MPa: 实际应力 σ，单位 MPa

    Returns:
        safety_factor: 安全系数 n
        is_qualified: 是否合格（n ≥ 1.5）
        process: 计算过程说明
    """
    if allowable_stress_MPa <= 0:
        return {"error": "许用应力必须大于 0", "matched": False}
    if actual_stress_MPa <= 0:
        return {"error": "实际应力必须大于 0", "matched": False}

    n = allowable_stress_MPa / actual_stress_MPa
    threshold = 1.5
    qualified = n >= threshold

    return {
        "matched": True,
        "calc_type": "安全系数校核",
        "safety_factor": round(n, 3),
        "is_qualified": qualified,
        "threshold": threshold,
        "allowable_stress_MPa": allowable_stress_MPa,
        "actual_stress_MPa": actual_stress_MPa,
        "process": (
            f"安全系数 n = [σ]/σ = {allowable_stress_MPa}/{actual_stress_MPa} = {n:.3f}\n"
            f"判定标准: n ≥ {threshold} 为合格\n"
            f"结论: {'合格' if qualified else '不合格（安全裕度不足）'}"
        ),
    }


# ---------------------------------------------------------------------------
# 4. 螺栓预紧力
# ---------------------------------------------------------------------------
def _estimate_thread_pitch(diameter_mm: float) -> float:
    """根据公称直径估算标准粗牙螺距（GB/T 193）。"""
    if diameter_mm <= 3:
        return 0.5
    if diameter_mm <= 6:
        return 1.0
    if diameter_mm <= 10:
        return 1.5
    if diameter_mm <= 14:
        return 2.0
    if diameter_mm <= 18:
        return 2.5
    if diameter_mm <= 24:
        return 3.0
    if diameter_mm <= 30:
        return 3.5
    if diameter_mm <= 36:
        return 4.0
    if diameter_mm <= 42:
        return 4.5
    if diameter_mm <= 48:
        return 5.0
    return 6.0


def calc_bolt_preload(
    diameter_mm: float,
    yield_strength_MPa: float,
    tightening_factor: float = 0.7,
) -> dict[str, Any]:
    """计算螺栓预紧力。

    公式: F = factor × σs × As
    As = (π/4) × (d - 0.9382×P)² —— ISO 898-1 应力截面积

    Args:
        diameter_mm: 螺栓公称直径，单位 mm
        yield_strength_MPa: 材料屈服强度 σs，单位 MPa
        tightening_factor: 拧紧系数，默认 0.7（一般拧紧）

    Returns:
        preload_N: 预紧力，单位 N
        stress_area_mm2: 应力截面积，单位 mm²
        pitch_mm: 采用的螺距估算值
        process: 计算过程说明
    """
    if diameter_mm <= 0:
        return {"error": "公称直径必须大于 0", "matched": False}
    if yield_strength_MPa <= 0:
        return {"error": "屈服强度必须大于 0", "matched": False}
    if not 0 < tightening_factor <= 1.0:
        return {"error": "拧紧系数应在 (0, 1.0] 范围内", "matched": False}

    P = _estimate_thread_pitch(diameter_mm)
    ds = diameter_mm - 0.9382 * P  # 应力直径
    As = math.pi / 4 * ds**2  # mm²
    F = tightening_factor * yield_strength_MPa * As  # N

    return {
        "matched": True,
        "calc_type": "螺栓预紧力",
        "preload_N": round(F, 1),
        "stress_area_mm2": round(As, 2),
        "pitch_mm": P,
        "tightening_factor": tightening_factor,
        "yield_strength_MPa": yield_strength_MPa,
        "diameter_mm": diameter_mm,
        "process": (
            f"① 估算粗牙螺距: P ≈ {P} mm（按 GB/T 193 M{diameter_mm} 粗牙）\n"
            f"② 应力截面积: As = π/4 × (d - 0.9382P)² = π/4×({diameter_mm} - 0.9382×{P})² = {As:.2f} mm²\n"
            f"③ 预紧力: F = {tightening_factor} × {yield_strength_MPa} × {As:.2f} = {F:.1f} N"
        ),
    }


# ---------------------------------------------------------------------------
# 5. 齿轮模数估算（赫兹接触强度）
# ---------------------------------------------------------------------------
def calc_gear_module(
    torque_Nm: float,
    z_teeth: int,
    width_mm: float,
    sigma_hlim_MPa: float,
) -> dict[str, Any]:
    """按赫兹接触强度估算齿轮模数（简化公式）。

    基于《机械设计》齿面接触强度设计公式简化:
    d₁ ≥ ³√[(2·K·T₁·(u+1))/(φ_d·u) · (Z_H·Z_E·Z_ε/[σ_H])²]
    其中 m = d₁/z₁，通过迭代求解 φ_d = b/d₁

    Args:
        torque_Nm: 小齿轮传递扭矩，单位 N·m
        z_teeth: 小齿轮齿数
        width_mm: 齿宽，单位 mm
        sigma_hlim_MPa: 齿轮接触疲劳极限应力 [σ_H]，单位 MPa

    Returns:
        suggested_module_mm: 建议模数，单位 mm（已圆整到标准值）
        pitch_diameter_mm: 分度圆直径，单位 mm
        process: 计算过程说明
    """
    if torque_Nm <= 0:
        return {"error": "扭矩必须大于 0", "matched": False}
    if z_teeth <= 0:
        return {"error": "齿数必须大于 0", "matched": False}
    if width_mm <= 0:
        return {"error": "齿宽必须大于 0", "matched": False}
    if sigma_hlim_MPa <= 0:
        return {"error": "接触疲劳极限必须大于 0", "matched": False}

    K = 1.4        # 简化载荷系数（含使用系数、动载系数）
    u = 1.0         # 默认齿数比 = 1（未指定时假设等速传动）
    Z_H = 2.5       # 节点区域系数（α=20° 标准压力角）
    Z_E = 189.8     # 弹性影响系数 √MPa（钢-钢配对）
    Z_eps = 0.9     # 重合度系数（简化取值）
    Z_sum = Z_H * Z_E * Z_eps  # ≈ 427
    T_nmm = torque_Nm * 1000  # N·m → N·mm

    # 迭代求解：φ_d = b/d₁ = b/(m·z)，m 影响 φ_d
    phi_d = 0.8  # 初始值（通用减速器典型范围 0.6~1.2）
    for _ in range(8):
        # d₁³ = (2·K·T₁·(u+1)·Z_sum²) / (φ_d·u·[σ_H]²)
        d1_cubed = (2 * K * T_nmm * (u + 1) * Z_sum**2) / (phi_d * u * sigma_hlim_MPa**2)
        d1 = math.pow(d1_cubed, 1.0 / 3.0)
        new_phi = width_mm / d1
        if abs(new_phi - phi_d) / phi_d < 0.001:
            phi_d = new_phi
            break
        phi_d = new_phi

    d1_final = math.pow((2 * K * T_nmm * (u + 1) * Z_sum**2) / (phi_d * u * sigma_hlim_MPa**2), 1.0 / 3.0)
    m_final = d1_final / z_teeth

    # 圆整到标准模数（第一系列优先）
    std_modules = [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40]
    m_std = min((m_val for m_val in std_modules if m_val >= m_final), default=m_final)

    d = m_std * z_teeth  # 分度圆直径

    return {
        "matched": True,
        "calc_type": "齿轮模数估算",
        "suggested_module_mm": m_std,
        "calculated_module_mm": round(m_final, 3),
        "pitch_diameter_mm": round(d, 2),
        "z_teeth": z_teeth,
        "width_mm": width_mm,
        "torque_Nm": torque_Nm,
        "sigma_hlim_MPa": sigma_hlim_MPa,
        "phi_d_final": round(phi_d, 3),
        "Z_sum": round(Z_sum, 1),
        "process": (
            f"① 载荷系数: K={K}, 弹性系数: Z_H={Z_H}, Z_E={Z_E}√MPa, Z_ε={Z_eps}, ΣZ={Z_sum:.0f}\n"
            f"② 接触强度设计公式: d₁ ≥ ³√[(2·K·T·(u+1)·ΣZ²)/(φ_d·u·[σ_H]²)]\n"
            f"   扭矩转换: T={torque_Nm} N·m = {T_nmm:.0f} N·mm\n"
            f"③ 迭代求解 φ_d=b/d₁ → φ_d={phi_d:.3f}, d₁={d1_final:.2f} mm, m={m_final:.3f} mm\n"
            f"④ 圆整至标准模数: {m_std} mm（第一系列）\n"
            f"⑤ 分度圆直径: d=m×z={m_std}×{z_teeth}={d:.2f} mm"
        ),
    }


# ---------------------------------------------------------------------------
# 6. 滚动轴承基本额定寿命
# ---------------------------------------------------------------------------
def calc_bearing_life(
    C_kN: float,
    P_kN: float,
    speed_rpm: float,
    exponent: float = 3,
) -> dict[str, Any]:
    """计算滚动轴承基本额定寿命 L10。

    公式: L10 = (C/P)^p × (10⁶/(60n))

    Args:
        C_kN: 基本额定动载荷，单位 kN
        P_kN: 当量动载荷，单位 kN
        speed_rpm: 轴承转速，单位 r/min
        exponent: 寿命指数，球轴承=3，滚子轴承=10/3≈3.33

    Returns:
        L10_hours: 基本额定寿命，单位 小时
        L10_million_rev: 以百万转为单位的寿命
        process: 计算过程说明
    """
    if C_kN <= 0:
        return {"error": "额定动载荷 C 必须大于 0", "matched": False}
    if P_kN <= 0:
        return {"error": "当量动载荷 P 必须大于 0", "matched": False}
    if speed_rpm <= 0:
        return {"error": "转速必须大于 0", "matched": False}
    if C_kN < P_kN:
        return {"error": f"额定动载荷 C({C_kN} kN) 小于当量动载荷 P({P_kN} kN)，轴承会立即失效", "matched": False}

    L10_rev = (C_kN / P_kN) ** exponent  # 百万转
    L10_hours = L10_rev * 1e6 / (60 * speed_rpm)

    bearing_type = "球轴承 (p=3)" if exponent == 3 else f"滚子轴承 (p={exponent})"

    return {
        "matched": True,
        "calc_type": "轴承寿命",
        "L10_hours": round(L10_hours, 1),
        "L10_million_rev": round(L10_rev, 2),
        "C_kN": C_kN,
        "P_kN": P_kN,
        "speed_rpm": speed_rpm,
        "exponent": exponent,
        "bearing_type": bearing_type,
        "process": (
            f"轴承类型: {bearing_type}\n"
            f"① 基本额定寿命（百万转）: L10 = (C/P)^p = ({C_kN}/{P_kN})^{exponent} = {L10_rev:.2f} (10⁶ rev)\n"
            f"② 转换为小时: L10h = L10×10⁶/(60×n) = {L10_rev:.2f}×10⁶/(60×{speed_rpm}) = {L10_hours:.1f} h"
        ),
    }


# ---------------------------------------------------------------------------
# 7. 简支梁中点挠度
# ---------------------------------------------------------------------------
def calc_beam_deflection(
    force_N: float,
    length_mm: float,
    E_GPa: float,
    I_mm4: float,
) -> dict[str, Any]:
    """简支梁中点集中力挠度计算。

    公式: f = F·L³ / (48·E·I)

    Args:
        force_N: 集中力，单位 N
        length_mm: 梁的跨度（支点间距），单位 mm
        E_GPa: 材料弹性模量，单位 GPa
        I_mm4: 截面惯性矩，单位 mm⁴

    Returns:
        max_deflection_mm: 最大挠度，单位 mm
        deflection_ratio: 挠跨比 f/L
        process: 计算过程说明
    """
    if force_N <= 0:
        return {"error": "集中力必须大于 0", "matched": False}
    if length_mm <= 0:
        return {"error": "跨度必须大于 0", "matched": False}
    if E_GPa <= 0:
        return {"error": "弹性模量必须大于 0", "matched": False}
    if I_mm4 <= 0:
        return {"error": "截面惯性矩必须大于 0", "matched": False}

    E_MPa = E_GPa * 1000  # GPa → MPa = N/mm²
    f = force_N * length_mm**3 / (48 * E_MPa * I_mm4)
    ratio = f / length_mm

    return {
        "matched": True,
        "calc_type": "简支梁挠度",
        "max_deflection_mm": round(f, 4),
        "deflection_ratio": round(ratio, 6),
        "force_N": force_N,
        "length_mm": length_mm,
        "E_GPa": E_GPa,
        "I_mm4": I_mm4,
        "process": (
            f"① 弹性模量换算: E = {E_GPa} GPa = {E_MPa} N/mm²\n"
            f"② 最大挠度: f = FL³/(48EI)\n"
            f"   = {force_N}×{length_mm}³/(48×{E_MPa}×{I_mm4})\n"
            f"   = {f:.4f} mm\n"
            f"③ 挠跨比: f/L = {f:.4f}/{length_mm} = {ratio:.6f}（即 1/{round(1/ratio) if ratio > 0 else '∞'}）"
        ),
    }


# ---------------------------------------------------------------------------
# 8. 计算分派器
# ---------------------------------------------------------------------------
def dispatch_calculation(user_query: str, params: dict[str, Any]) -> dict[str, Any]:
    """根据用户问句中的关键词自动分派到对应的计算函数。

    关键词映射规则:
    - "扭转"、"扭矩"、"切应力"、"剪应力"、"τ" → calc_shaft_torsion
    - "弯曲"、"弯矩"、"正应力"、"σ"、"抗弯" → calc_shaft_bending
    - "安全系数"、"校核"、"强度校核"、"许用应力" → calc_safety_factor
    - "螺栓"、"预紧力"、"拧紧"、"螺纹" → calc_bolt_preload
    - "齿轮"、"模数"、"分度圆"、"接触强度" → calc_gear_module
    - "轴承"、"寿命"、"L10"、"额定寿命" → calc_bearing_life
    - "挠度"、"挠曲"、"变形"、"简支梁"、"挠跨比" → calc_beam_deflection

    Args:
        user_query: 用户自然语言问句
        params: 参数字典，各函数所需键名见对应计算函数的文档

    Returns:
        若匹配成功，返回对应计算函数的结果（含 matched=True）
        若无法匹配，返回 {"matched": False, "message": "未识别计算类型"}
    """
    q = (user_query or "").strip().lower()
    if not q:
        return {"matched": False, "message": "未识别计算类型"}

    # 关键词 → (函数, 所需参数键列表)
    routes: list[tuple[tuple[str, ...], Any, list[str]]] = [
        (("扭转", "扭矩", "切应力", "剪应力", "τ"), calc_shaft_torsion, ["torque_Nm", "diameter_mm"]),
        (("弯曲", "弯矩", "正应力", "抗弯", "σ"), calc_shaft_bending, ["moment_Nm", "diameter_mm"]),
        (("安全系数", "校核", "强度校核", "许用应力"), calc_safety_factor, ["allowable_stress_MPa", "actual_stress_MPa"]),
        (("螺栓", "预紧力", "拧紧", "螺纹"), calc_bolt_preload, ["diameter_mm", "yield_strength_MPa"]),
        (("齿轮", "模数", "分度圆", "接触强度"), calc_gear_module, ["torque_Nm", "z_teeth", "width_mm", "sigma_hlim_MPa"]),
        (("轴承", "寿命", "L10", "额定寿命"), calc_bearing_life, ["C_kN", "P_kN", "speed_rpm"]),
        (("挠度", "挠曲", "变形", "简支梁", "挠跨比"), calc_beam_deflection, ["force_N", "length_mm", "E_GPa", "I_mm4"]),
    ]

    for keywords, func, required_keys in routes:
        if any(kw in q for kw in keywords):
            # 检查必要参数是否存在（允许 tightening_factor、exponent 为可选）
            optional_keys = {"tightening_factor", "exponent"}
            missing = [k for k in required_keys if k not in params and k not in optional_keys]
            kw_label = "、".join(keywords[:2])
            if missing:
                return {
                    "matched": False,
                    "message": f"识别为「{kw_label}」计算，但缺少参数: {', '.join(missing)}",
                }
            try:
                return func(**{k: params[k] for k in required_keys if k in params})
            except TypeError:
                return {
                    "matched": False,
                    "message": f"「{kw_label}」计算参数类型错误，请检查输入",
                }

    return {"matched": False, "message": "未识别计算类型"}


# ---------------------------------------------------------------------------
# 测试块
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("机械参数计算模块 — 单元测试")
    print("=" * 60)

    # 1. 轴扭转
    print("\n[1] calc_shaft_torsion — 正常输入")
    r = calc_shaft_torsion(torque_Nm=500, diameter_mm=40)
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n[1] calc_shaft_torsion — 异常输入（负扭矩）")
    r = calc_shaft_torsion(torque_Nm=-100, diameter_mm=40)
    print(f"  {r}")

    # 2. 轴弯曲
    print("\n[2] calc_shaft_bending — 正常输入")
    r = calc_shaft_bending(moment_Nm=300, diameter_mm=50)
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n[2] calc_shaft_bending — 异常输入（零直径）")
    r = calc_shaft_bending(moment_Nm=300, diameter_mm=0)
    print(f"  {r}")

    # 3. 安全系数
    print("\n[3] calc_safety_factor — 正常输入（合格）")
    r = calc_safety_factor(allowable_stress_MPa=250, actual_stress_MPa=120)
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n[3] calc_safety_factor — 正常输入（不合格）")
    r = calc_safety_factor(allowable_stress_MPa=200, actual_stress_MPa=180)
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n[3] calc_safety_factor — 异常输入（零应力）")
    r = calc_safety_factor(allowable_stress_MPa=250, actual_stress_MPa=0)
    print(f"  {r}")

    # 4. 螺栓预紧力
    print("\n[4] calc_bolt_preload — 正常输入")
    r = calc_bolt_preload(diameter_mm=16, yield_strength_MPa=640, tightening_factor=0.7)
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n[4] calc_bolt_preload — 异常输入（拧紧系数>1）")
    r = calc_bolt_preload(diameter_mm=16, yield_strength_MPa=640, tightening_factor=1.5)
    print(f"  {r}")

    # 5. 齿轮模数
    print("\n[5] calc_gear_module — 正常输入")
    r = calc_gear_module(torque_Nm=200, z_teeth=20, width_mm=50, sigma_hlim_MPa=1200)
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n[5] calc_gear_module — 异常输入（齿数为0）")
    r = calc_gear_module(torque_Nm=200, z_teeth=0, width_mm=50, sigma_hlim_MPa=1200)
    print(f"  {r}")

    # 6. 轴承寿命
    print("\n[6] calc_bearing_life — 正常输入（球轴承）")
    r = calc_bearing_life(C_kN=35, P_kN=8, speed_rpm=1450, exponent=3)
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n[6] calc_bearing_life — 异常输入（C < P）")
    r = calc_bearing_life(C_kN=5, P_kN=10, speed_rpm=1000)
    print(f"  {r}")

    # 7. 简支梁挠度
    print("\n[7] calc_beam_deflection — 正常输入")
    r = calc_beam_deflection(force_N=5000, length_mm=2000, E_GPa=206, I_mm4=1.5e7)
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n[7] calc_beam_deflection — 异常输入（负惯性矩）")
    r = calc_beam_deflection(force_N=5000, length_mm=2000, E_GPa=206, I_mm4=-100)
    print(f"  {r}")

    # 8. 分派器
    print("\n[8] dispatch_calculation — 正常分派（扭矩）")
    r = dispatch_calculation("计算这根轴的扭转切应力", {"torque_Nm": 500, "diameter_mm": 40})
    print(f"  matched: {r.get('matched')}, calc_type: {r.get('calc_type', 'N/A')}")

    print("\n[8] dispatch_calculation — 缺少参数")
    r = dispatch_calculation("校核这个零件安全系数", {})
    print(f"  {r}")

    print("\n[8] dispatch_calculation — 无法匹配")
    r = dispatch_calculation("今天天气怎么样", {})
    print(f"  {r}")

    print("\n" + "=" * 60)
    print("所有测试完成，无异常报错。")
    print("=" * 60)
