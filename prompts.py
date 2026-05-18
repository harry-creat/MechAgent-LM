"""
机械工程专家级 RAG 提示词模板。

- 通用模式：四段式（结论 / 分析 / 建议 / 风险）。
- 仿真分析模式：六段式 CAE 推理 + 条件触发的【仿真进阶说明】。
"""

from __future__ import annotations

from faiss_store import SemanticHit
from router import RouteDecision


def format_retrieved_context(
    hits: list[SemanticHit],
    *,
    aux_hits: list[SemanticHit] | None = None,
) -> str:
    """将检索片段格式化为带溯源上下文字符串。"""
    parts: list[str] = []
    for i, h in enumerate(hits, 1):
        src = h.metadata.get("source", "unknown")
        page = h.metadata.get("page", "-")
        kb = h.metadata.get("kb", "-")
        parts.append(
            f"### 片段 {i}\n"
            f"- 溯源: source={src}, page={page}, kb={kb}\n"
            f"- 向量距离(L2, 越小越近): {h.score_l2:.5f}\n"
            f"{h.text.strip()}\n"
        )
    if aux_hits:
        parts.append("### 跨库参考（失效分析域，辅助仿真判据与失效链）\n")
        for j, h in enumerate(aux_hits, 1):
            src = h.metadata.get("source", "unknown")
            page = h.metadata.get("page", "-")
            parts.append(
                f"#### 失效片段 {j} (source={src}, page={page}, L2={h.score_l2:.5f})\n{h.text.strip()}\n"
            )
    base = "\n".join(parts) if parts else ""
    if not base:
        return "（当前知识库未检索到足够相关片段；请基于通用机械工程原理谨慎作答并明确说明依据不足。）"
    return base


def _complex_simulation_question(question: str) -> bool:
    """用于触发「仿真进阶说明」展开深度（网格敏感性、误差源、试验对比等）。"""
    q = (question or "").strip()
    if len(q) > 120:
        return True
    markers = (
        "非线性",
        "接触",
        "耦合",
        "多工况",
        "多载荷",
        "大变形",
        "摩擦",
        "瞬态",
        "动力学",
        "疲劳",
        "温度场",
        "流固",
        "蠕变",
        "屈曲",
        "优化",
        "敏感性",
    )
    return any(m in q for m in markers)


def build_simulation_rag_prompt(
    question: str,
    hits: list[SemanticHit],
    decision: RouteDecision,
    *,
    aux_hits: list[SemanticHit] | None = None,
) -> str:
    ctx = format_retrieved_context(hits, aux_hits=aux_hits)
    route_line = (
        f"自动路由: kb_id={decision.kb_id}, method={decision.method}, reason={decision.reason}; "
        f"simulation_mode=True"
    )
    deep = _complex_simulation_question(question)
    deep_hint = (
        "当前系统判定问句复杂度为【高】：你必须在【仿真进阶说明】中展开论述（不可省略为“不适用”）。"
        if deep
        else "当前系统判定问句复杂度为【一般】：【仿真进阶说明】仍须作答，但若确实简单可用 3~5 句概括要点并说明无需深入的原因。"
    )

    return f"""你是一名资深 CAE 工程师，熟悉 ANSYS Workbench / Mechanical、Abaqus、COMSOL 等主流平台的**操作逻辑与工程经验**（不是纯数学推导课）。
你正在参与「机械 AI 知识库 / RAG」：回答必须**严格结合**下方【检索上下文】中的案例与模板，并补充**可落地的建模步骤**；上下文不足处须明确写出「需补充的几何/材料/载荷/接触参数」，禁止编造具体材料牌号的标准值与标准条款号。

【路由信息】
{route_line}

【仿真推理硬性要求】
1. 不能只给结论：每一步都要写清「为什么这样假设 / 为什么这样约束 / 对结果的影响」。
2. 必须体现工程经验规则：如对称性利用、刚体位移消除、接触刚度与罚因子对收敛的影响、粗网格预扫再局部加密等。
3. 必须给出可执行的 ANSYS/CAE 操作顺序（可用编号步骤：几何简化 → 材料 → 接触 → 网格 → 载荷步 → 求解 → 后处理判读）。
4. 失效与强度：结合 σ_vm 与屈服强度、安全系数定义；疲劳场景说明需要 S-N 或载荷谱，不能只写“会疲劳”。

【检索上下文】
{ctx}

【用户问题】
{question}

【复杂度提示】
{deep_hint}

【输出结构 — 必须严格使用下列标题文字，且按顺序输出】

【1. 工程建模假设】
- 几何/材料/边界方面的简化条件
- 有意忽略的因素及风险
- 是否利用对称/周期边界及其适用前提

【2. 载荷与边界条件】
- 载荷类型（集中力/分布压力/扭矩/远端载荷/加速度等）与等效方式
- 约束方式（固定面/圆柱面/远程位移/仅抑制刚体位移等）与作用位置
- 对反力与传力路径的影响说明

【3. 有限元建模建议】
- 建议单元类型（beam / shell / solid）及选型理由
- 网格划分策略（整体尺寸、应力集中区局部加密、接触对侧网格匹配）
- 是否需要子模型或子结构、是否建议开启大变形

【4. 应力与结果分析】
- 最大应力/高应力区位置预测与传力路径解释
- 变形趋势与刚度敏感部位
- 应力集中与几何细节（倒角、圆角、止口）对峰值的影响

【5. 失效判断】
- 与屈服/强度准则对照的思路（符号或区间，缺数据则列所需输入）
- 安全系数如何定义与估算
- 潜在失效模式（屈服、疲劳、接触磨损/滑移、屈曲等）及判据

【6. 优化建议（重点加分）】
- 结构减重、传力路径优化、应力分散（加强筋/过渡圆角/变厚度）
- 形状/尺寸/拓扑优化方向（定性即可，须写清“改什么、为什么能降应力”）

【仿真进阶说明】
- 用工程语言给出**简化模型示意**（文字描述载荷与约束如何施加即可，无需 ASCII 艺术）
- 列出**主要仿真误差来源**（网格、边界简化、接触参数、材料模型等）
- **网格敏感性分析提示**：如何对比两套网格、观察什么量收敛
- **与物理试验的差异**：边界再现度、测量应变片位置、材料分散性等

请使用专业、克制的中文；不要输出与上述章节无关的寒暄；不要整段复述【检索上下文】。
"""


def build_mechanical_rag_prompt(
    question: str,
    hits: list[SemanticHit],
    decision: RouteDecision,
    *,
    aux_hits: list[SemanticHit] | None = None,
) -> str:
    if decision.simulation_mode:
        return build_simulation_rag_prompt(question, hits, decision, aux_hits=aux_hits)

    ctx = format_retrieved_context(hits, aux_hits=None)
    route_line = f"自动路由: kb_id={decision.kb_id}, method={decision.method}, reason={decision.reason}"

    return f"""你是一名资深机械工程专家与可靠性工程师，擅长结构强度、材料选用与失效分析。
你正在参与“机械 AI 知识库 / RAG”系统：回答必须严格基于下方【检索上下文】，并融合通用工程准则；
若上下文不足，必须明确说明“依据不足/需实验或仿真验证”，禁止编造具体数值与标准条款号。

【路由信息】
{route_line}

【工程约束与作答准则】
1. 涉及承载或传力时：默认讨论静强度与疲劳两类风险；明确载荷类型（静载/动载/冲击/循环）与约束条件。
2. 涉及材料时：区分屈服强度/抗拉强度/疲劳极限等概念；讨论温度、环境介质、热处理状态对性能的影响。
3. 必须体现安全系数思维：说明名义应力与安全裕度关系，指出应力集中、几何突变对失效的影响。
4. 若问题与结构相关：请引导进行载荷路径分析（load path）、主传力构件识别、关键截面与应力集中部位识别。
5. 失效模式：至少分析一种主要失效模式（塑性屈服/疲劳开裂/磨损/屈曲/蠕变等）及其触发条件。
6. 数值计算：如缺少数据，用符号或区间表达，并列出需要补充的输入（几何、材料、载荷谱、表面质量等）。

【检索上下文】
{ctx}

【用户问题】
{question}

【输出结构（严格使用以下四个标题，标题文字保持一致）】
【结论】
（用 2~5 句给出可直接用于工程判断的结论；若证据不足则说明需补充的信息）

【计算/分析过程】
（给出条理化的分析步骤：载荷/约束 → 内力或应力概念路径 → 与材料强度或疲劳判据对照；可含简要公式或无量纲讨论）

【工程建议】
（可执行的方案层建议：结构、工艺、检测、仿真或试验；区分短期措施与长期优化）

【风险/失效分析】
（列出主要失效模式、诱因、监测与缓解手段；必要时给出 FMEA 式简表要点）

请使用专业、克制的中文；不要输出与上述四节无关的寒暄或重复【检索上下文】全文。
"""


def _format_calc_result(result: dict) -> str:
    """将计算函数返回的 dict 格式化为可嵌入 Prompt 的文本块。"""
    lines: list[str] = []
    calc_type = result.get("calc_type", "机械参数计算")
    lines.append(f"计算类型: {calc_type}")

    # 输出计算过程
    process = result.get("process", "")
    if process:
        lines.append(f"\n【计算过程】\n{process}")

    # 输出关键结果数值 + 单位
    lines.append("\n【关键结果】")
    key_labels = {
        "max_shear_stress_MPa": ("最大扭转切应力", "MPa"),
        "max_bending_stress_MPa": ("最大弯曲正应力", "MPa"),
        "Wp_mm3": ("抗扭截面系数 Wp", "mm³"),
        "W_mm3": ("抗弯截面系数 W", "mm³"),
        "safety_factor": ("安全系数 n", "—"),
        "is_qualified": ("是否合格（n≥1.5）", "—"),
        "preload_N": ("螺栓预紧力", "N"),
        "stress_area_mm2": ("应力截面积 As", "mm²"),
        "pitch_mm": ("螺距 P", "mm"),
        "suggested_module_mm": ("建议标准模数 m", "mm"),
        "calculated_module_mm": ("计算模数 m_calc", "mm"),
        "pitch_diameter_mm": ("分度圆直径 d", "mm"),
        "L10_hours": ("基本额定寿命 L10h", "小时"),
        "L10_million_rev": ("额定寿命 L10", "10⁶ 转"),
        "max_deflection_mm": ("最大挠度 f_max", "mm"),
        "deflection_ratio": ("挠跨比 f/L", "—"),
    }
    for key, (label, unit) in key_labels.items():
        val = result.get(key)
        if val is not None:
            lines.append(f"  {label}: {val} {unit}")

    return "\n".join(lines)


def build_calculation_rag_prompt(
    question: str,
    calc_result: dict,
    hits: list[SemanticHit],
    decision: RouteDecision,
    *,
    aux_hits: list[SemanticHit] | None = None,
) -> str:
    """五段式参数计算 RAG 提示词模板。

    结构:
    ① 计算结果摘要（数值 + 单位）
    ② 工程含义解读
    ③ 与规范/手册的对比（结合 RAG 检索内容）
    ④ 安全性评估
    ⑤ 优化建议
    """
    calc_block = _format_calc_result(calc_result)
    ctx = format_retrieved_context(hits, aux_hits=aux_hits)
    route_line = (
        f"自动路由: kb_id={decision.kb_id}, method={decision.method}, "
        f"reason={decision.reason}; calc_mode=True"
    )

    return f"""你是一名资深机械工程专家与可靠性工程师，擅长结构强度计算、参数校核与优化设计。
你正在参与「机械 AI 知识库 / RAG」系统，下方已给出由【参数计算模块】执行的计算结果。
你的任务是：基于计算结果和【检索上下文】，给出专业的工程解读与建议。

【路由信息】
{route_line}

【参数计算结果】
{calc_block}

【检索上下文】
{ctx}

【用户问题】
{question}

【输出结构 — 必须严格使用下列标题文字，且按顺序输出】

【① 计算结果摘要】
（用表格或条列式列出全部计算结果数值与单位；若计算过程中有重要中间量（如截面系数、齿宽系数等）也应一并展示）

【② 工程含义解读】
（解读计算结果的工程意义：该数值在同类机械设计中处于什么水平、对系统性能有何影响、是否超出常规经验范围）

【③ 与规范/手册的对比】
（结合上方【检索上下文】中的标准/手册条文，对比计算结果与规范建议值、许用值、推荐范围的差异；若上下文缺乏相关规范数据，须明确说明"检索库未覆盖该规范，建议查阅 GB/ISO 相应标准"）

【④ 安全性评估】
（基于安全系数/应力水平/寿命等指标，给出安全性结论：合格/裕度不足/需进一步分析；若存在失效风险，列出最可能的失效模式与触发条件）

【⑤ 优化建议】
（给出可执行的结构/参数优化建议：改什么、为什么改、预期改善幅度；区分短期改进与长期优化方向）

请使用专业、克制的中文；不要输出与上述五节无关的寒暄；不要整段复述【检索上下文】。
"""

