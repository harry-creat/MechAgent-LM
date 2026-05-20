import streamlit as st
from ai_engine import ask_ai_stream
from health_check import run_health_check, format_health_report
from logger import export_session_report

# ======================
# 引导性问题模板
# ======================
_GUIDED_TEMPLATES = {
    "轴系设计向导": (
        "请为以下轴系设计需求推荐合适的结构方案：\n"
        "【工况】传递扭矩约（ ）N·m，转速约（ ）r/min，\n"
        "【约束】轴长（ ）mm，安装空间（ ），\n"
        "【特殊要求】轻量化/高刚度/低成本（请勾选或补充）"
    ),
    "齿轮传动选型": (
        "请为以下齿轮传动需求推荐传动方案：\n"
        "【工况】传递功率（ ）kW，输入转速（ ）r/min，传动比约（ ），\n"
        "【布置】平行轴/相交轴/交错轴（请选择），\n"
        "【要求】低噪声/高效率/自锁（请勾选或补充）"
    ),
    "轴承选型计算": (
        "请为以下轴承需求推荐选型方案：\n"
        "【载荷】径向力 Fr=（ ）kN，轴向力 Fa=（ ）kN，\n"
        "【转速】（ ）r/min，\n"
        "【寿命要求】（ ）小时，工作温度（ ）°C"
    ),
    "联接方式推荐": (
        "请为以下轴毂联接需求推荐联接方式：\n"
        "【载荷】传递扭矩（ ）N·m，有无轴向力（ ），\n"
        "【轴径】（ ）mm，\n"
        "【要求】可拆/不可拆、频繁正反转（请勾选或补充）"
    ),
    "密封方案选择": (
        "请为以下密封需求推荐密封方案：\n"
        "【介质】润滑油/润滑脂/水/气体（请选择），\n"
        "【工况】轴径（ ）mm，转速（ ）r/min，压力（ ）MPa，\n"
        "【环境】温度（ ）°C，粉尘/腐蚀性（请说明）"
    ),
}

_QUICK_QUESTIONS = [
    "如何选择轴的材料？",
    "计算直径40mm轴能承受的最大扭矩",
    "推荐一种适合重载的齿轮传动方案",
    "ANSYS静力学分析步骤",
]

# ======================
# 页面配置
# ======================
st.set_page_config(page_title="机械AI助手", layout="wide")

# ======================
# 初始化 session_state
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = {"total": 0, "calc": 0, "recommend": 0}
if "last_route" not in st.session_state:
    st.session_state.last_route = "—"
if "last_meta" not in st.session_state:
    st.session_state.last_meta = {}
if "health_report" not in st.session_state:
    st.session_state.health_report = run_health_check()
if "context_rounds" not in st.session_state:
    st.session_state.context_rounds = 3
if "session_id" not in st.session_state:
    import uuid
    st.session_state["session_id"] = str(uuid.uuid4())[:8]
if "feedback_given" not in st.session_state:
    st.session_state["feedback_given"] = set()


def get_recent_history(n_rounds: int = 3) -> list[dict]:
    """从 session_state 提取最近 N 轮对话历史。"""
    messages = st.session_state.get("messages", [])
    history = messages[:-1] if messages else []  # 排除最后一条（当前问题）
    return history[-(n_rounds * 2):]  # 每轮含 user + assistant

# ======================
# 左侧控制栏
# ======================
with st.sidebar:
    st.title("机械AI控制台")

    # ── 系统自检状态 ──
    report = st.session_state.health_report
    all_ok = report.all_ok
    st.markdown(
        f"### {'✅' if all_ok else '⚠️'} 系统状态: "
        f"{'正常' if all_ok else f'{report.failed}项异常'}"
    )
    with st.expander("查看详情"):
        for item in report.items:
            st.caption(f"{item.icon} {item.name}: {item.detail}")

    st.markdown("---")

    # ── 知识库状态 ──
    st.subheader("知识库状态")
    try:
        from config_kb import ALL_KB_IDS
        from vector_store import VectorStore
        vs = VectorStore()
        for kb_id in ALL_KB_IDS:
            idx = vs._get_index(kb_id)
            count = idx.count
            if count > 0:
                st.caption(f"📗 {kb_id}: {count} 条")
            else:
                st.caption(f"📭 {kb_id}: 空库")
    except Exception:
        st.caption("知识库状态暂不可用")

    st.markdown("---")

    # ── 对话统计 ──
    st.subheader("本次对话统计")
    s = st.session_state.stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("提问", s["total"])
    with col2:
        st.metric("计算", s["calc"])
    col3, col4 = st.columns(2)
    with col3:
        st.metric("推荐", s["recommend"])
    with col4:
        st.metric("当前路由", st.session_state.last_route[:6] + "…" if len(st.session_state.last_route) > 6 else st.session_state.last_route)

    st.markdown("---")

    # ── 上下文记忆 ──
    st.markdown("**🧠 上下文记忆**")
    n_rounds = st.slider(
        "携带历史轮数",
        min_value=0,
        max_value=5,
        value=st.session_state.get("context_rounds", 3),
        help="0 = 每次独立问答；数字越大记忆越长，但响应略慢",
    )
    st.session_state["context_rounds"] = n_rounds

    st.markdown("---")

    # ── 反馈统计 ──
    st.markdown("**📊 反馈统计**")
    from logger import get_feedback_stats
    stats = get_feedback_stats()
    if stats["total"] > 0:
        st.metric("满意度", f"{stats['satisfaction_rate']:.0f}%")
        st.caption(f"👍 {stats['positive']}  👎 {stats['negative']}  共 {stats['total']} 条")
    else:
        st.caption("暂无反馈数据")

    st.markdown("---")

    # ── 快捷设计场景 ──
    st.subheader("快捷设计场景")
    st.caption("点击按钮自动填入引导性问题模板（不自动发送）")

    cols = st.columns(2)
    with cols[0]:
        if st.button("轴系设计向导", use_container_width=True):
            st.session_state["guided_template"] = _GUIDED_TEMPLATES["轴系设计向导"]
        if st.button("轴承选型计算", use_container_width=True):
            st.session_state["guided_template"] = _GUIDED_TEMPLATES["轴承选型计算"]
        if st.button("密封方案选择", use_container_width=True):
            st.session_state["guided_template"] = _GUIDED_TEMPLATES["密封方案选择"]
    with cols[1]:
        if st.button("齿轮传动选型", use_container_width=True):
            st.session_state["guided_template"] = _GUIDED_TEMPLATES["齿轮传动选型"]
        if st.button("联接方式推荐", use_container_width=True):
            st.session_state["guided_template"] = _GUIDED_TEMPLATES["联接方式推荐"]

    if st.session_state.get("guided_template"):
        if st.button("清除模板", use_container_width=True):
            st.session_state.pop("guided_template", None)
            st.rerun()

    st.markdown("---")

    # ── 导出报告 ──
    st.subheader("会话管理")
    if st.button("导出本次报告", use_container_width=True):
        report_md = export_session_report(st.session_state.messages)
        st.download_button(
            label="下载 Markdown 报告",
            data=report_md,
            file_name=f"mechanical_ai_report_{__import__('time').strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if st.button("清空对话", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.stats = {"total": 0, "calc": 0, "recommend": 0}
        st.session_state.last_route = "—"
        st.session_state.last_meta = {}
        st.rerun()

    st.markdown("---")

    # ── 文件上传 ──
    st.subheader("上传设计需求")
    uploaded_file = st.file_uploader(
        "支持 .txt 格式",
        type=["txt"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8")
            st.session_state["guided_template"] = content
            st.success(f"已读取文件: {uploaded_file.name} ({len(content)} 字符)")
        except Exception as e:
            st.error(f"文件读取失败: {e}")

# ======================
# 页面标题
# ======================
st.title("基于Deepseek的机械工程助手")

# ======================
# 显示历史消息（带增强卡片）
# ======================
for i, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    msg_type = msg.get("type", "qa")

    if role == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        # 根据消息类型选择图标
        icon_map = {"calc": "📐", "recommend": "🔧", "qa": "🤖"}
        icon = icon_map.get(msg_type, "🤖")

        with st.chat_message("assistant", avatar=icon):
            # 计算模式：展示关键数值卡片
            if msg_type == "calc" and msg.get("calc_meta"):
                calc_meta = msg["calc_meta"]
                calc_data = {}
                for key, label in [
                    ("max_shear_stress_MPa", "切应力"),
                    ("max_bending_stress_MPa", "弯曲应力"),
                    ("safety_factor", "安全系数"),
                    ("preload_N", "预紧力"),
                    ("suggested_module_mm", "建议模数"),
                    ("L10_hours", "轴承寿命"),
                    ("max_deflection_mm", "最大挠度"),
                ]:
                    val = calc_meta.get(key)
                    if val is not None:
                        suffix = ""
                        if key == "max_shear_stress_MPa":
                            suffix = " MPa"
                        elif key == "max_bending_stress_MPa":
                            suffix = " MPa"
                        elif key == "preload_N":
                            suffix = " N"
                        elif key == "suggested_module_mm":
                            suffix = " mm"
                        elif key == "L10_hours":
                            suffix = " h"
                        elif key == "max_deflection_mm":
                            suffix = " mm"
                        calc_data[label] = f"{val}{suffix}"
                if calc_data:
                    cols = st.columns(len(calc_data))
                    for ci, (label, val_str) in enumerate(calc_data.items()):
                        with cols[ci]:
                            st.metric(label, val_str)

            # 推荐模式：expander 展示方案
            if msg_type == "recommend" and msg.get("rec_meta"):
                recs = msg["rec_meta"]
                if recs:
                    with st.expander("查看推荐方案详情"):
                        for ri, rec in enumerate(recs, 1):
                            name = rec.get("name", f"方案{ri}")
                            score = rec.get("match_score", 0)
                            st.markdown(f"**{ri}. {name}** （匹配度: {score}）")
                            st.caption(f"场景: {rec.get('description', '')}")
                            col_adv, col_dis = st.columns(2)
                            with col_adv:
                                st.markdown("**优点**")
                                for a in rec.get("advantages", [])[:4]:
                                    st.caption(f"+ {a}")
                            with col_dis:
                                st.markdown("**缺点**")
                                for d in rec.get("disadvantages", [])[:4]:
                                    st.caption(f"- {d}")
                            with st.expander("参数范围 & 标准"):
                                for pk, pv in rec.get("typical_params", {}).items():
                                    st.caption(f"· {pk}: {pv}")
                                for std in rec.get("related_standards", []):
                                    st.caption(f"📋 {std}")
                            st.markdown("---")

            # AI 回复正文
            st.write(msg["content"])

            # 路由信息
            route_info = msg.get("route_info", "")
            if route_info:
                st.caption(f"路由: {route_info}")

            # ── 反馈按钮 ──
            if i not in st.session_state["feedback_given"]:
                c1, c2, c3 = st.columns([1, 1, 10])
                with c1:
                    if st.button("👍", key=f"fb_up_{i}", help="有帮助"):
                        from logger import save_feedback
                        user_msg = st.session_state["messages"][i - 1]["content"] if i > 0 else ""
                        save_feedback(
                            question=user_msg,
                            answer=msg["content"],
                            rating=1,
                            route=st.session_state.get("last_route", "unknown"),
                            use_hybrid=True,
                            session_id=st.session_state["session_id"],
                        )
                        st.session_state["feedback_given"].add(i)
                        st.rerun()
                with c2:
                    if st.button("👎", key=f"fb_down_{i}", help="无帮助"):
                        from logger import save_feedback
                        user_msg = st.session_state["messages"][i - 1]["content"] if i > 0 else ""
                        save_feedback(
                            question=user_msg,
                            answer=msg["content"],
                            rating=0,
                            route=st.session_state.get("last_route", "unknown"),
                            use_hybrid=True,
                            session_id=st.session_state["session_id"],
                        )
                        st.session_state["feedback_given"].add(i)
                        st.rerun()
            else:
                st.caption("✅ 已记录反馈")

# ======================
# 用户输入区
# ======================
guided = st.session_state.get("guided_template", None)

if guided:
    with st.chat_message("user"):
        edited = st.text_area(
            "编辑您的问题（可修改模板中的参数后再发送）：",
            value=guided,
            height=180,
            key="guided_editor",
        )
        col_send, col_cancel = st.columns([1, 5])
        with col_send:
            send_clicked = st.button("发送", type="primary", use_container_width=True)
else:
    edited = None
    send_clicked = False

user_input = st.chat_input("请输入机械设计/材料/结构问题...")

# ── 快捷问题标签 ──
st.caption("快捷提问：")
cols_quick = st.columns(len(_QUICK_QUESTIONS))
quick_clicked = None
for ci, qq in enumerate(_QUICK_QUESTIONS):
    with cols_quick[ci]:
        if st.button(qq, key=f"quick_{ci}", use_container_width=True):
            quick_clicked = qq

# ======================
# 核心逻辑
# ======================
final_input: str | None = None
if guided and send_clicked and edited:
    final_input = edited.strip()
    st.session_state.pop("guided_template", None)
elif quick_clicked:
    final_input = quick_clicked
elif user_input:
    final_input = user_input.strip()

if final_input:
    # 1️⃣ 存用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": final_input,
        "type": "qa",
    })

    # 2️⃣ 调用 AI（流式）
    try:
        n_rounds = st.session_state.get("context_rounds", 3)
        stream_gen, meta = ask_ai_stream(final_input, history=get_recent_history(n_rounds))
    except Exception as e:
        answer = f"系统处理异常，请稍后重试。错误信息: {e}"
        meta = {"route_label": "异常", "calc_result": None, "recommendations": None, "has_hits": False}
        from logger import log_error
        log_error("ask_ai异常", str(e), context=final_input[:100])
        # 非流式降级
        st.session_state.messages.append({
            "role": "assistant", "content": answer, "type": "qa",
            "route_info": "异常", "calc_meta": None, "rec_meta": None,
        })
        st.session_state.stats["total"] += 1
        st.rerun()
    else:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            for chunk in stream_gen:
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            answer = full_response

    # 3️⃣ 确定消息类型
    route_label = meta.get("route_label", "")
    if "计算" in route_label and meta.get("calc_result"):
        msg_type = "calc"
        st.session_state.stats["calc"] += 1
    elif "推荐" in route_label and meta.get("recommendations"):
        msg_type = "recommend"
        st.session_state.stats["recommend"] += 1
    else:
        msg_type = "qa"

    st.session_state.stats["total"] += 1
    st.session_state.last_route = route_label
    st.session_state.last_meta = meta

    # 4️⃣ 准备消息元数据
    msg_meta = {
        "role": "assistant",
        "content": answer,
        "type": msg_type,
        "route_info": route_label,
        "calc_meta": meta.get("calc_result"),
        "rec_meta": meta.get("recommendations"),
    }

    # 若无检索命中，追加提示
    if not meta.get("has_hits", True):
        msg_meta["content"] = (
            "> 知识库中暂无相关内容，以下回答基于通用机械工程知识生成。\n\n"
            + answer
        )

    # 5️⃣ 存AI回复
    st.session_state.messages.append(msg_meta)

    st.rerun()
