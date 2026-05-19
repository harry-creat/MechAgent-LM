import streamlit as st
from ai_engine import ask_ai

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

# ======================
# 页面配置
# ======================
st.set_page_config(page_title="机械AI助手", layout="wide")

# ======================
# 左侧控制栏
# ======================
with st.sidebar:
    st.title("⚙️ 机械AI控制台")

    mode = st.selectbox(
        "选择知识领域",
        ["通用机械", "材料分析", "结构设计", "支撑系统"]
    )

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
    st.info("💡 提示：本系统已接入机械知识库 + DeepSeek")

# ======================
# 初始化聊天记录
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ======================
# 页面标题
# ======================
st.title("🤖 基于Deepseek的机械工程助手")

# ======================
# 显示历史消息
# ======================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ======================
# 用户输入区
# ======================
guided = st.session_state.get("guided_template", None)

if guided:
    # 引导模板模式：显示可编辑文本框 + 发送按钮
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

# 正常 chat_input（始终显示，即使有引导模板）
user_input = st.chat_input("请输入机械设计/材料/结构问题...")

# ======================
# 核心逻辑
# ======================
# 确定最终的用户输入（优先级：引导发送 > chat_input）
final_input: str | None = None
if guided and send_clicked and edited:
    final_input = edited.strip()
    st.session_state.pop("guided_template", None)  # 清除模板
elif user_input:
    final_input = user_input.strip()

if final_input:
    # 1️⃣ 存用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": final_input
    })

    # 2️⃣ 调用AI
    with st.spinner("正在分析..."):
        answer = ask_ai(final_input)

    # 3️⃣ 存AI回复
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()
