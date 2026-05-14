import streamlit as st
from ai_engine import ask_ai

# ======================
# 页面配置
# ======================
st.set_page_config(page_title="机械AI助手", layout="wide")

# ======================
# 左侧控制栏（增强功能）
# ======================
with st.sidebar:
    st.title("⚙️ 机械AI控制台")

    mode = st.selectbox(
        "选择知识领域",
        ["通用机械", "材料分析", "结构设计", "支撑系统"]
    )

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
# 用户输入框（ChatGPT风格）
# ======================
user_input = st.chat_input("请输入机械设计/材料/结构问题...")

# ======================
# 核心逻辑
# ======================
if user_input:

    # 1️⃣ 存用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # 2️⃣ 调用AI
    answer = ask_ai(user_input)

    # 3️⃣ 存AI回复
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.write(answer)