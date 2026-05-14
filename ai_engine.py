from retriever import simple_search
from deepseek_api import call_deepseek  # 你已有的API封装

def ask_ai(user_question):

    # 1️⃣ 先查知识库
    knowledge = simple_search(user_question)

    # 2️⃣ 拼接成“增强提示词”
    context = ""

    if knowledge:
        context = "以下是机械知识库信息：\n"
        for k in knowledge:
            context += str(k) + "\n"

    # 3️⃣ 构造Prompt
    prompt = f"""
你是一名机械设计专家，请根据以下资料回答问题：

{context}

用户问题：
{user_question}

要求：
- 专业
- 结构化
- 尽量结合工程经验
"""

    # 4️⃣ 调用大模型
    response = call_deepseek(prompt)

    return response