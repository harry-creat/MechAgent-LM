import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")

def ask_ai(question):
    url = "https://api.deepseek.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个机械设计助手，回答要专业、简洁"},
            {"role": "user", "content": question}
        ]
    }

    response = requests.post(url, json=data, headers=headers)
    result = response.json()

    return result["choices"][0]["message"]["content"]