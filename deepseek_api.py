import requests

# 👉 换成你自己的 DeepSeek API Key
API_KEY = "sk-a8718a803b09496386f3ab8b7d7ab90a"

API_URL = "https://api.deepseek.com/v1/chat/completions"


def call_deepseek(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7
    }

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"API调用失败: {response.text}"