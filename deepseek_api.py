"""
DeepSeek Chat Completions 调用封装。

API Key 从环境变量读取，避免密钥进入版本库。
"""

from __future__ import annotations

import os
from typing import Any

import requests

API_URL = "https://api.deepseek.com/v1/chat/completions"


def _headers() -> dict[str, str]:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("未设置环境变量 DEEPSEEK_API_KEY，无法调用大模型。")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def chat_messages(
    messages: list[dict[str, Any]],
    *,
    model: str = "deepseek-chat",
    temperature: float = 0.7,
) -> str:
    """通用多轮消息接口（供路由、RAG 等复用）。"""
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    response = requests.post(API_URL, headers=_headers(), json=data, timeout=120)
    if response.status_code == 200:
        result = response.json()
        return str(result["choices"][0]["message"]["content"])
    return f"API调用失败: HTTP {response.status_code} {response.text}"


def call_deepseek(prompt: str) -> str:
    """单轮 user 提示（兼容旧代码）。"""
    try:
        return chat_messages([{"role": "user", "content": prompt}], temperature=0.7)
    except RuntimeError as e:
        return str(e)
