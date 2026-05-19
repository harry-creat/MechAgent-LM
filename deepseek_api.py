"""
DeepSeek Chat Completions 调用封装。

API Key 从环境变量读取，避免密钥进入版本库。
包含超时/限流/网络错误的中文友好提示。
"""

from __future__ import annotations

import os
from typing import Any

import requests

# 自动加载 .env 文件中的环境变量（若文件存在）
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

from logger import log_error

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
    try:
        response = requests.post(API_URL, headers=_headers(), json=data, timeout=120)
    except requests.exceptions.Timeout:
        log_error("API超时", "DeepSeek API 请求超时（120s）")
        return "AI 服务响应超时，请稍后重试或缩短问题长度。"
    except requests.exceptions.ConnectionError:
        log_error("API连接失败", "无法连接到 DeepSeek API")
        return "无法连接到 AI 服务，请检查网络连接后重试。"

    if response.status_code == 200:
        result = response.json()
        return str(result["choices"][0]["message"]["content"])
    elif response.status_code == 429:
        log_error("API限流", f"HTTP 429: {response.text[:200]}")
        return "AI 服务繁忙（请求过于频繁），请稍后重试。"
    elif response.status_code >= 500:
        log_error("API服务端错误", f"HTTP {response.status_code}: {response.text[:200]}")
        return "AI 服务暂时不可用，请稍后重试。"
    else:
        log_error("API请求失败", f"HTTP {response.status_code}: {response.text[:200]}")
        return f"AI 服务返回异常（HTTP {response.status_code}），请稍后重试。"


def call_deepseek(prompt: str) -> str:
    """单轮 user 提示（兼容旧代码）。"""
    try:
        return chat_messages([{"role": "user", "content": prompt}], temperature=0.7)
    except RuntimeError as e:
        log_error("API密钥缺失", str(e))
        return str(e)
