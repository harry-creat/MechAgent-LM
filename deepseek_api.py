"""
DeepSeek Chat Completions 调用封装。

API Key 从环境变量读取，避免密钥进入版本库。
包含超时/限流/网络错误的中文友好提示。
"""

from __future__ import annotations

import os
from typing import Any, Generator

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


def _stream_post(payload: dict) -> "Generator[str, None, None]":
    """流式 POST 请求的内部实现，逐行 yield SSE 数据中的文本增量。"""
    import json as _json

    try:
        response = requests.post(
            API_URL,
            headers=_headers(),
            json=payload,
            stream=True,
            timeout=(10, 60),
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    return
                try:
                    chunk = _json.loads(data_str)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except (_json.JSONDecodeError, KeyError, IndexError):
                    continue
    except requests.exceptions.Timeout:
        yield "\n\n⚠️ AI 服务响应超时，请稍后重试"
    except requests.exceptions.ConnectionError:
        yield "\n\n⚠️ 网络连接失败，请检查网络和代理设置"
    except Exception as e:
        log_error("流式API异常", str(e))
        yield f"\n\n⚠️ AI 服务异常: {str(e)[:80]}"


def stream_deepseek(prompt: str) -> "Generator[str, None, None]":
    """流式调用 DeepSeek API，逐块 yield 文本增量。

    用法:
        for chunk in stream_deepseek(prompt):
            full_text += chunk
            render(full_text)
    """
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.3,
        "max_tokens": 4096,
    }
    yield from _stream_post(payload)


def stream_chat_messages(messages: list[dict]) -> "Generator[str, None, None]":
    """多轮对话流式版本，messages 格式同 OpenAI：
    [{"role": "user/assistant/system", "content": "..."}]
    """
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
        "max_tokens": 4096,
    }
    yield from _stream_post(payload)
