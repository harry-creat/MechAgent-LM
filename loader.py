"""
文档加载：UTF-8 文本 → LangChain Document，并写入 source 元数据（文件名）。
"""

from __future__ import annotations

import os

from langchain_community.document_loaders import TextLoader


def load_doc(path: str):
    print(f"读取 {path}...")
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    source = os.path.basename(path)
    for d in docs:
        d.metadata["source"] = source
    print(f"成功读取 {len(docs)} 个文档")
    return docs
