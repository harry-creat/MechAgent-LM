"""
文本切分：LangChain RecursiveCharacterTextSplitter。

- split_docs：面向 LangChain Document（Markdown 等）。
- split_records：面向 {"text","metadata"} 块列表（PDF 块 / 已带溯源字段的片段）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _splitter(chunk_size: int = 420, chunk_overlap: int = 80) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", ". ", "; ", " ", ""],
    )


def split_docs(docs: list[Document], chunk_size: int = 420, chunk_overlap: int = 80) -> list[Document]:
    splitter = _splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"成功切分 {len(chunks)} 个文本块（Document）")
    return chunks


def split_text(text: str, chunk_size: int = 420, chunk_overlap: int = 80) -> list[str]:
    splitter = _splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_text(text)
    print(f"成功切分 {len(chunks)} 个文本块（纯文本）")
    return chunks


def split_records(
    records: list[dict[str, Any]],
    chunk_size: int = 420,
    chunk_overlap: int = 80,
) -> list[dict[str, Any]]:
    """
    将带 metadata 的记录列表切分为更小的检索单元；metadata 逐块继承。
    """
    splitter = _splitter(chunk_size, chunk_overlap)
    out: list[dict[str, Any]] = []
    for rec in records:
        text = rec.get("text") or ""
        meta = dict(rec.get("metadata") or {})
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if len(piece) < 2:
                continue
            out.append({"text": piece, "metadata": meta})
    print(f"成功切分 {len(out)} 个带 metadata 的文本块")
    return out
