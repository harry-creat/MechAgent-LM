"""
PDF 解析：PyMuPDF 按块（block）提取文本，保留页码与文件名。

相比整页拼接后粗切分，块级输出更利于向量检索的粒度控制；
下游仍可用 splitter.split_records 对过长块做语义滑窗切分。
"""

from __future__ import annotations

import os
from typing import Any

import fitz  # PyMuPDF


def load_pdf_blocks(path: str) -> list[dict[str, Any]]:
    """
    将 PDF 转为若干条记录，每条符合：
    {"text": str, "metadata": {"page": int, "source": str}}

    page 为 1-based，与工程图纸/标准引用习惯一致。
    """
    source = os.path.basename(path)
    doc = fitz.open(path)
    records: list[dict[str, Any]] = []

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            # blocks: (x0, y0, x1, y1, text, block_no, block_type) 取决于版本
            for block in page.get_text("blocks"):
                if len(block) < 5:
                    continue
                text = (block[4] or "").strip()
                if len(text) < 2:
                    continue
                records.append(
                    {
                        "text": text,
                        "metadata": {
                            "page": page_index + 1,
                            "source": source,
                        },
                    }
                )
    finally:
        doc.close()

    return records


def merge_pdf_blocks(
    records: list[dict[str, Any]],
    *,
    max_chars: int = 1600,
) -> list[dict[str, Any]]:
    """
    将同一页、同一文件下的相邻小块合并为较长段落，再交给 split_records，
    避免手册类 PDF 产生过量微块导致向量库爆炸。
    """
    if not records:
        return []

    merged: list[dict[str, Any]] = []
    buf_parts: list[str] = []
    buf_meta: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal buf_parts, buf_meta
        if buf_meta is None:
            return
        text = "\n".join(buf_parts).strip()
        if text:
            merged.append({"text": text, "metadata": dict(buf_meta)})
        buf_parts = []
        buf_meta = None

    for rec in records:
        t = (rec.get("text") or "").strip()
        m = dict(rec.get("metadata") or {})
        if buf_meta is None:
            buf_meta = m
            buf_parts = [t]
            continue
        same_page = m.get("page") == buf_meta.get("page") and m.get("source") == buf_meta.get("source")
        candidate = "\n".join(buf_parts + [t])
        if same_page and len(candidate) <= max_chars:
            buf_parts.append(t)
        else:
            flush()
            buf_meta = m
            buf_parts = [t]
    flush()
    return merged


def load_pdf(path: str) -> str:
    """
    兼容旧接口：返回整本 PDF 拼接的大字符串（不建议用于竞赛级 RAG 入库）。
    新管线请使用 load_pdf_blocks。
    """
    parts: list[str] = []
    for rec in load_pdf_blocks(path):
        parts.append(rec["text"])
    return "\n\n".join(parts)
