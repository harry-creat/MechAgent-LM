"""
句向量模型（sentence-transformers）。

竞赛级 RAG：所有入库与查询共用同一模型，维度固定，便于 FAISS 持久化。
"""

from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer

# 默认使用轻量英文模型（易缓存、依赖小）；中文竞赛场景可通过环境变量切换，例如：
# set ST_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
MODEL_NAME = os.environ.get(
    "ST_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_embedding_dimension() -> int:
    m = get_model()
    fn = getattr(m, "get_embedding_dimension", None)
    if callable(fn):
        return int(fn())
    return int(m.get_sentence_embedding_dimension())


def embed_text(text: str) -> np.ndarray:
    return get_model().encode(text, convert_to_numpy=True)


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, get_embedding_dimension()), dtype=np.float32)
    return get_model().encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )
