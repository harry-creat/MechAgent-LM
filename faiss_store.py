"""
单知识库 FAISS 向量索引封装。

设计要点（答辩可用）：
- sentence-transformers 生成语义向量；检索为向量最近邻，非关键词匹配。
- 向量 L2 归一化后使用 IndexFlatL2，排序与余弦相似度一致（单调变换）。
- 支持增量 add、save/load、topK。

说明：IndexIVFFlat 适合十万级以上离线批量构建与 train；本教学/竞赛场景以
IndexFlatL2 保证实现简洁、可增量更新。若需 IVF，可在离线全量重建管线中替换。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np

from embedding_model import embed_texts, get_embedding_dimension


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


@dataclass
class SemanticHit:
    """单条语义检索结果（含工程溯源 metadata）。"""

    text: str
    metadata: dict[str, Any]
    score_l2: float


class FAISSKnowledgeIndex:
    """一个知识库对应一个 FAISS IndexFlatL2 + 平行元数据列表。

    索引创建延迟到首次 add 或 search 时，避免启动阶段加载 embedding 模型。
    """

    def __init__(self, kb_id: str, persist_dir: str = "./faiss_db"):
        self.kb_id = kb_id
        self.persist_dir = persist_dir
        self._index = None      # 延迟创建
        self.dim = None         # 加载已有索引或首次使用时确定
        self._metas: list[dict[str, Any]] = []

    def _ensure_index(self) -> None:
        """延迟初始化 FAISS 索引，首次调用时加载 embedding 模型获取维度。"""
        if self._index is not None:
            return
        if self.dim is None:
            self.dim = get_embedding_dimension()
        self._index = faiss.IndexFlatL2(self.dim)

    def _paths(self) -> tuple[str, str]:
        base = os.path.join(self.persist_dir, self.kb_id)
        return base + ".index", base + ".meta.json"

    def add_records(self, records: list[dict[str, Any]]) -> int:
        """
        records: [{"text": str, "metadata": dict}, ...]
        metadata 建议含 page、source、kb 等；文本同时写入 _chunk_text 供检索还原。
        """
        if not records:
            return 0
        self._ensure_index()
        texts = [r["text"] for r in records]
        base_metas = [dict(r.get("metadata") or {}) for r in records]
        vectors = np.asarray(embed_texts(texts), dtype=np.float32)
        vectors = _l2_normalize(vectors)

        rows: list[dict[str, Any]] = []
        for t, m in zip(texts, base_metas):
            row = dict(m)
            row["_chunk_text"] = t
            rows.append(row)

        self._index.add(vectors)
        self._metas.extend(rows)
        return len(rows)

    def search(self, query: str, top_k: int = 5) -> list[SemanticHit]:
        if self._index is None or self._index.ntotal == 0:
            return []
        q = np.asarray(embed_texts([query]), dtype=np.float32)
        q = _l2_normalize(q)
        top_k = min(top_k, int(self._index.ntotal))
        dists, idxs = self._index.search(q, top_k)
        hits: list[SemanticHit] = []
        for d, i in zip(dists[0], idxs[0]):
            if i < 0 or i >= len(self._metas):
                continue
            meta = dict(self._metas[i])
            text = meta.pop("_chunk_text", "")
            hits.append(SemanticHit(text=text, metadata=meta, score_l2=float(d)))
        return hits

    def save(self) -> None:
        if self._index is None:
            return
        os.makedirs(self.persist_dir, exist_ok=True)
        idx_path, meta_path = self._paths()
        faiss.write_index(self._index, idx_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metas, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        idx_path, meta_path = self._paths()
        if not (os.path.isfile(idx_path) and os.path.isfile(meta_path)):
            return False
        try:
            self._index = faiss.read_index(idx_path)
            self.dim = self._index.d
            with open(meta_path, encoding="utf-8") as f:
                self._metas = json.load(f)
        except (json.JSONDecodeError, OSError, RuntimeError) as e:
            print(f"[{self.kb_id}] 索引文件损坏，跳过加载: {e}")
            self._index = None
            self._metas = []
            return False
        return True

    @property
    def count(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.ntotal)
