"""
多知识库向量存储：每个工程域独立 FAISS + BM25 双索引。

对外主类仍命名为 VectorStore，以兼容 build_kb / 旧 import。
"""

from __future__ import annotations

import os
from typing import Any

from bm25_store import BM25Index, reciprocal_rank_fusion
from config_kb import ALL_KB_IDS, FAISS_PERSIST_DIR, KB_ANSYS, KB_FAILURE, KB_MATERIAL, KB_STRUCTURE
from faiss_store import FAISSKnowledgeIndex, SemanticHit


class VectorStore:
    """四个独立 FAISS 索引 + 四个独立 BM25 索引 + 统一 save/load 入口。"""

    def __init__(self, persist_dir: str = FAISS_PERSIST_DIR, *, autoload: bool = True):
        self.persist_dir = persist_dir

        # FAISS 索引
        self.material_db = FAISSKnowledgeIndex(KB_MATERIAL, persist_dir)
        self.structure_db = FAISSKnowledgeIndex(KB_STRUCTURE, persist_dir)
        self.failure_db = FAISSKnowledgeIndex(KB_FAILURE, persist_dir)
        self.ansys_db = FAISSKnowledgeIndex(KB_ANSYS, persist_dir)
        self._by_id: dict[str, FAISSKnowledgeIndex] = {
            KB_MATERIAL: self.material_db,
            KB_STRUCTURE: self.structure_db,
            KB_FAILURE: self.failure_db,
            KB_ANSYS: self.ansys_db,
        }

        # BM25 索引
        bm25_dir = os.path.join(persist_dir, "bm25")
        self.material_bm25 = BM25Index(os.path.join(bm25_dir, "material.pkl"))
        self.structure_bm25 = BM25Index(os.path.join(bm25_dir, "structure.pkl"))
        self.failure_bm25 = BM25Index(os.path.join(bm25_dir, "failure.pkl"))
        self.ansys_bm25 = BM25Index(os.path.join(bm25_dir, "ansys.pkl"))
        self._bm25_map: dict[str, BM25Index] = {
            KB_MATERIAL: self.material_bm25,
            KB_STRUCTURE: self.structure_bm25,
            KB_FAILURE: self.failure_bm25,
            KB_ANSYS: self.ansys_bm25,
        }

        if autoload:
            self.load_all()

    def _get_index(self, kb_id: str) -> FAISSKnowledgeIndex:
        if kb_id not in self._by_id:
            raise KeyError(f"未知知识库: {kb_id}，可选: {ALL_KB_IDS}")
        return self._by_id[kb_id]

    # ── FAISS ──
    def store_records(self, kb_id: str, records: list[dict[str, Any]]) -> None:
        idx = self._get_index(kb_id)
        n = idx.add_records(records)
        print(f"[{kb_id}] 已追加 {n} 条向量，当前总量 {idx.count}")

    def store_chunks(self, chunks, kb_id: str) -> None:
        records = [{"text": c.page_content, "metadata": dict(c.metadata)} for c in chunks]
        self.store_records(kb_id, records)

    def store_texts(self, texts: list[str], kb_id: str, metadata: dict[str, Any] | None = None) -> None:
        meta = dict(metadata or {})
        records = [{"text": t, "metadata": meta} for t in texts]
        self.store_records(kb_id, records)

    def semantic_search(self, kb_id: str, query: str, top_k: int = 5) -> list[SemanticHit]:
        return self._get_index(kb_id).search(query, top_k=top_k)

    def search(self, query: str, kb_id: str | None = None, n_results: int = 3) -> list[str]:
        if kb_id:
            hits = self.semantic_search(kb_id, query, top_k=n_results)
            return [h.text for h in hits]
        texts: list[str] = []
        for kid in ALL_KB_IDS:
            hits = self.semantic_search(kid, query, top_k=max(1, n_results // 2))
            texts.extend(h.text for h in hits)
        return texts[: n_results * len(ALL_KB_IDS)]

    # ── BM25 ──
    def bm25_search(self, kb_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        bm25 = self._bm25_map.get(kb_id)
        if bm25 is None:
            return []
        return bm25.search(query, top_k=top_k)

    def add_to_bm25(self, kb_id: str, texts: list[str], metadata: list[dict] | None = None) -> None:
        bm25 = self._bm25_map.get(kb_id)
        if bm25 is None:
            print(f"[BM25] 未知知识库: {kb_id}")
            return
        bm25.add_documents(texts, metadata)
        print(f"[BM25] {kb_id} 已追加 {len(texts)} 条文档，当前总量 {len(bm25)}")

    def hybrid_search(self, kb_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        faiss_hits = self.semantic_search(kb_id, query, top_k=top_k * 2)
        bm25_hits = self.bm25_search(kb_id, query, top_k=top_k * 2)

        if not bm25_hits:
            # BM25 为空时降级为纯 FAISS
            return [
                {"text": h.text, "rrf_score": 0.0, "source": "faiss",
                 "faiss_rank": i + 1, "bm25_rank": None,
                 "metadata": h.metadata}
                for i, h in enumerate(faiss_hits[:top_k])
            ]

        fused = reciprocal_rank_fusion(faiss_hits, bm25_hits)
        return fused[:top_k]

    # ── 持久化 ──
    def save_all(self) -> None:
        for kid, idx in self._by_id.items():
            idx.save()
            print(f"已保存索引: {kid}")
        for kid, bm25 in self._bm25_map.items():
            bm25.save()
            print(f"已保存 BM25: {kid}（{len(bm25)} 文档）")

    def load_all(self) -> None:
        for kid, idx in self._by_id.items():
            if idx.load():
                print(f"已加载索引: {kid}（{idx.count} 向量）")
            else:
                print(f"未找到已保存索引（空库）: {kid}")
        # BM25 在 __init__ 中自动加载，此处检查状态
        for kid, bm25 in self._bm25_map.items():
            if len(bm25) > 0:
                print(f"已加载 BM25: {kid}（{len(bm25)} 文档）")
