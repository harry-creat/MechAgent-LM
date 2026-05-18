"""
多知识库向量存储：每个工程域独立 FAISS 索引（竞赛级隔离与可扩展路由）。

对外主类仍命名为 VectorStore，以兼容 build_kb / 旧 import；
底层实现为 faiss_store.FAISSKnowledgeIndex。
"""

from __future__ import annotations

from typing import Any

from config_kb import ALL_KB_IDS, FAISS_PERSIST_DIR, KB_ANSYS, KB_FAILURE, KB_MATERIAL, KB_STRUCTURE
from faiss_store import FAISSKnowledgeIndex, SemanticHit


class VectorStore:
    """四个独立 FAISS 索引 + 统一 save/load 入口。"""

    def __init__(self, persist_dir: str = FAISS_PERSIST_DIR, *, autoload: bool = True):
        self.persist_dir = persist_dir
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
        if autoload:
            self.load_all()

    def _get_index(self, kb_id: str) -> FAISSKnowledgeIndex:
        if kb_id not in self._by_id:
            raise KeyError(f"未知知识库: {kb_id}，可选: {ALL_KB_IDS}")
        return self._by_id[kb_id]

    def store_records(self, kb_id: str, records: list[dict[str, Any]]) -> None:
        """写入单库（覆盖式由调用方决定；此处为追加向量）。"""
        idx = self._get_index(kb_id)
        n = idx.add_records(records)
        print(f"[{kb_id}] 已追加 {n} 条向量，当前总量 {idx.count}")

    def store_chunks(self, chunks, kb_id: str) -> None:
        """兼容 LangChain Document 列表。"""
        records = [{"text": c.page_content, "metadata": dict(c.metadata)} for c in chunks]
        self.store_records(kb_id, records)

    def store_texts(self, texts: list[str], kb_id: str, metadata: dict[str, Any] | None = None) -> None:
        """兼容旧接口：纯文本列表（无分页信息时用于临时导入）。"""
        meta = dict(metadata or {})
        records = [{"text": t, "metadata": meta} for t in texts]
        self.store_records(kb_id, records)

    def semantic_search(self, kb_id: str, query: str, top_k: int = 5) -> list[SemanticHit]:
        return self._get_index(kb_id).search(query, top_k=top_k)

    def search(self, query: str, kb_id: str | None = None, n_results: int = 3) -> list[str]:
        """
        兼容旧 API：返回字符串列表（仅正文）。
        kb_id 为空时在四库各取若干条再拼接（用于调试，非最优检索）。
        """
        if kb_id:
            hits = self.semantic_search(kb_id, query, top_k=n_results)
            return [h.text for h in hits]
        texts: list[str] = []
        for kid in ALL_KB_IDS:
            hits = self.semantic_search(kid, query, top_k=max(1, n_results // 2))
            texts.extend(h.text for h in hits)
        return texts[: n_results * len(ALL_KB_IDS)]

    def save_all(self) -> None:
        for kid, idx in self._by_id.items():
            idx.save()
            print(f"已保存索引: {kid}")

    def load_all(self) -> None:
        for kid, idx in self._by_id.items():
            if idx.load():
                print(f"已加载索引: {kid}（{idx.count} 向量）")
            else:
                print(f"未找到已保存索引（空库）: {kid}")
