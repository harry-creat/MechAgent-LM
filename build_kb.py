"""
构建/增量构建四个独立 FAISS + BM25 双索引知识库。

流程：
1) Markdown 按 config_kb.MD_SOURCES 映射入库；
2) PDF 按块解析 → 滑窗切分 → 归入 structure_db（设计手册类）；
3) FAISS 入库后同步写入 BM25；
4) save_all 持久化。
"""

from __future__ import annotations

import glob
from pathlib import Path

from config_kb import FAISS_PERSIST_DIR, MD_SOURCES, PDF_TO_KB
from loader import load_doc
from pdf_loader import load_pdf_blocks, merge_pdf_blocks
from splitter import split_docs, split_records
from vector_store import VectorStore


def _sync_to_bm25(store: VectorStore, kb_id: str, records: list[dict], label: str = "") -> None:
    """将同一批记录同步写入 BM25 索引。"""
    texts = [r.get("text", r.get("_chunk_text", "")) for r in records]
    metas = [r.get("metadata", {}) for r in records]
    if texts:
        store.add_to_bm25(kb_id, texts, metas)
        tag = f" ({label})" if label else ""
        print(f"  [BM25] 同步 {len(texts)} 条 → {kb_id}{tag}")


def build(*, load_existing: bool = True) -> None:
    store = VectorStore(FAISS_PERSIST_DIR, autoload=load_existing)

    for rel_path, kb_id in MD_SOURCES.items():
        p = Path(rel_path)
        if not p.is_file():
            print(f"跳过（文件不存在）: {rel_path}")
            continue
        print(f"处理 Markdown → {kb_id}: {rel_path}")
        docs = load_doc(str(p))
        chunks = split_docs(docs)
        for ch in chunks:
            ch.metadata["kb"] = kb_id
        store.store_chunks(chunks, kb_id)
        # BM25 同步
        records = [{"text": c.page_content, "metadata": dict(c.metadata)} for c in chunks]
        _sync_to_bm25(store, kb_id, records, os.path.basename(rel_path))

    pdf_paths = sorted(glob.glob(str(Path("knowledge_base") / "*.pdf")))
    if not pdf_paths:
        print("未在 knowledge_base/ 下发现 PDF，跳过 PDF 入库。")
    else:
        for pdf_path in pdf_paths:
            print(f"处理 PDF（块级）→ {PDF_TO_KB}: {pdf_path}")
            blocks = merge_pdf_blocks(load_pdf_blocks(pdf_path), max_chars=1600)
            records = split_records(blocks, chunk_size=1200, chunk_overlap=120)
            for r in records:
                r["metadata"]["kb"] = PDF_TO_KB
            store.store_records(PDF_TO_KB, records)
            _sync_to_bm25(store, PDF_TO_KB, records, os.path.basename(pdf_path))

    store.save_all()
    print("知识库构建完成（FAISS + BM25 四库双索引 + 持久化）。")


if __name__ == "__main__":
    build()
