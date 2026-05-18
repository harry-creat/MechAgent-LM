"""
构建/增量构建四个独立 FAISS 知识库索引。

流程：
1) Markdown 按 config_kb.MD_SOURCES 映射入库；
2) PDF 按块解析 → 滑窗切分 → 归入 structure_db（设计手册类）；
3) save_all 持久化到 faiss_db/。
"""

from __future__ import annotations

import glob
from pathlib import Path

from config_kb import FAISS_PERSIST_DIR, MD_SOURCES, PDF_TO_KB
from loader import load_doc
from pdf_loader import load_pdf_blocks, merge_pdf_blocks
from splitter import split_docs, split_records
from vector_store import VectorStore


def build(*, load_existing: bool = True) -> None:
    """
    load_existing=True 时先尝试加载已有索引再追加；适合多次运行累积。
    若需从零重建，请手动删除 faiss_db 目录后再运行。
    """
    # load_existing=False 时不加载磁盘索引，避免在“误重复运行”时与空库混淆；
    # 从零构建请删除 faiss_db/ 目录或使用 load_existing=False。
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

    pdf_paths = sorted(glob.glob(str(Path("knowledge_base") / "*.pdf")))
    if not pdf_paths:
        print("未在 knowledge_base/ 下发现 PDF，跳过 PDF 入库。")
    else:
        for pdf_path in pdf_paths:
            print(f"处理 PDF（块级）→ {PDF_TO_KB}: {pdf_path}")
            blocks = merge_pdf_blocks(load_pdf_blocks(pdf_path), max_chars=1600)
            # 手册页信息密集：略增大 chunk 减少条目数，仍保持段落级粒度
            records = split_records(blocks, chunk_size=1200, chunk_overlap=120)
            for r in records:
                r["metadata"]["kb"] = PDF_TO_KB
            store.store_records(PDF_TO_KB, records)

    store.save_all()
    print("知识库构建完成（FAISS 四库 + 持久化）。")


if __name__ == "__main__":
    build()
