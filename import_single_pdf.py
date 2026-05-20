"""
一次性导入脚本：将《机械工程材料 第4版》导入 material_db。
"""

import os
from pathlib import Path

from pdf_loader import load_pdf_blocks, merge_pdf_blocks
from splitter import split_records
from vector_store import VectorStore


def main():
    # 用 glob 查找目标文件（避免中文空格/编码问题）
    candidates = [
        p for p in Path("knowledge_base").iterdir()
        if p.suffix == ".pdf" and "机械" in p.name and "材料" in p.name and "4" in p.name
    ]
    if not candidates:
        candidates = [
            p for p in Path("knowledge_base").iterdir()
            if p.suffix == ".pdf" and "材料" in p.name
        ]
    if not candidates:
        print("❌ 未找到机械工程材料 PDF 文件")
        import sys
        sys.exit(1)

    pdf_path = candidates[0]
    print(f"📄 找到文件: {pdf_path.name}")

    print(f"📄 文件: {pdf_path} ({pdf_path.stat().st_size // 1024 // 1024}MB)")

    # Step 1: 提取文本块
    print("1/4 提取文本块...")
    blocks = load_pdf_blocks(str(pdf_path))
    print(f"   提取 {len(blocks)} 个文本块")

    # Step 2: 合并相邻块
    print("2/4 合并段落...")
    merged = merge_pdf_blocks(blocks, max_chars=2000)
    print(f"   合并为 {len(merged)} 个段落")

    # Step 3: 切分
    print("3/4 文本切分...")
    records = split_records(merged, chunk_size=800, chunk_overlap=120)
    for r in records:
        r["metadata"].update({
            "kb": "material_db",
            "source": "机械工程材料 第4版.pdf",
            "title": "机械工程材料 第4版",
            "author": "沈莲",
            "category": "机械工程材料",
            "description": "沈莲主编《机械工程材料》第4版，涵盖金属材料/高分子/陶瓷/复合材料的结构-性能-工艺关系、热处理原理与工艺、典型机械零件选材与失效分析",
        })
    print(f"   切分为 {len(records)} 个片段")

    # Step 4: 写入 material_db（FAISS + BM25 双索引）
    print("4/4 写入 material_db...")
    store = VectorStore(autoload=True)
    print(f"   导入前 material_db: FAISS={store._get_index('material_db').count} 条")

    # FAISS
    store.store_records("material_db", records)

    # BM25
    texts = [r["text"] for r in records]
    metas = [r.get("metadata", {}) for r in records]
    store.add_to_bm25("material_db", texts, metas)

    store.save_all()
    print(f"   导入后 material_db: FAISS={store._get_index('material_db').count} 条, BM25={len(store.material_bm25)} 文档")
    print(f"✅ 导入完成! 新增 {len(records)} 条 FAISS 向量 + BM25 文档")


if __name__ == "__main__":
    main()
