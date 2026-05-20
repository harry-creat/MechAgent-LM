"""
检索对比工具 — 对同一查询比较 FAISS、BM25、混合三种方式的结果。
用于论文对比实验和检索质量评估。
"""

from __future__ import annotations

from typing import Any

from bm25_store import reciprocal_rank_fusion
from retriever import retrieve
from vector_store import VectorStore

_store = VectorStore()


def compare_search(query: str, top_k: int = 5) -> dict[str, Any]:
    """三路检索对比。

    Returns:
        {"faiss_only": [...], "bm25_only": [...], "hybrid": [...]}
        每条含 text/rrf_score/source/rank 等字段
    """

    # 先获取路由决策
    from router import route_query
    decision = route_query(query)
    kb_id = decision.kb_id

    # ── FAISS only ──
    faiss_raw = _store.semantic_search(kb_id, query, top_k=top_k)
    faiss_only = [
        {"text": h.text[:120], "score_l2": h.score_l2, "rank": i + 1, "source": h.metadata.get("source", "")[:40]}
        for i, h in enumerate(faiss_raw)
    ]

    # ── BM25 only ──
    bm25_raw = _store.bm25_search(kb_id, query, top_k=top_k)
    bm25_only = [
        {"text": h["text"][:120], "score": h["score"], "rank": h["rank"], "source": h.get("metadata", {}).get("source", "")[:40]}
        for h in bm25_raw[:top_k]
    ]

    # ── Hybrid ──
    hybrid_raw = _store.hybrid_search(kb_id, query, top_k=top_k)
    hybrid = [
        {"text": h["text"][:120], "rrf_score": h["rrf_score"], "source_type": h["source"],
         "faiss_rank": h.get("faiss_rank"), "bm25_rank": h.get("bm25_rank"),
         "metadata_source": h.get("metadata", {}).get("source", "")[:40]}
        for h in hybrid_raw
    ]

    return {
        "faiss_only": faiss_only,
        "bm25_only": bm25_only,
        "hybrid": hybrid,
        "kb_id": kb_id,
    }


if __name__ == "__main__":
    test_queries = [
        "渐开线齿轮",
        "赫兹接触应力",
        "GB/T 3077 合金钢",
        "疲劳裂纹扩展",
        "ANSYS网格划分",
        "轴承寿命计算",
        "45钢热处理工艺",
        "Workbench静力学分析步骤",
    ]

    for q in test_queries:
        result = compare_search(q)
        print(f"\n{'='*60}")
        print(f"查询: {q}  →  路由: {result['kb_id']}")

        for mode in ["faiss_only", "bm25_only", "hybrid"]:
            print(f"\n  ── {mode} ──")
            items = result[mode]
            if not items:
                print("    (无结果)")
                continue
            for i, hit in enumerate(items[:3]):
                if mode == "hybrid":
                    src_tag = f"[{hit['source_type']}]" if hit.get('source_type') else ""
                    print(f"    {i+1}. {src_tag} {hit['text'][:100]}")
                    if hit.get('rrf_score'):
                        print(f"       RRF={hit['rrf_score']:.4f} FAISS_rank={hit.get('faiss_rank')} BM25_rank={hit.get('bm25_rank')}")
                else:
                    print(f"    {i+1}. {hit['text'][:100]}")
                    if mode == "bm25_only":
                        print(f"       score={hit.get('score', 0):.2f}")
