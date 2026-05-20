"""
检索层：路由 → FAISS + BM25 混合检索 → 结构化 hits。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config_kb import KB_FAILURE
from router import RouteDecision, route_query
from vector_store import VectorStore


@dataclass
class HybridHit:
    """混合检索结果，兼容 SemanticHit 的 text/metadata/score_l2 使用方式。"""
    text: str
    rrf_score: float = 0.0
    source: str = "faiss"
    metadata: dict[str, Any] | None = None

    @property
    def score_l2(self) -> float:
        """兼容 SemanticHit.score_l2，将 RRF 分数映射为 L2 距离（越小越好）。"""
        return 1.0 / (1.0 + self.rrf_score) if self.rrf_score > 0 else 1.0


_store = VectorStore()


def retrieve(
    query: str,
    *,
    top_k: int = 5,
    vector_store: VectorStore | None = None,
    use_llm_router: bool = False,
    use_hybrid: bool = True,
) -> dict[str, Any]:
    """检索入口。

    Args:
        query: 用户问题
        top_k: 返回片段数
        vector_store: 可选外部 VectorStore 实例
        use_llm_router: 是否使用 LLM 路由
        use_hybrid: 是否启用 BM25+FAISS 混合检索（默认开启）

    Returns:
        {"decision": RouteDecision, "hits": list[HybridHit|SemanticHit],
         "aux_hits": list, "use_hybrid": bool}
    """
    vs = vector_store or _store
    decision: RouteDecision = route_query(query, use_llm=use_llm_router)

    if use_hybrid:
        raw_hits = vs.hybrid_search(decision.kb_id, query, top_k=top_k)
        hits = [
            HybridHit(
                text=h["text"],
                rrf_score=h["rrf_score"],
                source=h["source"],
                metadata=h.get("metadata", {}),
            )
            for h in raw_hits
        ]
    else:
        hits = vs.semantic_search(decision.kb_id, query, top_k=top_k)

    # 仿真模式补充 failure_db
    aux_hits: list = []
    if decision.simulation_mode:
        if use_hybrid:
            aux_raw = vs.hybrid_search(KB_FAILURE, query, top_k=min(3, max(2, top_k // 2)))
            aux_hits = [
                HybridHit(
                    text=h["text"],
                    rrf_score=h["rrf_score"],
                    source=h["source"],
                    metadata=h.get("metadata", {}),
                )
                for h in aux_raw
            ]
        else:
            aux_hits = vs.semantic_search(KB_FAILURE, query, top_k=min(3, max(2, top_k // 2)))

    return {
        "decision": decision,
        "hits": hits,
        "aux_hits": aux_hits,
        "use_hybrid": use_hybrid,
    }


def retrieve_legacy_doc_strings(query: str, vector_store: VectorStore | None = None) -> list[str]:
    pack = retrieve(query, vector_store=vector_store)
    return [h.text for h in pack["hits"]]
