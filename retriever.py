"""
检索层：路由 → 对应 FAISS 语义检索 → 结构化 hits（非关键词匹配）。
"""

from __future__ import annotations

from config_kb import KB_FAILURE
from faiss_store import SemanticHit
from router import RouteDecision, route_query
from vector_store import VectorStore

_store = VectorStore()


def retrieve(
    query: str,
    *,
    top_k: int = 5,
    vector_store: VectorStore | None = None,
    use_llm_router: bool = False,
) -> dict:
    """
    返回：
    - decision: RouteDecision
    - hits: list[SemanticHit]
    - aux_hits: 仿真模式下从 failure_db 补充的失效判据片段（可选）
    """
    vs = vector_store or _store
    decision: RouteDecision = route_query(query, use_llm=use_llm_router)
    hits: list[SemanticHit] = vs.semantic_search(decision.kb_id, query, top_k=top_k)
    aux_hits: list[SemanticHit] = []
    if decision.simulation_mode:
        aux_hits = vs.semantic_search(KB_FAILURE, query, top_k=min(3, max(2, top_k // 2)))
    return {"decision": decision, "hits": hits, "aux_hits": aux_hits}


def retrieve_legacy_doc_strings(query: str, vector_store: VectorStore | None = None) -> list[str]:
    """兼容极旧调用：仅返回文本列表。"""
    pack = retrieve(query, vector_store=vector_store)
    return [h.text for h in pack["hits"]]
