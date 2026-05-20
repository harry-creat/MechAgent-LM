"""
BM25 关键词检索索引封装。

与 FAISS 向量检索互补：对专业术语（渐开线/赫兹接触/GB/T 3077/材料牌号）
效果更好；与 FAISS 结果通过 RRF（倒数排名融合）合并后送入下游。
"""

from __future__ import annotations

import os
import pickle
import re
from typing import Any

# ── 停用词 ──
_STOP_WORDS: set[str] = {
    "的", "了", "是", "在", "和", "与", "或", "对", "从",
    "到", "为", "以", "于", "及", "等", "中", "上", "下",
    "也", "这", "那", "不", "都", "而", "但", "且", "其",
}


# ── 需要保护不被拆分的模式（正则预提取后冻结）──
_PROTECT_PATTERNS: list[tuple[str, str]] = [
    # 标准号
    (r"GB\s*/\s*T\s*\d+(?:\.\d+)?(?:[-–]\d+)?", "STD_GB"),
    (r"ISO\s*\d+(?:[-–:]\d+)?", "STD_ISO"),
    (r"ASTM\s*\w\d+", "STD_ASTM"),
    (r"DIN\s*\d+", "STD_DIN"),
    (r"JB\s*/\s*T\s*\d+", "STD_JB"),
    # 材料牌号
    (r"\d{2}\s*(?:Cr|CrMo|CrV|CrNi|Mn|SiMn|CrMnTi|CrNiMo)\w*", "MATL_STEEL"),
    (r"Q\d{3}\w*", "MATL_Q"),
    (r"HT\d{3}\w*", "MATL_HT"),
    (r"QT\d{3}\w*", "MATL_QT"),
    # 单位
    (r"\d+\.?\d*\s*(?:MPa|GPa|kN|Nm|rpm|mm|kW)", "UNIT"),
    # 电子邮件/URL（保护不拆分）
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "EMAIL"),
]


def chinese_tokenize(text: str) -> list[str]:
    """中文分词，保留英文单词、数字、标准号、材料牌号不被拆分。

    Args:
        text: 待分词文本

    Returns:
        分词后的 token 列表（已过滤停用词和单字符）
    """
    # Step 1: 预提取保护模式，替换为占位符
    protected: list[str] = []
    for pattern, ptype in _PROTECT_PATTERNS:
        def _repl(m: re.Match, _ptype: str = ptype) -> str:
            protected.append(m.group(0))
            return f" __{_ptype}_{len(protected)-1}__ "
        text = re.sub(pattern, _repl, text)

    # Step 2: jieba 分词
    try:
        import jieba
        tokens = list(jieba.cut(text))
    except ImportError:
        tokens = text.split()

    # Step 3: 过滤 + 还原保护词
    result: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token or len(token) <= 1:
            # 检查是否是保护词占位符
            m = re.match(r"__(\w+)_(\d+)__", token)
            if m:
                idx = int(m.group(2))
                if idx < len(protected):
                    result.append(protected[idx])
            continue
        if token in _STOP_WORDS:
            continue
        if re.match(r"^[\s\d\.\-,;:!?]+$", token):
            continue
        result.append(token.lower())

    return result


# ── BM25 索引类 ──
class BM25Index:
    """单知识库 BM25 索引，自动 pickle 持久化。"""

    def __init__(self, index_path: str):
        self.index_path = index_path
        self.corpus_tokens: list[list[str]] = []
        self.corpus_texts: list[str] = []
        self.corpus_metadata: list[dict[str, Any]] = []
        self._bm25: Any = None

        if os.path.isfile(index_path):
            self._load()
            self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        from rank_bm25 import BM25Okapi
        if self.corpus_tokens:
            self._bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self._bm25 = None

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        data = {
            "corpus_tokens": self.corpus_tokens,
            "corpus_texts": self.corpus_texts,
            "corpus_metadata": self.corpus_metadata,
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)

    def _load(self) -> None:
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self.corpus_tokens = data.get("corpus_tokens", [])
            self.corpus_texts = data.get("corpus_texts", [])
            self.corpus_metadata = data.get("corpus_metadata", [])
        except (pickle.UnpicklingError, EOFError, KeyError, OSError):
            self.corpus_tokens = []
            self.corpus_texts = []
            self.corpus_metadata = []

    def add_documents(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        """追加文档到索引。

        Args:
            texts: 文档文本列表
            metadata: 对应的元数据列表（可选）
        """
        if metadata is None:
            metadata = [{} for _ in texts]
        for t, m in zip(texts, metadata):
            tokens = chinese_tokenize(t)
            if not tokens:
                continue
            self.corpus_tokens.append(tokens)
            self.corpus_texts.append(t)
            self.corpus_metadata.append(dict(m or {}))
        self._rebuild_bm25()
        self.save()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """BM25 关键词检索。

        Args:
            query: 查询字符串
            top_k: 返回数量

        Returns:
            [{"text": str, "score": float, "rank": int, "metadata": dict}, ...]
        """
        if not self._bm25 or not self.corpus_tokens:
            return []

        query_tokens = chinese_tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        # 按分数降序取 top_k
        indexed = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        results: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(indexed[:top_k], 1):
            if score <= 0:
                break
            results.append({
                "text": self.corpus_texts[idx],
                "score": float(score),
                "rank": rank,
                "metadata": dict(self.corpus_metadata[idx]),
            })
        return results

    def __len__(self) -> int:
        return len(self.corpus_texts)


# ── RRF 倒数排名融合 ──
def reciprocal_rank_fusion(
    faiss_hits: list[Any],
    bm25_hits: list[dict[str, Any]],
    k: int = 60,
    faiss_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict[str, Any]]:
    """RRF 融合 FAISS 和 BM25 两路检索结果。

    公式: score(d) = Σ weight_i / (k + rank_i(d))

    Args:
        faiss_hits: FAISS 检索结果 (SemanticHit 列表)
        bm25_hits: BM25 检索结果 (BM25Index.search() 返回)
        k: RRF 平滑常数
        faiss_weight: FAISS 路权重
        bm25_weight: BM25 路权重

    Returns:
        融合后列表，按 rrf_score 降序排列，含 source 标注
    """
    # 建立 text→fusion 映射
    fusion_map: dict[str, dict[str, Any]] = {}

    # FAISS 路
    for rank, hit in enumerate(faiss_hits, 1):
        text = getattr(hit, "text", "")
        key = text[:200]  # 用前200字符作为去重键
        fusion_map[key] = {
            "text": text,
            "faiss_rank": rank,
            "faiss_score": getattr(hit, "score_l2", 0.0),
            "bm25_rank": None,
            "bm25_score": 0.0,
            "rrf_score": 0.0,
            "source": "faiss",
            "metadata": getattr(hit, "metadata", {}),
        }

    # BM25 路
    for hit in bm25_hits:
        text = hit.get("text", "")
        key = text[:200]
        rank = hit.get("rank", 99)
        if key in fusion_map:
            fusion_map[key]["bm25_rank"] = rank
            fusion_map[key]["bm25_score"] = hit.get("score", 0.0)
            fusion_map[key]["source"] = "both"
        else:
            fusion_map[key] = {
                "text": text,
                "faiss_rank": None,
                "faiss_score": 0.0,
                "bm25_rank": rank,
                "bm25_score": hit.get("score", 0.0),
                "rrf_score": 0.0,
                "source": "bm25",
                "metadata": hit.get("metadata", {}),
            }

    # 计算 RRF 分数
    for key, entry in fusion_map.items():
        score = 0.0
        if entry["faiss_rank"] is not None:
            score += faiss_weight / (k + entry["faiss_rank"])
        if entry["bm25_rank"] is not None:
            score += bm25_weight / (k + entry["bm25_rank"])
        entry["rrf_score"] = round(score, 6)

    # 按 rrf_score 降序
    sorted_results = sorted(
        fusion_map.values(), key=lambda x: x["rrf_score"], reverse=True
    )
    return sorted_results
