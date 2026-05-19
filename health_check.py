"""
系统自检模块。

启动时自动检查:
- FAISS 索引文件是否存在且可加载
- DeepSeek API Key 是否配置
- Embedding 模型是否可正常加载
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from config_kb import ALL_KB_IDS, FAISS_PERSIST_DIR


@dataclass
class CheckItem:
    name: str
    ok: bool
    detail: str = ""
    icon: str = ""


@dataclass
class HealthReport:
    items: list[CheckItem] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(item.ok for item in self.items)

    @property
    def passed(self) -> int:
        return sum(1 for item in self.items if item.ok)

    @property
    def failed(self) -> int:
        return len(self.items) - self.passed


def _check_faiss_indices() -> list[CheckItem]:
    results: list[CheckItem] = []
    for kb_id in ALL_KB_IDS:
        idx_path = os.path.join(FAISS_PERSIST_DIR, f"{kb_id}.index")
        meta_path = os.path.join(FAISS_PERSIST_DIR, f"{kb_id}.meta.json")

        idx_exists = os.path.isfile(idx_path)
        meta_exists = os.path.isfile(meta_path)

        if idx_exists and meta_exists:
            try:
                import faiss
                idx = faiss.read_index(idx_path)
                count = idx.ntotal
                results.append(CheckItem(
                    name=f"FAISS 索引: {kb_id}",
                    ok=True,
                    detail=f"已加载，{count} 条向量",
                    icon="✅",
                ))
            except Exception as e:
                results.append(CheckItem(
                    name=f"FAISS 索引: {kb_id}",
                    ok=False,
                    detail=f"索引文件存在但加载失败: {e}",
                    icon="❌",
                ))
        elif not idx_exists and not meta_exists:
            results.append(CheckItem(
                name=f"FAISS 索引: {kb_id}",
                ok=True,
                detail="空库（尚未导入文档）",
                icon="⚠️",
            ))
        else:
            missing = []
            if not idx_exists:
                missing.append(".index")
            if not meta_exists:
                missing.append(".meta.json")
            results.append(CheckItem(
                name=f"FAISS 索引: {kb_id}",
                ok=False,
                detail=f"缺少文件: {', '.join(missing)}",
                icon="❌",
            ))
    return results


def _check_api_key() -> CheckItem:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "***"
        return CheckItem(
            name="DeepSeek API Key",
            ok=True,
            detail=f"已配置 ({masked})",
            icon="✅",
        )
    return CheckItem(
        name="DeepSeek API Key",
        ok=False,
        detail="未设置环境变量 DEEPSEEK_API_KEY",
        icon="❌",
    )


def _check_embedding_model() -> CheckItem:
    try:
        from embedding_model import get_embedding_dimension, get_model
        m = get_model()
        dim = get_embedding_dimension()
        return CheckItem(
            name="Embedding 模型",
            ok=True,
            detail=f"已加载，维度={dim}",
            icon="✅",
        )
    except Exception as e:
        return CheckItem(
            name="Embedding 模型",
            ok=False,
            detail=f"加载失败: {e}",
            icon="❌",
        )


def run_health_check() -> HealthReport:
    items: list[CheckItem] = []
    items.extend(_check_faiss_indices())
    items.append(_check_api_key())
    items.append(_check_embedding_model())
    return HealthReport(items=items)


def format_health_report(report: HealthReport) -> str:
    lines = [f"系统自检: {report.passed}/{len(report.items)} 项通过"]
    for item in report.items:
        lines.append(f"  {item.icon} {item.name}: {item.detail}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 测试块
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    report = run_health_check()
    print(format_health_report(report))
    print(f"\n总体状态: {'全部通过' if report.all_ok else f'{report.failed} 项异常'}")
