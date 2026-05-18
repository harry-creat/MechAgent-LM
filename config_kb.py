"""
知识库工程分类配置（竞赛级 RAG：多域隔离 + 可审计路由）。

四个独立向量索引域：
- material_db：材料性能、热处理、选材
- structure_db：机械结构、传动、连接、设计手册类文本
- failure_db：失效模式、疲劳、断裂、磨损、可靠性
- ansys_db：ANSYS/Abaqus/COMSOL 与 FEA 全流程（案例、边界模板、网格收敛、接触与失效判读）
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# 对外稳定的知识库标识（与持久化目录名一致）
KB_MATERIAL: Final[str] = "material_db"
KB_STRUCTURE: Final[str] = "structure_db"
KB_FAILURE: Final[str] = "failure_db"
KB_ANSYS: Final[str] = "ansys_db"

ALL_KB_IDS: Final[tuple[str, ...]] = (
    KB_MATERIAL,
    KB_STRUCTURE,
    KB_FAILURE,
    KB_ANSYS,
)

# Markdown 源文件 → 目标知识库（工程分类逻辑）
MD_SOURCES: Final[dict[str, str]] = {
    "knowledge_base/materials.md": KB_MATERIAL,
    "knowledge_base/structures.md": KB_STRUCTURE,
    "knowledge_base/failure_modes.md": KB_FAILURE,
    "knowledge_base/ansys_cases.md": KB_ANSYS,
}

# PDF（机械设计手册等）体量较大，语义上偏“结构与机械设计规范”，归入 structure_db
# 若需单独“手册库”，可新增 KB 并在路由中扩展。
PDF_TO_KB: Final[str] = KB_STRUCTURE

# FAISS 与元数据落盘目录
FAISS_PERSIST_DIR: Final[str] = "./faiss_db"

ROOT = Path(__file__).resolve().parent
