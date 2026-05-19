# 机械工程 AI 助手 — 项目框架文档

> 生成时间：2026-05-19
> 基于 RAG（检索增强生成）架构，集成参数计算、结构推荐、仿真分析三大增强模块

---

## 一、目录结构

```
ai-mechanical-assistant/
├── app.py                  # Streamlit 主界面（ChatGPT 风格深色 UI）
├── ai_engine.py            # 核心编排层：路由→检索→Prompt→LLM，多模式分支
├── router.py               # 问题路由：规则版 4 库路由 + 仿真/计算/推荐意图识别
├── retriever.py            # 检索层：路由→FAISS 语义检索→结构化 hits
├── prompts.py              # Prompt 模板（通用四段式 / 仿真六段式 / 计算五段式 / 推荐六段式）
├── deepseek_api.py         # DeepSeek Chat Completions API 调用封装（含超时/限流处理）
│
├── calculator.py           # 机械参数计算模块：7 种公式 + 意图分派器
├── recommender.py          # 智能结构推荐模块：5 场景 18 方案知识库 + 匹配引擎
│
├── vector_store.py         # 多知识库向量存储门面（4 个独立 FAISS 索引）
├── faiss_store.py          # 单库 FAISS 索引封装（IndexFlatL2 + 元数据）
├── embedding_model.py      # sentence-transformers 句向量模型加载与推理
│
├── build_kb.py             # 知识库构建入口（Markdown + PDF → 向量索引）
├── loader.py               # Markdown 文档加载器（UTF-8 → LangChain Document）
├── pdf_loader.py           # PDF 解析器（PyMuPDF 块级提取 + 合并）
├── splitter.py             # 文本切分器（LangChain RecursiveCharacterTextSplitter）
│
├── config_kb.py            # 知识库分类配置（库 ID、MD→KB 映射、持久化路径）
├── simulation_mode.py      # 仿真分析模式识别（正则匹配 CAE/FEA 关键词）
├── health_check.py         # 系统自检（FAISS 索引 / API Key / Embedding 模型）
├── logger.py               # 对话日志（JSONL 存储 / Markdown 报告导出 / 错误日志）
│
├── requirements.txt        # Python 依赖列表
├── .env.example            # 环境变量配置模板
├── .env                    # 实际环境变量（不提交版本库）
├── .gitignore              # Git 忽略规则
├── README.md               # 项目说明文档
│
├── faiss_db/               # FAISS 索引持久化目录（.index + .meta.json）
├── knowledge_base/         # 原始知识文档（.md / .pdf）
├── logs/                   # 对话日志与错误日志
└── db/                     # Chroma 向量数据库目录（embedding 模型缓存）
```

---

## 二、架构分层

```
┌──────────────────────────────────────────────────────────────────┐
│  UI 层                                                           │
│  app.py                                                          │
├──────────────────────────────────────────────────────────────────┤
│  编排层                                                           │
│  ai_engine.py ─── 多模式分支（问答/计算/推荐/仿真）                  │
│  router.py    ─── 意图识别 + 知识库路由                            │
│  retriever.py ─── 路由→检索 桥接                                   │
├──────────────────────────────────────────────────────────────────┤
│  Prompt 层                                                        │
│  prompts.py  ─── 4 套专家级 Prompt 模板                            │
├──────────────────────────────────────────────────────────────────┤
│  能力增强层                                                        │
│  calculator.py ─── 7 种机械参数自动计算                            │
│  recommender.py ─── 5 场景 18 方案智能推荐                         │
│  simulation_mode.py ─── CAE/FEA 仿真模式识别                       │
├──────────────────────────────────────────────────────────────────┤
│  检索与向量存储层                                                   │
│  vector_store.py ─── 多库门面                                      │
│  faiss_store.py  ─── 单库 FAISS 索引                              │
│  embedding_model.py ─── 句向量模型                                 │
├──────────────────────────────────────────────────────────────────┤
│  知识库构建层                                                      │
│  build_kb.py   ─── 构建入口                                       │
│  loader.py     ─── Markdown 加载                                  │
│  pdf_loader.py ─── PDF 加载                                       │
│  splitter.py   ─── 文本切分                                       │
├──────────────────────────────────────────────────────────────────┤
│  工具 / 配置层                                                     │
│  config_kb.py      ─── 知识库分类配置                              │
│  deepseek_api.py   ─── LLM API 封装                               │
│  health_check.py   ─── 系统自检                                    │
│  logger.py         ─── 日志与报告                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 三、核心数据流

一次完整问答的数据流转路径：

```
用户输入 (app.py)
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  意图识别 (router.py)                                        │
│  · is_calculation_query()    → 参数计算意图？                 │
│  · is_recommendation_query() → 结构推荐意图？                 │
│  · detect_simulation_mode()  → 仿真分析意图？                 │
│  · route_query_rules()       → 四库关键词匹配路由              │
│  输出: RouteDecision(kb_id, calc_mode, recommend_mode, ...)  │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  向量检索 (retriever.py → vector_store.py → faiss_store.py) │
│  根据 kb_id 选择对应 FAISS 索引，查询 top_k 语义片段          │
│  输出: list[SemanticHit]（含 text + metadata + L2 距离）      │
└─────────────────────────────────────────────────────────────┘
  │
  ├── [若 calc_mode=True] ──→ calculator.dispatch_calculation()
  │                           将计算结果注入 Prompt
  │
  ├── [若 recommend_mode] ──→ recommender.recommend_structure()
  │                            将推荐方案注入 Prompt
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  Prompt 构建 (prompts.py)                                    │
│  · 通用: build_mechanical_rag_prompt()      四段式            │
│  · 仿真: build_simulation_rag_prompt()      六段式 CAE        │
│  · 计算: build_calculation_rag_prompt()     五段式 + 计算结果  │
│  · 推荐: build_recommendation_prompt()      六段式 + 方案列表  │
│  输出: 结构化 Prompt 字符串                                   │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM 调用 (deepseek_api.py)                                  │
│  POST https://api.deepseek.com/v1/chat/completions           │
│  输出: AI 回复文本                                           │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
返回答案 → app.py 渲染（消息气泡 + 计算卡片 / 推荐折叠 / 流式动画）
同时写入 logger.py（JSONL 日志 + 错误日志）
```

---

## 四、每个文件的详细说明

### app.py
- **所属层次**: UI 层
- **核心职责**: Streamlit 前端主入口，类 ChatGPT 深色模式 UI，管理聊天消息渲染、侧边栏控制、用户输入分发
- **对外暴露**: 无（直接运行入口）
- **主要函数**: 无（脚本化执行，调用 `ask_ai()`）
- **依赖**: `ai_engine`, `logger`, `health_check`

---

### ai_engine.py
- **所属层次**: 编排层
- **核心职责**: 多模式编排中枢。根据路由决策进入不同分支（计算/推荐/通用+仿真），构建对应 Prompt 并调用 LLM
- **对外暴露**: `ask_ai(question, *, use_llm_router, top_k, retrieval_pack, calc_params, calc_result) -> tuple[str, dict]`
- **辅助函数**: `_route_label(decision) -> str` — 路由决策→可读标签
- **依赖**: `calculator`, `deepseek_api`, `prompts`, `recommender`, `retriever`, `logger`

---

### router.py
- **所属层次**: 编排层
- **核心职责**: 问题意图识别与知识库路由。规则版（低延迟）+ 可选 LLM 版（语义更细）
- **对外暴露**:
  - `route_query(query, *, use_llm) -> RouteDecision` — 主路由入口（优先级: calc > recommend > simulation > 四库）
  - `is_calculation_query(query) -> bool` — 参数计算意图识别
  - `is_recommendation_query(query) -> bool` — 结构推荐意图识别
  - `route_query_rules(query) -> RouteDecision` — 规则版四库路由
  - `route_query_llm(query) -> RouteDecision | None` — LLM 版路由
- **数据类**: `RouteDecision(kb_id, reason, method, simulation_mode, calc_mode, recommend_mode)`
- **依赖**: `config_kb`, `simulation_mode`, `deepseek_api`

---

### retriever.py
- **所属层次**: 检索层
- **核心职责**: 将路由决策转化为实际 FAISS 语义检索动作，返回结构化 hits
- **对外暴露**:
  - `retrieve(query, *, top_k, vector_store, use_llm_router) -> dict` — 返回 `{decision, hits, aux_hits}`
  - `retrieve_legacy_doc_strings(query, vector_store) -> list[str]` — 兼容旧接口
- **依赖**: `config_kb`, `faiss_store`, `router`, `vector_store`

---

### prompts.py
- **所属层次**: Prompt 层
- **核心职责**: 构建 4 套专家级 Prompt 模板，控制 LLM 输出格式和质量
- **对外暴露**:
  - `format_retrieved_context(hits, *, aux_hits) -> str` — 检索片段→溯源上下文
  - `build_mechanical_rag_prompt(question, hits, decision) -> str` — 通用四段式（结论/分析/建议/风险）
  - `build_simulation_rag_prompt(question, hits, decision) -> str` — 仿真六段式（含进阶说明）
  - `build_calculation_rag_prompt(question, calc_result, hits, decision) -> str` — 计算五段式（含计算结果）
  - `build_recommendation_prompt(question, recs, rec_text, hits, decision) -> str` — 推荐六段式（需求理解/对比表/首选方案/参数建议/待确认/标准）
- **依赖**: `faiss_store`, `router`

---

### calculator.py
- **所属层次**: 能力增强层
- **核心职责**: 机械设计常用参数自动计算，含输入校验和分派路由
- **对外暴露**:
  - `calc_shaft_torsion(torque_Nm, diameter_mm) -> dict` — 轴扭转切应力 τ=16T/(πd³)
  - `calc_shaft_bending(moment_Nm, diameter_mm) -> dict` — 轴弯曲正应力 σ=32M/(πd³)
  - `calc_safety_factor(allowable, actual) -> dict` — 安全系数 n=[σ]/σ
  - `calc_bolt_preload(diameter_mm, yield_strength, factor) -> dict` — 螺栓预紧力
  - `calc_gear_module(torque_Nm, z_teeth, width_mm, sigma_hlim) -> dict` — 齿轮模数估算
  - `calc_bearing_life(C_kN, P_kN, speed_rpm, exponent) -> dict` — 轴承 L10 寿命
  - `calc_beam_deflection(force_N, length_mm, E_GPa, I_mm4) -> dict` — 简支梁挠度
  - `dispatch_calculation(user_query, params) -> dict` — 关键词分派器
- **依赖**: 标准库（math, re, typing）

---

### recommender.py
- **所属层次**: 能力增强层
- **核心职责**: 根据设计需求关键词匹配场景，从内置知识库推荐结构方案
- **对外暴露**:
  - `recommend_structure(query, calc_result) -> list[dict]` — 场景匹配 + 方案排序（max 3）
  - `format_recommendation(recommendations) -> str` — 格式化推荐文本
- **知识库规模**: 5 场景（轴系/齿轮/联接/轴承/密封）× 18 方案
- **依赖**: 标准库（typing）

---

### deepseek_api.py
- **所属层次**: 工具层
- **核心职责**: DeepSeek Chat Completions API 调用封装，含环境变量加载、代理支持、超时/限流/网络异常中文提示
- **对外暴露**:
  - `call_deepseek(prompt) -> str` — 单轮调用
  - `chat_messages(messages, *, model, temperature) -> str` — 多轮消息接口
- **依赖**: `requests`, `dotenv`, `logger`

---

### vector_store.py
- **所属层次**: 向量存储层
- **核心职责**: 四个独立 FAISS 索引的统一门面，对外提供一致存取接口
- **对外暴露**:
  - `VectorStore` 类:
    - `store_records(kb_id, records)` — 追加向量到指定库
    - `store_chunks(chunks, kb_id)` — 兼容 LangChain Document
    - `store_texts(texts, kb_id, metadata)` — 兼容纯文本
    - `semantic_search(kb_id, query, top_k) -> list[SemanticHit]` — 语义检索
    - `search(query, kb_id, n_results) -> list[str]` — 兼容旧 API
    - `save_all()` / `load_all()` — 持久化
- **依赖**: `config_kb`, `faiss_store`

---

### faiss_store.py
- **所属层次**: 向量存储层
- **核心职责**: 单知识库 FAISS IndexFlatL2 封装，支持增量 add、持久化、top-K 检索
- **对外暴露**:
  - `SemanticHit(text, metadata, score_l2)` — 检索结果数据类
  - `FAISSKnowledgeIndex` 类:
    - `add_records(records) -> int` — 向量化 + 追加
    - `search(query, top_k) -> list[SemanticHit]` — L2 最近邻
    - `save()` / `load() -> bool` — 持久化
    - `count -> int` — 向量总数
- **依赖**: `faiss`, `numpy`, `embedding_model`

---

### embedding_model.py
- **所属层次**: 向量存储层
- **核心职责**: sentence-transformers 句向量模型加载、缓存与批量推理
- **对外暴露**:
  - `get_model() -> SentenceTransformer` — 获取模型实例（全局单例）
  - `get_embedding_dimension() -> int` — 向量维度
  - `embed_text(text) -> np.ndarray` — 单句编码
  - `embed_texts(texts) -> np.ndarray` — 批量编码
- **配置**: 环境变量 `ST_MODEL_NAME` 切换模型（默认 `paraphrase-multilingual-MiniLM-L12-v2`）
- **依赖**: `sentence_transformers`, `numpy`

---

### build_kb.py
- **所属层次**: 知识库构建层
- **核心职责**: 一键构建/增量更新四个知识库的 FAISS 索引
- **对外暴露**: `build(*, load_existing) -> None`
- **处理流程**: Markdown（loader→splitter→store） + PDF（pdf_loader→merge→split_records→store） → save_all
- **依赖**: `config_kb`, `loader`, `pdf_loader`, `splitter`, `vector_store`

---

### loader.py
- **所属层次**: 知识库构建层
- **核心职责**: 加载 UTF-8 Markdown 文档为 LangChain Document 列表
- **对外暴露**: `load_doc(path) -> list[Document]`
- **依赖**: `langchain_community`

---

### pdf_loader.py
- **所属层次**: 知识库构建层
- **核心职责**: PyMuPDF 块级 PDF 解析，保留页码溯源，支持相邻块合并
- **对外暴露**:
  - `load_pdf_blocks(path) -> list[dict]` — 块级提取
  - `merge_pdf_blocks(records, *, max_chars) -> list[dict]` — 相邻小块合并
  - `load_pdf(path) -> str` — 兼容旧接口（全本拼接）
- **依赖**: `fitz`（PyMuPDF）

---

### splitter.py
- **所属层次**: 知识库构建层
- **核心职责**: LangChain RecursiveCharacterTextSplitter 文本切分，支持 Document 和纯文本两种输入
- **对外暴露**:
  - `split_docs(docs, chunk_size, chunk_overlap) -> list[Document]`
  - `split_text(text, chunk_size, chunk_overlap) -> list[str]`
  - `split_records(records, chunk_size, chunk_overlap) -> list[dict]`
- **依赖**: `langchain_core`, `langchain_text_splitters`

---

### config_kb.py
- **所属层次**: 工具/配置层
- **核心职责**: 知识库工程分类常量定义
- **对外暴露**:
  - `KB_MATERIAL`, `KB_STRUCTURE`, `KB_FAILURE`, `KB_ANSYS` — 库 ID 常量
  - `ALL_KB_IDS` — 四库 ID 元组
  - `MD_SOURCES` — Markdown 文件→库映射
  - `PDF_TO_KB` — PDF 默认归属库
  - `FAISS_PERSIST_DIR` — 索引持久化路径
- **依赖**: 标准库

---

### simulation_mode.py
- **所属层次**: 能力增强层
- **核心职责**: 正则匹配识别 CAE/FEA 仿真问题关键词
- **对外暴露**: `detect_simulation_mode(query) -> (bool, str)` — 返回是否命中 + 命中关键词
- **关键词数量**: 19 组正则模式（覆盖 ANSYS/Abaqus/COMSOL/Fluent 及通用 FEA 术语）
- **依赖**: 标准库（re）

---

### health_check.py
- **所属层次**: 工具层
- **核心职责**: 启动时状态自检（FAISS 索引文件、DeepSeek API Key、Embedding 模型）
- **对外暴露**:
  - `run_health_check() -> HealthReport` — 执行全项检查
  - `format_health_report(report) -> str` — 格式化输出
  - `CheckItem`, `HealthReport` 数据类
- **依赖**: `config_kb`, `faiss`, `embedding_model`

---

### logger.py
- **所属层次**: 工具层
- **核心职责**: 对话日志持久化、Markdown 报告导出、错误日志记录
- **对外暴露**:
  - `log_interaction(question, answer, route, elapsed, *, calc_result, recommendations)` — 追加问答日志
  - `load_session_history() -> list[dict]` — 读取全部历史
  - `export_session_report(session_messages) -> str` — 导出 Markdown 报告
  - `log_error(error_type, message, context)` — 记录错误日志
- **存储**: `logs/chat_history.jsonl`（JSONL 格式）、`logs/error.log`

---

## 五、知识库结构

| 库ID | 中文名 | 领域覆盖 | 索引文件 | 路由关键词（部分） |
|------|--------|----------|----------|-------------------|
| `material_db` | 材料库 | 材料性能、热处理、选材、力学性能指标 | `faiss_db/material_db.index` + `.meta.json` | 材料、钢材、合金、热处理、硬度、强度、韧性、弹性模量、屈服、抗拉 |
| `structure_db` | 结构库 | 结构方案、连接、传动、刚度强度设计、设计手册条文 | `faiss_db/structure_db.index` + `.meta.json` | 结构、轴承、齿轮、轴、螺栓、焊接、键连接、刚度、挠度、模态 |
| `failure_db` | 失效库 | 失效模式、疲劳、断裂、磨损、腐蚀、可靠性 | `faiss_db/failure_db.index` + `.meta.json` | 失效、疲劳、断裂、裂纹、磨损、腐蚀、屈曲、蠕变、可靠性 |
| `ansys_db` | 仿真库 | ANSYS/Abaqus/COMSOL、FEA 全流程案例 | `faiss_db/ansys_db.index` + `.meta.json` | ansys、abaqus、有限元、fea、仿真、网格、边界条件、后处理、求解器 |

**路由优先级**: 参数计算 > 结构推荐 > 仿真分析 > 四库规则路由 > `structure_db`（默认回退）

---

## 六、外部依赖

### 云服务

| 服务 | 用途 | 配置方式 |
|------|------|----------|
| DeepSeek API | 大模型问答生成 | `.env` 中 `DEEPSEEK_API_KEY` |
| HuggingFace Hub | 下载 Embedding 模型权重 | 自动缓存到 `~/.cache/huggingface/` |

### 关键第三方库

| 库 | 用途 |
|----|------|
| `streamlit` | Web 前端框架 |
| `sentence-transformers` | 句向量模型（`paraphrase-multilingual-MiniLM-L12-v2`，384 维） |
| `faiss-cpu` | Facebook 向量相似度搜索引擎（IndexFlatL2） |
| `numpy` | 向量运算 |
| `requests` | HTTP 请求（DeepSeek API） |
| `python-dotenv` | 环境变量加载 |
| `PyMuPDF` (`fitz`) | PDF 文档解析 |
| `langchain-core` / `langchain-text-splitters` / `langchain-community` | 文档加载与文本切分 |
| `plotly` | 管理面板图表（饼图） |
| `pandas` | 数据处理（plotly 依赖） |

### 本地存储

| 目录 | 内容 |
|------|------|
| `faiss_db/` | FAISS 向量索引（.index）和元数据（.meta.json），每库各一对 |
| `logs/` | 对话历史（chat_history.jsonl）和错误日志（error.log） |
| `db/` | Chroma SQLite 数据库（Embedding 模型缓存） |
| `knowledge_base/` | 原始 Markdown 和 PDF 知识文档 |

---

## 七、当前已实现的功能清单

### 核心 RAG
- [x] 四库独立 FAISS 语义检索（material / structure / failure / ansys）
- [x] 规则版 + 可选 LLM 版混合路由
- [x] 中英双语 Embedding 模型
- [x] Markdown + PDF 文档构建管线
- [x] 检索结果溯源展示（source + page + kb + L2 距离）

### Prompt 工程
- [x] 通用模式：四段式（结论 / 分析 / 建议 / 风险）
- [x] 仿真模式：六段式 CAE 推理（建模假设 / 载荷边界 / FEM 建议 / 应力分析 / 失效判断 / 优化 + 进阶说明）
- [x] 计算模式：五段式（结果摘要 / 工程解读 / 规范对比 / 安全性评估 / 优化建议）
- [x] 推荐模式：六段式（需求理解 / 对比表 / 首选方案 / 参数建议 / 待确认 / 标准参考）

### 机械参数计算
- [x] 轴扭转切应力（τ = 16T/πd³）
- [x] 轴弯曲正应力（σ = 32M/πd³）
- [x] 安全系数校核（n = [σ]/σ, n≥1.5 合格判定）
- [x] 螺栓预紧力（F = k·σs·As, ISO 898-1 应力面积）
- [x] 齿轮模数估算（赫兹接触强度简化迭代公式）
- [x] 滚动轴承 L10 寿命（球轴承/滚子轴承双指数）
- [x] 简支梁挠度（f = FL³/48EI, 挠跨比）
- [x] 关键词自动分派器

### 结构推荐
- [x] 5 大场景覆盖：轴系设计 / 齿轮传动 / 联接方式 / 轴承选型 / 密封方式
- [x] 18 种方案：含优缺点、典型参数范围、相关标准（GB/ISO）
- [x] 场景鉴别关键词精准匹配（100% 准确率）
- [x] 结合计算结果载荷水平排序

### UI / UX
- [x] ChatGPT 风格深色模式 UI
- [x] 消息气泡（用户右对齐 / AI 左对齐，计算卡片 / 推荐折叠）
- [x] 流式打字机输出动画
- [x] 快捷问题芯片（空对话时显示 4 个）
- [x] AI 回复复制按钮（JavaScript 剪贴板）
- [x] 侧边栏：知识库状态、对话统计、历史对话、路由显示
- [x] 引导性问题模板（5 个设计场景 + 文件上传）
- [x] 对话报告 Markdown 导出

### 系统运维
- [x] 启动时系统自检（FAISS 索引 / API Key / Embedding 模型）
- [x] 对话日志 JSONL 持久化
- [x] 错误日志记录（含类型标签和上下文）
- [x] DeepSeek API 超时/限流/网络异常中文友好提示
- [x] 代理配置支持（HTTPS_PROXY / gRPC 代理）
- [x] 游客模式（无需登录直接使用）

---

## 八、尚未实现 / 可扩展的方向

### 检索增强
- [ ] 多路召回融合（BM25 关键词 + FAISS 语义混合检索）
- [ ] 检索结果重排序（Cross-Encoder Reranker）
- [ ] 对话历史作为检索上下文（多轮对话记忆）
- [ ] 跨库检索权重动态调整

### 计算模块
- [ ] 自然语言参数提取（从用户输入中自动识别数值参数，无需手动传 dict）
- [ ] 更多计算类型（键强度校核、弹簧设计、带传动、链传动）
- [ ] 计算历史追溯与结果对比
- [ ] 单位自动换算

### 推荐模块
- [ ] 用户反馈收集（👍/👎 按钮数据写入日志）
- [ ] 基于反馈的方案排序优化
- [ ] 更多场景（焊接接头、弹簧选型、联轴器选型）
- [ ] 方案可视化（结构示意图生成）

### UI / 交互
- [ ] 真实流式输出（DeepSeek API streaming + SSE）
- [ ] 对话历史云端同步
- [ ] 多轮对话上下文管理
- [ ] 深色/浅色主题切换
- [ ] 移动端 PWA 支持
- [ ] 语音输入

### 知识库
- [ ] 知识库自动更新（监听文件变化）
- [ ] 版本管理（索引回滚）
- [ ] 知识片段质量评估与去重
- [ ] 多语言文档支持（英文/日文机械标准）

### 部署
- [ ] Docker 容器化
- [ ] Streamlit Cloud / HuggingFace Spaces 一键部署
- [ ] API 服务化（FastAPI 包装）
- [ ] 用户认证系统（可选恢复，当前已移除以减少复杂度）
