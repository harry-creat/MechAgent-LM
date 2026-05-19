# 机械工程 AI 助手

基于 RAG（检索增强生成）架构的机械工程智能问答系统，集成 **参数计算**、**结构推荐** 和 **仿真分析** 三大功能模块。

## 功能概览

- **智能问答** — 基于四个专业机械知识库的语义检索 + DeepSeek 大模型生成
- **参数计算** — 7 种常用机械设计参数自动计算（轴应力、安全系数、螺栓预紧力、齿轮模数、轴承寿命、梁挠度）
- **结构推荐** — 覆盖 5 大设计场景（轴系/齿轮/联接/轴承/密封），18 种方案的智能推荐
- **仿真分析** — CAE/FEA 仿真问题专用六段式工程 Prompt，支持 ANSYS/Abaqus/COMSOL 语境

## 环境要求

- Python 3.10+
- pip 包管理器
- DeepSeek API Key（[获取地址](https://platform.deepseek.com/)）

## 快速开始

### 1. 克隆项目

```bash
git clone <repo-url>
cd ai-mechanical-assistant
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY="your-api-key-here"

# Linux / macOS
export DEEPSEEK_API_KEY="your-api-key-here"
```

### 4. 构建知识库

将机械工程文档（Markdown、PDF）放入对应目录后：

```bash
python build_kb.py
```

系统会自动将文档分割、向量化并存入四个独立的 FAISS 索引。

### 5. 启动应用

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可使用。

## 知识库说明

系统维护 **四个独立知识库**，自动根据问题内容路由：

| 知识库 ID | 领域 | 推荐文档来源 |
|-----------|------|-------------|
| `material_db` | 材料性能、热处理、选材 | 《机械工程材料》《热处理手册》 |
| `structure_db` | 结构设计、传动、连接、设计手册 | 《机械设计手册》(成大先)、《机械设计》(濮良贵) |
| `failure_db` | 失效模式、疲劳、断裂、可靠性 | 《疲劳强度》《断裂力学》《FMEA 手册》 |
| `ansys_db` | 有限元 / CAE 仿真、案例参数 | ANSYS Help、Abaqus 文档、仿真案例集 |

知识库文档配置在 `config_kb.py` 的 `MD_SOURCES` 中，可按需增删。

## 项目结构

```
ai-mechanical-assistant/
├── app.py                 # Streamlit 前端界面
├── ai_engine.py           # 核心编排逻辑（路由 → 检索 → Prompt → LLM）
├── router.py              # 问题路由（calc / recommend / simulation / 四库）
├── prompts.py             # Prompt 模板（通用/仿真/计算/推荐 四种）
├── calculator.py          # 机械参数计算模块（7 种计算函数）
├── recommender.py         # 智能结构推荐模块（5 场景 18 方案）
├── deepseek_api.py        # DeepSeek API 调用封装
├── retriever.py           # 检索层（路由 → FAISS 语义检索）
├── vector_store.py        # 多知识库向量存储封装
├── faiss_store.py         # 单库 FAISS 索引封装
├── embedding_model.py     # sentence-transformers 句向量模型
├── health_check.py        # 系统自检（启动时检查索引/API/模型状态）
├── logger.py              # 对话日志与报告导出
├── config_kb.py           # 知识库工程分类配置
├── simulation_mode.py     # 仿真分析模式识别
├── loader.py              # Markdown 文档加载
├── pdf_loader.py          # PDF 文档加载（PyMuPDF）
├── splitter.py            # 文本分割（LangChain）
├── build_kb.py            # 知识库构建入口
├── requirements.txt       # Python 依赖列表
├── faiss_db/              # FAISS 索引持久化目录
├── logs/                  # 对话日志与错误日志
└── knowledge_base/        # 原始知识文档目录
    ├── materials.md
    ├── structures.md
    ├── failure_modes.md
    └── ansys_cases.md
```

## 使用指南

### 问答模式

直接输入机械工程问题，系统自动路由到相关知识库并生成回答：

```
"45钢的屈服强度是多少？"
"齿轮齿面点蚀的主要原因是什么？"
```

### 计算模式

输入包含计算关键词的问题，系统自动识别并调用对应计算函数：

```
"计算直径40mm轴在500Nm扭矩下的扭转切应力"
"校核这个零件安全系数，许用应力250MPa，实际应力120MPa"
```

支持的 7 种计算类型：轴扭转/弯曲、安全系数校核、螺栓预紧力、齿轮模数估算、轴承寿命、简支梁挠度。

### 推荐模式

输入选型/方案类问题，系统自动匹配场景并推荐结构方案：

```
"推荐一种适合高速主轴的轴承方案"
"重载轴毂联接怎么选？"
```

### 仿真模式

输入 CAE/FEA 相关问题，自动进入六段式仿真工程分析 Prompt：

```
"ANSYS 静力学分析中如何施加轴承约束？"
"齿轮接触分析中网格如何划分？"
```

### 快捷设计场景

侧边栏提供 5 个快捷按钮，点击自动填入引导性问题模板，补充参数后发送。

### 对话报告导出

侧边栏「导出本次报告」按钮可将当前会话导出为 Markdown 格式报告文件。

## 常见问题 FAQ

### Q1: 启动时显示"未设置环境变量 DEEPSEEK_API_KEY"？
需要在终端中设置 `DEEPSEEK_API_KEY` 环境变量，或创建 `.env` 文件并通过 `python-dotenv` 加载。

### Q2: 知识库显示"空库"怎么办？
运行 `python build_kb.py` 构建知识库索引。确保 `knowledge_base/` 目录下有对应的 Markdown 文件，或修改 `config_kb.py` 中的 `MD_SOURCES` 配置。

### Q3: 参数计算提示"缺少参数"？
计算功能需要提供具体的数值参数。可通过聊天界面直接描述参数（如 "扭矩500Nm，直径40mm"），系统会尝试识别。目前参数提取依赖 `calc_params` 字典传参，后续版本将支持自然语言参数提取。

### Q4: 如何切换 Embedding 模型？
设置环境变量 `ST_MODEL_NAME` 即可切换 sentence-transformers 模型。默认使用 `paraphrase-multilingual-MiniLM-L12-v2`（中英双语轻量模型）。中文场景可切换为 `shibing624/text2vec-base-chinese` 等中文专用模型。

### Q5: FAISS 索引文件在哪里？
索引文件存储在 `faiss_db/` 目录下，每个知识库对应 `.index`（向量）和 `.meta.json`（元数据）两个文件。删除后重新运行 `build_kb.py` 即可重建。

### Q6: API 调用失败或超时怎么办？
系统已内置错误处理：超时/限流/网络异常会返回中文友好提示并记录到 `logs/error.log`。如频繁遇到 429 限流，请检查 API 配额或降低请求频率。

### Q7: 如何添加新的计算函数？
在 `calculator.py` 中按现有函数格式添加新函数，然后在 `dispatch_calculation()` 的 `routes` 列表中增加对应的关键词映射即可。

### Q8: 如何添加新的推荐方案？
在 `recommender.py` 的 `_KNOWLEDGE_BASE` 字典中对应场景的 `schemes` 列表里添加新方案，按现有格式填写名称、描述、优缺点、典型参数、相关标准即可。
