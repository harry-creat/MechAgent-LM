# 机械工程 AI 助手 — 项目框架

> 基于 RAG 架构，集成参数计算、结构推荐、仿真分析、混合检索。

---

## 目录结构

```
ai-mechanical-assistant/
├── app.py                     # Streamlit 主界面（深色UI、消息气泡、反馈按钮、侧边栏控制台）
├── ai_engine.py               # 编排中枢：多模式分支（问答/计算/推荐）、多轮对话历史、流式输出
├── router.py                  # 问题路由：规则版四库路由 + 计算/推荐/仿真意图识别
├── retriever.py               # 检索层：FAISS+BM25 混合检索（可切换），HybridHit 数据类
├── prompts.py                 # 4 套 Prompt 模板（通用四段式/仿真六段式/计算五段式/推荐六段式）
├── deepseek_api.py            # DeepSeek API 封装：非流式 chat_messages + 流式 stream_chat_messages
├── calculator.py              # 7 种机械参数计算函数 + 关键词意图分派器
├── recommender.py             # 5 场景 18 方案结构推荐 + 载荷水平排序
├── param_extractor.py         # 自然语言参数自动提取（正则+单位换算+计算类型判断）
│
├── vector_store.py            # 四库 FAISS+BM25 双索引统一门面
├── faiss_store.py             # 单库 FAISS IndexFlatL2 封装（增删查改 + 持久化）
├── bm25_store.py              # 单库 BM25 关键词索引 + RRF 倒数排名融合 + jieba 中文分词
├── embedding_model.py         # sentence-transformers 句向量模型（384 维，默认 multilingual）
│
├── build_kb.py                # FAISS+BM25 双索引构建入口（Markdown + PDF）
├── loader.py                  # Markdown → LangChain Document
├── pdf_loader.py              # PyMuPDF 块级 PDF 解析 + 相邻块合并
├── splitter.py                # RecursiveCharacterTextSplitter 文本切分（LangChain）
│
├── config_kb.py               # 四库 ID / 路径 / MD→KB 映射常量
├── simulation_mode.py         # 19 组 CAE/FEA 正则匹配 → 仿真模式识别
├── health_check.py            # 启动自检：FAISS 索引 / API Key / Embedding 模型
├── logger.py                  # 对话日志（JSONL）、反馈日志、Markdown 报告导出、错误日志
├── search_compare.py          # 三路检索对比工具（FAISS vs BM25 vs Hybrid）
│
├── eval/
│   ├── questions.json         # 50 道评测题（四库覆盖，含 ground_truth）
│   ├── ragas_eval.py          # 评估引擎：LLM 评分（忠实度/相关性/召回率）+ 报告生成
│   └── feedback_to_eval.py    # 用户 👍 反馈 → 评测题转换
│
├── requirements.txt           # Python 依赖
├── .env / .env.example        # 环境变量配置
├── .gitignore
├── README.md                  # 项目说明
├── PROJECT_STRUCTURE.md       # 本文件
├── faiss_db/                  # FAISS 索引 + BM25 索引（.index + .meta.json + bm25/*.pkl）
├── logs/                      # 对话日志 + 反馈日志 + 错误日志
├── knowledge_base/            # 原始知识文档（.md + .pdf）
└── db/                        # Chroma SQLite 缓存
```

---

## 架构分层

```
UI 层        → app.py                  : Streamlit 前端，消息渲染、反馈按钮、侧边栏
编排层       → ai_engine.py            : 多模式分支 + 多轮历史 + 流式输出
             → router.py               : 意图识别 + 知识库路由（优先级：计算 > 推荐 > 仿真 > 四库）
             → retriever.py            : FAISS+BM25 混合检索桥接
Prompt 层    → prompts.py              : 四套专家级 Prompt 模板
增强层       → calculator.py           : 参数计算（7 种公式）
             → recommender.py          : 结构推荐（5 场景）
             → param_extractor.py      : 自然语言参数提取
             → simulation_mode.py      : 仿真关键词检测
检索存储层   → vector_store.py         : 四库 FAISS+BM25 双索引门面
             → faiss_store.py          : FAISS 向量索引
             → bm25_store.py           : BM25 关键词索引 + RRF 融合
             → embedding_model.py      : 句向量模型
构建层       → build_kb.py             : 索引构建主入口
             → loader.py / pdf_loader.py / splitter.py : 文档加载与切分
工具层       → deepseek_api.py         : LLM API 调用
             → config_kb.py            : 知识库配置
             → health_check.py         : 系统自检
             → logger.py               : 日志 / 反馈 / 报告
             → search_compare.py       : 检索对比
评估层       → eval/ragas_eval.py      : RAGAS 风格评估
             → eval/questions.json     : 50 道标注评测题
```

---

## 核心数据流

```
用户输入 → router (意图→kb_id) → retriever (FAISS+BM25→RRF融合)
→ prompts (构建专家Prompt) → deepseek_api (LLM生成/流式)
→ app.py (消息气泡 + 反馈按钮) → logger (日志/反馈持久化)
```

若命中计算意图，中间插入 param_extractor(提取参数) → calculator(执行公式) → 结果注入 Prompt。
若命中推荐意图，中间插入 recommender(方案匹配) → 结果注入 Prompt。

---

## 已实现功能

### 核心 RAG
- ✅ 四库独立 FAISS 语义检索 + BM25 关键词检索 + RRF 混合检索
- ✅ 规则版路由（计算 > 推荐 > 仿真 > 四库规则） + 可选 LLM 路由
- ✅ 中英双语 Embedding（paraphrase-multilingual-MiniLM-L12-v2, 384 维）
- ✅ Markdown + PDF 双格式知识库构建管线
- ✅ 多轮对话上下文管理（系统 Prompt + 最近 N 轮历史注入，侧边栏滑块控制）
- ✅ DeepSeek API 真实流式输出（SSE 逐 token 渲染）
- ✅ 检索结果溯源展示（source, page, kb, L2 距离）

### 机械领域增强
- ✅ 7 种参数自动计算（轴扭转/弯曲、安全系数、螺栓预紧力、齿轮模数、轴承寿命、梁挠度）
- ✅ 自然语言参数自动提取（19 组正则 + 单位换算 + 计算类型自动判断 + 参数不足友好引导）
- ✅ 5 场景 18 方案结构推荐（轴系/齿轮/联接/轴承/密封，含优缺点/典型参数/GB 标准）
- ✅ 仿真分析专用六段式 CAE Prompt（含进阶说明）
- ✅ 标准号/材料牌号在 BM25 分词中保护（GB/T, ISO, ASTM, 45 钢, 42CrMo 等不被拆分）

### UI / UX
- ✅ ChatGPT 风格深色 UI + 消息气泡 + 流式动画
- ✅ 计算模式：st.metric 数值卡片展示关键结果
- ✅ 推荐模式：st.expander 折叠方案详情（优缺点对比）
- ✅ 👍/👎 反馈按钮 + 去重逻辑 + 侧边栏满意度统计
- ✅ 快捷提问标签 + 引导性设计场景模板 + .txt 文件上传
- ✅ 侧边栏：知识库状态、对话统计、上下文记忆滑块、反馈统计、报告导出
- ✅ 系统启动自检（FAISS/API/Embedding 状态指示灯）

### 评估与运维
- ✅ 50 道标注评测题（四库覆盖，含 ground_truth 和难度分级）
- ✅ RAGAS 风格自动评估（忠实度/相关性/召回率三项 LLM 评分）
- ✅ 纯 FAISS vs 混合检索对比报告（柱状图 + Markdown + Excel）
- ✅ 对话日志 JSONL 持久化 + Markdown 报告导出
- ✅ 反馈日志 → 评测题转换
- ✅ DeepSeek API 超时/限流/网络异常中文友好提示

### 未实现
- ❌ 用户注册/登录系统（已移除 Firebase 依赖，当前为游客模式）
- ❌ 对话历史云端同步（仅本地 JSONL 存储）
- ❌ 多路召回 BM25 权重动态调整
- ❌ Cross-Encoder Reranker 重排序
- ❌ 知识库自动更新（文件变化监听）
- ❌ Docker 容器化部署

---

## 外部依赖

| 依赖 | 用途 |
|------|------|
| DeepSeek API | 大模型问答生成 + 评估评分 |
| sentence-transformers | 句向量模型（paraphrase-multilingual-MiniLM-L12-v2） |
| FAISS | 向量相似度搜索（IndexFlatL2） |
| rank-bm25 + jieba | BM25 关键词检索 + 中文分词 |
| PyMuPDF (fitz) | PDF 文档文本提取 |
| Streamlit | Web 前端框架 |
| LangChain (core/text-splitters/community) | 文档加载与递归切分 |
