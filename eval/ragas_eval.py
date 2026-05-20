"""
RAG 系统评估脚本 — 对比纯 FAISS vs 混合检索（BM25+FAISS）。

评估三项指标：答案忠实度、答案相关性、检索召回率。
使用 DeepSeek API 作为评估器 LLM。
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


def load_questions(path: str = "eval/questions.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── LLM 评估器 ──
def _eval_llm(system_prompt: str, user_prompt: str) -> str:
    """调用 DeepSeek API 进行评分。"""
    import requests

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY")

    resp = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        },
        timeout=60,
    )
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return "ERROR"


def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """评估答案忠实度：答案中的信息是否都能从检索上下文中找到依据。"""
    ctx_text = "\n---\n".join(c[:800] for c in contexts[:3])
    prompt = f"""请评估以下AI回答是否忠实于给定的检索上下文。

检索上下文：
{ctx_text}

AI回答：
{answer[:1500]}

评分标准（0-10分）：
- 9-10分：回答中的所有事实性陈述都能在上下文中找到明确依据
- 7-8分：绝大部分陈述有依据，仅个别细节推断超出上下文
- 5-6分：部分陈述有依据，也有明显超出上下文的推断
- 3-4分：仅少量陈述有依据，大量内容凭空编造
- 1-2分：几乎完全脱离上下文

请只输出一个数字（0-10），不要任何解释。"""
    result = _eval_llm("你是一个严格的评估专家。只输出数字评分。", prompt)
    try:
        return float(result.strip().split()[0]) / 10.0
    except ValueError:
        return 0.5


def score_relevancy(question: str, answer: str) -> float:
    """评估答案相关性：回答是否直接回应了用户的问题。"""
    prompt = f"""请评估以下AI回答与用户问题的相关程度。

用户问题：
{question}

AI回答：
{answer[:1500]}

评分标准（0-10分）：
- 9-10分：直接、完整地回答了问题的所有要点
- 7-8分：回答了主要问题，但个别次要要点不够完整
- 5-6分：部分回答了问题，但遗漏了重要方面或包含无关内容
- 3-4分：只有少量内容与问题相关，大量偏离主题
- 1-2分：几乎没有回答用户实际问的问题

请只输出一个数字（0-10），不要任何解释。"""
    result = _eval_llm("你是一个严格的评估专家。只输出数字评分。", prompt)
    try:
        return float(result.strip().split()[0]) / 10.0
    except ValueError:
        return 0.5


def score_recall(ground_truth: str, contexts: list[str]) -> float:
    """评估检索召回率：检索到的上下文中是否包含标准答案中的关键信息。"""
    ctx_text = "\n---\n".join(c[:800] for c in contexts[:3])
    prompt = f"""请评估以下检索到的上下文片段是否包含标准答案中的关键信息。

标准答案（关键信息点）：
{ground_truth[:600]}

检索到的上下文片段：
{ctx_text}

评分标准（0-10分）：
- 9-10分：上下文中包含了标准答案的几乎所有关键信息点
- 7-8分：包含了大部分关键信息，个别要点缺失
- 5-6分：包含了约一半关键信息，重要部分缺失
- 3-4分：仅包含少量关键信息，大部分缺失
- 1-2分：几乎没有包含标准答案中的信息

请只输出一个数字（0-10），不要任何解释。"""
    result = _eval_llm("你是一个严格的评估专家。只输出数字评分。", prompt)
    try:
        return float(result.strip().split()[0]) / 10.0
    except ValueError:
        return 0.5


# ── RAG Pipeline ──
def run_rag_pipeline(question: str, use_hybrid: bool = True) -> dict:
    from retriever import retrieve

    result = retrieve(question, top_k=5, use_hybrid=use_hybrid)
    hits = result["hits"] + result.get("aux_hits", [])
    contexts = []
    for h in hits:
        if hasattr(h, "text"):
            contexts.append(h.text)
        elif isinstance(h, dict):
            contexts.append(h.get("text", ""))

    # 用 DeepSeek 生成回答
    from deepseek_api import call_deepseek
    from prompts import build_mechanical_rag_prompt

    prompt = build_mechanical_rag_prompt(question, result["hits"], result["decision"])
    answer = call_deepseek(prompt)

    return {"question": question, "answer": answer, "contexts": contexts}


# ── 数据集构建 ──
def build_eval_dataset(questions: list[dict], use_hybrid: bool = True) -> list[dict]:
    data = []
    desc = "混合检索" if use_hybrid else "纯FAISS"
    for i, q in enumerate(questions, 1):
        print(f"  [{desc}] {i}/{len(questions)}: {q['id']} {q['question'][:40]}...")
        try:
            result = run_rag_pipeline(q["question"], use_hybrid=use_hybrid)
            result["ground_truth"] = q["ground_truth"]
            result["id"] = q["id"]
            result["domain"] = q.get("domain", "")
            data.append(result)
        except Exception as e:
            print(f"    [SKIP] {e}")
        time.sleep(0.3)  # 避免 API 限流
    return data


# ── 评估 ──
def evaluate_dataset(dataset: list[dict], mode_name: str) -> dict:
    """对数据集逐题评分，汇总平均。"""
    scores = {"faithfulness": [], "answer_relevancy": [], "context_recall": [], "per_question": []}
    for i, item in enumerate(dataset, 1):
        print(f"  评分 {i}/{len(dataset)}: {item.get('id', '?')}")
        f = score_faithfulness(item["answer"], item["contexts"])
        r = score_relevancy(item["question"], item["answer"])
        c = score_recall(item["ground_truth"], item["contexts"])
        scores["faithfulness"].append(f)
        scores["answer_relevancy"].append(r)
        scores["context_recall"].append(c)
        scores["per_question"].append({"id": item.get("id"), "f": f, "r": r, "c": round((f + r + c) / 3, 4)})
        time.sleep(0.3)

    return {
        "mode": mode_name,
        "faithfulness": round(sum(scores["faithfulness"]) / len(scores["faithfulness"]), 4),
        "answer_relevancy": round(sum(scores["answer_relevancy"]) / len(scores["answer_relevancy"]), 4),
        "context_recall": round(sum(scores["context_recall"]) / len(scores["context_recall"]), 4),
        "average": round(
            (sum(scores["faithfulness"]) + sum(scores["answer_relevancy"]) + sum(scores["context_recall"]))
            / (3 * len(scores["faithfulness"])),
            4,
        ),
        "per_question": scores["per_question"],
    }


# ── 报告生成 ──
def generate_report(faiss_scores: dict, hybrid_scores: dict, questions: list[dict], output_dir: str = "eval/results"):
    os.makedirs(output_dir, exist_ok=True)

    all_scores = {"faiss": {k: v for k, v in faiss_scores.items() if k != "per_question"},
                   "hybrid": {k: v for k, v in hybrid_scores.items() if k != "per_question"}}
    with open(f"{output_dir}/scores.json", "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)

    # detail.xlsx
    try:
        import pandas as pd
        rows = []
        for item in faiss_scores.get("per_question", []):
            hybrid_item = next((x for x in hybrid_scores.get("per_question", []) if x["id"] == item["id"]), None)
            rows.append({
                "题目ID": item["id"],
                "纯FAISS_忠实度": item["f"], "纯FAISS_相关性": item["r"], "纯FAISS_召回率": item["c"],
                "混合_忠实度": hybrid_item["f"] if hybrid_item else "",
                "混合_相关性": hybrid_item["r"] if hybrid_item else "",
                "混合_召回率": hybrid_item["c"] if hybrid_item else "",
            })
        df = pd.DataFrame(rows)
        df.to_excel(f"{output_dir}/detail.xlsx", index=False)
        print(f"  detail.xlsx 已生成")
    except Exception as e:
        print(f"  detail.xlsx 跳过: {e}")

    metric_names = {"faithfulness": "答案忠实度", "answer_relevancy": "答案相关性", "context_recall": "检索召回率", "average": "综合平均"}
    with open(f"{output_dir}/report.md", "w", encoding="utf-8") as f:
        f.write("# 机械 RAG 系统检索评估报告\n\n")
        f.write(f"评测题目数：{len(questions)} 道\n\n")
        f.write("## 指标对比表\n\n")
        f.write("| 指标 | 纯 FAISS | 混合检索 | 提升幅度 |\n")
        f.write("|------|---------|---------|--------|\n")
        for m in ["faithfulness", "answer_relevancy", "context_recall", "average"]:
            fs, hs = faiss_scores[m], hybrid_scores[m]
            delta = f"+{(hs-fs)*100:.1f}%" if hs >= fs else f"{(hs-fs)*100:.1f}%"
            f.write(f"| {metric_names[m]} | {fs:.4f} | {hs:.4f} | {delta} |\n")
        f.write("\n## 结论\n\n")
        avg_delta = (hybrid_scores["average"] - faiss_scores["average"]) * 100
        f.write(f"混合检索相比纯 FAISS，综合指标提升 **{avg_delta:.1f}%**。\n")

    # 柱状图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_display = ["faithfulness", "answer_relevancy", "context_recall"]
        faiss_vals = [faiss_scores[m] for m in metrics_display]
        hybrid_vals = [hybrid_scores[m] for m in metrics_display]
        x = range(len(metrics_display))
        width = 0.35
        b1 = ax.bar([i - width / 2 for i in x], faiss_vals, width, label="纯 FAISS", color="#4C72B0", alpha=0.85)
        b2 = ax.bar([i + width / 2 for i in x], hybrid_vals, width, label="混合检索 (BM25+FAISS)", color="#DD8452", alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels([metric_names[m] for m in metrics_display], fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("得分", fontsize=12)
        ax.set_title("检索策略对比评估", fontsize=14)
        ax.legend(fontsize=11)
        for bar in b1 + b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison.png", dpi=150)
        plt.close()
        print(f"  comparison.png 已生成")
    except Exception as e:
        print(f"  comparison.png 跳过: {e}")

    print(f"✅ 报告已生成到 {output_dir}/")


# ── main ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["faiss", "hybrid", "both"], default="both")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    questions = load_questions()[:args.limit]
    print(f"载入 {len(questions)} 道评测题\n")

    faiss_scores, hybrid_scores = None, None

    if args.mode in ("faiss", "both"):
        print("▶ 纯 FAISS 评估...")
        ds = build_eval_dataset(questions, use_hybrid=False)
        faiss_scores = evaluate_dataset(ds, "纯FAISS")
        print(f"  结果: F={faiss_scores['faithfulness']:.4f} R={faiss_scores['answer_relevancy']:.4f} RC={faiss_scores['context_recall']:.4f} AVG={faiss_scores['average']:.4f}\n")

    if args.mode in ("hybrid", "both"):
        print("▶ 混合检索评估...")
        ds = build_eval_dataset(questions, use_hybrid=True)
        hybrid_scores = evaluate_dataset(ds, "混合检索")
        print(f"  结果: F={hybrid_scores['faithfulness']:.4f} R={hybrid_scores['answer_relevancy']:.4f} RC={hybrid_scores['context_recall']:.4f} AVG={hybrid_scores['average']:.4f}\n")

    if faiss_scores and hybrid_scores:
        generate_report(faiss_scores, hybrid_scores, questions)
