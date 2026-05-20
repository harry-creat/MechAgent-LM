"""
优化 questions.json 中所有 ground_truth 字段，
使其贴近知识库原文风格（含具体数值、专业术语、标准号）。
"""

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


def retrieve_contexts(question: str, domain: str, use_hybrid: bool = True) -> list[str]:
    """检索知识库，返回 top-5 文本片段。"""
    from retriever import retrieve
    try:
        result = retrieve(question, top_k=5, use_hybrid=use_hybrid)
        hits = result.get("hits", [])
        contexts = []
        for h in hits:
            if hasattr(h, "text"):
                contexts.append(h.text)
            elif isinstance(h, dict):
                contexts.append(h.get("text", ""))
        return contexts
    except Exception:
        return []


def optimize_ground_truth(question: str, original_gt: str, contexts: list[str], keywords: list[str]) -> str:
    """用 LLM 基于检索内容优化 ground_truth。"""
    import requests

    ctx_block = "\n---\n".join(c[:500] for c in contexts[:4]) if contexts else "(无检索结果，请基于机械工程专业知识编写)"
    kw_str = "、".join(keywords) if keywords else "无"

    system = (
        "你是一名机械工程教材编写专家。你的任务是将一段答案改写为贴近教材原文的风格。"
        "改写要求：1) 包含具体数值、标准号、专业术语；2) 150-300字；"
        "3) 使用检索片段的原文表述；4) 必须包含给定的关键词；5) 直接输出改写结果，不要解释。"
    )

    user = f"""请改写以下参考答案，使其贴近检索内容的表述风格。

【问题】
{question}

【当前参考答案】
{original_gt}

【检索到的知识库片段（供参考原文表述）】
{ctx_block}

【必须包含的关键词】
{kw_str}

请直接输出改写后的 ground_truth（150-300字，含具体数值和标准号）："""

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return original_gt

    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
                "max_tokens": 600,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return original_gt


def main():
    questions = load_questions()
    print(f"载入 {len(questions)} 道题目")
    print()

    optimized = []
    total_orig_len = 0
    total_new_len = 0
    retrieval_ok = 0
    retrieval_fail = 0

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        domain = q.get("domain", "structure_db")
        orig_gt = q.get("ground_truth", "")
        keywords = q.get("keywords", [])

        print(f"[{i}/50] {qid}: {q['question'][:50]}...", end=" ", flush=True)

        # 检索
        contexts = retrieve_contexts(q["question"], domain)
        if contexts and any(len(c) > 30 for c in contexts):
            retrieval_ok += 1
        else:
            retrieval_fail += 1
            print("(检索不足)", end=" ", flush=True)

        # 优化
        new_gt = optimize_ground_truth(q["question"], orig_gt, contexts, keywords)
        total_orig_len += len(orig_gt)
        total_new_len += len(new_gt)

        new_q = dict(q)
        new_q["ground_truth"] = new_gt
        new_q["_optimized"] = True
        if not contexts:
            new_q["_source"] = "manual"
        optimized.append(new_q)

        print(f"OK ({len(orig_gt)}→{len(new_gt)}字)", flush=True)
        time.sleep(0.3)  # 避免 API 限流

    # 保存
    out_path = "eval/questions_optimized.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(optimized, f, ensure_ascii=False, indent=2)

    # 摘要
    print()
    print("=" * 50)
    print(f"✅ 保存到 {out_path}")
    print(f"共优化: {len(optimized)} 题")
    print(f"平均长度: {total_orig_len//len(questions)} → {total_new_len//len(questions)} 字")
    print(f"检索成功: {retrieval_ok}/{len(questions)} 题")
    print(f"依赖补充: {retrieval_fail}/{len(questions)} 题")
    print("=" * 50)

    # 打印5个样例
    import random
    samples = random.sample(optimized, min(5, len(optimized)))
    for s in samples:
        orig = questions[[q["id"] for q in questions].index(s["id"])]["ground_truth"]
        ctx = retrieve_contexts(s["question"], s.get("domain", "structure_db"))
        print(f"\n题目ID: {s['id']}")
        print(f"问题: {s['question'][:80]}")
        print(f"优化前: {orig[:150]}...")
        print(f"优化后: {s['ground_truth'][:200]}...")
        print(f"检索支撑: {(ctx[0][:100] if ctx else '无')}...")
        print("─" * 40)


if __name__ == "__main__":
    main()
