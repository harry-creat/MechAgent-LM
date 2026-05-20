"""
将用户反馈（👍）转化为评测集补充题目。

读取 logs/feedback.jsonl 中 rating=1 的问答对，
用 AI 回答作为 ground_truth（用户验证过的高质量回答）。
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def convert_feedback_to_questions(
    feedback_path: str = "logs/feedback.jsonl",
    output_path: str = "eval/questions_from_feedback.json",
    min_rating: int = 1,
) -> int:
    """将用户标注为👍的问答对转为评测题格式。

    Args:
        feedback_path: 反馈日志路径
        output_path: 输出 JSON 路径
        min_rating: 最低评分阈值（1=只要👍）

    Returns:
        成功转化的题目数量
    """
    if not os.path.isfile(feedback_path):
        print(f"[WARN] 反馈文件不存在: {feedback_path}")
        return 0

    questions = []
    with open(feedback_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("rating", 0) < min_rating:
                continue
            route = rec.get("route", "unknown")
            # 将 route 映射到 domain
            domain_map = {
                "material_db": "material_db",
                "structure_db": "structure_db",
                "failure_db": "failure_db",
                "ansys_db": "ansys_db",
            }
            domain = "structure_db"
            for k, v in domain_map.items():
                if k in route:
                    domain = v
                    break

            questions.append({
                "id": f"fb_{rec.get('session_id', 'unknown')}_{len(questions)+1}",
                "domain": domain,
                "question": rec.get("question", ""),
                "ground_truth": rec.get("answer", "")[:300],
                "difficulty": "medium",
                "keywords": [],
                "source": "user_feedback",
            })

    if questions:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
        print(f"[OK] 从反馈中转化 {len(questions)} 道评测题 -> {output_path}")
    else:
        print("未找到符合条件的反馈记录")
    return len(questions)


if __name__ == "__main__":
    convert_feedback_to_questions()
