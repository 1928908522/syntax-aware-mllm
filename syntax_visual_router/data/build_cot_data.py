"""Build Stage E Syntax-Route CoT SFT data from Stage D filtered jsonl.

Outputs train/test splits and keeps route/syntax/NLI metrics for later losses.

Example:
  python -m syntax_visual_router.data.build_cot_data --sample_fraction 0.05
"""

import argparse
import json
import os
import random
import sys

from syntax_visual_router.data.cot_templates import build_chosen_response, build_prompt

DEFAULT_INPUT = os.path.join("data", "syntax_negatives.jsonl")
DEFAULT_OUT_DIR = "data"


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("image_path") and rec.get("positive") and rec.get("negative"):
                rows.append(rec)
    return rows


def norm_score(value):
    if value is None:
        return 0.5
    try:
        v = float(value)
    except Exception:
        return 0.5
    return max(0.0, min(1.0, v))


def syntax_quality(rec):
    nli = (rec.get("nli_label") or "").lower()
    sim = float(rec.get("embedding_similarity") or 0.0)
    jac = float(rec.get("jaccard") or 0.0)
    return 1.0 if nli == "contradiction" and sim >= 0.75 and jac >= 0.3 else 0.0


def first_triple(rec):
    triples = rec.get("target_triples") or rec.get("selected_triples") or []
    return triples[0] if triples else {}


def convert(rec, rng):
    positive_first = rng.random() < 0.5
    if positive_first:
        caption_a, caption_b, answer = rec["positive"], rec["negative"], "A"
    else:
        caption_a, caption_b, answer = rec["negative"], rec["positive"], "B"

    triple = first_triple(rec)
    prompt = build_prompt(caption_a, caption_b)
    response = build_chosen_response(triple, rec.get("candidate_type"), answer)
    route_score = norm_score(rec.get("candidate_score"))
    s_quality = syntax_quality(rec)
    reward = 1.0 + 0.5 + 0.5 * route_score + 0.25 * s_quality

    return {
        "image_path": rec["image_path"],
        "caption_a": caption_a,
        "caption_b": caption_b,
        "answer": answer,
        "positive": rec["positive"],
        "negative": rec["negative"],
        "candidate_type": rec.get("candidate_type", "GENERIC"),
        "selected_triple": triple,
        "target_triples": rec.get("target_triples", []),
        "prompt": prompt,
        "response": response,
        "route_score": route_score,
        "reward": reward,
        "metrics": {
            "candidate_score": rec.get("candidate_score"),
            "candidate_rank": rec.get("candidate_rank"),
            "ted": rec.get("ted"),
            "jaccard": rec.get("jaccard"),
            "embedding_similarity": rec.get("embedding_similarity"),
            "nli_label": rec.get("nli_label"),
            "syntax_quality": s_quality,
        },
    }


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default=DEFAULT_INPUT)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="stage_e_cot_sft_topk")
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--sample_fraction", type=float, default=1.0,
                        help="Use e.g. 0.05 to build a smoke subset.")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_rows(args.input_jsonl)
    rng.shuffle(rows)
    if args.limit > 0:
        rows = rows[:args.limit]
    if 0 < args.sample_fraction < 1.0:
        rows = rows[: max(1, int(len(rows) * args.sample_fraction))]

    converted = [convert(r, rng) for r in rows]
    split = int(len(converted) * args.train_ratio)
    train_rows = converted[:split]
    test_rows = converted[split:]

    suffix = "" if args.sample_fraction >= 1.0 and args.limit <= 0 else "_smoke"
    train_path = os.path.join(args.out_dir, f"{args.prefix}{suffix}_train.jsonl")
    test_path = os.path.join(args.out_dir, f"{args.prefix}{suffix}_test.jsonl")
    all_path = os.path.join(args.out_dir, f"{args.prefix}{suffix}.jsonl")
    write_jsonl(all_path, converted)
    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)

    print(f"读取: {args.input_jsonl}")
    print(f"输出总样本: {len(converted)}")
    print(f"训练集: {len(train_rows)} -> {train_path}")
    print(f"测试集: {len(test_rows)} -> {test_path}")
    print(f"全集: {all_path}")


if __name__ == "__main__":
    main()
