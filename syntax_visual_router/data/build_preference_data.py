"""Build Stage E preference data from CoT SFT data."""

import argparse
import json
import os
import random
import sys

from syntax_visual_router.data.cot_templates import (
    build_wrong_answer_response,
    build_wrong_relation_response,
    build_wrong_visual_response,
)

DEFAULT_INPUT = os.path.join("data", "stage_e_cot_sft_topk_train.jsonl")
DEFAULT_OUTPUT = os.path.join("data", "stage_e_preference_topk_train.jsonl")


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rewards(row, rejected_type):
    route = float(row.get("route_score", 0.5))
    syntax = float((row.get("metrics") or {}).get("syntax_quality", 0.0))
    chosen = 1.0 + 0.5 + 0.5 * route + 0.25 * syntax
    if rejected_type == "wrong_answer":
        rejected = 0.0 + 0.5 + 0.5 * route + 0.25 * syntax
    elif rejected_type == "wrong_key_relation":
        rejected = 0.0 + 0.0 + 0.2 * route + 0.25 * syntax
    else:
        rejected = 0.0 + 0.2 + 0.0 + 0.25 * syntax
    return chosen, rejected


def build_rows(row, rng):
    out = []
    ctype = row.get("candidate_type", "GENERIC")
    triple = row.get("selected_triple", {})
    answer = row["answer"]
    wrong_triples = [t for t in row.get("target_triples", []) if t != triple]
    wrong_triple = rng.choice(wrong_triples) if wrong_triples else {
        "src": "alternative", "rel": "relation", "dst": "caption"
    }
    rejected_map = {
        "wrong_answer": build_wrong_answer_response(triple, ctype, answer),
        "wrong_visual_support": build_wrong_visual_response(triple, ctype, answer),
        "wrong_key_relation": build_wrong_relation_response(wrong_triple, ctype, answer),
    }
    for rtype, rejected in rejected_map.items():
        chosen_reward, rejected_reward = rewards(row, rtype)
        out.append({
            "image_path": row["image_path"],
            "prompt": row["prompt"],
            "chosen": row["response"],
            "rejected": rejected,
            "rejected_type": rtype,
            "answer": answer,
            "candidate_type": ctype,
            "selected_triple": triple,
            "route_score": row.get("route_score", 0.5),
            "reward_info": {
                "chosen_reward": chosen_reward,
                "rejected_reward": rejected_reward,
                "reward_delta": chosen_reward - rejected_reward,
                "metrics": row.get("metrics", {}),
            },
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default=DEFAULT_INPUT)
    parser.add_argument("--output_jsonl", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(args.input_jsonl)
    if args.limit > 0:
        rows = rows[:args.limit]
    pref = []
    for row in rows:
        pref.extend(build_rows(row, rng))
    rng.shuffle(pref)
    write_jsonl(args.output_jsonl, pref)
    print(f"输入: {len(rows)} -> {args.input_jsonl}")
    print(f"偏好对: {len(pref)} -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
