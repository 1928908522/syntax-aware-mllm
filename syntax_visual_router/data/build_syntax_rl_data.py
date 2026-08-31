"""Build route-independent RL targets from positive/negative caption syntax."""

import argparse
import json
import os
import sys

from syntax_visual_router.syntax import parse_all
from syntax_visual_router.data.cot_templates import build_chosen_response


DEFAULT_INPUT = os.path.join("data", "stage_e_cot_sft_topk_train.jsonl")
DEFAULT_OUTPUT = os.path.join("data", "stage_e_syntax_rl_train.jsonl")

NON_SEMANTIC_RELATIONS = {"det", "punct", "cc", "aux", "auxpass", "cop", "mark", "case"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def edge_triples(caption):
    _, edges = parse_all(caption)
    return [
        {"src": edge.child_text, "rel": edge.fine_relation, "dst": edge.head_text}
        for edge in edges
    ]


def triple_key(triple):
    return tuple(str(triple.get(key, "")).strip().casefold() for key in ("src", "rel", "dst"))


def unique_triples(triples):
    seen = set()
    result = []
    for triple in triples:
        key = triple_key(triple)
        if key not in seen:
            seen.add(key)
            result.append(triple)
    return result


def contrastive_triples(positive_triples, negative_triples):
    """Keep dependency relations present in the positive caption but not the negative."""
    negative_keys = {triple_key(triple) for triple in negative_triples}
    differences = [
        triple for triple in unique_triples(positive_triples)
        if triple_key(triple) not in negative_keys
    ]
    semantic = [
        triple for triple in differences
        if str(triple.get("rel", "")).casefold() not in NON_SEMANTIC_RELATIONS
    ]
    return semantic or differences


def convert(row, allow_route_fallback=False):
    positive_triples = edge_triples(row["positive"])
    negative_triples = edge_triples(row["negative"])
    targets = contrastive_triples(positive_triples, negative_triples)
    source = "text_contrast"
    if not targets:
        if allow_route_fallback:
            targets = row.get("target_triples") or [row.get("selected_triple", {})]
            targets = [triple for triple in targets if triple]
            source = "route_fallback"
        else:
            targets = []
            source = "answer_only"
    result = dict(row)
    result["route_selected_triple"] = row.get("selected_triple")
    result["route_selected_response"] = row.get("response")
    result["positive_syntax_triples"] = positive_triples
    result["negative_syntax_triples"] = negative_triples
    result["target_triples"] = targets
    result["syntax_target_source"] = source
    if targets:
        result["selected_triple"] = targets[0]
        result["response"] = build_chosen_response(
            targets[0], row.get("candidate_type"), row["answer"])
    else:
        result["selected_triple"] = {}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default=DEFAULT_INPUT)
    parser.add_argument("--output_jsonl", default=DEFAULT_OUTPUT)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--allow_route_fallback", action="store_true",
                        help="Ablation only: use the old Route-selected triple when text contrast is empty.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    total = 0
    text_contrast = 0
    fallback = 0
    target_count = 0
    with open(args.input_jsonl, encoding="utf-8") as source, open(
        args.output_jsonl, "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if args.max_samples > 0 and total >= args.max_samples:
                break
            line = line.strip()
            if not line:
                continue
            converted = convert(json.loads(line), allow_route_fallback=args.allow_route_fallback)
            destination.write(json.dumps(converted, ensure_ascii=False) + "\n")
            total += 1
            target_count += len(converted["target_triples"])
            if converted["syntax_target_source"] == "text_contrast":
                text_contrast += 1
            elif converted["syntax_target_source"] == "route_fallback":
                fallback += 1

    print(f"输入: {args.input_jsonl}")
    print(f"输出: {args.output_jsonl}")
    print(f"样本: {total}")
    print(f"文本差异三元组: {text_contrast}")
    print(f"Route fallback: {fallback}")
    print(f"仅答案监督: {total - text_contrast - fallback}")
    print(f"平均 target triples: {target_count / max(total, 1):.2f}")


if __name__ == "__main__":
    main()
