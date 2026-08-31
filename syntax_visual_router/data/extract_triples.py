"""Extract dependency triples from caption JSONL without Image Route."""

import argparse
import json
import os

from syntax_visual_router.syntax import parse_all


SEMANTIC_PRIORITY = {
    "nsubj": 1.0,
    "nsubjpass": 1.0,
    "obj": 1.0,
    "dobj": 1.0,
    "iobj": 1.0,
    "amod": 0.9,
    "prep": 0.85,
    "pobj": 0.85,
    "poss": 0.8,
    "compound": 0.75,
    "xcomp": 0.75,
    "acl": 0.7,
    "conj": 0.65,
}


def extract(text):
    _, edges = parse_all(text)
    triples = [
        {
            "src": edge.child_text,
            "rel": edge.fine_relation,
            "dst": edge.head_text,
            "selection_score": SEMANTIC_PRIORITY.get(edge.fine_relation, 0.5),
        }
        for edge in edges
    ]
    triples.sort(key=lambda item: -item["selection_score"])
    for rank, triple in enumerate(triples, start=1):
        triple["rank"] = rank
    return triples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with caption and image_path fields")
    parser.add_argument("--output", default=os.path.join("data", "ranked_triples.jsonl"))
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    written = 0
    with open(args.input, encoding="utf-8") as source, open(args.output, "w", encoding="utf-8") as target:
        for line in source:
            if args.max_samples > 0 and written >= args.max_samples:
                break
            row = json.loads(line)
            caption = row.get("caption") or row.get("positive")
            if not caption:
                continue
            output = {
                "image_path": row.get("image_path", ""),
                "caption": caption,
                "triples": extract(caption),
                "triple_rank_source": "syntax_priority",
            }
            target.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} records to {args.output}")


if __name__ == "__main__":
    main()
