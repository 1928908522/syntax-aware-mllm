"""
Part 5: 验证生成的困难负文本是否合格（放宽版）。

prompt/ctype 放宽后，不再做严格的结构扰动验证，只保留「错乱检测」硬门：
  text_valid_reason : 非空、非复制原句、纯英文(无中文)、长度合理、无严重重复

其余指标仅统计（不硬过滤）：
  - TED / Jaccard / EmbeddingSim（接受样本的均值）
  - NLI 分布（contradiction/neutral/entailment）

候选顺序重试（§6.6）:
  对每张图按 candidate_score 从高到低依次验证，接受第一条通过错乱检测的；
  全部失败则 Skip Image。

用法:
  python -m syntax_visual_router.data.filter_negatives --input data/generated_negatives.jsonl --output data/syntax_negatives.jsonl
"""

import json
import argparse
import os
import sys
from collections import defaultdict, Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from syntax_visual_router.evaluation.text_metrics import (
    embedding_similarity, nli_label, syntactic_jaccard, tree_edit_distance,
)

INPUT_JSONL = os.path.join("data", "generated_negatives.jsonl")
OUTPUT_JSONL = os.path.join("data", "syntax_negatives.jsonl")


def _norm(w):
    return w.strip().strip(".,;:!?\"'()[]").lower()


def _has_cjk(s):
    """是否包含中文字符（视为错乱输出）。"""
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)


def text_valid_reason(pos, neg):
    """错乱检测（唯一硬门）。返回 None 表示通过，否则返回失败原因。"""
    neg = (neg or "").strip()
    if not neg:
        return "empty"
    if _norm(pos) == _norm(neg):
        return "unchanged"
    if _has_cjk(neg):
        return "non_english"
    pw = [_norm(w) for w in pos.split() if _norm(w)]
    nw = [_norm(w) for w in neg.split() if _norm(w)]
    if not pw or not nw or len(nw) < 2:
        return "too_short"
    ratio = len(nw) / len(pw)
    if ratio < 0.5 or ratio > 2.0:
        return "length_bad"
    cnt = Counter(nw)
    if max(cnt.values()) / len(nw) > 0.5:  # 严重重复
        return "repetitive"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_JSONL)
    parser.add_argument("--output", default=OUTPUT_JSONL)
    parser.add_argument("--dump_fail", action="store_true",
                        help="将错乱候选 dump 到 {output}.rejected.jsonl 供排查")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"读取负文本条数: {len(rows)}")

    # 按 image 分组，保留文件内顺序（candidate 已按 score 降序）
    by_image = defaultdict(list)
    for r in rows:
        by_image[r["image_path"]].append(r)

    for img in by_image:
        by_image[img].sort(key=lambda r: -(r.get("candidate_score") or 0.0))

    n_images = len(by_image)
    n_accepted = 0
    n_skipped = 0

    invalid_counter = Counter()
    type_counter = Counter()
    accept_type_counter = Counter()
    nli_counter = Counter()
    accept_nli_counter = Counter()

    ted_vals, jac_vals, sim_vals = [], [], []

    out_lines = []
    rejected = []

    for img, cands in by_image.items():
        accepted_this = None
        for rank, r in enumerate(cands, start=1):
            pos = r["positive"]
            neg = r["negative"]
            ctype = r.get("candidate_type", "")
            type_counter[ctype] += 1

            lab = nli_label(pos, neg)
            nli_counter[lab] += 1

            reason = text_valid_reason(pos, neg)
            if reason is not None:
                invalid_counter[reason] += 1
                rejected.append({
                    "image_path": img,
                    "candidate_rank": rank,
                    "candidate_type": ctype,
                    "positive": pos,
                    "negative": neg,
                    "reason": reason,
                    "nli_label": lab,
                })
                continue

            # 错乱检测通过 → 接受第一条，统计指标
            ted = tree_edit_distance(pos, neg)
            jac = syntactic_jaccard(pos, neg)
            sim = embedding_similarity(pos, neg)
            if ted is not None:
                ted_vals.append(ted)
            if jac is not None:
                jac_vals.append(jac)
            if sim is not None:
                sim_vals.append(sim)

            accepted_this = {
                "image_path": img,
                "positive": pos,
                "negative": neg,
                "candidate_type": ctype,
                "candidate_score": r.get("candidate_score"),
                "candidate_rank": rank,
                "target_triples": r.get("candidate_triples", []),
                "ted": ted,
                "jaccard": jac,
                "embedding_similarity": sim,
                "nli_label": lab,
                "generator": r.get("generator", ""),
            }
            break

        if accepted_this is not None:
            n_accepted += 1
            accept_type_counter[accepted_this["candidate_type"]] += 1
            accept_nli_counter[accepted_this["nli_label"]] += 1
            out_lines.append(accepted_this)
        else:
            n_skipped += 1

    # ---- 写输出 ----
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in out_lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.dump_fail:
        rej_path = args.output + ".rejected.jsonl"
        with open(rej_path, "w", encoding="utf-8") as f:
            for rec in rejected:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"错乱样本已 dump: {rej_path} ({len(rejected)} 条)")

    # ---- 汇总 ----
    avg_ted = sum(ted_vals) / len(ted_vals) if ted_vals else 0.0
    avg_jac = sum(jac_vals) / len(jac_vals) if jac_vals else 0.0
    avg_sim = sum(sim_vals) / len(sim_vals) if sim_vals else 0.0

    print("\n" + "=" * 70)
    print("Part 5 过滤结果（放宽版：仅错乱硬门 + 指标统计）")
    print("=" * 70)
    print(f"  图片数            : {n_images}")
    print(f"  接受 (Accept)     : {n_accepted}  ({n_accepted/n_images:.1%})")
    print(f"  跳过 (Skip)       : {n_skipped}  ({n_skipped/n_images:.1%})")

    print(f"\n[错乱/无效统计]")
    if invalid_counter:
        for reason, cnt in invalid_counter.most_common():
            print(f"  {reason:20s}: {cnt}")
    else:
        print("  (无)")

    print(f"\n[整体指标]（接受的 {n_accepted} 条负样本）")
    print(f"  TED (均值)      : {avg_ted:.3f}")
    print(f"  Jaccard (均值)  : {avg_jac:.3f}")
    print(f"  Embedding sim   : {avg_sim:.3f}")

    print(f"\n[Candidate 类型分布] 接受/总")
    all_ctypes = sorted(
        set(type_counter) | set(accept_type_counter),
        key=lambda c: -(type_counter.get(c, 0)),
    )
    for ctype in all_ctypes:
        acc = accept_type_counter.get(ctype, 0)
        tot = type_counter.get(ctype, 0)
        print(f"  {ctype:24s}: {acc:4d}/{tot:4d}")

    print(f"\n[NLI 分布]（仅记录，不硬过滤）")
    total_nli = sum(nli_counter.values()) or 1
    for lab in ["contradiction", "neutral", "entailment"]:
        print(f"  全部候选 {lab:14s}: {nli_counter.get(lab, 0):4d} "
              f"({nli_counter.get(lab, 0)/total_nli:.1%})")
    acc_total = sum(accept_nli_counter.values()) or 1
    for lab in ["contradiction", "neutral", "entailment"]:
        print(f"  接受样本 {lab:14s}: {accept_nli_counter.get(lab, 0):4d} "
              f"({accept_nli_counter.get(lab, 0)/acc_total:.1%})")

    print(f"\n输出: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
