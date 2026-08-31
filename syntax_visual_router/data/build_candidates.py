"""
Part 3: Perturbation Candidate Builder + 统计（Stage D）。

对照 Stage_D_简明实施技术指南_更新版_v3.md §4.0~§4.4 与 §21。

职责:
  - 从 Part 2 的 ranked triples（stage_d_targets_coco.jsonl）出发，
    自上而下扫描，判断每条 Triple 能否单独或与其他 Triple 组成一个
    合法的结构扰动操作（Perturbation Candidate）。
  - 第一版支持 4 类 Candidate:
      ROLE_SWAP / ATTRIBUTE_BINDING_SWAP / SPATIAL_REVERSAL / PREDICATE_CONFLICT
  - 先做统计（不调用 Generator）:
      First Feasible Rank / Top-1 Relation 分布 / Candidate Coverage。

设计要点（§4.0.2）:
  - Triple      e = (src, rel, dst)，其中 src=child 词，dst=head 词，rel=spaCy 细关系
  - Candidate   G 可包含一条或多条 Triple
  - Candidate Score: 单条 = Score(e)；多条 = min(Score(e))  （§4.0.4）

用法:
  python -m syntax_visual_router.data.build_candidates --max_samples 200
"""

import os
import sys
import json
import argparse
from collections import defaultdict, Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INPUT_JSONL = os.path.join("data", "ranked_triples.jsonl")
OUTPUT_JSONL = os.path.join("data", "perturbation_candidates.jsonl")

# ---------------- 关系常量 ----------------
# ROLE_SWAP: 同一 predicate 下同时存在主语 + 宾语
SUBJ_RELS = {"nsubj", "nsubjpass", "csubj"}
OBJ_RELS = {"obj", "dobj", "iobj"}
ROLE_RELS = SUBJ_RELS | OBJ_RELS

# ATTRIBUTE_BINDING_SWAP: 属性修饰（第一版只做 amod）
AMOD_REL = "amod"

# SPATIAL_REVERSAL: 可反转空间关系词
SPATIAL_WORDS = {
    "left", "right", "above", "below", "behind", "front",
    "beside", "under", "over", "between", "top", "bottom",
    "next", "near", "inside", "outside", "beneath", "underneath",
}

# PREDICATE_CONFLICT: 只接受语义角色边指向视觉 predicate（排除 aux/prep/advmod 等功能边）
PREDICATE_SEMANTIC_RELS = {"nsubj", "nsubjpass", "csubj", "dobj", "obj", "iobj", "xcomp", "ccomp"}

# PREDICATE_CONFLICT: 明确视觉 predicate（动词）
PREDICATE_WORDS = {
    "ride", "riding", "hold", "holding", "chase", "chasing",
    "wear", "wearing", "stand", "standing", "sit", "sitting",
    "eat", "eating", "play", "playing", "kick", "kicking",
    "throw", "throwing", "catch", "catching", "watch", "watching",
    "read", "reading", "write", "writing", "walk", "walking",
    "run", "running", "jump", "jumping", "swim", "swimming",
    "drive", "driving", "cook", "cooking", "cut", "cutting",
    "carry", "carrying", "push", "pushing", "pull", "pulling",
    "fly", "flying", "climb", "climbing", "surf", "surfing",
    "ski", "skiing", "skate", "skating", "dance", "dancing",
    "sing", "singing", "sleep", "sleeping", "talk", "talking",
    "look", "looking", "smile", "smiling", "kiss", "kissing",
    "hug", "hugging", "feed", "feeding", "wash", "washing",
    "drink", "drinking", "pour", "pouring",
}


# ---------------- Candidate 构造 ----------------

def infer_ctype(triple):
    """从单条三元组推断候选类型（宽松版，含兜底 GENERIC）。

    放宽 prompt 后，type 仅作 soft hint，不再需要 ROLE_SWAP / ATTRIBUTE_BINDING_SWAP
    那种必须跨三元组组合才能命中的严格规则。这里对每条 triple 直接给一个最接近的类型，
    保证几乎每条 triple 都能得到一个 hint，避免原先因四类严格匹配而漏掉大量图片。
    """
    src = (triple.get("src") or "").lower()
    dst = (triple.get("dst") or "").lower()
    rel = (triple.get("rel") or "").lower()

    if src in SPATIAL_WORDS or dst in SPATIAL_WORDS:
        return "SPATIAL_REVERSAL"
    if rel in PREDICATE_SEMANTIC_RELS and dst in PREDICATE_WORDS:
        return "PREDICATE_CONFLICT"
    if rel == AMOD_REL:
        return "ATTRIBUTE_BINDING_SWAP"
    if rel in ROLE_RELS:
        return "ROLE_SWAP"
    return "GENERIC"


def _sig(members):
    """Candidate 去重签名（与顺序无关）。"""
    return frozenset(
        (t["src"].lower(), t["rel"], t["dst"].lower()) for t in members
    )


def _make_candidate(ctype, members):
    """由若干条 Triple 构造一个 Candidate。"""
    source_ranks = sorted(t["rank"] for t in members)
    score = min(t["selection_score"] for t in members)
    return {
        "type": ctype,
        "triples": [
            {"src": t["src"], "rel": t["rel"], "dst": t["dst"],
             "rank": t["rank"], "score": t["selection_score"]}
            for t in members
        ],
        "source_ranks": source_ranks,
        "candidate_score": round(score, 6),
        "first_feasible_rank": source_ranks[0],
    }


def build_candidates(triples, caption):
    """对单个样本的 ranked triples 构造 perturbation candidates。

    Args:
        triples: list[dict]，已按 rank 升序，每条含 src/rel/dst/selection_score/rank
        caption: 原始 caption（备用，当前主要依赖 triple 文本）

    Returns:
        list[dict]，按 candidate_score 降序
    """
    candidates = []
    seen = set()

    by_head = defaultdict(list)  # dst(lower) -> [triple]，用于 ROLE_SWAP
    amods = []
    spatial = []
    predicates = []

    for t in triples:
        rel = t["rel"]
        src = t["src"].lower()
        dst = t["dst"].lower()

        if rel in ROLE_RELS:
            by_head[dst].append(t)
        if rel == AMOD_REL:
            amods.append(t)
        if src in SPATIAL_WORDS or dst in SPATIAL_WORDS:
            spatial.append(t)
        if rel in PREDICATE_SEMANTIC_RELS and dst in PREDICATE_WORDS:
            predicates.append(t)

    # ---- ROLE_SWAP：同一 predicate 下 nsubj + obj/dobj ----
    for dst, ts in by_head.items():
        subjs = [t for t in ts if t["rel"] in SUBJ_RELS]
        objs = [t for t in ts if t["rel"] in OBJ_RELS]
        for ns in subjs:
            for ob in objs:
                members = [ns, ob]
                sig = _sig(members)
                if sig in seen:
                    continue
                seen.add(sig)
                candidates.append(_make_candidate("ROLE_SWAP", members))

    # ---- ATTRIBUTE_BINDING_SWAP：多条 amod 修饰不同实体 ----
    for i in range(len(amods)):
        for j in range(i + 1, len(amods)):
            a, b = amods[i], amods[j]
            if a["dst"].lower() == b["dst"].lower():
                continue  # 修饰同一实体，无法 swap
            members = [a, b]
            sig = _sig(members)
            if sig in seen:
                continue
            seen.add(sig)
            candidates.append(_make_candidate("ATTRIBUTE_BINDING_SWAP", members))

    # ---- SPATIAL_REVERSAL：单条 Triple 涉及可反转空间关系 ----
    for t in spatial:
        sig = _sig([t])
        if sig in seen:
            continue
        seen.add(sig)
        candidates.append(_make_candidate("SPATIAL_REVERSAL", [t]))

    # ---- PREDICATE_CONFLICT：单条 Triple 的 head 是明确视觉 predicate ----
    for t in predicates:
        sig = _sig([t])
        if sig in seen:
            continue
        seen.add(sig)
        candidates.append(_make_candidate("PREDICATE_CONFLICT", [t]))

    # 按 candidate_score 降序（§4.0.5）
    candidates.sort(key=lambda c: -c["candidate_score"])
    return candidates


# ---------------- 统计 ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_JSONL)
    parser.add_argument("--output", default=OUTPUT_JSONL)
    parser.add_argument("--max_samples", type=int, default=0, help="0=全部")
    parser.add_argument("--show_examples", type=int, default=5, help="打印前 N 个样本的 candidate 样例")
    args = parser.parse_args()

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if args.max_samples > 0:
        records = records[: args.max_samples]

    print(f"读取样本数: {len(records)}\n")

    # ---- 统计容器 ----
    ffr_values = []          # 每个样本的 first feasible rank（无 candidate 记 None）
    top1_rel_counter = Counter()
    n_with_candidate = 0
    type_counter = Counter()
    candidate_count_per_img = []

    out_lines = []

    for rec in records:
        caption = rec.get("caption", "")
        triples = rec.get("triples", [])

        # Top-1 relation 分布（§21.2）
        if triples:
            top1_rel_counter[triples[0]["rel"]] += 1

        candidates = build_candidates(triples, caption)

        ffr = None
        if candidates:
            ffr = min(c["first_feasible_rank"] for c in candidates)
            n_with_candidate += 1
            for c in candidates:
                type_counter[c["type"]] += 1

        ffr_values.append(ffr)
        candidate_count_per_img.append(len(candidates))

        out_lines.append({
            "image_path": rec.get("image_path", ""),
            "caption": caption,
            "num_triples": len(triples),
            "first_feasible_rank": ffr,
            "candidates": candidates,
        })

    # ---- 汇总输出 ----
    total = len(records)
    print("=" * 70)
    print("Part 3 Candidate Builder 统计结果")
    print("=" * 70)

    # §21.3 Coverage
    coverage = n_with_candidate / total if total else 0.0
    print(f"\n[Coverage] 至少能构造一个 Candidate 的图片占比")
    print(f"  {n_with_candidate}/{total} = {coverage:.1%}")

    # §21.1 First Feasible Rank 分布（累计桶，与指南一致）
    print(f"\n[First Feasible Rank] 第一个能参与构造 Candidate 的 Triple 的 Rank")
    def bucket(cond):
        n = sum(1 for r in ffr_values if r is not None and cond(r))
        return n, n / total if total else 0.0
    for label, cond in [
        ("Rank = 1", lambda r: r == 1),
        ("Rank <= 3", lambda r: r <= 3),
        ("Rank <= 5", lambda r: r <= 5),
        ("Rank <= 10", lambda r: r <= 10),
    ]:
        n, p = bucket(cond)
        print(f"  {label:>10s}: {n:4d}  ({p:5.1%})")
    n_gt10 = sum(1 for r in ffr_values if r is not None and r > 10)
    n_none = sum(1 for r in ffr_values if r is None)
    print(f"  {'Rank > 10':>10s}: {n_gt10:4d}  ({n_gt10/total:5.1%})")
    print(f"  {'No feasible':>10s}: {n_none:4d}  ({n_none/total:5.1%})")

    # §21.2 Top-1 Relation 分布
    print(f"\n[Top-1 Relation 分布] Stage B Rank-1 的 relation 类型")
    for rel, cnt in top1_rel_counter.most_common():
        print(f"  {rel:12s}: {cnt:4d}  ({cnt/total:5.1%})")

    # 各类 Candidate 数量分布
    print(f"\n[Candidate 类型分布]")
    for ctype in ["ROLE_SWAP", "ATTRIBUTE_BINDING_SWAP",
                  "SPATIAL_REVERSAL", "PREDICATE_CONFLICT"]:
        cnt = type_counter.get(ctype, 0)
        print(f"  {ctype:24s}: {cnt:4d}")

    # 每图 candidate 数量概览
    if candidate_count_per_img:
        avg = sum(candidate_count_per_img) / len(candidate_count_per_img)
        mx = max(candidate_count_per_img)
        print(f"\n[每图 Candidate 数量] mean={avg:.2f}  max={mx}")

    # ---- 样例 ----
    if args.show_examples > 0:
        print(f"\n{'='*70}")
        print(f"样例（前 {min(args.show_examples, len(out_lines))} 个样本的 Candidate）")
        print(f"{'='*70}")
        shown = 0
        for rec in out_lines:
            if not rec["candidates"]:
                continue
            shown += 1
            if shown > args.show_examples:
                break
            print(f"\n[{shown}] {os.path.basename(rec['image_path'])}")
            print(f"  caption: {rec['caption']}")
            print(f"  first_feasible_rank: {rec['first_feasible_rank']}")
            for c in rec["candidates"][:5]:
                members = " + ".join(
                    f"{t['src']}-{t['rel']}->{t['dst']}(r{t['rank']})"
                    for t in c["triples"]
                )
                print(f"    {c['type']:24s} score={c['candidate_score']:.6f}  [{members}]")

    # ---- 写输出 ----
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in out_lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nCandidate 已写入: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
