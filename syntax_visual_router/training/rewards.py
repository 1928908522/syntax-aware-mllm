"""Pure reward helpers for syntax-first Stage E policy optimization."""

import re


def _normalize_part(value):
    text = str(value or "").strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;:!?\"'")


def extract_answer(text):
    match = re.search(r"Answer\s*:\s*([AB])\b", text or "", re.I)
    return match.group(1).upper() if match else ""


def parse_key_relation(text):
    """Parse `Key relation: <src, rel, dst>` from a generated response."""
    match = re.search(
        r"Key\s+relation\s*:\s*<\s*([^,<>]+?)\s*,\s*([^,<>]+?)\s*,\s*([^<>]+?)\s*>",
        text or "",
        re.I,
    )
    if not match:
        return None
    return {
        "src": _normalize_part(match.group(1)),
        "rel": _normalize_part(match.group(2)),
        "dst": _normalize_part(match.group(3)),
    }


def _normalize_triple(triple):
    return {
        "src": _normalize_part(triple.get("src") or triple.get("subject")),
        "rel": _normalize_part(triple.get("rel") or triple.get("relation")),
        "dst": _normalize_part(triple.get("dst") or triple.get("object")),
    }


def triple_similarity(predicted, target):
    """Direction-sensitive component score for a dependency triple."""
    if not predicted or not target:
        return 0.0
    pred = _normalize_triple(predicted)
    gold = _normalize_triple(target)
    src_match = pred["src"] == gold["src"] and bool(gold["src"])
    rel_match = pred["rel"] == gold["rel"] and bool(gold["rel"])
    dst_match = pred["dst"] == gold["dst"] and bool(gold["dst"])
    if src_match and rel_match and dst_match:
        return 1.0
    score = 0.35 * src_match + 0.30 * rel_match + 0.35 * dst_match
    return max(0.0, min(1.0, score))


def target_triples(row):
    if "target_triples" in row:
        return [triple for triple in row.get("target_triples", []) if triple]
    selected = row.get("selected_triple")
    return [selected] if selected else []


def response_format_score(response):
    text = response or ""
    checks = (
        bool(extract_answer(text)),
        parse_key_relation(text) is not None,
        "visual check:" in text.casefold(),
        "contrast:" in text.casefold(),
    )
    return sum(checks) / len(checks)


def score_response(
    row,
    response,
    answer_coef=1.0,
    triple_coef=1.0,
    joint_coef=0.5,
    format_coef=0.2,
    missing_answer_penalty=0.5,
    missing_triple_penalty=0.5,
    non_ascii_penalty=0.2,
):
    """Return candidate-dependent syntax-first reward components."""
    answer = extract_answer(response)
    predicted_triple = parse_key_relation(response)
    triples = target_triples(row)
    triple_score = max(
        (triple_similarity(predicted_triple, target) for target in triples),
        default=0.0,
    )
    answer_ok = float(bool(answer) and answer == row.get("answer"))
    exact_triple = float(triple_score >= 1.0 - 1e-6)
    joint = answer_ok * exact_triple
    format_score = response_format_score(response)
    penalty = 0.0
    if not answer:
        penalty += float(missing_answer_penalty)
    if predicted_triple is None:
        penalty += float(missing_triple_penalty)
    if any(ord(char) > 127 for char in (response or "")):
        penalty += float(non_ascii_penalty)
    total = (
        float(answer_coef) * answer_ok
        + float(triple_coef) * triple_score
        + float(joint_coef) * joint
        + float(format_coef) * format_score
        - penalty
    )
    return {
        "total": total,
        "answer": answer_ok,
        "triple": triple_score,
        "joint": joint,
        "format": format_score,
        "penalty": penalty,
        "pred_answer": answer,
        "pred_triple": predicted_triple,
    }


def sample_confidence_weight(row, syntax_coef=0.25, route_coef=0.0):
    """Weight the policy loss; route is an optional, bounded auxiliary signal."""
    metrics = row.get("metrics") or {}
    syntax_quality = max(0.0, min(1.0, float(metrics.get("syntax_quality", 0.0))))
    route_score = max(0.0, min(1.0, float(row.get("route_score", 0.5))))
    weight = 1.0 + float(syntax_coef) * syntax_quality
    weight += float(route_coef) * (route_score - 0.5)
    return max(0.5, min(1.5, weight))
