"""Templates for Stage E syntax-route reasoning data."""

DISC_PROMPT = (
    "You are given an image and two captions.\n\n"
    "Caption A:\n{caption_a}\n\n"
    "Caption B:\n{caption_b}\n\n"
    "Analyze the key visual relation, check which caption is supported by the image, "
    "and end with \"Answer: A.\" or \"Answer: B.\""
)


def triple_text(triple):
    if not triple:
        return "<unknown, related_to, unknown>"
    src = triple.get("src") or triple.get("subject") or "unknown"
    rel = triple.get("rel") or triple.get("relation") or "related_to"
    dst = triple.get("dst") or triple.get("object") or "unknown"
    return f"<{src}, {rel}, {dst}>"


def candidate_description(candidate_type):
    ctype = (candidate_type or "GENERIC").upper()
    if "ATTRIBUTE" in ctype or "ATT" in ctype:
        return "the alternative caption changes an attribute binding."
    if "SPATIAL" in ctype or "REL" in ctype:
        return "the alternative caption changes a visual relation."
    if "ROLE" in ctype or "SWAP" in ctype:
        return "the alternative caption swaps the participants in a relation."
    if "PREDICATE" in ctype:
        return "the alternative caption changes the action or predicate."
    if "ALL" in ctype:
        return "the alternative caption changes one or more visual relations."
    return "the alternative caption changes a visually important detail."


def build_prompt(caption_a, caption_b):
    return DISC_PROMPT.replace("{caption_a}", caption_a).replace("{caption_b}", caption_b)


def build_chosen_response(triple, candidate_type, answer):
    t = triple_text(triple)
    desc = candidate_description(candidate_type)
    return (
        f"Key relation: {t}.\n"
        "Visual check: this relation should be verified against the image evidence.\n"
        f"Contrast: {desc}\n"
        f"Answer: {answer}."
    )


def build_wrong_answer_response(triple, candidate_type, correct_answer):
    wrong = "B" if correct_answer == "A" else "A"
    t = triple_text(triple)
    desc = candidate_description(candidate_type)
    return (
        f"Key relation: {t}.\n"
        "Visual check: this relation should be verified against the image evidence.\n"
        f"Contrast: {desc}\n"
        f"Answer: {wrong}."
    )


def build_wrong_visual_response(triple, candidate_type, correct_answer):
    t = triple_text(triple)
    desc = candidate_description(candidate_type)
    return (
        f"Key relation: {t}.\n"
        "Visual check: the changed relation is supported by the image.\n"
        f"Contrast: {desc}\n"
        f"Answer: {'B' if correct_answer == 'A' else 'A'}."
    )


def build_wrong_relation_response(wrong_triple, candidate_type, correct_answer):
    t = triple_text(wrong_triple)
    return (
        f"Key relation: {t}.\n"
        "Visual check: this relation is the decisive evidence in the image.\n"
        "Contrast: the other caption is less visually supported.\n"
        f"Answer: {'B' if correct_answer == 'A' else 'A'}."
    )

