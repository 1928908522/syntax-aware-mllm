"""
句法模块: Relation Vocabulary

把 spaCy 细粒度依存关系映射为粗类别 (技术指南 §6):
  nsubj → SUBJECT
  obj   → OBJECT
  amod  → ATTRIBUTE
  ...
"""
from typing import Dict, Optional

# 粗关系 → 索引
COARSE_RELATIONS = [
    "NONE",           # 0
    "SUBJECT",        # 1
    "OBJECT",         # 2
    "ATTRIBUTE",      # 3
    "SPATIAL",        # 4
    "CLAUSAL",        # 5
    "CONJUNCTION",    # 6
    "OTHER_VISUAL",   # 7
]

NUM_RELATIONS = len(COARSE_RELATIONS)


# spaCy dep → 粗关系
FINE_TO_COARSE: Dict[str, str] = {
    "nsubj":      "SUBJECT",
    "nsubjpass":  "SUBJECT",
    "csubj":      "SUBJECT",
    "csubjpass":  "SUBJECT",
    "obj":        "OBJECT",
    "dobj":       "OBJECT",
    "iobj":       "OBJECT",
    "pobj":       "OBJECT",
    "amod":       "ATTRIBUTE",
    "compound":   "ATTRIBUTE",
    "poss":       "ATTRIBUTE",
    "nummod":     "ATTRIBUTE",
    "advmod":     "ATTRIBUTE",
    "prep":       "SPATIAL",
    "obl":        "SPATIAL",
    "acl":        "CLAUSAL",
    "xcomp":      "CLAUSAL",
    "ccomp":      "CLAUSAL",
    "advcl":      "CLAUSAL",
    "relcl":      "CLAUSAL",
    "conj":       "CONJUNCTION",
}


def coarse_to_id(relation: str) -> int:
    """粗关系 → id"""
    return COARSE_RELATIONS.index(relation.upper())


def fine_to_coarse(fine_dep: str) -> str:
    """spaCy 细关系 → 粗关系"""
    r = FINE_TO_COARSE.get(fine_dep, "OTHER_VISUAL")
    return r


def coarse_id_to_str(cid: int) -> str:
    """id → 粗关系名"""
    return COARSE_RELATIONS[cid]


def func(fine_dep: str) -> int:
    """spaCy 细关系 → 粗关系 id"""
    return coarse_to_id(fine_to_coarse(fine_dep))
