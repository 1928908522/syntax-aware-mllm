"""
句法模块: Dependency Parser + Node Filtering

步骤 (技术指南 Step 3-4):
  Caption → spaCy → Dependency Tree → Visualizable Node Filtering → Dependency Triplets
"""
import spacy
from typing import List, Tuple
from dataclasses import dataclass

# ----------------- 常量 -----------------
KEEP_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

KEEP_DEP = {
    "nsubj", "nsubjpass", "obj", "dobj", "iobj",
    "amod", "compound", "poss", "prep", "pobj",
    "acl", "conj", "csubj", "xcomp",
}

# 关系粗类别映射
DEP_TO_COARSE = {
    "nsubj":      "SUBJECT",
    "nsubjpass":  "SUBJECT",
    "csubj":      "SUBJECT",
    "obj":        "OBJECT",
    "dobj":       "OBJECT",
    "iobj":       "OBJECT",
    "amod":       "ATTRIBUTE",
    "compound":   "ATTRIBUTE",
    "poss":       "ATTRIBUTE",
    "prep":       "SPATIAL",
    "pobj":       "SPATIAL",
    "acl":        "CLAUSAL",
    "xcomp":      "CLAUSAL",
    "conj":       "CONJUNCTION",
}


@dataclass
class DepNode:
    idx: int
    text: str       # lemma (lowercase) 或原词形（parse_all）
    pos: str
    dep: str
    head_idx: int
    char_offset: int = -1   # 原 token 在 caption 中的字符起始偏移（对齐用）
    char_end: int = -1      # 原 token 的字符结束偏移（char_offset + len(原文)）


@dataclass
class DepEdge:
    child: DepNode
    head: DepNode
    relation: str       # 粗关系: SUBJECT/OBJECT/ATTRIBUTE/...
    fine_relation: str  # 细关系: nsubj/obj/amod/...
    child_text: str
    head_text: str


# ==================== spaCy ====================
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_lg")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_lg' not found. "
                "Run: python -m spacy download en_core_web_lg"
            )
    return _nlp


# ==================== 依存解析 ====================
def parse_caption(caption: str) -> List[DepNode]:
    """对 Caption 做依存解析"""
    nlp = get_nlp()
    doc = nlp(caption)
    return [
        DepNode(idx=t.i, text=t.lemma_.lower(),
                pos=t.pos_, dep=t.dep_, head_idx=t.head.i,
                char_offset=t.idx, char_end=t.idx + len(t.text))
        for t in doc
    ]


# ==================== 节点过滤 ====================
def is_visualizable(node: DepNode) -> bool:
    return node.pos in KEEP_POS

def filter_nodes(nodes: List[DepNode]) -> List[DepNode]:
    return [n for n in nodes if is_visualizable(n)]


# ==================== 依存边 ====================
def extract_edges(nodes: List[DepNode], keep_nodes: List[DepNode]) -> List[DepEdge]:
    """提取依存边，两端都必须是 visualizable 节点"""
    keep_idx = {n.idx for n in keep_nodes}
    node_map = {n.idx: n for n in nodes}
    edges = []
    for node in nodes:
        if node.idx not in keep_idx:
            continue
        head = node_map.get(node.head_idx)
        if head is None or head.idx not in keep_idx:
            continue
        if node.idx == head.idx:
            continue
        # 只在保留的依存关系中取
        if node.dep not in KEEP_DEP:
            continue
        coarse = DEP_TO_COARSE.get(node.dep, "OTHER_VISUAL")
        edges.append(DepEdge(
            child=node, head=head,
            relation=coarse, fine_relation=node.dep,
            child_text=node.text, head_text=head.text,
        ))
    return edges


# ==================== 一步解析 ====================
def parse_and_filter(caption: str) -> Tuple[List[DepNode], List[DepEdge]]:
    nodes = parse_caption(caption)
    keep = filter_nodes(nodes)
    edges = extract_edges(nodes, keep)
    return keep, edges


# ==================== 全量三元组解析（Stage D Part 1 专用） ====================
def parse_all(caption: str) -> Tuple[List[DepNode], List[DepEdge]]:
    """提取 ALL 依存三元组，不做任何词性/关系/端点过滤。

    与 parse_and_filter 的区别:
      - 不做 KEEP_POS 词性过滤、KEEP_DEP 关系白名单过滤、端点可视觉化过滤
      - 仅排除标点/空格(PUNCT/SPACE/SYM)
      - text 用原词形 t.text（非 lemma），便于 Stage D 后续在原 caption 上做扰动

    返回 (nodes, edges)，edge 的 fine_relation 保留 spaCy 原始细粒度关系。
    """
    nlp = get_nlp()
    doc = nlp(caption)
    nodes = [
        DepNode(idx=t.i, text=t.text, pos=t.pos_, dep=t.dep_, head_idx=t.head.i,
                char_offset=t.idx, char_end=t.idx + len(t.text))
        for t in doc
        if t.pos_ not in ("PUNCT", "SPACE", "SYM")
    ]
    node_map = {n.idx: n for n in nodes}
    edges = []
    for node in nodes:
        if node.idx == node.head_idx:  # 跳过 root 自环
            continue
        head = node_map.get(node.head_idx)
        if head is None:  # head 是被排除的标点等
            continue
        coarse = DEP_TO_COARSE.get(node.dep, "OTHER")
        edges.append(DepEdge(
            child=node, head=head,
            relation=coarse, fine_relation=node.dep,
            child_text=node.text, head_text=head.text,
        ))
    return nodes, edges


# ==================== 测试 ====================
if __name__ == "__main__":
    caption = "A young man is riding a brown horse."
    print(f"Caption: {caption}\n")

    nodes, edges = parse_and_filter(caption)

    print(f"Visualizable 节点: {[n.text for n in nodes]}")
    print(f"\n依存边:")
    for e in edges:
        print(f"  {e.child_text} --{e.fine_relation}({e.relation})--> {e.head_text}")

    print("\n期望:")
    print("  young --amod--> man")
    print("  man --nsubj--> riding")
    print("  brown --amod--> horse")
    print("  horse --obj--> riding")
