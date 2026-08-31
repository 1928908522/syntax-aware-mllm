"""Lazy-loaded syntax similarity and NLI metrics for caption pairs."""

import os

import torch


SPACY_MODEL = os.environ.get("SYNTAX_SPACY_MODEL", "en_core_web_lg")
EMBEDDING_MODEL = os.environ.get("SYNTAX_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
NLI_MODEL = os.environ.get("SYNTAX_NLI_MODEL", "FacebookAI/roberta-large-mnli")

_nlp = None
_embedding_tokenizer = None
_embedding_model = None
_nli_pipeline = None


def _load_syntax():
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load(SPACY_MODEL)
    return _nlp


def _load_embedding():
    global _embedding_tokenizer, _embedding_model
    if _embedding_model is None:
        from transformers import AutoModel, AutoTokenizer

        _embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        _embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL)
        _embedding_model.eval()


def _build_tree(token):
    from zss import Node

    node = Node(token.text)
    for child in token.children:
        node.addkid(_build_tree(child))
    return node


def tree_edit_distance(first, second):
    from zss import simple_distance

    nlp = _load_syntax()
    roots_first = [token for token in nlp(first) if token.head == token]
    roots_second = [token for token in nlp(second) if token.head == token]
    if not roots_first or not roots_second:
        return None
    return simple_distance(_build_tree(roots_first[0]), _build_tree(roots_second[0]))


def dependency_edges(text):
    nlp = _load_syntax()
    return {
        (token.head.text, token.dep_, token.text)
        for token in nlp(text)
        if token.head != token
    }


def syntactic_jaccard(first, second):
    first_edges = dependency_edges(first)
    second_edges = dependency_edges(second)
    union = first_edges | second_edges
    return len(first_edges & second_edges) / len(union) if union else 0.0


def _encode(text):
    _load_embedding()
    encoded = _embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        hidden = _embedding_model(**encoded).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def embedding_similarity(first, second):
    return float(torch.nn.functional.cosine_similarity(_encode(first), _encode(second), dim=-1).item())


def nli_label(premise, hypothesis):
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        _nli_pipeline = pipeline("text-classification", model=NLI_MODEL, device=device)
    text = f"{premise} </s></s> {hypothesis}"
    return _nli_pipeline(text, truncation=True)[0]["label"].lower()


def combined_similarity(first, second, ted, jaccard, semantic_similarity):
    denominator = max(len(first.split()), len(second.split()))
    ted_term = 0.0 if not denominator or ted is None else 1.0 - ted / denominator
    return 0.25 * ted_term + 0.25 * jaccard + 0.5 * max(0.0, semantic_similarity)
