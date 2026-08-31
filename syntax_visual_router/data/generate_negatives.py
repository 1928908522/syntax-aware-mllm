"""
Part 4: 生成困难负文本（结构化 prompt + 三元组提示；默认本地 Qwen2.5，可选 DeepSeek）。

对照 Stage_D_简明实施技术指南_更新版_v4.md §5。

v4 改进:
  1. 去掉多模态，只用 DeepSeek 纯文本 LLM（caption 本身已描述图像内容）。
  2. 放宽版 prompt：只给「可触碰的概念(依存三元组)」+「改动意图」，不强制机械 swap，
     允许 LLM 在给定范围内自然发挥，并显式禁止拆分固定复合词（cast iron / close-up 等）。
  3. 默认 thinking=disabled（非思考模式），直接输出单个句子。

用法:
  python -m syntax_visual_router.data.generate_negatives --input data/perturbation_candidates.jsonl
  测试看输入: ... --max_samples 3 --print_prompt
"""

import os
import sys
import json
import time
import random
import argparse

import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from syntax_visual_router.data.build_candidates import infer_ctype

DEFAULT_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_MODEL = "deepseek-v4-pro"

LOCAL_MODEL_PATH = os.environ.get("NEGATIVE_GENERATOR_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

TARGETS_JSONL = os.path.join("data", "perturbation_candidates.jsonl")
OUTPUT_JSONL = os.path.join("data", "generated_negatives.jsonl")


# ---------------- 改动提示（放宽版：三元组 + 类型仅作 HINT，不强制机械操作） ----------------

REL_GLOSS = {
    "amod": "is an attribute/quality of",
    "compound": "is part of the compound",
    "nsubj": "is the subject of",
    "nsubjpass": "is the subject of",
    "csubj": "is the subject of",
    "obj": "is the object of",
    "dobj": "is the object of",
    "iobj": "is the object of",
    "prep": "is a preposition linking",
    "pobj": "is the object of the preposition",
    "conj": "is coordinated with",
    "acl": "is a clause modifying",
    "xcomp": "is a complement of",
    "poss": "belongs to",
}


def _triples_text(triples):
    """把 candidate 的三元组转成给 LLM 看的、可读的『可编辑范围』。"""
    lines = []
    for t in triples:
        gloss = REL_GLOSS.get(t["rel"], t["rel"])
        lines.append(f'- "{t["src"]}" {gloss} "{t["dst"]}"')
    return "\n".join(lines)


def get_ctype_hint(ctype):
    """每种 candidate 类型的『改动提示』（软性，仅作参考，不强制机械操作）。"""
    if ctype == "ROLE_SWAP":
        return (
            "A useful place to introduce the mistake may be the semantic roles "
            "of the entities in the event—for example, who performs the action "
            "and who receives or is affected by it. You may alter this relation "
            "if it can be done naturally, but do not force a literal subject-object "
            "swap if that would produce an awkward or implausible sentence."
        )
    if ctype == "ATTRIBUTE_BINDING_SWAP":
        return (
            "A useful place to introduce the mistake may be an attribute or quality "
            "and which object it describes. You may change or reassign an attribute "
            "if doing so creates a natural but incorrect description. Do not "
            "mechanically swap words, and do not break compound or multi-word "
            "expressions just to force an attribute change."
        )
    if ctype == "SPATIAL_REVERSAL":
        return (
            "A useful place to introduce the mistake may be the spatial or positional "
            "relationship between the entities. Consider changing the relation to a "
            "different, incompatible one if this can be expressed naturally. A literal "
            "left-right or above-below reversal is not required."
        )
    if ctype == "PREDICATE_CONFLICT":
        return (
            "A useful place to introduce the mistake may be the action, state, or "
            "event described in the caption. Consider replacing it with a different "
            "but natural action or state that makes the description incorrect. "
            "Prefer a small change and keep the entities and surrounding wording "
            "unchanged whenever possible."
        )
    return (
        "Use the listed semantic relations as hints for where a small factual "
        "mistake could be introduced. Choose the most natural minimal change."
    )


# topk 模式：Top-k 三元组 + 改动类型提示（原放宽版，主方法）
PROMPT_TOPK = (
    "You are generating a hard negative caption for vision-language training.\n\n"
    "The caption below is CORRECT. Rewrite it into a plausible but WRONG caption by "
    "introducing one small factual mistake.\n\n"
    "CORRECT caption:\n"
    '"{caption}"\n\n'
    "Potentially useful semantic relations from the caption:\n"
    "{relations}\n\n"
    "Suggested type of mistake:\n"
    "{ctype_hint}\n\n"
    "IMPORTANT:\n"
    "The relations and suggested mistake type above are only HINTS.\n"
    "You do NOT need to literally swap words, reverse syntax, or mechanically apply "
    "the suggested operation. Use them only to identify a good place where a small "
    "semantic mistake could be introduced.\n\n"
    "Your goal is to produce a sentence that:\n\n"
    "1. Is clearly different in meaning from the correct caption and contains at "
    "least one factual error.\n\n"
    "2. Stays as close as possible to the original caption in wording, sentence "
    "structure, grammar, and length.\n\n"
    "3. Makes only a small, localized semantic change. Avoid rewriting the whole "
    "sentence.\n\n"
    "4. Remains a natural and grammatically correct English sentence that a person "
    "could realistically use to describe an image.\n\n"
    "5. Preserves words and phrases that do not need to change.\n\n"
    "6. Never creates unnatural expressions by mechanically swapping individual "
    "words.\n\n"
    "7. Treats multi-word expressions and compound concepts as indivisible units "
    "whenever appropriate, for example:\n"
    '   "cast iron", "close-up", "traffic light", "fire hydrant",\n'
    '   "tennis court", "double faucet".\n'
    "   Do not split, reorder, or recombine their internal words merely to create "
    "an error.\n\n"
    "8. Does NOT need to preserve the exact dependency relation suggested above. "
    "Semantic correctness of the negative example is more important than "
    "performing a literal structural transformation.\n\n"
    "9. If the suggested relation cannot be changed naturally while producing a "
    "clearly wrong sentence, choose another nearby, minimal factual change "
    "instead.\n\n"
    "10. Do not simply paraphrase the correct caption. The rewritten caption must "
    "contain a genuine semantic error.\n\n"
    "11. Do not output the original caption unchanged.\n\n"
    "Before answering, internally check:\n"
    "- Is the new sentence natural?\n"
    "- Is it actually different in meaning?\n"
    "- Does it contain a genuine factual mistake?\n"
    "- Did I accidentally break a compound expression or create a nonsense phrase?\n\n"
    "Output ONLY the rewritten caption.\n"
    "Do not provide quotes, explanations, labels, or reasoning."
)

# all 模式：给出全部三元组（无类型提示），让 LLM 自行决定改哪里
PROMPT_ALL = (
    "You are generating a hard negative caption for vision-language training.\n\n"
    "The caption below is CORRECT. Rewrite it into a plausible but WRONG caption by "
    "introducing one small factual mistake.\n\n"
    "CORRECT caption:\n"
    '"{caption}"\n\n'
    "Semantic relations extracted from the caption:\n"
    "{all_relations}\n\n"
    "The relations above describe the structure of the original caption.\n"
    "Use them only as optional hints to understand the sentence and to identify "
    "a suitable place where a small semantic mistake could be introduced.\n\n"
    "IMPORTANT:\n"
    "You are free to decide which relation or fact is most suitable to modify.\n"
    "You do NOT need to modify every relation, and you do NOT need to mechanically "
    "swap, reverse, reorder, or replace words according to the listed relations.\n\n"
    "Your goal is to produce a rewritten caption that:\n\n"
    "1. Contains one clear factual or semantic mistake relative to the correct "
    "caption.\n\n"
    "2. Stays as close as reasonably possible to the original caption in wording, "
    "sentence structure, grammar, and length.\n\n"
    "3. Makes only a small, localized semantic change rather than rewriting the "
    "whole sentence.\n\n"
    "4. Remains a natural and grammatically correct English sentence that a person "
    "could realistically use to describe an image.\n\n"
    "5. Preserves words, entities, phrases, and relations that do not need to change.\n\n"
    "6. Never sacrifices grammaticality or semantic plausibility merely to follow "
    "one of the listed relations.\n\n"
    "7. Never creates an error by mechanically swapping individual words.\n\n"
    "8. Treats meaningful compound and multi-word expressions as indivisible units "
    "whenever appropriate, for example:\n"
    '   "cast iron", "close-up", "traffic light", "fire hydrant",\n'
    '   "tennis court", "double faucet".\n'
    "   Do not split, reorder, or recombine their internal words merely to create "
    "an error.\n\n"
    "9. If one listed relation cannot be changed naturally, simply ignore it and "
    "consider another relation or another nearby factual detail.\n\n"
    "10. Do not merely paraphrase the original caption. The rewritten caption must "
    "contain a genuine semantic error.\n\n"
    "11. You MUST change the caption. Never output the original caption unchanged.\n\n"
    "Before answering, internally check:\n"
    "- Is the rewritten sentence natural and grammatical?\n"
    "- Is it genuinely different in meaning?\n"
    "- Does it contain a real factual mistake?\n"
    "- Is the edit small and localized?\n"
    "- Did I accidentally break a compound or multi-word expression?\n\n"
    "Output ONLY the rewritten caption.\n"
    "Do not provide quotes, explanations, labels, or reasoning."
)

# none 模式：不给出任何三元组，让 LLM 自行决定改动位置
PROMPT_NONE = (
    "You are generating a hard negative caption for vision-language training.\n\n"
    "The caption below is CORRECT. Rewrite it into a plausible but WRONG caption by "
    "introducing one small factual mistake.\n\n"
    "CORRECT caption:\n"
    '"{caption}"\n\n'
    "Decide for yourself which fact in the caption can be changed most naturally "
    "to create a small but genuine semantic error.\n\n"
    "Your goal is to produce a rewritten caption that:\n\n"
    "1. Contains one clear factual or semantic mistake relative to the correct "
    "caption.\n\n"
    "2. Stays as close as reasonably possible to the original caption in wording, "
    "sentence structure, grammar, and length.\n\n"
    "3. Makes only a small, localized semantic change rather than rewriting the "
    "whole sentence.\n\n"
    "4. Remains a natural and grammatically correct English sentence that a person "
    "could realistically use to describe an image.\n\n"
    "5. Preserves words, entities, phrases, and relations that do not need to change.\n\n"
    "6. Never sacrifices grammaticality or semantic plausibility merely to create "
    "an error.\n\n"
    "7. Never creates an error by mechanically swapping individual words.\n\n"
    "8. Treats meaningful compound and multi-word expressions as indivisible units "
    "whenever appropriate, for example:\n"
    '   "cast iron", "close-up", "traffic light", "fire hydrant",\n'
    '   "tennis court", "double faucet".\n'
    "   Do not split, reorder, or recombine their internal words merely to create "
    "an error.\n\n"
    "9. Choose the most natural location in the caption where a small factual error "
    "can be introduced.\n\n"
    "10. Do not merely paraphrase the original caption. The rewritten caption must "
    "contain a genuine semantic error.\n\n"
    "11. You MUST change the caption. Never output the original caption unchanged.\n\n"
    "Before answering, internally check:\n"
    "- Is the rewritten sentence natural and grammatical?\n"
    "- Is it genuinely different in meaning?\n"
    "- Does it contain a real factual mistake?\n"
    "- Is the edit small and localized?\n"
    "- Did I accidentally break a compound or multi-word expression?\n\n"
    "Output ONLY the rewritten caption.\n"
    "Do not provide quotes, explanations, labels, or reasoning."
)


def build_prompt(caption, candidate, triple_mode="topk"):
    """按 triple_mode（topk/all/none）选择 prompt 模板并填充。"""
    if triple_mode == "none":
        return PROMPT_NONE.format(caption=caption)
    if triple_mode == "all":
        return PROMPT_ALL.format(
            caption=caption,
            all_relations=_triples_text(candidate.get("triples", [])),
        )
    return PROMPT_TOPK.format(
        caption=caption,
        relations=_triples_text(candidate.get("triples", [])),
        ctype_hint=get_ctype_hint(candidate["type"]),
    )


# ---------------- 本地 LLM（Qwen2.5-1.5B-Instruct） ----------------

def load_local_model(model_path=LOCAL_MODEL_PATH, device="cuda"):
    """加载本地 Qwen2.5-Instruct（纯文本）与 tokenizer。"""
    use_cuda = device == "cuda" and torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
    )
    model.eval()
    return model, tokenizer


def local_generate(prompt, model, tokenizer, max_new_tokens=64, temperature=0.8):
    """本地推理，返回生成的单个负文本。"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer([text], return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------- DeepSeek API ----------------

def deepseek_generate(prompt, api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL,
                      model=DEFAULT_MODEL, max_new_tokens=64, temperature=0.8,
                      thinking=False):
    """调用 DeepSeek chat/completions，返回生成的单个负文本。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        # deepseek-v4-pro 默认开启思考模式；显式关闭，避免推理链吞掉 max_tokens
        # 导致 content 为空。正确参数是 thinking.type=disabled，而非 enable_thinking。
        "thinking": {"type": "enabled" if thinking else "disabled"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = base_url.rstrip("/") + "/chat/completions"

    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            if not content:
                raise RuntimeError(
                    f"DeepSeek 返回空 content，原始响应: "
                    f"{json.dumps(data, ensure_ascii=False)[:500]}")
            return content.strip()
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(2 ** attempt)

    raise RuntimeError(f"DeepSeek 调用失败: {last_err}")


# ---------------- 主流程 ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=TARGETS_JSONL)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--triple_mode", default="topk",
                        choices=["topk", "all", "none"],
                        help="topk=Top-k三元组+类型提示；all=全部三元组；none=不给三元组")
    parser.add_argument("--select_mode", default="topk", choices=["topk", "random"],
                        help="topk=按 selection_score 取 Top-k；random=随机取 k 个三元组（对照组，不用 ImageRouter 分数）")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="select_mode=random 时的随机种子，保证对照组三元组选择可复现")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--backend", default="local", choices=["local", "deepseek"],
                        help="生成后端：local=本地 Qwen2.5，deepseek=API")
    parser.add_argument("--local_model_path", default=LOCAL_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--api_key", default=DEFAULT_API_KEY)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking", action="store_true",
                        help="开启思考模式（默认关闭，直接输出单句）")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="每次请求间隔秒数，用于避开限流")
    parser.add_argument("--output_jsonl", default=OUTPUT_JSONL)
    parser.add_argument("--print_prompt", action="store_true",
                        help="打印每个样本真正给模型的 prompt 文本")
    args = parser.parse_args()

    if args.select_mode == "random":
        random.seed(args.random_seed)
        print(f"select_mode=random，随机种子={args.random_seed}")

    if args.backend == "local":
        print(f"backend: local | model: {args.local_model_path} | device: {args.device}")
    else:
        if not args.api_key:
            parser.error("--backend deepseek requires --api_key or DEEPSEEK_API_KEY")
        print(f"backend: deepseek | model: {args.model} | base_url: {args.base_url}")
    print(f"temperature: {args.temperature} | max_new_tokens: {args.max_new_tokens}")

    output_jsonl = args.output_jsonl
    ckpt_path = output_jsonl + ".checkpoint"

    # ---- 加载生成后端 ----
    local_model = None
    local_tokenizer = None
    if args.backend == "local":
        local_model, local_tokenizer = load_local_model(
            args.local_model_path, args.device)

    # Accept either typed candidates from build_candidates or ranked triples directly.
    with open(args.input, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    samples = []
    for rec in records:
        triples = rec.get("triples", [])
        if args.triple_mode == "topk":
            existing_candidates = rec.get("candidates") or []
            if existing_candidates:
                if args.select_mode == "random":
                    cands = random.sample(
                        existing_candidates,
                        min(args.top_k, len(existing_candidates)),
                    )
                else:
                    cands = existing_candidates[: args.top_k]
            else:
                if not triples:
                    continue
                chosen = (
                    random.sample(triples, min(args.top_k, len(triples)))
                    if args.select_mode == "random"
                    else triples[: args.top_k]
                )
                cands = [
                    {
                        "type": infer_ctype(triple),
                        "triples": [{key: triple[key] for key in ("src", "rel", "dst")}],
                        "candidate_score": triple.get("selection_score"),
                        "first_feasible_rank": triple.get("rank"),
                    }
                    for triple in chosen
                ]
        elif args.triple_mode == "all":
            if not triples:
                continue
            cands = [{
                "type": "ALL_TRIPLES",
                "triples": [{"src": t["src"], "rel": t["rel"], "dst": t["dst"]}
                            for t in triples],
                "candidate_score": None,
                "first_feasible_rank": None,
            }]
        else:  # none
            cands = [{
                "type": "NO_TRIPLES",
                "triples": [],
                "candidate_score": None,
                "first_feasible_rank": None,
            }]
        samples.append({
            "image_path": rec["image_path"],
            "caption": rec.get("caption", ""),
            "candidates": cands,
        })

    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if args.triple_mode == "topk":
        desc = f"每样本 Top-{args.top_k} candidate"
    elif args.triple_mode == "all":
        desc = "每样本 1 条（全部三元组）"
    else:
        desc = "每样本 1 条（无三元组）"
    print(f"待生成样本数: {len(samples)}（{desc}）")

    # ---- 断点续传 ----
    done_set = set()
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            done_set = {line.strip() for line in f if line.strip()}
        if done_set:
            print(f"断点续传: 已完成 {len(done_set)} 个样本，将跳过")

    out_f = open(output_jsonl, "a", encoding="utf-8")
    ckpt_f = open(ckpt_path, "a", encoding="utf-8")

    n_done = 0
    n_skip = 0

    try:
        for i, s in enumerate(samples):
            img_path = s["image_path"]
            caption = s["caption"]

            if img_path in done_set:
                n_skip += 1
                continue

            for cidx, cand in enumerate(s["candidates"]):
                prompt = build_prompt(caption, cand, args.triple_mode)

                if args.print_prompt:
                    print("\n" + "=" * 70)
                    print(f"[Prompt] sample #{i+1} | type={cand['type']}")
                    print(f"  image: {os.path.basename(img_path)}")
                    print("-" * 70)
                    print(prompt)
                    print("=" * 70)

                if args.backend == "local":
                    negative = local_generate(
                        prompt, local_model, local_tokenizer,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                else:
                    negative = deepseek_generate(
                        prompt,
                        api_key=args.api_key,
                        base_url=args.base_url,
                        model=args.model,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        thinking=args.thinking,
                    )
                negative = negative.strip().strip('"').strip("'")

                out_f.write(json.dumps({
                    "image_path": img_path,
                    "positive": caption,
                    "candidate_type": cand["type"],
                    "candidate_score": cand.get("candidate_score"),
                    "candidate_first_feasible_rank": cand.get("first_feasible_rank"),
                    "candidate_triples": [
                        {"src": t["src"], "rel": t["rel"], "dst": t["dst"]}
                        for t in cand.get("triples", [])
                    ],
                    "negative": negative,
                    "generator": args.local_model_path if args.backend == "local" else args.model,
                    "temperature": args.temperature,
                }, ensure_ascii=False) + "\n")
                out_f.flush()

            ckpt_f.write(img_path + "\n")
            ckpt_f.flush()
            n_done += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

            if n_done % 20 == 0 or n_done == len(samples):
                print(f"  [{n_done}/{len(samples)}] 完成 (skip={n_skip})")
                if not args.print_prompt:
                    print(f"      pos: {caption}")
                    print(f"      neg: {negative}")
    finally:
        out_f.close()
        ckpt_f.close()

    print(f"\n完成: 生成 {n_done} 个样本，跳过 {n_skip} 个")
    print(f"输出: {output_jsonl}")


if __name__ == "__main__":
    main()
