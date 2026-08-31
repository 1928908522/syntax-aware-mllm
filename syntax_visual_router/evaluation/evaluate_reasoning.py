"""Evaluate Stage E reasoning on the held-out Stage E test split.

This is the project-specific test for Syntax-Route CoT training:
given image + Caption A/B, generate a short reasoning trace and parse
the final "Answer: A/B".

Example:
  python -m syntax_visual_router.evaluation.evaluate_reasoning --adapter checkpoints/cot_sft/stage_e1_epoch01.pt --input_jsonl data/stage_e_cot_sft_topk_test.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import torch
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from syntax_visual_router.training.rewards import (
    parse_key_relation,
    target_triples,
    triple_similarity,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_PATH = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
INPUT_JSONL = os.path.join("data", "stage_e_cot_sft_topk_test.jsonl")
DEFAULT_ADAPTER = os.path.join("checkpoints", "cot_sft", "stage_e1_epoch01.pt")


def read_jsonl(path, max_samples=-1, start_index=0, end_index=-1):
    rows = []
    seen = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if seen < start_index:
                seen += 1
                continue
            if end_index >= 0 and seen >= end_index:
                break
            row = json.loads(line)
            if row.get("image_path") and os.path.exists(row["image_path"]):
                rows.append(row)
            seen += 1
            if 0 < max_samples <= len(rows):
                break
    return rows


def extract_answer(text):
    if not text:
        return ""
    m = re.search(r"Answer\s*:\s*([AB])", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([AB])\b", text.strip())
    return m.group(1).upper() if m else ""


def has_required_format(text):
    text = text or ""
    return "Key relation:" in text and "Visual check:" in text and bool(extract_answer(text))


def key_relation_hit(row, gen_text):
    return key_relation_score(row, gen_text) >= 1.0 - 1e-6


def key_relation_score(row, gen_text):
    predicted = parse_key_relation(gen_text)
    return max(
        (triple_similarity(predicted, target) for target in target_triples(row)),
        default=0.0,
    )


def load_model(args, device):
    print(f"加载 Qwen-VL: {args.model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if device == "cuda":
        model = model.to(device)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.image_processor.size["longest_edge"] = args.max_pixels

    if args.adapter and args.adapter.lower() not in ("none", "off", ""):
        ckpt = torch.load(args.adapter, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {})
        lora_r = ckpt_args.get("lora_r", args.lora_r)
        lora_alpha = ckpt_args.get("lora_alpha", args.lora_alpha)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        set_peft_model_state_dict(model, ckpt["adapter_state"])
        print(f"已加载 LoRA adapter: {args.adapter}", flush=True)
    else:
        print("未加载 LoRA adapter", flush=True)
    model.eval()
    return model, processor


def generate_one(model, processor, row, device, args):
    user_msg = {
        "role": "user",
        "content": [
            {"type": "image", "image": row["image_path"]},
            {"type": "text", "text": row["prompt"]},
        ],
    }
    text = processor.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info([user_msg])
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    gen_text = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    del output_ids, inputs
    if device == "cuda":
        torch.cuda.empty_cache()
    return gen_text


def summarize(results):
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    fmt = sum(1 for r in results if r["format_ok"])
    key = sum(1 for r in results if r["key_relation_hit"])
    key_score = sum(float(r.get("key_relation_score", 0.0)) for r in results)
    print("\n" + "=" * 72)
    print("Stage E Reasoning Held-out Test")
    print("=" * 72)
    print(f"  n                 : {n}")
    print(f"  Answer Acc         : {correct}/{n} = {correct / max(n, 1) * 100:.2f}%")
    print(f"  Format OK          : {fmt}/{n} = {fmt / max(n, 1) * 100:.2f}%")
    print(f"  Key relation hit   : {key}/{n} = {key / max(n, 1) * 100:.2f}%")
    print(f"  Key relation score : {key_score / max(n, 1):.4f}")

    by_type = defaultdict(list)
    for r in results:
        by_type[r.get("candidate_type", "unknown")].append(r)
    print("\n[By candidate_type]")
    print(f"  {'type':28s} {'n':>6s} {'Acc':>8s} {'Fmt':>8s} {'KeyHit':>8s}")
    for t, rows in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        nn = len(rows)
        aa = sum(1 for r in rows if r["correct"]) / nn * 100
        ff = sum(1 for r in rows if r["format_ok"]) / nn * 100
        kk = sum(1 for r in rows if r["key_relation_hit"]) / nn * 100
        print(f"  {t:28s} {nn:6d} {aa:7.2f}% {ff:7.2f}% {kk:7.2f}%")

    by_nli = defaultdict(list)
    for r in results:
        by_nli[r.get("nli_label", "unknown")].append(r)
    print("\n[By NLI]")
    for t, rows in sorted(by_nli.items(), key=lambda kv: -len(kv[1])):
        nn = len(rows)
        aa = sum(1 for r in rows if r["correct"]) / nn * 100
        print(f"  {t:14s} {nn:6d}  Acc={aa:7.2f}%")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--input_jsonl", default=INPUT_JSONL)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1,
                        help="exclusive end index in the original jsonl; -1 means no limit")
    parser.add_argument("--max_pixels", type=int, default=313600)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    rows = read_jsonl(args.input_jsonl, args.max_samples, args.start_index, args.end_index)
    print(f"测试样本: {len(rows)}")
    if not rows:
        raise RuntimeError("没有可用测试样本")

    model, processor = load_model(args, device)
    dev = next(model.parameters()).device
    results = []
    out_f = None
    if args.output_jsonl:
        out_dir = os.path.dirname(args.output_jsonl)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_f = open(args.output_jsonl, "w", encoding="utf-8")
    with torch.no_grad():
        for i, row in enumerate(rows, 1):
            gen_text = generate_one(model, processor, row, dev, args)
            pred = extract_answer(gen_text)
            correct = pred == row.get("answer")
            rec = {
                "image_path": row["image_path"],
                "answer": row.get("answer"),
                "pred": pred,
                "correct": correct,
                "candidate_type": row.get("candidate_type"),
                "nli_label": (row.get("metrics") or {}).get("nli_label"),
                "route_score": row.get("route_score"),
                "format_ok": has_required_format(gen_text),
                "key_relation_hit": key_relation_hit(row, gen_text),
                "key_relation_score": key_relation_score(row, gen_text),
                "selected_triple": row.get("selected_triple"),
                "gen_text": gen_text,
            }
            results.append(rec)
            if out_f is not None:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
            if i % 25 == 0 or i == len(rows):
                acc = sum(1 for r in results if r["correct"]) / len(results) * 100
                print(f"  [{i}/{len(rows)}] Acc={acc:.2f}% pred={pred} gt={row.get('answer')}", flush=True)

    summarize(results)
    if out_f is not None:
        out_f.close()
        print(f"\n详细结果已保存: {args.output_jsonl}")


if __name__ == "__main__":
    main()
