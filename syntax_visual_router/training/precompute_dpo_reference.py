"""Precompute reference logprobs for Stage E2 DPO.

This avoids loading policy and reference Qwen-VL models at the same time during DPO.

Example:
  python -m syntax_visual_router.training.precompute_stage_e2_ref_logps ^
    --input_jsonl data/stage_e_preference_topk_train.jsonl ^
    --adapter checkpoints/cot_sft/stage_e1_epoch01.pt ^
    --output_jsonl data/stage_e_preference_topk_train_ref_mean.jsonl
"""

import argparse
import json
import os
import sys

import torch
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from syntax_visual_router.training.train_dpo import response_logp

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_PATH = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
INPUT_JSONL = os.path.join("data", "stage_e_preference_topk_train.jsonl")
E1_CKPT = os.path.join("checkpoints", "cot_sft", "stage_e1_epoch01.pt")
OUTPUT_JSONL = os.path.join("data", "stage_e_preference_topk_train_ref_mean.jsonl")


def iter_rows(path, max_samples=-1):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("image_path") and os.path.exists(row["image_path"]):
                yield row
                n += 1
                if 0 < max_samples <= n:
                    break


def count_rows(path, max_samples=-1):
    n = 0
    for _ in iter_rows(path, max_samples=max_samples):
        n += 1
    return n


def load_ref_model(args, device):
    print(f"加载 reference Qwen-VL: {args.model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if device == "cuda":
        model = model.to(device)

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
    model.eval()
    print(f"已加载 reference adapter: {args.adapter}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--adapter", default=E1_CKPT)
    parser.add_argument("--input_jsonl", default=INPUT_JSONL)
    parser.add_argument("--output_jsonl", default=OUTPUT_JSONL)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_pixels", type=int, default=313600)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    total = count_rows(args.input_jsonl, args.max_samples)
    print(f"设备: {device}")
    print(f"待预计算 preference pairs: {total}")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.image_processor.size["longest_edge"] = args.max_pixels
    model = load_ref_model(args, device)
    dev = next(model.parameters()).device

    out_dir = os.path.dirname(args.output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    done = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        with torch.no_grad():
            for row in iter_rows(args.input_jsonl, args.max_samples):
                ref_chosen = response_logp(
                    model, processor, row, row["chosen"], dev, length_normalize=True)
                ref_rejected = response_logp(
                    model, processor, row, row["rejected"], dev, length_normalize=True)
                row["ref_chosen_logp_mean"] = float(ref_chosen.item())
                row["ref_rejected_logp_mean"] = float(ref_rejected.item())
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"  [{done}/{total}] ref_chosen_mean={row['ref_chosen_logp_mean']:.3f} "
                          f"ref_rejected_mean={row['ref_rejected_logp_mean']:.3f}", flush=True)
                if device == "cuda":
                    torch.cuda.empty_cache()

    print(f"\n完成: {done} -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
