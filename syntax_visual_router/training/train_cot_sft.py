r"""Stage E1: Syntax-Route CoT SFT.

Example smoke run:
  python -m syntax_visual_router.training.train_stage_e1_cot_sft ^
    --input_jsonl data/stage_e_cot_sft_topk_train.jsonl ^
    --max_samples 64
"""

import argparse
import json
import os
import sys

import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from syntax_visual_router.training.losses import response_lm_loss

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_PATH = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
INPUT_JSONL = os.path.join("data", "stage_e_cot_sft_topk_train.jsonl")
OUTPUT_DIR = os.path.join("checkpoints", "cot_sft")


def read_rows(path, max_samples=-1, sample_fraction=1.0):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if os.path.exists(row.get("image_path", "")):
                    rows.append(row)
    if 0 < sample_fraction < 1.0:
        rows = rows[: max(1, int(len(rows) * sample_fraction))]
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def build_inputs(processor, row, device):
    user_msg = {
        "role": "user",
        "content": [
            {"type": "image", "image": row["image_path"]},
            {"type": "text", "text": row["prompt"]},
        ],
    }
    assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": row["response"]}]}
    full_messages = [user_msg, assistant_msg]
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(full_messages)
    full_inputs = processor(text=[full_text], images=image_inputs, return_tensors="pt", padding=True)
    prompt_inputs = processor(text=[prompt_text], images=image_inputs, return_tensors="pt", padding=True)
    labels = full_inputs["input_ids"][0].clone()
    labels[:prompt_inputs["input_ids"].shape[1]] = -100
    return {k: v.to(device) for k, v in full_inputs.items()}, labels.to(device).unsqueeze(0)


def forward_loss(model, processor, row, device, route_strength=0.0):
    inputs, labels = build_inputs(processor, row, device)
    outputs = model(
        input_ids=inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
        attention_mask=inputs.get("attention_mask"),
        labels=labels,
    )
    route_score = torch.tensor(float(row.get("route_score", 0.5)), device=outputs.loss.device)
    return response_lm_loss(outputs.loss, route_score, route_strength=route_strength)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--input_jsonl", default=INPUT_JSONL)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--route_strength", type=float, default=0.0,
                        help="Optional weak route-based loss weighting; 0 disables it.")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max_pixels", type=int, default=313600)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    rows = read_rows(args.input_jsonl, args.max_samples, args.sample_fraction)
    if not rows:
        raise RuntimeError("没有可用 Stage E1 样本")
    print(f"设备: {device}")
    print(f"Stage E1 样本: {len(rows)}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, trust_remote_code=True)
    if device == "cuda":
        model = model.to(device)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.image_processor.size["longest_edge"] = args.max_pixels

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, ckpt["adapter_state"])
        print(f"加载续训 LoRA: {args.resume}")
    model.train()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    os.makedirs(args.output_dir, exist_ok=True)
    dev = next(model.parameters()).device

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        total = 0.0
        valid = 0
        for i, row in enumerate(rows, 1):
            loss = forward_loss(model, processor, row, dev, args.route_strength) / args.grad_accum
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            total += float(loss.item()) * args.grad_accum
            valid += 1
            if i % args.grad_accum == 0 or i == len(rows):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
            avg = total / max(valid, 1)
            sys.stdout.write(f"  [{i}/{len(rows)}] loss={avg:.4f}\r")
            sys.stdout.flush()
        avg = total / max(valid, 1)
        print(f"\nEpoch {epoch}: loss={avg:.4f}")
        save_path = os.path.join(args.output_dir, f"stage_e1_epoch{epoch:02d}.pt")
        torch.save({
            "adapter_state": get_peft_model_state_dict(model),
            "epoch": epoch,
            "loss": avg,
            "args": vars(args),
        }, save_path)
        print(f"checkpoint -> {save_path}")


if __name__ == "__main__":
    main()
