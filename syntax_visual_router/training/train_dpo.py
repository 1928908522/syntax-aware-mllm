"""Stage E2a: Standard reward-weighted DPO for Syntax-Route CoT."""

import argparse
import json
import os
import re
import sys

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from syntax_visual_router.training.losses import dpo_loss, reward_weight

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_PATH = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
INPUT_JSONL = os.path.join("data", "stage_e_preference_topk_train.jsonl")
E1_CKPT = os.path.join("checkpoints", "cot_sft", "stage_e1_epoch01.pt")
OUTPUT_DIR = os.path.join("checkpoints", "syntax_dpo")


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


def rows_have_precomputed_ref(rows):
    return bool(rows) and all(
        ("ref_chosen_logp_mean" in r and "ref_rejected_logp_mean" in r)
        or ("ref_chosen_logp" in r and "ref_rejected_logp" in r)
        for r in rows
    )


def get_ref_logps(row, device):
    if "ref_chosen_logp_mean" in row and "ref_rejected_logp_mean" in row:
        return (
            torch.tensor(float(row["ref_chosen_logp_mean"]), device=device),
            torch.tensor(float(row["ref_rejected_logp_mean"]), device=device),
        )
    return (
        torch.tensor(float(row["ref_chosen_logp"]), device=device),
        torch.tensor(float(row["ref_rejected_logp"]), device=device),
    )


def extract_answer(text):
    m = re.search(r"Answer\s*:\s*([AB])", text or "", re.I)
    return m.group(1).upper() if m else ""


def format_ok(text):
    text = text or ""
    return (
        "Key relation:" in text
        and "Visual check:" in text
        and bool(extract_answer(text))
    )


def compact_text(text, limit=180):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def generate_monitor_response(model, processor, row, device, max_new_tokens):
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
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def print_generation_monitor(model, processor, rows, device, args, global_step):
    if args.monitor_generate_every <= 0:
        return
    if global_step <= 0 or global_step % args.monitor_generate_every != 0:
        return
    model.eval()
    ok = 0
    with torch.no_grad():
        print(f"\n[monitor generation] step={global_step}", flush=True)
        for idx, row in enumerate(rows[: max(1, args.monitor_samples)], 1):
            text = generate_monitor_response(
                model, processor, row, device, args.monitor_max_new_tokens)
            pred = extract_answer(text)
            fmt = format_ok(text)
            ok += int(fmt)
            print(
                f"  sample{idx}: pred={pred or '<empty>'} gt={row.get('answer')} "
                f"format_ok={fmt} text={compact_text(text)}",
                flush=True,
            )
    model.train()
    print(f"  generation_format_ok={ok}/{max(1, args.monitor_samples)}", flush=True)


def preference_weight(row, args, device):
    if args.reward_weight_mode == "none":
        return None
    delta = (row.get("reward_info") or {}).get("reward_delta", 1.0)
    base = reward_weight(delta).to(device)
    if args.reward_weight_strength <= 0:
        return torch.ones((), device=device)
    return 1.0 + float(args.reward_weight_strength) * (base - 1.0)


def build_inputs(processor, row, response, device):
    user_msg = {
        "role": "user",
        "content": [
            {"type": "image", "image": row["image_path"]},
            {"type": "text", "text": row["prompt"]},
        ],
    }
    assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": response}]}
    full_messages = [user_msg, assistant_msg]
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(full_messages)
    full_inputs = processor(text=[full_text], images=image_inputs, return_tensors="pt", padding=True)
    prompt_inputs = processor(text=[prompt_text], images=image_inputs, return_tensors="pt", padding=True)
    labels = full_inputs["input_ids"][0].clone()
    labels[:prompt_inputs["input_ids"].shape[1]] = -100
    return {k: v.to(device) for k, v in full_inputs.items()}, labels.to(device).unsqueeze(0)


def build_batch_inputs(processor, rows, responses, device):
    """Build a padded multimodal batch; labels are active only on response tokens."""
    full_texts = []
    prompt_texts = []
    batch_messages = []
    for row, response in zip(rows, responses):
        user_msg = {
            "role": "user",
            "content": [
                {"type": "image", "image": row["image_path"]},
                {"type": "text", "text": row["prompt"]},
            ],
        }
        assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": response}]}
        full_messages = [user_msg, assistant_msg]
        full_texts.append(processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False))
        prompt_texts.append(processor.apply_chat_template(
            [user_msg], tokenize=False, add_generation_prompt=True))
        batch_messages.append(full_messages)

    image_inputs, _ = process_vision_info(batch_messages)
    full_inputs = processor(text=full_texts, images=image_inputs, return_tensors="pt", padding=True)
    prompt_inputs = processor(text=prompt_texts, images=image_inputs, return_tensors="pt", padding=True)

    labels = full_inputs["input_ids"].clone()
    prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()
    for b, plen in enumerate(prompt_lens):
        labels[b, :int(plen)] = -100
    if "attention_mask" in full_inputs:
        labels = labels.masked_fill(full_inputs["attention_mask"].eq(0), -100)
    return {k: v.to(device) for k, v in full_inputs.items()}, labels.to(device)


def _masked_sequence_logp(logits, labels, length_normalize=True):
    logits = logits[:, :-1, :]
    target = labels[:, 1:]
    mask = target.ne(-100)
    safe_target = target.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    seq_logp = (token_logp * mask).sum(dim=1)
    if length_normalize:
        seq_logp = seq_logp / mask.sum(dim=1).clamp_min(1)
    return seq_logp


def response_logp(model, processor, row, response, device, length_normalize=True):
    inputs, labels = build_inputs(processor, row, response, device)
    outputs = model(
        input_ids=inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
        attention_mask=inputs.get("attention_mask"),
    )
    return _masked_sequence_logp(
        outputs.logits, labels, length_normalize=length_normalize)[0]


def batch_response_logps(model, processor, rows, responses, device, length_normalize=True):
    inputs, labels = build_batch_inputs(processor, rows, responses, device)
    outputs = model(
        input_ids=inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
        attention_mask=inputs.get("attention_mask"),
    )
    return _masked_sequence_logp(
        outputs.logits, labels, length_normalize=length_normalize)


def pair_logps(model, processor, row, device, length_normalize=True):
    """Compute chosen/rejected logprobs in one policy forward."""
    logps = batch_response_logps(
        model, processor, [row, row], [row["chosen"], row["rejected"]], device,
        length_normalize=length_normalize,
    )
    return logps[0], logps[1]


def load_lora_model(args, adapter_path, device, trainable):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, trust_remote_code=True)
    if device == "cuda":
        model = model.to(device)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if adapter_path:
        ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, ckpt["adapter_state"])
        print(f"加载 adapter: {adapter_path}")
    model.train(trainable)
    for p in model.parameters():
        p.requires_grad_(trainable and p.requires_grad)
    return model


def save_checkpoint(policy, optimizer, args, epoch, global_step, next_index, avg, suffix):
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, suffix)
    torch.save({
        "adapter_state": get_peft_model_state_dict(policy),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "next_index": next_index,
        "loss": avg,
        "args": vars(args),
    }, save_path)
    print(f"\ncheckpoint -> {save_path}", flush=True)
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--input_jsonl", default=INPUT_JSONL)
    parser.add_argument("--init_adapter", default=E1_CKPT)
    parser.add_argument("--reference_adapter", default=E1_CKPT)
    parser.add_argument("--resume", default=None,
                        help="Resume from a DPO checkpoint saved by this script.")
    parser.add_argument("--use_precomputed_ref", action="store_true",
                        help="Use ref_chosen_logp/ref_rejected_logp fields from input_jsonl and do not load a reference model.")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=500,
                        help="Save a rolling checkpoint every N preference pairs; <=0 disables.")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum preference pairs to process in this run; -1 means all remaining.")
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--sft_coef", type=float, default=0.2,
                        help="Chosen-response SFT anchor weight to preserve CoT format during DPO.")
    parser.add_argument("--reward_weight_mode", choices=["none", "delta"], default="none",
                        help="Use no preference weight by default; delta uses reward_info.reward_delta as a weak optional signal.")
    parser.add_argument("--reward_weight_strength", type=float, default=0.25,
                        help="Interpolate reward weight toward 1.0; only used when --reward_weight_mode delta.")
    parser.add_argument("--max_ref_gap", type=float, default=3.0,
                        help="Skip pairs where normalized ref chosen-rejected gap is already too large; <=0 disables.")
    parser.add_argument("--monitor_every", type=int, default=100,
                        help="Print rolling numeric training diagnostics every N valid preference pairs.")
    parser.add_argument("--monitor_generate_every", type=int, default=500,
                        help="Generate fixed monitor samples every N valid preference pairs; <=0 disables.")
    parser.add_argument("--monitor_samples", type=int, default=2,
                        help="Number of fixed training prompts to generate for monitor output.")
    parser.add_argument("--monitor_max_new_tokens", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_pixels", type=int, default=313600)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    rows = read_rows(args.input_jsonl, args.max_samples, args.sample_fraction)
    if not rows:
        raise RuntimeError("没有可用 preference 样本")
    print(f"设备: {device}")
    print(f"Stage E2 DPO 偏好对: {len(rows)}")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.image_processor.size["longest_edge"] = args.max_pixels
    adapter_to_load = args.resume or args.init_adapter
    policy = load_lora_model(args, adapter_to_load, device, trainable=True)
    use_precomputed_ref = args.use_precomputed_ref or rows_have_precomputed_ref(rows)
    if use_precomputed_ref:
        ref = None
        print("使用预计算 reference logprobs，不加载 reference 模型")
    else:
        ref = load_lora_model(args, args.reference_adapter, device, trainable=False)
        ref.eval()
    dev = next(policy.parameters()).device
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    os.makedirs(args.output_dir, exist_ok=True)

    start_epoch = 1
    start_index = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(dev)
        start_epoch = int(ckpt.get("epoch", 1))
        start_index = int(ckpt.get("next_index", 0))
        global_step = int(ckpt.get("global_step", 0))
        print(f"续训 DPO: {args.resume}")
        print(f"  epoch={start_epoch}, next_index={start_index}, global_step={global_step}")
        del ckpt

    for epoch in range(start_epoch, start_epoch + args.epochs):
        optimizer.zero_grad()
        total = 0.0
        valid = 0
        processed_this_run = 0
        skipped_ref_gap = 0
        skipped_bad_loss = 0
        mon_count = 0
        mon_loss = 0.0
        mon_pref = 0.0
        mon_sft = 0.0
        mon_pi_gap = 0.0
        mon_ref_gap = 0.0
        for i0, row in enumerate(rows[start_index:], start=start_index):
            i = i0 + 1
            pc, pr = pair_logps(policy, processor, row, dev, length_normalize=True)
            if use_precomputed_ref:
                rc, rr = get_ref_logps(row, dev)
            else:
                with torch.no_grad():
                    rc = response_logp(ref, processor, row, row["chosen"], dev, length_normalize=True)
                    rr = response_logp(ref, processor, row, row["rejected"], dev, length_normalize=True)
            if args.max_ref_gap > 0 and float((rc - rr).detach().item()) > args.max_ref_gap:
                skipped_ref_gap += 1
                continue
            weight = preference_weight(row, args, dev)
            preference_loss = dpo_loss(pc, pr, rc, rr, beta=args.dpo_beta, weight=weight)
            sft_anchor_loss = -pc
            loss = (preference_loss + args.sft_coef * sft_anchor_loss) / args.grad_accum
            if torch.isnan(loss) or torch.isinf(loss):
                skipped_bad_loss += 1
                continue
            loss.backward()
            raw_loss = float(loss.item()) * args.grad_accum
            pref_value = float(preference_loss.detach().item())
            sft_value = float(sft_anchor_loss.detach().item())
            pi_gap_value = float((pc - pr).detach().item())
            ref_gap_value = float((rc - rr).detach().item())
            total += raw_loss
            valid += 1
            processed_this_run += 1
            global_step += 1
            mon_count += 1
            mon_loss += raw_loss
            mon_pref += pref_value
            mon_sft += sft_value
            mon_pi_gap += pi_gap_value
            mon_ref_gap += ref_gap_value
            if valid % args.grad_accum == 0 or i == len(rows):
                torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
            avg = total / max(valid, 1)
            if args.monitor_every > 0 and global_step % args.monitor_every == 0:
                denom = max(mon_count, 1)
                print(
                    f"\n[monitor] i={i}/{len(rows)} step={global_step} "
                    f"loss={mon_loss / denom:.4f} pref={mon_pref / denom:.4f} "
                    f"sft={mon_sft / denom:.4f} pi_gap={mon_pi_gap / denom:.4f} "
                    f"ref_gap={mon_ref_gap / denom:.4f} "
                    f"skipped_ref_gap={skipped_ref_gap} skipped_bad_loss={skipped_bad_loss}",
                    flush=True,
                )
                mon_count = 0
                mon_loss = 0.0
                mon_pref = 0.0
                mon_sft = 0.0
                mon_pi_gap = 0.0
                mon_ref_gap = 0.0
            if args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(
                    policy, optimizer, args, epoch, global_step, i, avg,
                    "stage_e2_dpo_latest.pt",
                )
            print_generation_monitor(policy, processor, rows, dev, args, global_step)
            sys.stdout.write(
                f"  [{i}/{len(rows)}] step={global_step} dpo_loss={avg:.4f}\r"
            )
            sys.stdout.flush()
            if args.max_steps > 0 and processed_this_run >= args.max_steps:
                avg = total / max(valid, 1)
                save_checkpoint(
                    policy, optimizer, args, epoch, global_step, i, avg,
                    f"stage_e2_dpo_step{global_step:06d}.pt",
                )
                print(f"\n达到 --max_steps={args.max_steps}，本次运行停止。")
                return
        start_index = 0
        avg = total / max(valid, 1)
        print(f"\nEpoch {epoch}: dpo_loss={avg:.4f}")
        save_path = os.path.join(args.output_dir, f"stage_e2_dpo_epoch{epoch:02d}.pt")
        torch.save({
            "adapter_state": get_peft_model_state_dict(policy),
            "epoch": epoch,
            "loss": avg,
            "args": vars(args),
        }, save_path)
        print(f"checkpoint -> {save_path}")


if __name__ == "__main__":
    main()
