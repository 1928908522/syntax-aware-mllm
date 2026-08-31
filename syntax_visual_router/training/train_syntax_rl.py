"""Stage E2b: syntax-first group-relative policy optimization.

The primary rewards are answer correctness and parsed dependency-triple agreement.
Image Route is retained only as an optional, bounded sample-weighting ablation.
"""

import argparse
import json
import os
import sys

import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from syntax_visual_router.training.losses import group_advantages
from syntax_visual_router.training.rewards import (
    sample_confidence_weight,
    score_response,
)
from syntax_visual_router.training.train_dpo import response_logp

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_PATH = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
INPUT_JSONL = os.path.join("data", "stage_e_syntax_rl_train.jsonl")
DPO_CKPT = os.path.join("checkpoints", "syntax_dpo", "stage_e2_dpo_epoch01.pt")
OUTPUT_DIR = os.path.join("checkpoints", "syntax_rl")


def read_rows(path, max_samples=-1, sample_fraction=1.0, include_answer_only=False):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if not include_answer_only and row.get("syntax_target_source") == "answer_only":
                    continue
                if os.path.exists(row.get("image_path", "")):
                    rows.append(row)
    if 0 < sample_fraction < 1.0:
        rows = rows[: max(1, int(len(rows) * sample_fraction))]
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def load_policy(args, device):
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
    if args.init_adapter:
        ckpt = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, ckpt["adapter_state"])
        print(f"加载 adapter: {args.init_adapter}")
    model.train()
    return model


def generate_response(model, processor, row, device, max_new_tokens, temperature, top_p, top_k):
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
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def reward(row, response, args):
    return score_response(
        row,
        response,
        answer_coef=args.answer_reward_coef,
        triple_coef=args.triple_reward_coef,
        joint_coef=args.joint_reward_coef,
        format_coef=args.format_reward_coef,
        missing_answer_penalty=args.missing_answer_penalty,
        missing_triple_penalty=args.missing_triple_penalty,
        non_ascii_penalty=args.non_ascii_penalty,
    )


def compact_text(text, limit=180):
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + "..."


def save_checkpoint(policy, optimizer, args, epoch, sample_index, step, loss, path):
    torch.save({
        "adapter_state": get_peft_model_state_dict(policy),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "sample_index": sample_index,
        "step": step,
        "loss": loss,
        "args": vars(args),
    }, path)
    print(f"checkpoint -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--input_jsonl", default=INPUT_JSONL)
    parser.add_argument("--init_adapter", default=DPO_CKPT)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--include_answer_only", action="store_true",
                        help="Include rows without text-contrast triples; disabled by default.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--temperature_max", type=float, default=1.3)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--exploration_generations", type=int, default=2,
                        help="Extra high-temperature candidates when the initial group has one reward.")
    parser.add_argument("--answer_reward_coef", type=float, default=1.0)
    parser.add_argument("--triple_reward_coef", type=float, default=1.0)
    parser.add_argument("--joint_reward_coef", type=float, default=0.5)
    parser.add_argument("--format_reward_coef", type=float, default=0.2)
    parser.add_argument("--missing_answer_penalty", type=float, default=0.5)
    parser.add_argument("--missing_triple_penalty", type=float, default=0.5)
    parser.add_argument("--non_ascii_penalty", type=float, default=0.2,
                        help="Penalize mixed-language artifacts in the English reasoning format.")
    parser.add_argument("--sft_coef", type=float, default=0.1,
                        help="Anchor to the gold CoT response to prevent format drift.")
    parser.add_argument("--syntax_sample_weight_coef", type=float, default=0.25)
    parser.add_argument("--route_sample_weight_coef", type=float, default=0.0,
                        help="Optional Image Route ablation; keep 0.0 for syntax-first training.")
    parser.add_argument("--monitor_every", type=int, default=25)
    parser.add_argument("--monitor_generate_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--warn_answer_below", type=float, default=0.5)
    parser.add_argument("--warn_triple_below", type=float, default=0.2)
    parser.add_argument("--warn_format_below", type=float, default=0.8)
    parser.add_argument("--warn_update_rate_below", type=float, default=0.1)
    parser.add_argument("--warn_grad_above", type=float, default=20.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_pixels", type=int, default=313600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    rows = read_rows(
        args.input_jsonl,
        args.max_samples,
        args.sample_fraction,
        include_answer_only=args.include_answer_only,
    )
    if not rows:
        raise RuntimeError("没有可用 RL 样本")
    print(f"设备: {device}")
    print(f"Stage E2 syntax-first RL 样本: {len(rows)}")
    print(
        "奖励权重: "
        f"answer={args.answer_reward_coef} triple={args.triple_reward_coef} "
        f"joint={args.joint_reward_coef} format={args.format_reward_coef}"
    )
    print(
        "样本权重: "
        f"syntax={args.syntax_sample_weight_coef} route={args.route_sample_weight_coef} "
        f"sft_anchor={args.sft_coef}"
    )

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.image_processor.size["longest_edge"] = args.max_pixels
    policy = load_policy(args, device)
    dev = next(policy.parameters()).device
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    skipped_same_reward = 0
    skipped_bad_loss = 0
    policy_signal_count = 0
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        valid = 0
        window = []
        train_window = []
        for i, row in enumerate(rows, 1):
            responses = []
            reward_parts = []
            policy.eval()
            with torch.no_grad():
                for generation_index in range(args.num_generations):
                    fraction = generation_index / max(args.num_generations - 1, 1)
                    temperature = args.temperature + fraction * (
                        args.temperature_max - args.temperature)
                    y = generate_response(
                        policy, processor, row, dev, args.max_new_tokens,
                        temperature, args.top_p, args.top_k,
                    )
                    responses.append(y)
                    reward_parts.append(reward(row, y, args))
                initial_rewards = [part["total"] for part in reward_parts]
                if len({round(value, 6) for value in initial_rewards}) == 1:
                    for extra_index in range(args.exploration_generations):
                        temperature = args.temperature_max + 0.1 * extra_index
                        y = generate_response(
                            policy, processor, row, dev, args.max_new_tokens,
                            temperature, 1.0, args.top_k,
                        )
                        responses.append(y)
                        reward_parts.append(reward(row, y, args))
            policy.train()
            rewards = [part["total"] for part in reward_parts]
            unique_responses = len({compact_text(response, limit=10000) for response in responses})
            unique_rewards = len({round(value, 6) for value in rewards})
            adv = group_advantages(rewards).to(dev)
            has_policy_signal = not torch.allclose(adv, torch.zeros_like(adv))
            if not has_policy_signal:
                skipped_same_reward += 1
            else:
                policy_signal_count += 1
            optimizer.zero_grad()
            sample_weight = sample_confidence_weight(
                row,
                syntax_coef=args.syntax_sample_weight_coef,
                route_coef=args.route_sample_weight_coef,
            )
            loss_value = 0.0
            policy_loss_value = 0.0
            bad_loss = False
            if has_policy_signal:
                for advantage, response in zip(adv, responses):
                    logp = response_logp(policy, processor, row, response, dev)
                    term = sample_weight * (-advantage * logp) / len(responses)
                    if not torch.isfinite(term):
                        bad_loss = True
                        break
                    term.backward()
                    term_value = float(term.detach().item())
                    policy_loss_value += term_value
                    loss_value += term_value
            sft_loss_value = 0.0
            if not bad_loss:
                sft_loss = -response_logp(policy, processor, row, row["response"], dev)
                anchor_term = args.sft_coef * sft_loss
                if torch.isfinite(anchor_term):
                    anchor_term.backward()
                    sft_loss_value = float(sft_loss.detach().item())
                    loss_value += float(anchor_term.detach().item())
                else:
                    bad_loss = True
            if bad_loss:
                skipped_bad_loss += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total += loss_value
            valid += 1
            global_step += 1
            window.extend(reward_parts)
            group_reward_std = float(torch.as_tensor(rewards).std(unbiased=False).item())
            train_window.append({
                "loss": loss_value,
                "policy": policy_loss_value,
                "sft": sft_loss_value,
                "grad": float(grad_norm.item()),
                "sample_weight": sample_weight,
                "group_std": group_reward_std,
                "unique_responses": unique_responses,
                "unique_rewards": unique_rewards,
                "group_size": len(responses),
            })

            if args.monitor_every > 0 and global_step % args.monitor_every == 0:
                count = max(len(window), 1)
                mean = lambda key: sum(part[key] for part in window) / count
                train_count = max(len(train_window), 1)
                train_mean = lambda key: sum(item[key] for item in train_window) / train_count
                answer_mean = mean("answer")
                triple_mean = mean("triple")
                format_mean = mean("format")
                update_rate = policy_signal_count / max(i, 1)
                print(
                    f"[monitor] i={i}/{len(rows)} step={global_step} "
                    f"loss={train_mean('loss'):.4f} policy={train_mean('policy'):.4f} "
                    f"sft={train_mean('sft'):.4f} grad={train_mean('grad'):.3f} "
                    f"reward={mean('total'):.3f} answer={answer_mean:.3f} "
                    f"triple={triple_mean:.3f} joint={mean('joint'):.3f} "
                    f"format={format_mean:.3f} penalty={mean('penalty'):.3f} "
                    f"group_std={train_mean('group_std'):.3f} "
                    f"unique_response={train_mean('unique_responses'):.2f}/{train_mean('group_size'):.2f} "
                    f"unique_reward={train_mean('unique_rewards'):.2f} "
                    f"sample_w={train_mean('sample_weight'):.3f} policy_signal_rate={update_rate:.2%} "
                    f"skipped_same_reward={skipped_same_reward} skipped_bad_loss={skipped_bad_loss}"
                )
                warnings = []
                if answer_mean < args.warn_answer_below:
                    warnings.append(f"answer={answer_mean:.3f}")
                if triple_mean < args.warn_triple_below:
                    warnings.append(f"triple={triple_mean:.3f}")
                if format_mean < args.warn_format_below:
                    warnings.append(f"format={format_mean:.3f}")
                if train_mean("grad") > args.warn_grad_above:
                    warnings.append(f"grad={train_mean('grad'):.3f}")
                if warnings:
                    print("[WARNING possible collapse] " + " ".join(warnings))
                if update_rate < args.warn_update_rate_below:
                    print(
                        "[WARNING weak exploration] "
                        f"policy_signal_rate={update_rate:.2%} "
                        f"unique_response={train_mean('unique_responses'):.2f} "
                        f"unique_reward={train_mean('unique_rewards'):.2f}"
                    )
                window = []
                train_window = []

            if args.monitor_generate_every > 0 and global_step % args.monitor_generate_every == 0:
                best = max(range(len(rewards)), key=rewards.__getitem__)
                worst = min(range(len(rewards)), key=rewards.__getitem__)
                print(f"[monitor generation] i={i} step={global_step}")
                for label, index in (("best ", best), ("worst", worst)):
                    part = reward_parts[index]
                    print(
                        f"  {label}: reward={part['total']:.3f} "
                        f"pred={part['pred_answer']} gt={row.get('answer')} "
                        f"answer={part['answer']:.0f} triple={part['triple']:.2f} "
                        f"joint={part['joint']:.0f} format={part['format']:.2f} "
                        f"text={compact_text(responses[index])}"
                    )

            if args.save_every > 0 and global_step % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"stage_e2_syntax_rl_step{global_step:06d}.pt")
                save_checkpoint(
                    policy, optimizer, args, epoch, i, global_step,
                    total / max(valid, 1), save_path,
                )
                latest_path = os.path.join(args.output_dir, "stage_e2_syntax_rl_latest.pt")
                save_checkpoint(
                    policy, optimizer, args, epoch, i, global_step,
                    total / max(valid, 1), latest_path,
                )
        avg = total / max(valid, 1)
        print(f"\nEpoch {epoch}: rl_loss={avg:.4f}")
        save_path = os.path.join(args.output_dir, f"stage_e2_syntax_rl_epoch{epoch:02d}.pt")
        save_checkpoint(policy, optimizer, args, epoch, len(rows), global_step, avg, save_path)


if __name__ == "__main__":
    main()
