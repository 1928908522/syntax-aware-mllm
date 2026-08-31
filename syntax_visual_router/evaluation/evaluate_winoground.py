"""
Winoground 评测：视觉-语言组合推理（visio-linguistic compositional reasoning）。

任务：400 组样本，每组两张图 (I0, I1) + 两个 caption (C0, C1)，两 caption 词集完全相同、仅词序不同。
      正确匹配是 C0<->I0、C1<->I1。

对生成式 VLM 的匹配分数：图像条件下 caption 的「长度归一化条件对数似然」
  s(caption, image) = (1/|c|) * Σ_t log P(c_t | image, c_<t)
                    = -mean_per_token_NLL(caption | image)
  Qwen2-VL 是自回归文本概率模型，没有 CLIP 式图文特征余弦相似度，
  因此用语言模型对 caption 的 per-token 平均对数似然作为匹配分数（越高越匹配）。
  lm_forward 返回的 outputs.loss 是 cross-entropy，transformers 默认对非 -100
  的 label 位置求平均，即已按 caption token 数做长度归一化，故 score = -loss。

指标（官方）：
  text_correct : s(C0,I0) > s(C1,I0) 且 s(C1,I1) > s(C0,I1)   （给定图选对 caption）
  image_correct: s(C0,I0) > s(C0,I1) 且 s(C1,I1) > s(C1,I0)   （给定 caption 选对图）
  group_correct: text_correct 且 image_correct
  Text/Image score = 100 * mean(text/image_correct)，随机基线 25%
  Group score     = 100 * mean(group_correct)，随机基线 16.67%

数据：D:\\ljq\\Winoground\\data\\test-00000-of-00001.parquet
字段：id, image_0, image_1, caption_0, caption_1, tag, secondary_tag, num_main_preds, collapsed_tag

用法（原始模型 / stage_d 各跑一次）：
  python -m syntax_visual_router.eval.evaluate_winoground --adapter none --output_jsonl D:\\ljq\\Winoground\\qwen_base.jsonl
  python -m syntax_visual_router.eval.evaluate_winoground --adapter c:\\daobenben\\vlm\\checkpoints\\stage_d\\stage_d_epoch01.pt --output_jsonl D:\\ljq\\Winoground\\stage_d.jsonl
"""

import os
import sys
import json
import time
import argparse
from io import BytesIO

import torch
import pyarrow.parquet as pq
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, TaskType
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from qwen_vl_utils import process_vision_info
MODEL_PATH = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
PROMPT = "Describe this image in one sentence."

WINO_PARQUET = os.path.join("data", "winoground.parquet")


def load_winoground(parquet_path):
    """读 Winoground parquet，image 字段为 {bytes, path} 结构。"""
    table = pq.read_table(parquet_path)
    samples = []
    for i in range(table.num_rows):
        def get_img(col):
            f = table[col][i].as_py()
            return f["bytes"] if isinstance(f, dict) else f
        samples.append({
            "id": table["id"][i].as_py(),
            "image_0": get_img("image_0"),
            "image_1": get_img("image_1"),
            "caption_0": table["caption_0"][i].as_py(),
            "caption_1": table["caption_1"][i].as_py(),
            "tag": table["tag"][i].as_py(),
            "collapsed_tag": table["collapsed_tag"][i].as_py(),
        })
    return samples


def encode_image(model, processor, pil_image, device):
    """对单张图做一次 ViT + projector，返回视觉特征（tuple，每个图一个 tensor）。

    同一张图会对两个 caption 各算一次 score，视觉特征与 caption 无关，
    因此只编码一次并复用，避免每组样本多做 2 次 ViT 前向。
    """
    user_msg = {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": PROMPT},
        ],
    }
    image_inputs, _ = process_vision_info([user_msg])
    text = processor.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
    tmp = processor(text=[text], images=image_inputs, return_tensors="pt", padding=False)
    pixel_values = tmp["pixel_values"].to(device)
    image_grid_thw = tmp["image_grid_thw"].to(device)
    return model.get_image_features(pixel_values, image_grid_thw).pooler_output


def caption_score(model, processor, pil_image, caption, device, cached_feats):
    """计算 s(caption, image) = 长度归一化的条件对数似然。

    直接做一次前向传播，不调用 model.generate()。只统计 caption 文本 token
    位置的 log probability，并取 per-token 平均（长度归一化）。

    排除项：system 提示、user/assistant 角色与聊天模板特殊 token、图像 token、
    前缀 PROMPT 文本、<|im_end|>、换行、padding token 均不参与平均计算。
    """
    user_msg = {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": PROMPT},
        ],
    }
    assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": caption}]}
    full_messages = [user_msg, assistant_msg]

    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False)

    image_inputs, _ = process_vision_info(full_messages)
    # padding=False：单样本不引入 padding token
    full_inputs = processor(
        text=[full_text], images=image_inputs, return_tensors="pt", padding=False)

    full_ids = full_inputs["input_ids"][0]
    caption_ids = processor.tokenizer.encode(caption, add_special_tokens=False)
    caption_len = len(caption_ids)

    # caption 位于 full_ids 末尾，其后仅跟 <|im_end|> 与换行两个特殊 token，
    # 因此可直接反推 caption 起始位置，避免为算 prompt_len 再做一次图像预处理。
    prompt_len = full_ids.shape[0] - caption_len - 2
    if prompt_len < 0 or full_ids[prompt_len:prompt_len + caption_len].tolist() != caption_ids:
        prompt_text = processor.apply_chat_template(
            [user_msg], tokenize=False, add_generation_prompt=True)
        prompt_inputs = processor(
            text=[prompt_text], images=image_inputs, return_tensors="pt", padding=False)
        prompt_len = prompt_inputs["input_ids"].shape[1]

    # label 只落在 caption 文本 token 区间 [prompt_len, prompt_len + caption_len)
    labels = full_ids.clone()
    labels[:prompt_len] = -100                    # system/user/图像/PROMPT/assistant 前缀
    labels[prompt_len + caption_len:] = -100      # <|im_end|>、换行等特殊 token

    input_ids = full_inputs["input_ids"].to(device)
    attention_mask = full_inputs["attention_mask"].to(device)
    image_grid_thw = full_inputs["image_grid_thw"].to(device)
    mm_token_type_ids = full_inputs.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        mm_token_type_ids = mm_token_type_ids.to(device)

    # 用缓存的视觉特征构造 inputs_embeds，跳过重复的 ViT 前向。
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_embeds = torch.cat(cached_feats, dim=0).to(device, inputs_embeds.dtype)
    image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    labels = labels.to(device).unsqueeze(0)

    outputs = model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        labels=labels,
    )
    # outputs.loss 是 cross-entropy，对非 -100 位置求平均（即 per-token 长度归一化）
    return -float(outputs.loss.item())


def text_correct(r):
    return r["c0_i0"] > r["c1_i0"] and r["c1_i1"] > r["c0_i1"]


def image_correct(r):
    return r["c0_i0"] > r["c0_i1"] and r["c1_i1"] > r["c1_i0"]


def group_correct(r):
    return text_correct(r) and image_correct(r)


def print_report(results, title):
    n = len(results)
    text = sum(1 for r in results if r["text_correct"]) / n if n else 0.0
    image = sum(1 for r in results if r["image_correct"]) / n if n else 0.0
    group = sum(1 for r in results if r["group_correct"]) / n if n else 0.0

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"  样本数       : {n}")
    print(f"  Text  score  : {text * 100:.2f}%  (随机基线 25%)")
    print(f"  Image score  : {image * 100:.2f}%  (随机基线 25%)")
    print(f"  Group score  : {group * 100:.2f}%  (随机基线 16.67%)")

    by_tag = {}
    for r in results:
        by_tag.setdefault(r["collapsed_tag"], []).append(r)
    print(f"\n[按 collapsed_tag 分组]（Object / Relation / Both）")
    print(f"  {'tag':12s} {'n':>5s} {'Text':>8s} {'Image':>8s} {'Group':>8s}")
    for tag in sorted(by_tag, key=lambda t: -len(by_tag[t])):
        g = by_tag[tag]
        t = sum(1 for r in g if r["text_correct"]) / len(g) * 100
        im = sum(1 for r in g if r["image_correct"]) / len(g) * 100
        gr = sum(1 for r in g if r["group_correct"]) / len(g) * 100
        print(f"  {tag:12s} {len(g):5d} {t:7.2f}% {im:7.2f}% {gr:7.2f}%")
    print("=" * 60)
    return text, image, group


def run_eval(args, samples, device):
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

    load_lora = bool(args.adapter) and args.adapter.lower() not in ("none", "off", "")
    if load_lora:
        ckpt = torch.load(args.adapter, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {})
        lora_r = ckpt_args.get("lora_r", 8)
        lora_alpha = ckpt_args.get("lora_alpha", 16)
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
        print(f"已加载 LoRA adapter: {args.adapter} (r={lora_r}, alpha={lora_alpha})")
    else:
        print("未加载 LoRA（原始模型）")
    model.eval()

    dev = next(model.parameters()).device
    if device == "cuda":
        torch.cuda.synchronize()
        print(f"[显存] 已分配 {torch.cuda.memory_allocated()/1024**2:.0f} MB / "
              f"保留 {torch.cuda.memory_reserved()/1024**2:.0f} MB / "
              f"总量 {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB")

    results = []

    with torch.no_grad():
        for idx, s in enumerate(samples, 1):
            t0 = time.time()
            try:
                pil0 = Image.open(BytesIO(s["image_0"])).convert("RGB")
                pil1 = Image.open(BytesIO(s["image_1"])).convert("RGB")
            except Exception as e:
                print(f"  [{idx}/{len(samples)}] 图片解码失败: {e}", flush=True)
                continue

            # 每张图只编码一次视觉特征，两个 caption 复用
            t_v0 = time.time()
            feats0 = encode_image(model, processor, pil0, dev)
            feats1 = encode_image(model, processor, pil1, dev)
            if device == "cuda":
                torch.cuda.synchronize()
            t_enc = time.time() - t_v0

            scores = {}
            t_c0 = time.time()
            for cname, cap in [("c0", s["caption_0"]), ("c1", s["caption_1"])]:
                scores[f"{cname}_i0"] = caption_score(model, processor, pil0, cap, dev, feats0)
                scores[f"{cname}_i1"] = caption_score(model, processor, pil1, cap, dev, feats1)
            if device == "cuda":
                torch.cuda.synchronize()
            t_cap = time.time() - t_c0

            r = {
                "id": s["id"],
                "tag": s["tag"],
                "collapsed_tag": s["collapsed_tag"],
                **scores,
                "text_correct": text_correct(scores),
                "image_correct": image_correct(scores),
                "group_correct": group_correct(scores),
            }
            results.append(r)

            dt = time.time() - t0
            acc = sum(1 for x in results if x["group_correct"]) / len(results)
            print(f"  [{idx}/{len(samples)}] 总 {dt:.1f}s (ViT {t_enc:.1f}s / 文本 {t_cap:.1f}s)，Group(累计)={acc:.3f}", flush=True)

            if device == "cuda":
                torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--adapter", default=None,
                        help="LoRA checkpoint；'none' 或留空 = 原始模型")
    parser.add_argument("--parquet", default=WINO_PARQUET)
    parser.add_argument("--max_samples", type=int, default=-1, help="-1=全部(400)")
    parser.add_argument("--max_pixels", type=int, default=313600)
    parser.add_argument("--output_jsonl", default=None, help="可选：保存每组结果")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    samples = load_winoground(args.parquet)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"Winoground 样本数: {len(samples)}")

    results = run_eval(args, samples, device)
    title = "Winoground（{}）".format(
        "LoRA: " + args.adapter
        if args.adapter and args.adapter.lower() not in ("none", "off", "")
        else "原始 Qwen-VL"
    )
    print_report(results, title)

    if args.output_jsonl:
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n详细结果已保存: {args.output_jsonl}")


if __name__ == "__main__":
    main()
