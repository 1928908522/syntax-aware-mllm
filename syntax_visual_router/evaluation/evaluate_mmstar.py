"""
MMStar 评测：跑 1500 条多选 VQA，支持三种 setting，用于计算 Acc / MG / ML。

三种 setting（官方 MG/ML 协议）：
  S_v  带图评测（默认）：完整视觉输入，即常规 Acc。
  S_wv 盲测（--blind）：同一 LVLM 去掉图像，只给文本+选项。
  S_t  纯文本 LLM backbone（--text_llm PATH）：无多模态训练、无图像。

指标：
  Acc = S_v
  MG  = S_v - S_wv
  ML  = max(0, S_wv - S_t)

用法（先各跑一次，再用 compute_mgml.py 汇总）：
  # S_v（带图）
  python -m syntax_visual_router.eval.evaluate_mmstar \
      --adapter c:\\daobenben\\vlm\\checkpoints\\stage_d_baseline_posonly\\stage_d_baseline_epoch01.pt \
      --output_jsonl D:\\ljq\\MMStar\\baseline_sv.jsonl

  # S_wv（盲测）
  python -m syntax_visual_router.eval.evaluate_mmstar \
      --adapter c:\\daobenben\\vlm\\checkpoints\\stage_d_baseline_posonly\\stage_d_baseline_epoch01.pt \
      --blind --output_jsonl D:\\ljq\\MMStar\\baseline_swv.jsonl

  # S_t（纯文本 LLM backbone，与 --adapter 无关）
  python -m syntax_visual_router.eval.evaluate_mmstar \
      --text_llm "D:\\ljq\\models\\Qwen2.5-1.5B-Instruct\\Qwen\\Qwen2___5-1___5B-Instruct" \
      --output_jsonl D:\\ljq\\MMStar\\st.jsonl
"""

import sys
import os
import csv
import json
import base64
import argparse
from io import BytesIO

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, TaskType
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_PATH = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
MMSTAR_TSV = os.path.join("data", "MMStar.tsv")

DEFAULT_ADAPTER = None
# 注：Qwen2-VL-2B 的 LLM backbone 严格是 Qwen2-1.5B；本机仅有 Qwen2.5-1.5B，
# 用作 S_t 的近似。对 baseline vs stage_d 对比无影响（S_t 是共享常数）。
DEFAULT_TEXT_LLM = os.environ.get("TEXT_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

POST_PROMPT = "\nAnswer with the option's letter from the given choices directly"


def load_mmstar(tsv_path):
    """读 MMStar.tsv，返回样本列表。csv 模块能正确处理带引号含换行的 question。"""
    # image 字段为 base64 内嵌 JPEG，远超 csv 默认字段上限(128KB)，必须先调大。
    csv.field_size_limit(2147483647)
    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            samples.append({
                "index": row["index"],
                "question": row["question"],
                "answer": (row.get("answer") or "").strip().upper(),
                "image_b64": row["image"],
                "category": row.get("category", ""),
                "l2_category": row.get("l2_category", ""),
            })
    return samples


def b64_to_pil(b64):
    data = base64.b64decode(b64.strip())
    return Image.open(BytesIO(data)).convert("RGB")


_ANSWER_PHRASES = [
    "the answer is", "answer is", "the correct answer is", "correct answer is",
    "the best answer is", "best answer is", "the correct option is", "correct option is",
    "the best option is", "best option is", "the choice is", "choice is",
    "the correct choice is", "correct choice is", "i choose", "i select", "i pick",
    "my answer is", "my choice is",
]
_FORMAT_PRIORITY = {
    "start": 10, "end": 9, "phrase": 7, "parentheses": 6,
    "period": 5, "colon": 4, "right_paren": 3, "space": 2, "fallback": 0,
}


def extract_option(text, choices=("A", "B", "C", "D")):
    """按 lmms-eval 官方 MMStar 的 extract_mcq_answer 逻辑提取选项字母。

    覆盖 (A)/A./A:/A)/A /「the answer is X」/开头/结尾等格式，
    并优先取高优先级格式、同格式取靠后出现者，避免误匹配选项正文。
    """
    if not text or not text.strip():
        return ""
    all_choices = list(choices)
    for ch in [",", ".", "!", "?", ";", ":", "'", '"']:
        text = text.strip(ch)
    text = " " + text + " "
    candidates = []
    for ch in all_choices:
        if f"({ch})" in text:
            candidates.append((ch, text.rfind(f"({ch})"), "parentheses"))
    for ch in all_choices:
        if f"{ch}." in text:
            candidates.append((ch, text.rfind(f"{ch}."), "period"))
    for ch in all_choices:
        if f"{ch}:" in text:
            candidates.append((ch, text.rfind(f"{ch}:"), "colon"))
    for ch in all_choices:
        if f"{ch})" in text:
            candidates.append((ch, text.rfind(f"{ch})"), "right_paren"))
    for ch in all_choices:
        if f"{ch} " in text:
            candidates.append((ch, text.rfind(f"{ch} "), "space"))
    text_lower = text.lower()
    for phrase in _ANSWER_PHRASES:
        idx = text_lower.find(phrase)
        if idx != -1:
            after = idx + len(phrase)
            for ch in all_choices:
                ch_pos = text.find(ch, after)
                if ch_pos != -1:
                    candidates.append((ch, ch_pos, "phrase"))
    stripped = text.strip()
    for ch in all_choices:
        if stripped.startswith(ch) and (len(stripped) == 1 or not stripped[1].isalpha()):
            candidates.append((ch, 0, "start"))
    for ch in all_choices:
        if stripped.endswith(ch) and (len(stripped) == 1 or not stripped[-2].isalpha()):
            candidates.append((ch, len(text) - 1, "end"))
    if not candidates:
        for ch in all_choices:
            if ch in text:
                candidates.append((ch, text.rfind(ch), "fallback"))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: (_FORMAT_PRIORITY.get(x[2], 0), x[1]), reverse=True)
    return candidates[0][0]


def build_question(sample):
    """官方 MMStar prompt：question 已含选项，追加标准 post_prompt。"""
    q = (sample["question"] or "").strip()
    q = q.replace(" Please answer yes or no.", "")
    return f"{q}{POST_PROMPT}"


def _record(sample, pred, gen_text):
    return {
        "index": sample["index"],
        "category": sample["category"],
        "l2_category": sample["l2_category"],
        "answer": sample["answer"],
        "pred": pred,
        "correct": (pred == sample["answer"]),
        "gen_text": gen_text,
    }


def print_report(results, title):
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    acc = correct / total if total else 0.0
    cat_correct = {}
    cat_total = {}
    for r in results:
        c = r["category"]
        cat_total[c] = cat_total.get(c, 0) + 1
        if r["correct"]:
            cat_correct[c] = cat_correct.get(c, 0) + 1

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"  总体 Acc: {correct}/{total} = {acc:.4f}")

    print(f"\n[按 category 分组 Acc]（6 大能力）")
    for cat in sorted(cat_total, key=lambda c: -cat_total[c]):
        cc = cat_correct.get(cat, 0)
        tt = cat_total[cat]
        print(f"  {cat:32s}: {cc:4d}/{tt:4d} = {cc/tt:.4f}")
    print("=" * 70)
    return acc


def run_vl(args, samples, device):
    """S_v（带图）或 S_wv（--blind 盲测）。"""
    mode = "S_wv（盲测，无图）" if args.blind else "S_v（带图）"
    print(f"[{mode}] 加载 Qwen-VL: {args.model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    # 8GB 笔记本 GPU 下 device_map="auto" 在 WDDM 上会错误切分/offload，
    # 导致 k_proj 的 cuBLAS 执行失败。显式整体放到单块 GPU 更稳。
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
    results = []

    with torch.no_grad():
        for i, s in enumerate(samples, 1):
            question = build_question(s)

            if args.blind:
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": question},
                ]}]
                image_inputs = None
            else:
                try:
                    pil = b64_to_pil(s["image_b64"])
                except Exception as e:
                    print(f"  [{i}/{len(samples)}] 图片解码失败: {e}")
                    continue
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": question},
                ]}]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            if not args.blind:
                image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, return_tensors="pt", padding=True)
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            output_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            gen_ids = output_ids[0][input_len:]
            gen_text = processor.decode(gen_ids, skip_special_tokens=True).strip()
            # 及时释放显存，避免 8GB 显存累积导致 cuBLAS 执行失败。
            del output_ids, inputs
            if device == "cuda":
                torch.cuda.empty_cache()

            pred = extract_option(gen_text)
            results.append(_record(s, pred, gen_text))

            if i % 50 == 0 or i == len(samples):
                acc = sum(1 for r in results if r["correct"]) / len(results)
                print(f"  [{i}/{len(samples)}] Acc={acc:.3f} (pred={pred}, gt={s['answer']})")

    return results, f"MMStar {mode} ({args.adapter})"


def run_text_llm(args, samples, device):
    """S_t：纯文本 LLM backbone（无图、无多模态训练）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[S_t 纯文本 LLM backbone] 加载: {args.text_llm}")
    tokenizer = AutoTokenizer.from_pretrained(args.text_llm, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.text_llm, torch_dtype=torch.float16, trust_remote_code=True)
    if device == "cuda":
        model = model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for i, s in enumerate(samples, 1):
            question = build_question(s)
            messages = [{"role": "user", "content": question}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            output_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            gen_text = tokenizer.decode(
                output_ids[0][input_len:], skip_special_tokens=True).strip()
            del output_ids, inputs
            if device == "cuda":
                torch.cuda.empty_cache()

            pred = extract_option(gen_text)
            results.append(_record(s, pred, gen_text))

            if i % 50 == 0 or i == len(samples):
                acc = sum(1 for r in results if r["correct"]) / len(results)
                print(f"  [{i}/{len(samples)}] Acc={acc:.3f} (pred={pred}, gt={s['answer']})")

    return results, f"MMStar S_t（纯文本 LLM backbone）: {args.text_llm}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER,
                        help="LoRA checkpoint；'none' 表示原始模型（S_v/S_wv 用）")
    parser.add_argument("--mmstar_tsv", default=MMSTAR_TSV)
    parser.add_argument("--max_samples", type=int, default=-1, help="-1=全部(1500)")
    parser.add_argument("--max_pixels", type=int, default=313600)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--output_jsonl", default=None, help="可选：保存每题结果")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blind", action="store_true",
                        help="盲测：不给图像，算 S_wv")
    parser.add_argument("--text_llm", default=None,
                        help="纯文本 LLM backbone 路径，算 S_t（与 --adapter 无关）")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    samples = load_mmstar(args.mmstar_tsv)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"MMStar 样本数: {len(samples)}")

    if args.text_llm:
        results, title = run_text_llm(args, samples, device)
    else:
        results, title = run_vl(args, samples, device)

    acc = print_report(results, title)

    if args.output_jsonl:
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n详细结果已保存: {args.output_jsonl}")


if __name__ == "__main__":
    main()
