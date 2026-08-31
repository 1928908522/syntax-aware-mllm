"""
Part 2: Stage B 可恢复视觉特异性排序（Recoverable Visual Specificity Ranking）。

对照 Stage_D_简明实施技术指南_更新版.md §3。

核心变化（相对旧版 raw recoverability）:
  1. 去掉所有 Route 的公共视觉成分（Teacher / Student 各自中心化）
  2. 主排序分数不再是 raw cosine，而是:
       Score_m = VisualImpact_m * max(0, CenteredStructRecov_m)
  3. raw recoverability / margin 降级为诊断指标

对每个 COCO teacher trajectory:
  1. 提取 X = V_raw [N, D]
  2. Stage B ImageRouter（只输入图像）→ K 个 student routes
  3. Hungarian 匹配 teacher triple ↔ student slot
  4. 计算 7 个指标（RawRecov / Specificity / CRoute / VImpact / CStruct / Margin / Score）
  5. 输出 JSONL（triples 按 Score 降序），供 Part 3 选 Top-k

用法:
  python -m syntax_visual_router.image_route.rank_targets --max_samples 200
"""

import os
import sys
import re
import json
import argparse

import torch
import torch.nn.functional as F
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from syntax_visual_router.image_route.hungarian_matcher import HungarianMatcher
from syntax_visual_router.image_route.image_router import ImageRouter
from syntax_visual_router.image_route.visual_projector import QwenVisualExtractor

MODEL_PATH = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
TEACHER_DIR = os.path.join("data", "teacher_trajectories")
STAGE_B_CKPT = os.path.join("checkpoints", "image_route", "router.pt")
CAPTION_JSON = os.path.join("data", "captions.json")
OUTPUT_JSONL = os.path.join("data", "ranked_triples.jsonl")


# ---------------- 基础函数 ----------------

def normalize_routes(routes: torch.Tensor) -> torch.Tensor:
    """逐行归一化为概率分布（sum=1）。"""
    routes = routes.clamp_min(0.0)
    s = routes.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return routes / s


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return float((a * b).sum().item())


def js_div(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon divergence（两个分布）。"""
    p = p.clamp_min(1e-9)
    q = q.clamp_min(1e-9)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum()
    kl_qm = (q * (q / m).log()).sum()
    return float((0.5 * (kl_pm + kl_qm)).item())


# ---------------- 加载 ----------------

def load_router(ckpt_path, dim, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("image_router", ckpt))
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    K = int(args.get("K", 8))
    L = int(args.get("L", 1))
    tau = float(args.get("tau", 0.2))
    router = ImageRouter(dim=dim, K=K, L=L, tau=tau).to(device)
    router.load_state_dict(state_dict, strict=False)
    router.eval()
    return router, K, L, tau


def load_caption_map(caption_json, min_words=4, max_words=25):
    """image_id(int) -> 最短合法 caption（与 CocoDataset 口径一致）。"""
    with open(caption_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    caption_map = {}
    for img_id, info in data.items():
        captions = [c.strip() for c in info.get("caption", [])]
        valid = [c for c in captions
                 if min_words <= len(c.split()) <= max_words]
        if not valid:
            continue
        caption_map[int(img_id)] = min(valid, key=lambda c: len(c.split()))
    return caption_map


def extract_image_id(image_path: str):
    m = re.search(r"COCO_train2014_(\d+)\.jpg", image_path)
    return int(m.group(1)) if m else None


# ---------------- 主流程 ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--teacher_dir", default=TEACHER_DIR)
    parser.add_argument("--stage_b_ckpt", default=STAGE_B_CKPT)
    parser.add_argument("--caption_json", default=CAPTION_JSON)
    parser.add_argument("--output_jsonl", default=OUTPUT_JSONL)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    index_path = os.path.join(args.teacher_dir, "index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    dim = index["dim"]
    all_files = [os.path.join(args.teacher_dir, fn) for fn in index["files"]]
    if args.max_samples > 0:
        all_files = all_files[: args.max_samples]
    print(f"评估样本数: {len(all_files)}")

    caption_map = load_caption_map(args.caption_json)

    print("加载 Qwen2-VL 视觉编码器...")
    extractor = QwenVisualExtractor(args.model_path, device=device)

    print(f"加载 Stage B Router: {args.stage_b_ckpt}")
    router, K, L, tau = load_router(args.stage_b_ckpt, dim, device)
    print(f"  K={K}, L={L}, tau={tau}")

    matcher = HungarianMatcher()

    # ---- 断点续传 ----
    ckpt_path = args.output_jsonl + ".checkpoint"
    done_files = set()
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            done_files = {line.strip() for line in f if line.strip()}
        if done_files:
            print(f"断点续传: 检测到 {len(done_files)} 个已完成样本，将跳过")

    # ---- 统计 ----
    all_score, all_vimpact, all_cstruct, all_croute, all_spec, all_raw = [], [], [], [], [], []
    n_images = 0
    n_skipped = 0
    n_triples = 0
    n_soft_fallback = 0

    out_f = open(args.output_jsonl, "a", encoding="utf-8")
    ckpt_f = open(ckpt_path, "a", encoding="utf-8")

    print(f"\n{'='*80}")
    print("Stage B 可恢复视觉特异性排序中...")
    print(f"{'='*80}")

    for i, fpath in enumerate(all_files):
        fn = os.path.basename(fpath)
        if fn in done_files:
            n_skipped += 1
            continue
        data = torch.load(fpath, map_location="cpu", weights_only=False)
        img_path = data["image_path"]
        trajectories = data["trajectories"]

        X = extractor.extract(img_path).to(dtype=torch.float32).detach()  # [N, D]
        N = X.shape[0]

        # ---- 收集所有 teacher 边（跨所有 round）----
        teacher_edges = []  # (src, rel, dst, A [N])
        for t in trajectories:
            for s in t.get("samples", []):
                A = torch.tensor(s["A_in"], device=device, dtype=torch.float32)
                # round>0 的 A_in 在 Part 1 保存时被 pad 到 N+K_prev；统一截断/补齐到 N，
                # 使每条依存边对应其原始 [N] teacher route（与 X [N,D] 对齐）
                if A.shape[0] > N:
                    A = A[:N]
                elif A.shape[0] < N:
                    A = F.pad(A, (0, N - A.shape[0]), value=0.0)
                teacher_edges.append((
                    s.get("src_text", ""),
                    s.get("relation_text", ""),
                    s.get("dst_text", ""),
                    A,
                ))

        if not teacher_edges:
            continue

        # ---- student routes（round 0，只看图像）----
        with torch.no_grad():
            rounds = router.forward(X)
        if not rounds:
            continue
        student_routes = normalize_routes(rounds[0]["routes"])   # [K, N]
        student_logits = rounds[0]["logits"]                     # [K, N]

        # ---- teacher routes 矩阵 ----
        teacher_routes = normalize_routes(
            torch.stack([e[3] for e in teacher_edges]))           # [M, N]
        M = teacher_routes.shape[0]

        # ---- Hungarian 匹配（student logits vs teacher routes）----
        matched, _ = matcher.match(student_logits, teacher_routes)  # matched[k]=m or -1
        matching = [-1] * M
        for k, m in enumerate(matched):
            if m >= 0 and m < M:
                matching[m] = k

        # ---- 中心化（去公共视觉成分）----
        teacher_mean = teacher_routes.mean(dim=0)   # [N]
        student_mean = student_routes.mean(dim=0)   # [N]
        delta_teacher = teacher_routes - teacher_mean  # [M, N]
        delta_student = student_routes - student_mean  # [K, N]

        # ---- 中心化后的 cosine 矩阵（用于 CRoute + soft fallback）----
        dt_norm = F.normalize(delta_teacher, dim=-1)
        ds_norm = F.normalize(delta_student, dim=-1)
        centered_cos = dt_norm @ ds_norm.T            # [M, K]

        # ---- raw cosine 矩阵（用于 RawRecov + Margin）----
        tr_norm = F.normalize(teacher_routes, dim=-1)
        sr_norm = F.normalize(student_routes, dim=-1)
        raw_cos = tr_norm @ sr_norm.T                 # [M, K]

        # ---- 结构特征聚合（一次矩阵乘）----
        dS_t_all = delta_teacher @ X                   # [M, D]
        dS_i_all = delta_student @ X                   # [K, D]
        common_struct = teacher_mean @ X               # [D]
        common_norm = common_struct.norm() + 1e-8

        # ---- 逐三元组打分 ----
        triples = []
        for m in range(M):
            src, rel, dst, _ = teacher_edges[m]
            k = matching[m]
            if k < 0:
                k = int(centered_cos[m].argmax().item())  # soft fallback
                n_soft_fallback += 1

            A_t = teacher_routes[m]
            dA_t = delta_teacher[m]
            dA_i = delta_student[k]

            raw_recov = float(raw_cos[m, k].item())
            specificity = js_div(A_t, teacher_mean)
            centered_route_recov = float(centered_cos[m, k].item())

            dS_t = dS_t_all[m]
            dS_i = dS_i_all[k]
            visual_impact = float(dS_t.norm().item()) / float(common_norm.item())
            centered_struct_recov = cosine(dS_t, dS_i)
            score = visual_impact * max(0.0, centered_struct_recov)

            # margin（诊断）：该 triple 与所有 student slot 的 raw cosine top1-top2
            rc = raw_cos[m]
            top1, _ = rc.max(dim=0)
            rc2 = rc.clone()
            rc2[rc2.argmax()] = -1.0
            top2 = rc2.max()
            margin = float((top1 - top2).item())

            all_score.append(score)
            all_vimpact.append(visual_impact)
            all_cstruct.append(centered_struct_recov)
            all_croute.append(centered_route_recov)
            all_spec.append(specificity)
            all_raw.append(raw_recov)
            n_triples += 1

            triples.append({
                "src": src,
                "rel": rel,
                "dst": dst,
                "raw_recoverability": round(raw_recov, 4),
                "teacher_specificity": round(specificity, 4),
                "centered_route_recoverability": round(centered_route_recov, 4),
                "visual_impact": round(visual_impact, 4),
                "centered_struct_recoverability": round(centered_struct_recov, 4),
                "margin": round(margin, 4),
                "selection_score": round(score, 6),
                "matched_slot": k,
            })

        # 按 Score 降序
        triples.sort(key=lambda x: -x["selection_score"])
        # 标注 Rank（1-based），供 Part 3 按 Top-K 顺序逐个尝试
        for rank, t in enumerate(triples, start=1):
            t["rank"] = rank

        img_id = extract_image_id(img_path)
        caption = caption_map.get(img_id, "") if img_id is not None else ""

        rec = {
            "image_path": img_path,
            "caption": caption,
            "triples": triples,
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        ckpt_f.write(fn + "\n")
        ckpt_f.flush()
        n_images += 1

        # 每 100 个样本打印该样本排序结果（仅观察）
        if n_images % 100 == 0:
            print(f"\n{'='*100}")
            print(f"[样本 #{n_images}] {os.path.basename(img_path)}")
            print(f"  caption: {caption}")
            print(f"  {'src':16s} {'rel':12s} {'dst':16s} {'RawRecov':>9s} {'Spec':>7s} "
                  f"{'CRoute':>8s} {'VImpact':>8s} {'CStruct':>8s} {'Score':>9s} {'Rank':>4s}")
            print(f"  {'-'*16} {'-'*12} {'-'*16} {'-'*9} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*4}")
            for t in triples:
                print(f"  {t['src'][:15]:16s} {t['rel'][:11]:12s} {t['dst'][:15]:16s} "
                      f"{t['raw_recoverability']:9.4f} {t['teacher_specificity']:7.4f} "
                      f"{t['centered_route_recoverability']:8.4f} {t['visual_impact']:8.4f} "
                      f"{t['centered_struct_recoverability']:8.4f} {t['selection_score']:9.5f} "
                      f"{t['rank']:4d}")
            print(f"{'='*100}")

        pct = 100.0 * (i + 1) / len(all_files)
        line = f"  [{pct:5.1f}%] {i+1}/{len(all_files)}"
        sys.stdout.write(line + "\n" if i == len(all_files) - 1 else line + "\r")
        sys.stdout.flush()

    out_f.close()
    ckpt_f.close()

    # ---- 汇总 ----
    print(f"\n{'='*80}")
    print("Stage B 可恢复视觉特异性排序 汇总")
    print(f"{'='*80}")
    print(f"  本次处理图片数: {n_images}")
    print(f"  跳过(续传)样本数: {n_skipped}")
    print(f"  三元组数(本次): {n_triples}")
    print(f"  soft fallback 次数(本次): {n_soft_fallback}")

    def stat(vals):
        return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))

    if all_score:
        m, s = stat(all_score)
        print(f"\n  selection_score            : mean={m:.5f}  std={s:.5f}")
        m, s = stat(all_vimpact)
        print(f"  visual_impact              : mean={m:.4f}  std={s:.4f}")
        m, s = stat(all_cstruct)
        print(f"  centered_struct_recoverability: mean={m:.4f}  std={s:.4f}")
        m, s = stat(all_croute)
        print(f"  centered_route_recoverability : mean={m:.4f}  std={s:.4f}")
        m, s = stat(all_spec)
        print(f"  teacher_specificity        : mean={m:.4f}  std={s:.4f}")
        m, s = stat(all_raw)
        print(f"  raw_recoverability         : mean={m:.4f}  std={s:.4f}")
    print(f"\n  输出: {args.output_jsonl}")


if __name__ == "__main__":
    main()
