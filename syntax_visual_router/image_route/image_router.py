"""
Image Router V2 —— 图像条件化的 Cosine Routing 模块。

核心: K 个可学习 Base Query + 图像条件偏移 → 每个样本动态产生 query。
      S = A @ X（不加 MLP、不做减法），保持 routing lineage 可追踪。

架构:
  X [N, D] → mean → img_repr [D] → Linear → offset [D]
  Q_k = base_k + offset      [K, D] — 图像相关的
  X̄ = L2-norm(X), Q̄ = L2-norm(Q)
  A = softmax(Q̄ @ X̄ᵀ / τ)    [K, N]
  S = A @ X                    [K, D] — 用原始 X，非 norm 后的 X̄

删除了旧版的:
  - Sinkhorn normalization
  - GRU-based Slot Attention iterative update
  - A_in / A_out 双路设计
  - relation classifier
  - S = v_in - v_out
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ImageRouter(nn.Module):
    """图像条件化 Image Router。

    Q = base_queries + query_proj(mean(X))
    每张图产生各自的 query 方向，K 个 base query 提供跨图像共享的知识。
    """

    def __init__(
        self,
        dim: int = 1536,
        K: int = 8,
        L: int = 1,
        tau: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.K = K
        self.L = L
        self.tau = tau

        # K 个跨图像共享的 Base Query（小初始化）
        self.base_queries = nn.Parameter(torch.randn(K, dim) * 0.02)

        # 图像条件投影: mean(X) → 每个 query 独立的 offset [K, dim]
        # 避免所有 K 个 query 共享同一个方向导致塌缩
        self.query_proj = nn.Linear(dim, K * dim)
        # 小初始化，让 offset 从接近 0 开始
        nn.init.xavier_uniform_(self.query_proj.weight, gain=0.01)
        nn.init.zeros_(self.query_proj.bias)

    def set_tau(self, tau: float):
        self.tau = tau

    def _compute_queries(self, token_bank: torch.Tensor) -> torch.Tensor:
        """图像条件化: base + proj(mean(X)) → L2-norm。

        Args:
            token_bank: [B, N, dim] 或 [N, dim]
        Returns:
            q_norm: [B, K, dim] 或 [K, dim], L2 normalized
        """
        has_batch = token_bank.dim() == 3

        if has_batch:
            # [B, N, dim] → [B, dim]
            img_repr = token_bank.mean(dim=1)
            # [B, dim] → [B, K, dim]（每个 query 独立 offset）
            offset = self.query_proj(img_repr).view(-1, self.K, self.dim)
            # base [K, dim] + offset [B, K, dim] → [B, K, dim]
            queries = self.base_queries.unsqueeze(0) + offset
            q_norm = F.normalize(queries, dim=-1)  # [B, K, dim]
        else:
            # [N, dim] → [dim]
            img_repr = token_bank.mean(dim=0)
            # [dim] → [K, dim]（每个 query 独立 offset）
            offset = self.query_proj(img_repr).view(self.K, self.dim)
            queries = self.base_queries + offset  # [K, dim]
            q_norm = F.normalize(queries, dim=-1)

        return q_norm

    def forward(
        self,
        token_bank: torch.Tensor,
        teacher_S: Optional[List[torch.Tensor]] = None,
        teacher_p: float = 0.0,
    ) -> List[dict]:
        """L 轮递归路由。

        Args:
            token_bank: [N, dim] 或 [B, N, dim] 视觉 token bank
            teacher_S: 可选，每轮 teacher 的 S [K_teacher, dim]，用于 teacher forcing
            teacher_p: teacher forcing 概率 (0=纯推理, 1=纯 teacher)

        Returns:
            rounds: List[dict], 每轮 {"routes": [K, N_total], "struct_tokens": [K, dim], "logits": [K, N_total]}
        """
        if token_bank.dim() == 2:
            token_bank = token_bank.unsqueeze(0)  # [1, N, dim]
            squeeze = True
        else:
            squeeze = False

        B, N, dim = token_bank.shape
        device = token_bank.device
        all_rounds = []

        for round_idx in range(self.L):
            # ============================================================
            # 1. 图像条件化 Query + L2-norm
            # ============================================================
            q_norm = self._compute_queries(token_bank)       # [B, K, dim] or [K, dim]

            # X L2-norm（仅用于 routing）
            x_norm = F.normalize(token_bank, dim=-1)          # [B, N_total, dim]

            # ============================================================
            # 2. Cosine routing logits
            # ============================================================
            if q_norm.dim() == 3:
                # batch mode: [B, K, dim] @ [B, N, dim]ᵀ → [B, K, N]
                logits = torch.bmm(q_norm, x_norm.transpose(1, 2))
            else:
                # single sample: [K, dim] @ [N, dim]ᵀ → [K, N]
                logits = torch.matmul(q_norm, x_norm.T)
                logits = logits.unsqueeze(0)  # [1, K, N]
            logits = logits / self.tau

            # ============================================================
            # 3. Routing distribution
            # ============================================================
            routes = F.softmax(logits, dim=-1)                # [B, K, N_total]

            # ============================================================
            # 4. IMPORTANT: S 用原始 X 加权（非 norm 后的 X̄）
            # ============================================================
            struct_tokens = torch.bmm(routes, token_bank) if routes.dim() == 3 \
                else torch.matmul(routes, token_bank)       # [B, K, dim]

            round_output = {
                "routes": routes.squeeze(0) if squeeze else routes,          # [K, N_total]
                "struct_tokens": struct_tokens.squeeze(0) if squeeze else struct_tokens,  # [K, dim]
                "logits": logits.squeeze(0) if squeeze else logits,          # [K, N_total]
            }
            all_rounds.append(round_output)

            # ============================================================
            # 5. 扩展 token bank: [B, N+K, dim]
            #    teacher_p: scheduled sampling — 概率用 teacher S，否则用模型自己的 S
            # ============================================================
            S = struct_tokens
            if teacher_S is not None and round_idx < len(teacher_S):
                S_t = teacher_S[round_idx]
                if S_t.dim() == 2:
                    S_t = S_t.unsqueeze(0)
                S_t = S_t.to(device=device, dtype=S.dtype)
                # Pad/truncate teacher S to match model's K
                K_t = S_t.shape[1]
                if K_t < self.K:
                    pad = torch.zeros(B, self.K - K_t, dim, device=device, dtype=S.dtype)
                    S_t = torch.cat([S_t, pad], dim=1)
                elif K_t > self.K:
                    S_t = S_t[:, :self.K, :]
                # Scheduled sampling
                if teacher_p >= 1.0:
                    S = S_t
                elif teacher_p > 0.0:
                    mask = torch.rand(B, 1, 1, device=device) < teacher_p
                    S = torch.where(mask, S_t, S)
            token_bank = torch.cat([token_bank, S], dim=1)

        return all_rounds


# ==================== 测试 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("Image Router V2 基础测试")
    print("=" * 50)

    # 模拟 V_raw
    N, D = 100, 1536
    V_raw = torch.randn(N, D)

    router = ImageRouter(dim=D, K=8, L=1, tau=0.5)
    print(f"K={router.K}, L={router.L}, tau={router.tau}")
    print(f"参数量: {sum(p.numel() for p in router.parameters()):,}")

    rounds = router.forward(V_raw)
    for r_idx, rd in enumerate(rounds):
        print(f"\nRound {r_idx}:")
        print(f"  routes:        {rd['routes'].shape}")         # [K, N]
        print(f"  struct_tokens: {rd['struct_tokens'].shape}")  # [K, D]
        print(f"  logits:        {rd['logits'].shape}")         # [K, N]

        # 检查 routes 每行 sum=1
        row_sums = rd["routes"].sum(dim=-1)
        print(f"  route row sums:  {row_sums.tolist()}")

        # 检查 S norm
        S_norms = rd["struct_tokens"].norm(dim=-1)
        print(f"  S norms mean:    {S_norms.mean():.4f}")

    # L=2 测试
    print(f"\n--- L=2 递归测试 ---")
    router2 = ImageRouter(dim=D, K=8, L=2, tau=0.5)
    rounds2 = router2.forward(V_raw)
    for r_idx, rd in enumerate(rounds2):
        N_total = rd["routes"].shape[-1]
        print(f"Round {r_idx}: A [{router2.K}, {N_total}], S [{router2.K}, {D}]")

    print(f"\n✓ 测试通过")
