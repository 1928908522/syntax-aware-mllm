"""
Hungarian 匹配 V2 —— 仅用 Soft CE 作为代价矩阵。

文档建议:
  C_km = -Σ A_T(m) * log(A_I(k) + ε)
  即 Student slot k 与 Teacher route m 的 Soft Cross Entropy。

不再包含 relation cost、struct cost、A_in/A_out 分离。
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment


class HungarianMatcher:
    """Hungarian 匹配器: 预测 slot 与目标路由的最优配对。"""

    def __init__(self):
        pass

    def compute_cost_matrix(
        self,
        pred_logits: torch.Tensor,       # [K, N]
        teacher_routes: torch.Tensor,    # [M, N]
    ) -> torch.Tensor:
        """计算 K×M Soft CE 代价矩阵。

        C[k, m] = -Σ_j A_T[m, j] * log_softmax(logits_I[k])[j]

        Returns:
            cost_matrix [K, M]
        """
        K = pred_logits.shape[0]
        M = teacher_routes.shape[0]

        # log_softmax over N dimension for each of K slots
        log_probs = F.log_softmax(pred_logits, dim=-1)  # [K, N]

        # C[k, m] = -Σ_j teacher_routes[m, j] * log_probs[k, j]
        # = -(teacher_routes @ log_probs.T)
        cost = -torch.matmul(teacher_routes, log_probs.T)  # [M, K]
        cost = cost.T  # [K, M]

        return cost

    def match(
        self,
        pred_logits: torch.Tensor,       # [K, N]
        teacher_routes: torch.Tensor,    # [M, N]
    ) -> Tuple[List[int], torch.Tensor]:
        """执行 Hungarian 匹配。

        Returns:
            matched_indices: List[int] 长度 K, pred_slot_k → target_m 或 -1
            cost_matrix: [K, M]
        """
        K = pred_logits.shape[0]
        M = teacher_routes.shape[0]

        cost_matrix = self.compute_cost_matrix(pred_logits, teacher_routes)

        # M=0 时无法匹配
        if M == 0:
            return [-1] * K, cost_matrix

        # 如果 K > M: 补 padding 列（高代价）
        if K > M:
            pad_cost = torch.full((K, K - M), 1e6, device=cost_matrix.device)
            cost_np = torch.cat([cost_matrix, pad_cost], dim=1).detach().cpu().numpy()
        else:
            cost_np = cost_matrix.detach().cpu().numpy()

        row_ind, col_ind = linear_sum_assignment(cost_np)

        matched = [-1] * K
        for k, m in zip(row_ind, col_ind):
            if m < M:
                matched[k] = m

        return matched, cost_matrix
