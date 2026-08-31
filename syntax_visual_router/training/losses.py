"""Loss helpers for Stage E CoT, DPO, and route-aware RL."""

import torch
import torch.nn.functional as F


def route_weight(route_score, low=0.5, high=1.5):
    if not torch.is_tensor(route_score):
        route_score = torch.tensor(float(route_score))
    return torch.clamp(route_score.float(), min=0.0, max=1.0) * (high - low) + low


def response_lm_loss(outputs_loss, route_score=None, route_strength=1.0):
    """Weight response-only LM loss by route consistency when available."""
    if route_score is None or route_strength <= 0:
        return outputs_loss
    weight = route_weight(route_score).to(outputs_loss.device)
    if route_strength != 1.0:
        weight = 1.0 + float(route_strength) * (weight - 1.0)
    return outputs_loss * weight


def normalize_route(a):
    a = a.float().clamp_min(1e-9)
    return a / a.sum(dim=-1, keepdim=True).clamp_min(1e-9)


def route_alignment_loss(teacher_route=None, image_route=None):
    """KL(A_teacher || A_image). Returns zero if routes are missing."""
    if teacher_route is None or image_route is None:
        return None
    p = normalize_route(teacher_route)
    q = normalize_route(image_route)
    return F.kl_div(q.log(), p, reduction="batchmean")


def route_contrast_loss(teacher_route=None, pos_route=None, neg_route=None, margin=0.2):
    """Max-margin route contrast. Returns zero if routes are missing."""
    if teacher_route is None or pos_route is None or neg_route is None:
        return None
    t = F.normalize(teacher_route.flatten().float(), dim=0)
    p = F.normalize(pos_route.flatten().float(), dim=0)
    n = F.normalize(neg_route.flatten().float(), dim=0)
    sim_pos = torch.dot(t, p)
    sim_neg = torch.dot(t, n)
    return torch.clamp(margin - sim_pos + sim_neg, min=0.0)


def route_diversity_loss(routes=None):
    """Mean off-diagonal cosine similarity across route slots."""
    if routes is None:
        return None
    if routes.dim() == 3:
        routes = routes[0]
    if routes.shape[0] < 2:
        return routes.new_tensor(0.0)
    z = F.normalize(routes.float(), dim=-1)
    sim = z @ z.T
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    return sim.masked_select(~eye).mean()


def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta=0.1, weight=None):
    pi_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits)
    if weight is not None:
        loss = loss * weight.to(loss.device)
    return loss.mean()


def reward_weight(delta, low=0.5, high=2.0):
    if not torch.is_tensor(delta):
        delta = torch.tensor(float(delta))
    return torch.clamp(delta.float(), min=low, max=high)


def group_advantages(rewards, eps=1e-6):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.numel() <= 1:
        return rewards * 0.0
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(eps)
