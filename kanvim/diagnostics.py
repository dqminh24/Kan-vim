"""Diagnostics of the reducers, logged by train.py (no effect on training):

pool_stats   -- is a learned pool (KANPool / MLP Pool) really weighting tokens, or is it an Avg Pool in disguise?
                Inside every group the merge weights are softmax(logits). Avg Pool = uniform weights: normalised
                entropy 1, weight std 0. A learned pool that prefers some tokens has entropy < 1.
spread_stats -- where the kept tokens lie on the patch grid. After pruning, tokens far apart in the image become
                neighbours in the sequence that Mamba's causal conv1d (kernel 4) and scan read; Static Drop keeps
                an even spread. gap = 2-D distance (in patches) between consecutive kept tokens; adjacent = share of
                consecutive kept tokens that are grid neighbours (distance 1); cover = mean distance from every
                token of the grid to its nearest kept token (lower = better coverage).
"""
import math

import torch


@torch.no_grad()
def pool_stats(logits, assign, m):
    """logits (B, N) of the merge weights, assign (N,) group of every token, m groups."""
    logits = logits.float()
    b, n = logits.shape
    idx = assign.view(1, n).expand(b, n).to(logits.device)
    gmax = logits.new_full((b, m), float("-inf")).scatter_reduce(1, idx, logits, "amax", include_self=True)
    e = torch.exp(logits - gmax.gather(1, idx))
    w = e / e.new_zeros(b, m).scatter_add(1, idx, e).gather(1, idx)              # softmax inside each group
    size = torch.bincount(assign.cpu(), minlength=m).to(logits.device).float()  # tokens per group
    ent = e.new_zeros(b, m).scatter_add(1, idx, -w * torch.log(w.clamp_min(1e-12)))
    multi = size > 1
    norm_ent = (ent[:, multi] / torch.log(size[multi])).mean().item() if multi.any() else float("nan")
    mean_w = (1.0 / size)[assign.to(logits.device)]                                # uniform weight of each token
    wstd = ((w - mean_w) ** 2).mean().sqrt().item()
    wmax = e.new_full((b, m), 0.0).scatter_reduce(1, idx, w, "amax", include_self=True)[:, multi]
    return {"logit_std": logits.std(1).mean().item(), "logit_absmean": logits.abs().mean().item(),
            "weight_std": wstd, "weight_entropy": norm_ent,
            "weight_max": wmax.mean().item() if multi.any() else float("nan"),
            "uniform_max": (1.0 / size[multi]).mean().item() if multi.any() else float("nan")}


@torch.no_grad()
def spread_stats(idx, grid_hw):
    """idx (B, K) kept token indices (raster order of the grid), sorted."""
    h, w = grid_hw
    idx = idx.long().cpu()
    ys, xs = (idx // w).float(), (idx % w).float()
    d = torch.sqrt((ys[:, 1:] - ys[:, :-1]) ** 2 + (xs[:, 1:] - xs[:, :-1]) ** 2)
    all_y, all_x = (torch.arange(h * w) // w).float(), (torch.arange(h * w) % w).float()
    dist = torch.sqrt((all_y[None, :, None] - ys[:, None, :]) ** 2 + (all_x[None, :, None] - xs[:, None, :]) ** 2)
    return {"gap": d.mean().item(), "adjacent": (d <= 1.0 + 1e-6).float().mean().item(),
            "cover": dist.min(2).values.mean().item()}


def static_spread(n_tokens, keep, grid_hw):
    """Same numbers for Static Drop (evenly spaced along the sequence): reference for the pruners."""
    from .reducers import num_keep
    m = num_keep(n_tokens, keep)
    return spread_stats(torch.linspace(0, n_tokens - 1, m).round().long()[None], grid_hw)


def reducer_stats(reducer, grid_hw):
    """Diagnostics of the last forward of a reducer (its aux), {} if it has nothing to report."""
    aux = getattr(reducer, "aux", {}) or {}
    out = {}
    if hasattr(reducer, "assign") and "logits" in aux:
        out.update(pool_stats(aux["logits"], reducer.assign, reducer.m))
    if "idx" in aux:  # pruners: score spread (after the per-image z-score when --prune_norm) and kept positions
        if "logits" in aux:
            out["logit_std"] = aux["logits"].float().std(1).mean().item()
        out.update(spread_stats(aux["idx"], grid_hw))
    return out


def fmt_stats(s):
    return " ".join(f"{k} {v:.3g}" for k, v in s.items() if isinstance(v, float) and not math.isnan(v))
