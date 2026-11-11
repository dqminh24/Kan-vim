"""Token reduction WITHOUT retraining: the fine-tuned 100%-token baseline, then a fixed reducer is plugged in and the
test set is scored, no gradient step. Shows how much each way of reducing breaks the trained model by itself, before
fine-tuning repairs it (e.g. Mamba's causal conv1d (kernel 4) and scan suddenly see tokens that were far apart in the
image as neighbours). Only fixed reducers: a learned scorer would still be at its random init here. Also the
lower-resolution control (same fine-tuned weights, pos-embed resampled) and, with the kept positions, their spread.

  python scripts/eval_noretrain.py --dataset cifar100 --teacher_ckpt teacher/teacher_cifar100.pt --impl official
  -> <out>/noretrain_<dataset>.json and .md
"""
import argparse
import contextlib
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kanvim import metrics  # noqa: E402
from kanvim.data import build_loaders_val  # noqa: E402
from kanvim.diagnostics import static_spread  # noqa: E402
from kanvim.model import ReducedVim, load_vim  # noqa: E402
from kanvim.reducers import build_reducer  # noqa: E402
from kanvim.vim_pure import VimPure, resize_pos_embed  # noqa: E402

DIMS = {"tiny": (192, 24), "small": (384, 24), "stub": (64, 2)}
FIXED = ["static_drop", "random_drop", "avg_pool", "max_pool"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--data_root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--backbone", default="tiny", choices=list(DIMS))
    ap.add_argument("--impl", default="pure", choices=["pure", "official"])
    ap.add_argument("--teacher_ckpt", required=True, help="fine-tuned 224 px identity baseline (train.py --save_ckpt)")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.75, 0.5, 0.25])
    ap.add_argument("--img_sizes", type=int, nargs="*", default=[192, 160, 112])
    ap.add_argument("--reduce_at", type=int, default=0,
                    help=">0: reduce before block L (tokens already contextualised); adds the Delta-scored control "
                         "(delta_prune: keep large Delta, delta_prune_low: keep small Delta; needs --impl pure)")
    ap.add_argument("--val_size", type=int, default=5000)
    ap.add_argument("--val_from", default="train", choices=["train", "test"])
    ap.add_argument("--test_subset", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="random_drop")
    ap.add_argument("--out", default=os.path.join(ROOT, "results_noretrain"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dim, depth = DIMS[a.backbone]
    _, _, te, nc = build_loaders_val(a.dataset, a.data_root, a.batch, a.workers, a.val_size, 0, a.test_subset,
                                     a.val_from)
    sd = torch.load(a.teacher_ckpt, map_location="cpu")

    def model_for(name, ratio, size):
        g = size // 16
        bb = VimPure(dim, depth, nc, img=size) if a.backbone == "stub" else load_vim(a.backbone, None, nc, a.impl,
                                                                                      img_size=size)
        s = dict(sd)
        s["backbone.pos_embed"] = resize_pos_embed(sd["backbone.pos_embed"], (g, g))
        m = ReducedVim(bb, build_reducer("identity", dim, (g, g), 1.0), img_size=size)
        m.load_state_dict(s)
        red = build_reducer(name, dim, (g, g), ratio)
        if a.reduce_at > 0 and name != "identity":
            m = ReducedVim(m.backbone, m.reducer, stages=[(a.reduce_at, red)], img_size=size)
        else:
            m.reducer = red
        return m.to(dev).eval()

    @torch.no_grad()
    def score(m):
        torch.manual_seed(a.seed)
        hit = n = 0
        for x, y in te:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.float16) if dev == "cuda" else contextlib.nullcontext():
                out = m(x)
            hit += (out.argmax(1) == y).sum().item()
            n += y.numel()
        return 100.0 * hit / n

    rows = []
    fixed = FIXED + (["delta_prune", "delta_prune_low"] if a.reduce_at > 0 and a.impl == "pure" else [])
    for name, ratio, size in ([("identity", 1.0, 224)] + [(r, q, 224) for q in a.ratios for r in fixed]
                              + [("identity", 1.0, s) for s in a.img_sizes]):
        m = model_for(name, ratio, size)
        row = {"method": name if size == 224 else f"lowres{size}", "keep": ratio, "img_size": size,
               "tokens": (m.stages[0] if len(m.stages) else m.reducer).m, "top1": score(m), "reduce_at": a.reduce_at,
               "GMACs": metrics.total_macs(m.cpu(), dim, depth, num_classes=nc) / 1e9}
        if name in ("static_drop",) and not a.reduce_at:
            row.update(static_spread(m.n_patches, ratio, (size // 16, size // 16)))
        rows.append(row)
        print(row, flush=True)
    base = rows[0]["top1"]
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"noretrain_{a.dataset}" + (f"_L{a.reduce_at}" if a.reduce_at else ""))
    json.dump({"args": vars(a), "rows": rows}, open(stem + ".json", "w"), indent=1)
    L = [f"# Giảm token không huấn luyện lại: {a.dataset}", "",
         f"Baseline đã fine-tune ({os.path.basename(a.teacher_ckpt)}), cắm bộ giảm token cố định vào rồi chấm test, "
         "không cập nhật trọng số nào. Baseline 100% token: **" + f"{base:.2f}".replace(".", ",") + "**.", "",
         "| Cách giảm | Giữ | Token | GMACs | Test top-1 | Sụt so với baseline |", "|---|---|---|---|---|---|"]
    for r in rows[1:]:
        L.append(f"| {r['method']} | {round(r['keep'] * 100)}% | {r['tokens']} | {r['GMACs']:.3f} | {r['top1']:.2f} | "
                 f"{r['top1'] - base:+.2f} |".replace(".", ","))
    open(stem + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("saved", stem + ".json / .md")


if __name__ == "__main__":
    main()
