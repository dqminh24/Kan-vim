"""Efficiency of every method x keep-ratio WITHOUT training (inference only):
GMACs, reducer params, throughput (img/s), peak VRAM.
  python scripts/bench.py --ckpt vim_t.pth --out RESULTS_DIR
  python scripts/bench.py --ckpt vim_t.pth --impl official --img_sizes 192 160 112 --batches 1 64
Run it alone on an idle GPU (the throughput written by train.py is measured while the other GPU trains).
"""
import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kanvim import metrics  # noqa: E402
from kanvim.model import ReducedVim, load_vim  # noqa: E402
from kanvim.reducers import ALL_REDUCERS, build_reducer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", default="tiny")
ap.add_argument("--ckpt", default="vim_t.pth")
ap.add_argument("--impl", default="pure", choices=["pure", "official"],
                help="the training runs used official (scan_impl in their json): use the same for speed numbers")
ap.add_argument("--batches", type=int, nargs="+", default=[64], help="e.g. 1 64: latency-bound and throughput-bound")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--ratios", type=float, nargs="+", default=[0.75, 0.5, 0.25])
ap.add_argument("--reducers", nargs="+", default=[r for r in ALL_REDUCERS if r != "random_drop"])
ap.add_argument("--img_sizes", type=int, nargs="*", default=[],
                help="extra rows: no reducer, input img_size x img_size ((img_size/16)^2 tokens), e.g. 192 160 112")
ap.add_argument("--out", default="results")
a = ap.parse_args()

dim, depth = {"tiny": (192, 24), "small": (384, 24)}[a.backbone]
dev = "cuda" if torch.cuda.is_available() else "cpu"
from kanvim.vim_pure import _cuda_scan  # noqa: E402
scan = {"official": "official Vim kernels (mamba-ssm)",
        "pure": "pure PyTorch blocks + " + ("mamba-ssm CUDA selective_scan" if _cuda_scan is not None and dev == "cuda"
                                            else "PyTorch chunked scan")}[a.impl]
print("scan:", scan, "| device:", torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", flush=True)
backbones = {224: load_vim(a.backbone, a.ckpt, 100, a.impl)}
rows = []
for name, ratio, size in ([("identity", 1.0, 224)] + [(r, q, 224) for q in a.ratios for r in a.reducers]
                          + [("identity", 1.0, s) for s in a.img_sizes]):
    if size not in backbones:
        backbones[size] = load_vim(a.backbone, a.ckpt, 100, a.impl, img_size=size)
    g = size // 16
    model = ReducedVim(backbones[size], build_reducer(name, dim, (g, g), ratio), img_size=size)
    row = {"method": name if size == 224 else f"lowres{size}", "keep": ratio, "img_size": size,
           "tokens": model.reducer.m,
           "GMACs": round(metrics.total_macs(model.cpu(), dim, depth, num_classes=100) / 1e9, 3),
           "reducer_params": sum(p.numel() for p in model.reducer.parameters()), "impl": a.impl}
    if dev == "cuda":
        for bs in a.batches:
            it = a.iters * max(1, 64 // bs) if bs < 64 else a.iters  # batch 1: more iterations for a stable time
            row[f"img_per_s_b{bs}"] = round(metrics.throughput(model, batch=bs, iters=it, warmup=5, img_size=size), 1)
            row[f"ms_per_batch_b{bs}"] = round(1000 * bs / row[f"img_per_s_b{bs}"], 2)
            row[f"vram_MB_b{bs}"] = round(metrics.peak_vram_mb(model, batch=bs, img_size=size), 0)
        model.cpu(); torch.cuda.empty_cache()
    rows.append(row)
    print(row, flush=True)

os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, f"bench_{a.backbone}_{a.impl}.csv")
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
    w.writeheader(); w.writerows(rows)
print("saved", path)
