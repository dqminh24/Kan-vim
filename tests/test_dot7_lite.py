"""run_fix.py --dot7 --lite: job list of the reduced round 7 and how its report marks the 1-seed cells.
No GPU, no training: the job lists are plain tuples, the report reads fake result jsons."""
import json
import os
import sys
import types

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_fix as F  # noqa: E402


def args(out="results_dot7", lite=True, quick=False, ext=("tiny-imagenet",)):
    return types.SimpleNamespace(out=out, food_epochs=30, ext_epochs=10, quick=quick, lite=lite, ext_ds=list(ext),
                                 dot6=False, dot7=True)


def key(j):
    """(dataset, reducer, ratio, seed, tag)"""
    return (j[5] if len(j) > 5 else "cifar100",) + tuple(j[:4])


def test_lite_job_list():
    p1, p2 = F.jobs_dot7(args())
    assert len(p1) == 11 and len(p2) == 32 and len(p1) + len(p2) == 43
    keys = [key(j) for j in p1 + p2]
    assert len(set(keys)) == len(keys)
    # phase 1: the Food-101 teacher first, then the jobs that do not need it
    assert key(p1[0]) == ("food101", "identity", 1.0, 0, F.DOT6_TAG) and "--save_ckpt" in p1[0][4]
    assert all(j[5] != "food101" for j in p1[1:] if len(j) > 5)
    assert sum(k[0] == "cifar100" and k[4].startswith(F.DOT6_TAG + "_res") for k in keys) == 9
    assert ("tiny-imagenet", "static_drop", 0.5, 0, F.DOT6_TAG) in keys
    # phase 2: Food-101 seed 0 = full matrix (7 methods x 3 ratios + 3 lower resolutions), seeds 1 / 2 only the 25% set
    food = [k for k in keys if k[0] == "food101" and k[1:] != ("identity", 1.0, 0, F.DOT6_TAG)]
    assert sum(k[3] == 0 for k in food) == 24
    later = {(k[1], k[2], k[4]) for k in food if k[3] > 0}
    assert later == F.LITE_SEEDS12 and sum(k[3] > 0 for k in food) == 8
    assert all("--teacher_ckpt" in j[4] for j in p2)
    # nothing of the full round that the reduced one drops
    assert not any(k[1] == "delta_prune" for k in keys)
    assert not any(k[0] == "food101" and k[1] == "identity" and k[2] == 1.0 and k[3] > 0 and k[4] == F.DOT6_TAG
                   for k in keys)
    assert not any(k[0] == "imagenet-100" for k in keys)


def test_full_round_unchanged():
    p1, p2 = F.jobs_dot7(args(lite=False, ext=("tiny-imagenet", "imagenet-100")))
    assert len(p1) + len(p2) == 99
    assert sum(j[0] == "delta_prune" for j in p2) == 6


def test_lite_quick_has_no_delta_job():
    p1, p2 = F.jobs_dot7(args(quick=True))
    assert not any(j[0] == "delta_prune" for j in p1 + p2)
    assert any(j[0] == "delta_prune" for j in F.jobs_dot7(args(quick=True, lite=False))[1])


def test_planned_seeds():
    a = args()
    assert F.planned_seeds(a, "food101", "kan_pool", 0.25) == 3
    assert F.planned_seeds(a, "food101", F.LOWRES, 0.25) == 3
    assert F.planned_seeds(a, "food101", "kan_pool", 0.5) == 1
    assert F.planned_seeds(a, "food101", "avg_pool", 0.25) == 1
    assert F.planned_seeds(a, "cifar100", "kan_pool", 0.5) == 3
    assert F.planned_seeds(args(lite=False), "food101", "kan_pool", 0.5) == 3


def _fake(out, ds, reducer, ratio, seed, tag, top1, epochs=30):
    d = {"args": {"dataset": ds, "reducer": reducer, "ratio": ratio, "seed": seed, "tag": tag, "epochs": epochs},
         "top1": top1, "val_top1": top1, "best_epoch": epochs - 1, "GMACs": 1.467 * ratio,
         "log": [{"epoch": epochs - 1, "top1": top1}]}
    json.dump(d, open(os.path.join(out, F.result_name(reducer, ratio, seed, tag, ds)), "w"))


def test_lite_report_marks_one_seed_cells(tmp_path):
    out = str(tmp_path)
    _fake(out, "food101", "identity", 1.0, 0, F.DOT6_TAG, 85.0)
    for s in range(3):
        for r, base in (("kan_pool", 80.0), ("static_drop", 79.0)):
            _fake(out, "food101", r, 0.25, s, F.DOT6_TAG, base + 0.1 * s)
    for r in ("kan_pool", "mlp_pool", "static_drop"):
        _fake(out, "food101", r, 0.5, 0, F.DOT6_TAG, 82.0)
    F.report_dot7(args(out=out))
    md = open(os.path.join(out, "KET_QUA_DOT7.md"), encoding="utf-8").read()
    food = md[md.index("## Food-101"):]  # the CIFAR-100 section (empty here) comes first
    row = next(l for l in food.splitlines() if l.startswith("| KANPool − Static Drop"))
    cells = row.split(" | ")
    assert "1 seed: tham khảo, không tính KPI" in cells[2]  # giữ 50%: 1 seed by design
    assert "đạt" in cells[3] and "tham khảo" not in cells[3]  # giữ 25%: 3 seeds -> real KPI
    assert "bản gọn" in md and "Δ" not in md and "ImageNet-100 (làm lại" not in md
