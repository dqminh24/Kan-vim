"""--img_size (lower-resolution control), pos-embed resampling, and the data config written into the result json.
CPU only, no download: Food-101 / Tiny-ImageNet / ImageNet-100 are faked with their real folder layouts."""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from kanvim import metrics
from kanvim.data import (_tf_imagenet, _tf_small, build_datasets, build_loaders_val, data_config,
                         describe_transform, format_transform, transform_of)
from kanvim.model import ReducedVim
from kanvim.reducers import build_reducer
from kanvim.vim_pure import VimPure, load_vim_pure, resize_pos_embed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 32


# ----------------------------------------------------------------------------- pos-embed
@pytest.mark.parametrize("new", [12, 10, 7])
def test_resize_pos_embed_keeps_cls_in_the_middle(new):
    """(1, 197, D), CLS at index 98 -> (1, n+1, D), CLS at index n//2 (n = 144 / 100 / 49: 49 is odd), and the patch
    rows are exactly the 14x14 grid resampled (CLS row NOT mixed into the grid, unlike the DeiT recipe)."""
    torch.manual_seed(0)
    pos = torch.randn(1, 197, D)
    out = resize_pos_embed(pos, (new, new))
    n = new * new
    assert out.shape == (1, n + 1, D)
    assert torch.equal(out[:, n // 2], pos[:, 98])
    grid = torch.cat([pos[:, :98], pos[:, 99:]], 1).reshape(1, 14, 14, D).permute(0, 3, 1, 2)
    ref = F.interpolate(grid, size=(new, new), mode="bicubic", align_corners=False, antialias=True)
    ref = ref.permute(0, 2, 3, 1).reshape(1, n, D)
    got = torch.cat([out[:, :n // 2], out[:, n // 2 + 1:]], 1)
    assert torch.allclose(got, ref, atol=1e-6)


def test_resize_pos_embed_same_grid_is_identity_and_smooth_field_survives():
    pos = torch.randn(1, 197, D)
    assert resize_pos_embed(pos, (14, 14)) is pos
    # a smooth field over the grid (cos of the row / column) stays the same field after 14x14 -> 10x10
    r = (torch.arange(14) + 0.5) / 14
    field = (torch.cos(torch.pi * r)[:, None] * torch.ones(14)[None]).reshape(196, 1).expand(196, D)
    pos = torch.cat([field[:98], torch.full((1, D), 7.0), field[98:]], 0)[None]
    out = resize_pos_embed(pos, (10, 10))
    r10 = (torch.arange(10) + 0.5) / 10
    want = (torch.cos(torch.pi * r10)[:, None] * torch.ones(10)[None]).reshape(100, 1).expand(100, D)
    got = torch.cat([out[0, :50], out[0, 51:]], 0)
    assert (got - want).abs().max() < 0.05 and torch.all(out[0, 50] == 7.0)


def test_load_vim_pure_lowres_from_224_checkpoint(tmp_path):
    """the 224 px checkpoint loads at 160 px: pos-embed resampled, every other weight unchanged"""
    torch.manual_seed(0)
    src = VimPure(dim=192, depth=24, num_classes=1000)
    with torch.no_grad():
        src.pos_embed.normal_(0, 0.02)
    ck = tmp_path / "vim.pth"
    torch.save({"model": src.state_dict()}, ck)
    m = load_vim_pure("tiny", str(ck), num_classes=10, img_size=160)
    assert m.patch_embed.grid_size == (10, 10) and m.pos_embed.shape == (1, 101, 192)
    assert torch.equal(m.pos_embed, resize_pos_embed(src.pos_embed.detach(), (10, 10)))
    assert torch.equal(m.patch_embed.proj.weight, src.patch_embed.proj.weight)
    assert torch.equal(m.layers[5].mixer.in_proj.weight, src.layers[5].mixer.in_proj.weight)
    assert torch.equal(load_vim_pure("tiny", str(ck), 10).pos_embed, src.pos_embed)  # 224: untouched


# ----------------------------------------------------------------------------- model at another resolution
@pytest.mark.parametrize("size", [192, 160, 112])
def test_lowres_identity_matches_backbone_forward_and_token_count(size):
    """ReducedVim(identity) at img_size = the backbone's own forward (CLS in the middle, also for odd N = 49);
    every block sees (size/16)^2 + 1 tokens"""
    torch.manual_seed(0)
    g = size // 16
    bb = VimPure(dim=D, depth=2, num_classes=5, img=size).eval()
    with torch.no_grad():
        bb.pos_embed.normal_(0, 0.02)
        bb.cls_token.normal_(0, 0.02)
    m = ReducedVim(bb, build_reducer("identity", D, (g, g), 1.0), img_size=size).eval()
    assert m.n_patches == g * g and m.layer_tokens() == [g * g + 1] * 2
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        assert bb.patch_embed(x).shape[1] == g * g
        assert torch.allclose(m(x), bb(x), atol=1e-5)


def test_lowres_model_resizes_the_224_input_itself():
    """data.py gives 224 px: the model must see exactly F.interpolate(bilinear, antialias) of it"""
    torch.manual_seed(0)
    bb = VimPure(dim=D, depth=2, num_classes=5, img=160).eval()
    m = ReducedVim(bb, build_reducer("identity", D, (10, 10), 1.0), img_size=160).eval()
    x = torch.randn(2, 3, 224, 224)
    small = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False, antialias=True)
    with torch.no_grad():
        assert torch.allclose(m(x), m(small), atol=1e-6)
        assert torch.allclose(m(x), bb(small), atol=1e-5)


def test_wrong_input_size_without_img_size_fails_clearly():
    bb = VimPure(dim=D, depth=1, num_classes=5, img=160)
    m = ReducedVim(bb, build_reducer("identity", D, (10, 10), 1.0))  # img_size not given: no resize
    with pytest.raises(AssertionError, match="patch embedding gave 196 tokens"):
        m(torch.randn(1, 3, 224, 224))


def test_lowres_macs_and_throughput():
    """GMACs use the model's own patch count; 160 px ~ keep 50%, 112 px < keep 25% (no reducer cost)"""
    dim, depth = 192, 24
    full = ReducedVim(VimPure(dim, depth, 100), build_reducer("identity", dim, (14, 14), 1.0))
    low = {s: ReducedVim(VimPure(dim, depth, 100, img=s), build_reducer("identity", dim, (s // 16,) * 2, 1.0),
                         img_size=s) for s in (192, 160, 112)}
    g = {s: metrics.total_macs(m, dim, depth, num_classes=100) / 1e9 for s, m in low.items()}
    assert abs(metrics.total_macs(full, dim, depth, num_classes=100) / 1e9 - 1.467) < 0.01
    assert g[160] == pytest.approx(metrics.vim_macs_layers([101] * 24, dim, n_patches=100, num_classes=100) / 1e9)
    keep = {r: metrics.total_macs(ReducedVim(VimPure(dim, depth, 100), build_reducer("kan_pool", dim, (14, 14), r)),
                                  dim, depth, num_classes=100) / 1e9 for r in (0.75, 0.5, 0.25)}
    assert g[192] < keep[0.75] and abs(g[160] - keep[0.5]) < 0.02 and g[112] < keep[0.25]
    small = ReducedVim(VimPure(D, 1, 5, img=112), build_reducer("identity", D, (7, 7), 1.0), img_size=112)
    assert metrics.throughput(small, batch=2, iters=1, warmup=1, device="cpu", img_size=112) > 0


def test_lowres_with_reducer_uses_its_own_grid():
    """a reducer on top of a lower resolution: grid (img/16)^2 tokens, keep ratio of those"""
    bb = VimPure(dim=D, depth=2, num_classes=5, img=160)
    m = ReducedVim(bb, build_reducer("kan_pool", D, (10, 10), 0.5, kan_hidden=8), img_size=160)
    assert m.layer_tokens() == [51, 51]
    out = m.train()(torch.randn(2, 3, 224, 224))
    out.sum().backward()
    assert out.shape == (2, 5)


# ----------------------------------------------------------------------------- data
def _jpg(path, size, mode="RGB"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new(mode, size, 128 if mode == "L" else (200, 100, 50)).save(path)


@pytest.fixture
def fake_root(tmp_path):
    """Food-101 / Tiny-ImageNet / ImageNet-100 with their real folder layouts, 3 classes, a few images each"""
    root = tmp_path / "data"
    f = root / "food-101"
    meta = {"train": {}, "test": {}}
    for c in ("apple_pie", "baklava", "ramen"):
        for i in range(8):
            split = "train" if i < 6 else "test"
            meta[split].setdefault(c, []).append(f"{c}/{i}")
            _jpg(str(f / "images" / c / f"{i}.jpg"), (512, 384) if i % 2 else (384, 512))
    os.makedirs(f / "meta")
    for split, m in meta.items():
        (f / "meta" / f"{split}.json").write_text(json.dumps(m))
    t = root / "tiny-imagenet-200"
    ann = []
    for c in ("n01443537", "n01629819", "n01641577"):
        for i in range(6):
            _jpg(str(t / "train" / c / "images" / f"{c}_{i}.JPEG"), (64, 64), "L" if i == 0 else "RGB")
        ann.append(f"val_{c}.JPEG\t{c}\t0\t0\t63\t63")
        _jpg(str(t / "val" / "images" / f"val_{c}.JPEG"), (64, 64))
    (t / "val" / "val_annotations.txt").write_text("\n".join(ann) + "\n")
    for split, n in (("train", 6), ("val", 2)):
        for c in ("n01440764", "n01443537", "n01484850"):
            for i in range(n):
                _jpg(str(root / "imagenet-100" / split / c / f"{i}.JPEG"), (500, 375))
    return str(root)


def test_food101_uses_exactly_the_imagenet100_pipeline(fake_root):
    f_tr, f_te, nc = build_datasets("food101", fake_root)
    i_tr, i_te, _ = build_datasets("imagenet-100", fake_root)
    assert nc == 101 and len(f_tr) == 18 and len(f_te) == 6
    assert describe_transform(f_tr.transform) == describe_transform(i_tr.transform) == describe_transform(_tf_imagenet(True))
    assert describe_transform(f_te.transform) == describe_transform(i_te.transform) == describe_transform(_tf_imagenet(False))
    names = [s["name"] for s in describe_transform(f_tr.transform)]
    assert names == ["RandomResizedCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"]
    assert [s["name"] for s in describe_transform(f_te.transform)][:2] == ["Resize", "CenterCrop"]
    for ds in (f_tr, f_te):  # non-square photos -> 3 x 224 x 224
        assert ds[0][0].shape == (3, 224, 224) and ds[1][0].shape == (3, 224, 224)


def test_small_image_pipeline_and_padding_mode_is_recorded(fake_root):
    tr, te, nc = build_datasets("tiny-imagenet", fake_root)
    assert nc == 200 and tr[0][0].shape == (3, 224, 224) and te[0][0].shape == (3, 224, 224)  # tr[0]: grayscale
    crop = describe_transform(tr.transform)[1]
    assert crop["name"] == "RandomCrop" and crop["padding"] == 16 and crop["padding_mode"] == "reflect"
    assert "padding_mode" not in repr(tr.transform.transforms[1])  # what repr() would have lost
    assert describe_transform(te.tf) == describe_transform(_tf_small(False))


@pytest.mark.parametrize("name", ["food101", "tiny-imagenet", "imagenet-100"])
def test_loaders_val_uses_eval_transform_and_config_reads_the_real_loaders(fake_root, name):
    tr, va, te, nc = build_loaders_val(name, fake_root, batch=2, workers=0, val_size=3)
    cfg = data_config(name, tr, va, te, nc)
    json.dumps(cfg)  # json-safe
    assert cfg["n_val"] == 3 and cfg["n_train"] + cfg["n_val"] == len(tr.dataset.dataset)
    assert cfg["transform"]["train"] == describe_transform(transform_of(tr.dataset))
    assert cfg["transform"]["val"] == cfg["transform"]["test"] and cfg["val_uses_eval_transform"]
    assert cfg["transform"]["train"] != cfg["transform"]["val"]
    assert transform_of(tr.dataset) is not transform_of(va.dataset)  # copy.copy did not share the train transform
    # the val subset really returns the deterministic eval transform of the raw image
    base, i = va.dataset.dataset, va.dataset.indices[0]
    raw = base.loader(base.samples[i][0]) if hasattr(base, "samples") else \
        Image.open(base._image_files[i]).convert("RGB")
    assert torch.equal(va.dataset[0][0], transform_of(te.dataset)(raw))
    assert " -> " in format_transform(cfg["transform"]["train"])


# ----------------------------------------------------------------------------- train.py end to end
def _train(tmp, *args):
    cmd = [sys.executable, os.path.join(ROOT, "train.py"), "--dataset", "fake", "--backbone", "stub", "--val_size", "200",
           "--epochs", "1", "--max_steps", "2", "--batch", "8", "--workers", "0", "--no_bench", "--mixup", "0.2",
           "--out", str(tmp), "--tag", "lr5e-5", *args]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def test_train_lowres_end_to_end_json(tmp_path):
    """224 baseline (saved as teacher) -> 160 px control: own result file, 100 tokens, distilled from the 224 teacher,
    and the json records the data / input / recipe config"""
    _train(tmp_path, "--save_ckpt", str(tmp_path / "t.pt"))
    out = _train(tmp_path, "--img_size", "160", "--teacher_ckpt", str(tmp_path / "t.pt"))
    assert "distill=on (teacher 224 px)" in out and "100 patch tokens" in out
    base = json.load(open(tmp_path / "fake_stub_identity_r1.0_s0_lr5e-5.json"))
    low = json.load(open(tmp_path / "fake_stub_identity_r1.0_s0_lr5e-5_res160.json"))
    assert base["mean_tokens"] == 196 and low["mean_tokens"] == 100 and low["GMACs"] < base["GMACs"]
    assert not base["train_recipe"]["distill"] and low["train_recipe"]["distill"]
    assert low["train_recipe"]["teacher_img_size"] == 224 and low["train_recipe"]["mixup_alpha"] == 0.2
    assert low["train_recipe"]["label_smoothing"] == 0.1
    assert low["model_input"]["patch_tokens"] == 100 and low["model_input"]["grid"] == [10, 10]
    assert base["model_input"]["resize_in_model"] is None and "bilinear" in low["model_input"]["resize_in_model"]
    assert low["data"]["train_batch_seen"] == [8, 3, 224, 224] and low["data"]["val_uses_eval_transform"]
    assert low["data"]["transform"]["train"][1]["padding_mode"] == "reflect"
    assert low["args"]["tag"] == "lr5e-5_res160"


def test_train_reducer_at_224_unchanged_and_tag_not_duplicated(tmp_path):
    _train(tmp_path, "--save_ckpt", str(tmp_path / "t.pt"))
    _train(tmp_path, "--reducer", "kan_pool", "--ratio", "0.5", "--kan_hidden", "8", "--teacher_ckpt", str(tmp_path / "t.pt"))
    d = json.load(open(tmp_path / "fake_stub_kan_pool_r0.5_s0_lr5e-5.json"))
    assert d["mean_tokens"] == 98 and d["train_recipe"]["distill"] and d["model_input"]["img_size"] == 224
    _train(tmp_path, "--img_size", "112", "--tag", "lr5e-5_res112")  # tag already says res112: not appended twice
    assert (tmp_path / "fake_stub_identity_r1.0_s0_lr5e-5_res112.json").exists()

def test_val_from_test_splits_the_official_test_set_in_clean_halves(fake_root):
    """ImageNet-100 fix: val / test = two stratified halves of the official val (not in the pretraining set);
    the whole train set is used; both halves use the eval transform; the split is fixed"""
    tr, va, te, nc = build_loaders_val("imagenet-100", fake_root, batch=2, workers=0, val_from="test")
    full_te = build_datasets("imagenet-100", fake_root)[1]
    vi, ti = va.dataset.indices, te.dataset.indices
    assert len(tr.dataset) == len(build_datasets("imagenet-100", fake_root)[0])
    assert not set(vi) & set(ti) and sorted(vi + ti) == list(range(len(full_te)))
    assert sorted(full_te.targets[i] for i in vi) == sorted(full_te.targets[i] for i in ti)  # one image per class each
    cfg = data_config("imagenet-100", tr, va, te, nc)
    assert cfg["val_from"] == "test" and cfg["val_uses_eval_transform"] and cfg["n_val"] == cfg["n_test"] == 3
    assert build_loaders_val("imagenet-100", fake_root, 2, 0, val_from="test")[1].dataset.indices == vi
    assert data_config("imagenet-100", *build_loaders_val("imagenet-100", fake_root, 2, 0, val_size=3))["val_from"] == "train"


def test_diagnostics_pool_and_spread():
    """Avg Pool = uniform weights (entropy 1); a pool whose logits prefer one token per group has entropy < 1.
    Static Drop spreads the kept tokens (no grid neighbours at 50%); keeping a contiguous block does the opposite."""
    from kanvim.diagnostics import pool_stats, spread_stats, static_spread
    r = build_reducer("avg_pool", D, (14, 14), 0.5)
    u = pool_stats(torch.zeros(2, 196), r.assign, r.m)
    assert abs(u["weight_entropy"] - 1) < 1e-6 and u["weight_std"] < 1e-7 and abs(u["weight_max"] - u["uniform_max"]) < 1e-6
    peaked = pool_stats(torch.arange(196.).repeat(2, 1) * 5, r.assign, r.m)
    assert peaked["weight_entropy"] < 0.2 and peaked["weight_max"] > 0.9
    s = static_spread(196, 0.5, (14, 14))
    block = spread_stats(torch.arange(98)[None], (14, 14))
    assert s["adjacent"] == 0 and block["adjacent"] > 0.9 and s["cover"] < block["cover"]


@pytest.mark.parametrize("name,sign", [("delta_prune", 1.0), ("delta_prune_low", -1.0)])
def test_delta_prune_selects_by_the_previous_blocks_delta(name, sign):
    """--reduce_at L with delta_prune: kept tokens = top (sign * Delta) of block L-1, CLS slot excluded, original order,
    plus one fused token; the backbone still gets gradients; refuses --reduce_at 0"""
    torch.manual_seed(0)
    L, depth = 2, 4
    bb = VimPure(dim=D, depth=depth, num_classes=5).eval()
    m = ReducedVim(bb, build_reducer("identity", D, (14, 14), 1.0), stages=[(L, build_reducer(name, D, (14, 14), 0.5))])
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        m(x)
        score = bb.layers[L - 1].mixer.last_score                     # (B, 197) Delta of block L-1, CLS at 98
    ref = torch.cat([score[:, :98], score[:, 99:]], 1)
    want = (sign * ref).topk(97, 1).indices.sort(1).values
    assert torch.equal(m.stages[0].aux["idx"], want) and m.layer_tokens() == [197] * L + [99] * (depth - L)
    out = m.train()(x)
    out.sum().backward()
    assert out.shape == (2, 5) and bb.layers[0].mixer.in_proj.weight.grad is not None
    with pytest.raises(AssertionError, match="reduce_at"):
        ReducedVim(VimPure(D, 2, 5), build_reducer(name, D, (14, 14), 0.5))
