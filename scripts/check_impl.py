"""Can the fast official Vim kernels (hustvl/Vim + mamba-ssm) be used?  python scripts/check_impl.py vim_t.pth
Prints USE_OFFICIAL and exits 0 only if
  ReducedVim(official, identity) == official Vim  and  official Vim == kanvim.vim_pure  (same checkpoint),
at 224 px and at 160 px (--img_size control: pos-embed resampled from 14x14, CLS row in the middle).
Otherwise run_plan.py falls back to the pure-PyTorch backbone (slower, same accuracy)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for p in [os.environ.get("VIM_PATH", ""), os.path.join(ROOT, "..", "Vim", "vim"), "/workspace/Vim/vim",
          "/kaggle/working/Vim/vim"]:
    if p and os.path.isdir(p):
        sys.path.insert(0, os.path.abspath(p))

import torch  # noqa: E402

ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "vim_t.pth")
if not torch.cuda.is_available():
    sys.exit("no CUDA -> pure")
try:
    import models_mamba  # noqa: F401
    import mamba_ssm  # noqa: F401
except Exception as e:  # noqa: BLE001
    sys.exit(f"official Vim not importable ({e!r}) -> pure")

from kanvim.model import ReducedVim, load_vim  # noqa: E402
from kanvim.reducers import build_reducer  # noqa: E402

torch.manual_seed(0)
ok = True
for size in (224, 160):
    g = size // 16
    x = torch.randn(4, 3, size, size, device="cuda")
    off = load_vim("tiny", ckpt, 1000, "official", img_size=size).cuda().eval()
    pure = load_vim("tiny", ckpt, 1000, "pure", img_size=size).cuda().eval()
    with torch.no_grad():
        a = off(x).float()
        b = ReducedVim(off, build_reducer("identity", 192, (g, g), 1.0), img_size=size).cuda().eval()(x).float()
        c = pure(x).float()
    scale = a.abs().max().item()
    d1, d2 = (a - b).abs().max().item() / scale, (a - c).abs().max().item() / scale
    print(f"{size} px: relative max diff: ReducedVim vs official {d1:.2e} | pure vs official {d2:.2e} | "
          f"top-1 agree pure/official: {(a.argmax(1) == c.argmax(1)).float().mean().item()}")
    ok = ok and d1 < 2e-2 and d2 < 2e-2
    del off, pure
if ok:
    print("USE_OFFICIAL")
else:
    sys.exit("mismatch -> pure")
