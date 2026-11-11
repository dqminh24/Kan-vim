"""CIFAR-10 / CIFAR-100 / Tiny-ImageNet / Food-101 (auto-download), ImageNet-100 (ImageFolder you prepare).
Every loader gives 224x224 images, as specified in the proposal (PIPELINE_SIZE). Two pipelines:
  small square images (CIFAR, Tiny-ImageNet): Resize(224) -> RandomCrop(224, pad 16, reflect) -> flip | Resize(224)
  large photos (Food-101, ImageNet-100):      RandomResizedCrop(224) -> flip | Resize(256) -> CenterCrop(224)
Mixup / CutMix mix whole batches, so they live in the training loop (train.py), not here.
data_config() reads the transforms back from the loaders actually used; train.py writes it into the result json."""
import copy
import enum
import os
import urllib.request
import zipfile

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms as T
from torchvision.datasets.folder import default_loader

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PIPELINE_SIZE = 224  # every loader's output size; a model with another --img_size resizes it itself (ReducedVim)
VAL_SPLIT_SEED = 12345
TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
# where the test set comes from (Tiny-ImageNet's own test split has no labels)
TEST_SPLIT = {"cifar10": "official test", "cifar100": "official test", "food101": "official test",
              "tiny-imagenet": "official val", "imagenet-100": "official val", "fake": "FakeData"}


def _tf_small(train):
    """CIFAR / Tiny-ImageNet: small square images (32 / 64 px), upscaled to 224."""
    if train:
        return T.Compose([T.Resize(224), T.RandomCrop(224, padding=16, padding_mode="reflect"),
                          T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(MEAN, STD)])


def _tf_imagenet(train):
    """ImageNet-100 and Food-101: large non-square photos. Standard ImageNet recipe (crop ratio 224/256)."""
    if train:
        return T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


class _TinyVal(Dataset):
    def __init__(self, root, class_to_idx, tf):
        d = os.path.join(root, "val")
        self.items = []
        for line in open(os.path.join(d, "val_annotations.txt")):
            f, c = line.split("\t")[:2]
            self.items.append((os.path.join(d, "images", f), class_to_idx[c]))
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(default_loader(p)), y


def build_datasets(name, root):
    if name == "cifar10":
        tr = datasets.CIFAR10(root, True, _tf_small(True), download=True)
        te = datasets.CIFAR10(root, False, _tf_small(False), download=True)
        return tr, te, 10
    if name == "cifar100":
        tr = datasets.CIFAR100(root, True, _tf_small(True), download=True)
        te = datasets.CIFAR100(root, False, _tf_small(False), download=True)
        return tr, te, 100
    if name == "tiny-imagenet":
        base = os.path.join(root, "tiny-imagenet-200")
        if not os.path.isdir(base):
            os.makedirs(root, exist_ok=True)
            z = os.path.join(root, "tiny.zip")
            urllib.request.urlretrieve(TINY_URL, z)
            zipfile.ZipFile(z).extractall(root)
        tr = datasets.ImageFolder(os.path.join(base, "train"), _tf_small(True))
        te = _TinyVal(base, tr.class_to_idx, _tf_small(False))
        return tr, te, 200
    if name == "food101":  # 101 classes, 75,750 train / 25,250 test, long side <= 512 px, not from ImageNet (~5 GB)
        tr = datasets.Food101(root, "train", _tf_imagenet(True), download=True)
        te = datasets.Food101(root, "test", _tf_imagenet(False), download=True)
        tr.targets = list(tr._labels)  # for the stratified train / val split
        return tr, te, 101
    if name == "imagenet-100":  # root/imagenet-100/{train,val}/<class>/*.JPEG
        base = os.path.join(root, "imagenet-100")
        return (datasets.ImageFolder(os.path.join(base, "train"), _tf_imagenet(True)),
                datasets.ImageFolder(os.path.join(base, "val"), _tf_imagenet(False)), 100)
    raise ValueError(name)


def _subset(ds, n, seed=0):
    if not n or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())


def build_loaders(name, root, batch, workers=2, train_subset=0, test_subset=0):
    """train_subset / test_subset > 0: fixed random subset (same images for every method -> fair)."""
    tr, te, nc = build_datasets(name, root)
    tr, te = _subset(tr, train_subset), _subset(te, test_subset)
    return (DataLoader(tr, batch, shuffle=True, num_workers=workers, drop_last=True, pin_memory=True,
                       persistent_workers=workers > 0),
            DataLoader(te, batch * 2, shuffle=False, num_workers=workers, pin_memory=True), nc)


def _targets(ds):
    t = getattr(ds, "targets", None)
    if t is None:
        t = getattr(ds, "_labels", None)  # Food101
    if t is None and isinstance(ds, _TinyVal):
        t = [y for _, y in ds.items]
    return None if t is None else [int(v) for v in t]


def split_train_val(n, targets, val_size, seed=VAL_SPLIT_SEED):
    """Fixed stratified split (independent of the run seed -> every method sees the same val set)."""
    g = torch.Generator().manual_seed(seed)
    if targets is None:
        p = torch.randperm(n, generator=g).tolist()
        return p[val_size:], p[:val_size]
    by_cls = {}
    for i in torch.randperm(n, generator=g).tolist():
        by_cls.setdefault(targets[i], []).append(i)
    per = max(1, val_size // len(by_cls))
    va = [i for idx in by_cls.values() for i in idx[:per]]
    tr = [i for idx in by_cls.values() for i in idx[per:]]
    return tr, va


def build_loaders_val(name, root, batch, workers=2, val_size=5000, train_subset=0, test_subset=0, val_from="train"):
    """train / val / test loaders. Model selection and early stopping use val only; test is touched once, at the end.
    val_from="train": val = val_size images held out of the official train set (eval transform).
    val_from="test": the official test set (ImageNet-100: the official ImageNet val) is split in two halves,
      stratified per class with a fixed seed: one half is val, the other the test; the whole train set is used.
      For ImageNet-100: its train images are all in the ImageNet-1K pretraining set, so a val split of them is
      already "seen" by the checkpoint (val 3-6 points above test, best epoch 0-1 for Prune); the official val
      images are not. val_size is ignored."""
    assert val_from in ("train", "test"), val_from
    if name == "fake":  # CPU end-to-end test of the pipeline
        tr = datasets.FakeData(1200, (3, 224, 224), 10, _tf_small(True))
        te = datasets.FakeData(200, (3, 224, 224), 10, _tf_small(False), random_offset=10 ** 6)
        nc = 10
    else:
        tr, te, nc = build_datasets(name, root)
    if val_from == "test":
        te_idx, va_idx = split_train_val(len(te), _targets(te), len(te) // 2)
        tr_idx = list(range(len(tr)))
        va, te = Subset(te, va_idx), Subset(te, te_idx)  # both keep the eval transform of the test set
    else:
        ev = copy.copy(tr)
        ev.transform = te.transform if hasattr(te, "transform") else te.tf
        tr_idx, va_idx = split_train_val(len(tr), _targets(tr), val_size)
        va = Subset(ev, va_idx)
    if train_subset and train_subset < len(tr_idx):
        g = torch.Generator().manual_seed(0)
        tr_idx = [tr_idx[i] for i in torch.randperm(len(tr_idx), generator=g)[:train_subset].tolist()]
    tr, te = Subset(tr, tr_idx), _subset(te, test_subset)
    mk = lambda ds, b, shuf: DataLoader(ds, b, shuffle=shuf, num_workers=workers, drop_last=shuf, pin_memory=True,
                                        persistent_workers=workers > 0)
    return mk(tr, batch, True), mk(va, batch * 2, False), mk(te, batch * 2, False), nc


# ----------------------------------------------------------------------------- record of the data config
def _jsonable(v):
    if isinstance(v, enum.Enum):  # InterpolationMode.BILINEAR -> "bilinear"
        return v.value
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, torch.Tensor):
        return v.tolist()
    if v is None or isinstance(v, (str, bool, int, float)):
        return v
    return repr(v)


def describe_transform(tf):
    """Every step of a transform pipeline with ALL its attributes -> json-safe list of dicts.
    repr() is not enough: RandomCrop's repr omits padding_mode, Resize's omits nothing but antialias depends on
    the torchvision version. Nested Compose are flattened."""
    if tf is None:
        return None
    if isinstance(tf, T.Compose):
        return [step for t in tf.transforms for step in describe_transform(t)]
    attrs = {k: _jsonable(v) for k, v in vars(tf).items() if not k.startswith("_") and k != "training"}
    return [{"name": type(tf).__name__, **attrs}]


def _base(ds):
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


def transform_of(ds):
    """The transform a dataset (possibly wrapped in Subset) really applies: .transform, or .tf for _TinyVal."""
    b = _base(ds)
    t = getattr(b, "transform", None)
    return t if t is not None else getattr(b, "tf", None)


def data_config(name, train_loader, val_loader, test_loader, num_classes):
    """Read back from the loaders that are used (not from a description written by hand): split sizes and the
    train / val / test transforms. val must use the eval transform (same as test)."""
    va = val_loader.dataset if val_loader is not None else None
    tf = {"train": describe_transform(transform_of(train_loader.dataset)),
          "val": describe_transform(transform_of(va)) if va is not None else None,
          "test": describe_transform(transform_of(test_loader.dataset))}
    out = {"dataset": name, "num_classes": num_classes, "n_train": len(train_loader.dataset),
           "n_val": len(va) if va is not None else 0, "n_test": len(test_loader.dataset),
           "test_split": TEST_SPLIT.get(name, "?"), "pipeline_size": PIPELINE_SIZE, "transform": tf,
           "resize_note": "transforms get PIL images: PIL resizing is always antialiased, so antialias='warn' "
                          "(torchvision < 0.17 default) changes nothing here"}
    if va is not None:
        from_test = _base(va) is _base(test_loader.dataset)
        out["val_from"] = "test" if from_test else "train"
        out["val_split"] = (f"half of the {TEST_SPLIT.get(name, 'official test')} set, the other half is the test; "
                            f"stratified per class, fixed seed {VAL_SPLIT_SEED}; eval transform" if from_test else
                            f"held out from the official train set, stratified per class, fixed seed {VAL_SPLIT_SEED} "
                            "(same images for every run); eval transform")
        out["val_uses_eval_transform"] = tf["val"] == tf["test"]
    return out


def format_transform(desc):
    """describe_transform() output -> one readable line (logs, KET_QUA_*.md)."""
    if desc is None:
        return "-"
    def val(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        if isinstance(v, list):
            return "(" + ", ".join(val(x) for x in v) + ")"
        return str(v)
    steps = []
    for s in desc:
        args = ", ".join(f"{k}={val(v)}" for k, v in s.items() if k != "name")
        steps.append(f"{s['name']}({args})" if args else s["name"])
    return " -> ".join(steps)
