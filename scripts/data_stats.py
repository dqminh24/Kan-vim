"""Đo kích thước ảnh thật của các tập dữ liệu (chỉ đọc header ảnh, không cần torch).

  python scripts/data_stats.py data/tiny-imagenet-200/train data/tiny-imagenet-200/val
  python scripts/data_stats.py data/imagenet-100/train data/imagenet-100/val
  python scripts/data_stats.py data/food-101/images
  python scripts/data_stats.py --cifar data/cifar-100-python      # CIFAR lưu dạng mảng, không phải file ảnh

In ra: số ảnh, số lớp, min / trung vị / max của rộng, cao, cạnh ngắn, cạnh dài, 5 kích thước hay gặp nhất,
tỉ lệ ảnh không phải RGB (ảnh xám, CMYK...), và một patch 16x16 sau khi resize về 224 ứng với bao nhiêu pixel gốc.
"""
import argparse
import os
import pickle
import statistics
from collections import Counter

from PIL import Image

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def scan(folder, limit):
    sizes, modes, classes, n = [], Counter(), set(), 0
    for d, _, files in os.walk(folder, followlinks=True):
        for f in files:
            if not f.lower().endswith(EXT):
                continue
            try:
                with Image.open(os.path.join(d, f)) as im:
                    sizes.append(im.size)
                    modes[im.mode] += 1
            except OSError:
                modes["LỖI"] += 1
            classes.add(os.path.relpath(d, folder).split(os.sep)[0])
            n += 1
            if limit and n >= limit:
                return sizes, modes, classes
    return sizes, modes, classes


def report(name, sizes, modes, classes):
    if not sizes:
        print(f"\n== {name}: không thấy ảnh")
        return
    w = [s[0] for s in sizes]
    h = [s[1] for s in sizes]
    short = [min(s) for s in sizes]
    long_ = [max(s) for s in sizes]
    q = lambda v: f"{min(v)} / {statistics.median(v):g} / {max(v)}"
    print(f"\n== {name}")
    print(f"ảnh: {len(sizes)}   lớp (thư mục cấp 1): {len(classes)}")
    print(f"rộng      min/trung vị/max: {q(w)}")
    print(f"cao       min/trung vị/max: {q(h)}")
    print(f"cạnh ngắn min/trung vị/max: {q(short)}")
    print(f"cạnh dài  min/trung vị/max: {q(long_)}")
    print("hay gặp nhất: " + ", ".join(f"{a}x{b} ({c / len(sizes):.1%})" for (a, b), c in Counter(sizes).most_common(5)))
    print("chế độ màu: " + ", ".join(f"{m} {c / len(sizes):.2%}" for m, c in modes.most_common()))
    m = statistics.median(short)
    print(f"cạnh ngắn trung vị {m:g} px -> resize 224: hệ số x{224 / m:.2f}; "
          f"1 patch 16x16 ứng với ~{m / 14:.1f} px ảnh gốc")


def cifar(folder):
    for split in ("train", "test"):
        p = os.path.join(folder, split)
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        x = d["data"]
        n = len(x)
        labels = d.get("fine_labels", d.get("labels"))
        per_img = len(x[0]) if n else 0
        print(f"\n== CIFAR {split}: {n} ảnh, {len(set(labels))} lớp, {per_img} giá trị/ảnh "
              f"(= 3 x 32 x 32 nếu là 3072) -> resize 224: hệ số x7, 1 patch 16x16 ~ {32 / 14:.1f} px gốc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--cifar", help="thư mục cifar-100-python")
    ap.add_argument("--limit", type=int, default=0, help="chỉ đọc N ảnh đầu mỗi thư mục (0 = tất cả)")
    a = ap.parse_args()
    if a.cifar:
        cifar(a.cifar)
    for folder in a.folders:
        report(folder, *scan(folder, a.limit))


if __name__ == "__main__":
    main()
