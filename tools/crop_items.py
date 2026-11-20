"""Crop item icons out of a 720-wide snapshot.

Takes x,y,w,h boxes read off a snapshot, trims a fixed inset so the coloured
level frame and the "N级" text stay out of the template, and writes the
results into resource/image/.

    python tools/crop_items.py --snap snap/20261108-130821_720.png ^
        --item alcohol=163,568,87,91 --item honey=460,683,95,93

    # preview only, nothing written
    python tools/crop_items.py --snap snap/20261108-130821_720.png --item x=1,2,3,4 --dry-run

    # different inset (default 6 px, keeps the icon, drops the frame)
    python tools/crop_items.py ... --inset 8
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(ROOT, "screenshots")
OUT = os.path.join(ROOT, "resource", "image")


def parse_item(text: str):
    if "=" not in text:
        raise argparse.ArgumentTypeError("want name=x,y,w,h")
    name, box = text.split("=", 1)
    try:
        parts = [int(v) for v in box.replace(" ", "").split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad box for {name}: {box}")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"bad box for {name}: {box}")
    return name, tuple(parts)


def frame_colour(patch: np.ndarray) -> str:
    strip = patch[0:3, patch.shape[1] // 2 - 10:patch.shape[1] // 2 + 10]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    h = int(np.median(hsv[:, :, 0]))
    s = int(np.median(hsv[:, :, 1]))
    if s < 55:
        return "gray"
    if h < 22 or h > 170:
        return "orange"
    if 22 <= h < 40:
        return "yellow"
    if 40 <= h < 92:
        return "green"
    if 92 <= h <= 112:
        return "blue"
    return "purple"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True, help="path under screenshots/, or absolute")
    ap.add_argument("--item", action="append", required=True, type=parse_item,
                    metavar="name=x,y,w,h")
    ap.add_argument("--inset", type=int, default=6,
                    help="pixels trimmed from every side (default 6)")
    ap.add_argument("--inset-x", type=int, default=None,
                    help="horizontal inset, overrides --inset")
    ap.add_argument("--inset-top", type=int, default=None,
                    help="top inset, overrides --inset")
    ap.add_argument("--drop-bottom", type=int, default=0,
                    help="extra pixels trimmed from the bottom (level text)")
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--prefix", default="item_")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    snap = args.snap if os.path.isabs(args.snap) else os.path.join(SHOTS, args.snap)
    if not os.path.exists(snap):
        print(f"[error] snapshot not found: {snap}", file=sys.stderr)
        return 1
    img = cv2.imread(snap)
    if img is None:
        print(f"[error] cannot read {snap}", file=sys.stderr)
        return 1
    print(f"[snap] {os.path.relpath(snap, ROOT)}  {img.shape[1]}x{img.shape[0]}")

    made = []
    for name, (x, y, w, h) in args.item:
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > img.shape[1] or y + h > img.shape[0]:
            print(f"[skip] {name}: box {x},{y},{w},{h} outside {img.shape[1]}x{img.shape[0]}")
            continue
        ix = args.inset if args.inset_x is None else args.inset_x
        it = args.inset if args.inset_top is None else args.inset_top
        x0, y0 = x + ix, y + it
        x1, y1 = x + w - ix, y + h - it - args.drop_bottom
        if x1 - x0 < 16 or y1 - y0 < 16:
            print(f"[skip] {name}: too small after inset ({x1-x0}x{y1-y0})")
            continue
        patch = img[y0:y1, x0:x1]
        col = frame_colour(img[y:y+h, x:x+w])
        g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        dst = os.path.join(args.out_dir, f"{args.prefix}{name}.png")
        made.append((name, patch, col, g.std(), dst))
        if not args.dry_run:
            os.makedirs(args.out_dir, exist_ok=True)
            cv2.imwrite(dst, patch)
        print(f"  {name:10s} {patch.shape[1]:3d}x{patch.shape[0]:<3d} frame={col:7s} "
              f"std={g.std():5.1f}  -> {os.path.relpath(dst, ROOT)}")

    if not made:
        return 1

    # preview strip
    H = 120
    tiles = []
    for name, patch, col, _, _ in made:
        t = cv2.resize(patch, (int(patch.shape[1] * H / patch.shape[0]), H),
                       interpolation=cv2.INTER_NEAREST)
        t = cv2.copyMakeBorder(t, 22, 0, 0, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, name, (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        tiles.append(t)
    prev = os.path.join(SHOTS, "80_crop_preview.png")
    cv2.imwrite(prev, np.hstack(tiles))
    print(f"[preview] {os.path.relpath(prev, ROOT)}")

    sizes = {(m[1].shape[1], m[1].shape[0]) for m in made}
    if len(sizes) > 1:
        print(f"[warn] mixed sizes {sorted(sizes)} -- templates should share one size")
    print(f"[done] {len(made)} template(s)"
          f"{' (dry run, nothing written)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
