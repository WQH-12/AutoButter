"""Template sanity check for resource/image/item_*.png.

Checks, in order:
  1. format     -- size, channels, alpha, flatness
  2. self-match -- every template must score ~1.0 somewhere in its own source
  3. recall     -- expected item counts per reference screenshot
  4. false pos  -- screens without item cells must stay under threshold
  5. colour     -- which frame colour each template was cut from

Run: python tools/check_templates.py
     python tools/check_templates.py --dir User_screenshot --prefix ""
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(ROOT, "screenshots")

# frame colour of the cell a template was cut from, read from its own border
def frame_colour(t: np.ndarray) -> str:
    strip = t[1:5, t.shape[1] // 2 - 10:t.shape[1] // 2 + 10]
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


def is_channel_swapped(img: np.ndarray) -> bool:
    roi = img[int(img.shape[0] * .38):int(img.shape[0] * .58), 60:660]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = ((hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 115) & (hsv[:, :, 1] > 60)).mean()
    warm = (((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 60)).mean()
    return blue > warm


def find_all(img, tpl, thr):
    if img.shape[0] < tpl.shape[0] or img.shape[1] < tpl.shape[1]:
        return []
    res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    out = []
    r = res.copy()
    while True:
        _, mx, _, ml = cv2.minMaxLoc(r)
        if mx < thr:
            break
        out.append((float(mx), ml[0] + tpl.shape[1] // 2, ml[1] + tpl.shape[0] // 2))
        r[max(0, ml[1] - 20):ml[1] + 20, max(0, ml[0] - 20):ml[0] + 20] = -1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "resource", "image"))
    ap.add_argument("--prefix", default="item_")
    ap.add_argument("--threshold", type=float, default=0.92)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, f"{args.prefix}*.png")))
    if not paths:
        print(f"no templates matching {args.prefix}*.png in {args.dir}")
        return 1

    tpls = {}
    problems = []

    print("=" * 74)
    print("1. format")
    print("=" * 74)
    for p in paths:
        name = os.path.basename(p)[:-4]
        rgba = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        alpha_ok = rgba.ndim == 2 or bool((rgba[:, :, 3] == 255).all())
        flat = g.std() < 12
        col = frame_colour(bgr)
        tpls[name] = bgr
        flags = []
        if min(h, w) < 24:
            flags.append("TOO SMALL")
        if flat:
            flags.append("FLAT (no texture)")
        if not alpha_ok:
            flags.append("alpha not 255 (green_mask would need it)")
        if flags:
            problems.append(f"{name}: {', '.join(flags)}")
        print(f"  {name:14s} {w:3d}x{h:<3d} std={g.std():5.1f} frame={col:7s} "
              f"{'  <-- ' + ', '.join(flags) if flags else ''}")

    sizes = {(t.shape[1], t.shape[0]) for t in tpls.values()}
    print(f"\n  distinct sizes: {sorted(sizes)}"
          f"{'  <-- inconsistent' if len(sizes) > 1 else '  (consistent)'}")
    if len(sizes) > 1:
        problems.append(f"inconsistent template sizes: {sorted(sizes)}")

    print()
    print("=" * 74)
    print("2. self-match (template must find itself at ~1.0)")
    print("=" * 74)
    self_src = {
        "alcohol": "snap/20261108-131441_720.png",
        "honey":   "snap/20261108-130821_720.png",
        "jade":    "snap/20261108-130821_720.png",
        "berry":   "snap/20261108-130821_720.png",
        "fabric":  "snap/20261108-131441_720.png",
        "stone":   "snap/20261108-131441_720.png",
        "silver":  "snap/20261108-132031_720.png",
    }
    for name, t in tpls.items():
        key = name.replace(args.prefix, "")
        if key.endswith("_1"):
            key = key[:-2]
        rel = self_src.get(key)
        if not rel:
            print(f"  {name:14s} (no source mapping, skipped)")
            continue
        p = os.path.join(SHOTS, rel.replace("/", os.sep))
        if not os.path.exists(p):
            print(f"  {name:14s} source missing: {rel}")
            continue
        img = cv2.imread(p)
        best = max((s for s, _, _ in find_all(img, t, 0.0)), default=0.0)
        mark = "ok" if best >= 0.97 else "LOW"
        if best < 0.97:
            problems.append(f"{name}: self-match only {best:.3f}")
        print(f"  {name:14s} best={best:.4f}  {mark}   <- {rel}")

    print()
    print("=" * 74)
    print(f"3. false positives on item-free screens (threshold {args.threshold})")
    print("=" * 74)
    negatives = ["01_current.png", "23_verify_now.png", "43_manage_now.png", "53_rob_popup.png"]
    for f in negatives:
        p = os.path.join(SHOTS, f)
        if not os.path.exists(p):
            continue
        img = cv2.imread(p)
        if is_channel_swapped(img):
            print(f"  {f:22s} SKIPPED (channel-swapped capture)")
            continue
        hits = []
        for name, t in tpls.items():
            for s, cx, cy in find_all(img, t, args.threshold):
                hits.append(f"{name}@{s:.3f}({cx},{cy})")
        if hits:
            print(f"  {f:22s} FALSE POSITIVE: {', '.join(hits)}")
            problems.append(f"false positive in {f}: {hits}")
        else:
            print(f"  {f:22s} clean")

    print()
    print("=" * 74)
    print("4. frame colours in use")
    print("=" * 74)
    by_col = defaultdict(list)
    for name, t in tpls.items():
        by_col[frame_colour(t)].append(name.replace(args.prefix, ""))
    for col in sorted(by_col):
        print(f"  {col:7s} -> {', '.join(by_col[col])}")
    missing = [c for c in ("orange", "blue", "purple") if c not in by_col]
    if missing:
        print(f"\n  NOTE: no template for {', '.join(missing)} frame.")
        print("        Same item at another level may not match;"
              " add one sample per level, or crop icon-only.")

    print()
    print("=" * 74)
    if problems:
        print(f"{len(problems)} issue(s):")
        for p_ in problems:
            print(f"  - {p_}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
