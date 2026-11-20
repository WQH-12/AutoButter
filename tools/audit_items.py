"""审计 7 张物品模板在实机截图上的覆盖情况。

回答两个问题：
  1. 已知 7 种物品，各自在哪些截图里能被 0.88 阈值命中
  2. 画面上出现了但没有模板的物品（漏检）

用法:
    python tools/audit_items.py
    python tools/audit_items.py --shots 159_before_drain.png 160_after_wait.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter"
IMAGE_DIR = os.path.join(ROOT, "resource", "image")
SHOT_DIR = os.path.join(ROOT, "screenshots")

# 秘境内容区：排除 y>1300 的底栏（树果图标会稳定误报 0.888~0.890）
CONTENT_ROI = (0, 380, 720, 920)
THR = 0.88


def find_all(img, tpl, thr, roi):
    x, y, w, h = roi
    crop = img[y:y + h, x:x + w]
    if crop.shape[0] < tpl.shape[0] or crop.shape[1] < tpl.shape[1]:
        return []
    res = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
    out = []
    r = res.copy()
    while True:
        _, mx, _, ml = cv2.minMaxLoc(r)
        if mx < thr:
            break
        out.append((float(mx), x + ml[0] + tpl.shape[1] // 2,
                    y + ml[1] + tpl.shape[0] // 2))
        r[max(0, ml[1] - 30):ml[1] + 30, max(0, ml[0] - 30):ml[0] + 30] = -1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", nargs="*", default=[])
    ap.add_argument("--thr", type=float, default=THR)
    args = ap.parse_args()

    tpls = {}
    for p in sorted(glob.glob(os.path.join(IMAGE_DIR, "item_*.png"))):
        name = os.path.basename(p)[:-4].replace("item_", "")
        tpl = cv2.imread(p)
        if tpl is not None:
            tpls[name] = tpl
    print(f"templates: {sorted(tpls)}  (threshold {args.thr})\n")

    shots = args.shots or [
        "159_before_drain.png", "160_after_wait.png",
        "154_after_fix.png", "153_before.png", "132_live_popup.png",
    ]

    for shot in shots:
        path = os.path.join(SHOT_DIR, shot)
        img = cv2.imread(path)
        if img is None:
            print(f"{shot:26s} MISSING")
            continue
        h, w = img.shape[:2]
        # 模板都是 720 短边裁的，大图先缩到 720 再匹配
        if w != 720:
            scale = 720 / w
            img = cv2.resize(img, (720, int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
            print(f"{shot:26s} {w}x{h} -> resized to 720x{img.shape[0]}")
        else:
            print(f"{shot:26s} {w}x{h}")
        total = 0
        for name, tpl in tpls.items():
            hits = find_all(img, tpl, args.thr, CONTENT_ROI)
            if hits:
                total += len(hits)
                desc = ", ".join(f"{s:.3f}@({cx},{cy})" for s, cx, cy in hits)
                print(f"    {name:8s} {len(hits)}x  {desc}")
        if total == 0:
            print("    (no item template matched)")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
