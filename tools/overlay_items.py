"""把物品模板的命中位置画到截图上，用于人工核对是否真命中。

用法:
    python tools/overlay_items.py 159_before_drain.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter"
IMAGE_DIR = os.path.join(ROOT, "resource", "image")
SHOT_DIR = os.path.join(ROOT, "screenshots")
CONTENT_ROI = (0, 380, 720, 920)
THR = 0.88


def find_all(img, tpl, thr, roi):
    x, y, w, h = roi
    crop = img[y:y + h, x:x + w]
    res = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
    out = []
    r = res.copy()
    while True:
        _, mx, _, ml = cv2.minMaxLoc(r)
        if mx < thr:
            break
        out.append((float(mx), x + ml[0], y + ml[1],
                    tpl.shape[1], tpl.shape[0]))
        r[max(0, ml[1] - 30):ml[1] + 30, max(0, ml[0] - 30):ml[0] + 30] = -1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shot")
    ap.add_argument("--thr", type=float, default=THR)
    args = ap.parse_args()

    path = os.path.join(SHOT_DIR, args.shot)
    img = cv2.imread(path)
    if img is None:
        print(f"cannot read {path}")
        return 1
    h, w = img.shape[:2]
    if w != 720:
        img = cv2.resize(img, (720, int(round(h * 720 / w))), interpolation=cv2.INTER_AREA)

    out = img.copy()
    colors = {
        "alcohol": (0, 0, 255), "berry": (0, 128, 255), "fabric": (0, 200, 200),
        "honey": (255, 0, 255), "jade": (0, 255, 0), "silver": (255, 0, 0),
        "stone": (128, 0, 255),
    }
    for p in sorted(glob.glob(os.path.join(IMAGE_DIR, "item_*.png"))):
        name = os.path.basename(p)[:-4].replace("item_", "")
        tpl = cv2.imread(p)
        if tpl is None:
            continue
        for s, x, y, tw, th in find_all(img, tpl, args.thr, CONTENT_ROI):
            c = colors.get(name, (255, 255, 255))
            cv2.rectangle(out, (x, y), (x + tw, y + th), c, 3)
            cv2.putText(out, f"{name} {s:.2f}", (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
            print(f"  {name:8s} {s:.4f} box=({x},{y},{tw},{th})")

    dst = os.path.join(SHOT_DIR, f"_overlay_{args.shot}")
    cv2.imwrite(dst, out)
    print(f"\nsaved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
