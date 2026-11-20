"""把 7 张物品模板放大拼成一张图，用于人工确认每个模板对应的物品。"""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter"
IMAGE_DIR = os.path.join(ROOT, "resource", "image")
SCALE = 5
COLS = 4


def main() -> int:
    items = []
    for p in sorted(glob.glob(os.path.join(IMAGE_DIR, "item_*.png"))):
        name = os.path.basename(p)[:-4].replace("item_", "")
        tpl = cv2.imread(p)
        if tpl is None:
            continue
        big = cv2.resize(tpl, (tpl.shape[1] * SCALE, tpl.shape[0] * SCALE),
                         interpolation=cv2.INTER_NEAREST)
        items.append((name, big))
        print(f"{name:10s} {tpl.shape[1]}x{tpl.shape[0]}")

    if not items:
        print("no templates")
        return 1

    cell_w = max(t.shape[1] for _, t in items) + 24
    cell_h = max(t.shape[0] for _, t in items) + 56
    rows = (len(items) + COLS - 1) // COLS
    sheet = np.full((rows * cell_h, COLS * cell_w, 3), 30, np.uint8)

    for i, (name, t) in enumerate(items):
        r, c = divmod(i, COLS)
        ox = c * cell_w + 12
        oy = r * cell_h + 40
        sheet[oy:oy + t.shape[0], ox:ox + t.shape[1]] = t
        cv2.putText(sheet, f"item_{name}", (ox, r * cell_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    dst = os.path.join(ROOT, "screenshots", "_template_sheet.png")
    cv2.imwrite(dst, sheet)
    print(f"\nsaved {dst}  ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
