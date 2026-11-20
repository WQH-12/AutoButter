"""把三帧的斗笠区域并排放大，确认姿态差异到底有多大。"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BURST = r"E:\work\AutoButter\screenshots\burst\pad_test2"
SCALE = 5
BOX = (15, 0, 110, 75)  # x, y, w, h


def main() -> int:
    tiles = []
    for i in range(3):
        img = cv2.imread(os.path.join(BURST, f"cell0_f{i:02d}.png"))
        if img is None:
            raise SystemExit("missing frame")
        x, y, w, h = BOX
        crop = img[y:y + h, x:x + w]
        big = cv2.resize(crop, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, f"f{i:02d}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(big)
        print(f"f{i:02d} crop {w}x{h} from ({x},{y})")

    pad = 10
    total_w = sum(t.shape[1] for t in tiles) + pad * (len(tiles) + 1)
    h = tiles[0].shape[0] + pad * 2
    sheet = np.full((h, total_w, 3), 30, np.uint8)
    x = pad
    for t in tiles:
        sheet[pad:pad + t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + pad

    dst = os.path.join(BURST, "_hat_compare.png")
    cv2.imwrite(dst, sheet)
    print(f"\nsaved {dst} ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
