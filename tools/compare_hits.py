"""把命中区域和模板并排放大，人工核对是不是真的同一个物品。"""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter"
IMAGE_DIR = os.path.join(ROOT, "resource", "image")
SHOT = os.path.join(ROOT, "screenshots", "159_before_drain.png")

# 从 overlay 输出抄下来的命中框
HITS = [
    ("alcohol", 0.9452, (174, 583, 66, 47)),
    ("berry", 0.9462, (277, 546, 66, 47)),
    ("silver", 0.9986, (73, 697, 66, 47)),
    ("berry", 0.9808, (480, 703, 66, 47)),
    ("berry", 0.9357, (378, 723, 66, 47)),
    ("honey", 0.9862, (582, 718, 66, 47)),
]

SCALE = 4


def main() -> int:
    img = cv2.imread(SHOT)
    if img is None:
        print("cannot read shot")
        return 1

    rows = []
    for name, score, (x, y, w, h) in HITS:
        tpl = cv2.imread(os.path.join(IMAGE_DIR, f"item_{name}.png"))
        hit = img[y:y + h, x:x + w]
        # 命中区域再扩一圈，看格子底色和等级文字
        ctx = img[max(0, y - 20):y + h + 24, max(0, x - 20):x + w + 20]

        tpl_big = cv2.resize(tpl, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
        hit_big = cv2.resize(hit, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
        ctx_big = cv2.resize(ctx, (ctx.shape[1] * SCALE, ctx.shape[0] * SCALE),
                             interpolation=cv2.INTER_NEAREST)

        pad = 12
        row_h = max(tpl_big.shape[0], hit_big.shape[0], ctx_big.shape[0]) + pad * 2
        row_w = tpl_big.shape[1] + hit_big.shape[1] + ctx_big.shape[1] + pad * 4
        row = np.full((row_h, row_w, 3), 30, np.uint8)
        cx = pad
        for tile in (tpl_big, hit_big, ctx_big):
            row[pad:pad + tile.shape[0], cx:cx + tile.shape[1]] = tile
            cx += tile.shape[1] + pad
        cv2.putText(row, f"{name} {score:.3f}", (pad, row_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
        print(f"{name:8s} {score:.4f}  box=({x},{y})")

    max_w = max(r.shape[1] for r in rows)
    total_h = sum(r.shape[0] for r in rows)
    sheet = np.full((total_h, max_w, 3), 30, np.uint8)
    y = 0
    for r in rows:
        sheet[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0]

    dst = os.path.join(ROOT, "screenshots", "_compare_hits.png")
    cv2.imwrite(dst, sheet)
    print(f"\nsaved {dst}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print("每行: [模板] [命中区域] [命中区域+上下文]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
