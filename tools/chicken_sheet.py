"""把 frontage_* / back_* 拼成一张对照图，用于人工确认两种鸡妖姿态。"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

D = r"E:\work\AutoButter\User_screenshot"
OUT = r"E:\work\AutoButter\screenshots"
SCALE = 4


def load(name):
    p = os.path.join(D, name + ".png")
    raw = np.fromfile(p, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read {p}")
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        bgr = img[:, :, :3]
        # 把透明部分铺成中灰，避免棋盘格干扰观察
        bg = np.full_like(bgr, 90)
        a = (alpha.astype(np.float32) / 255.0)[:, :, None]
        img = (bgr.astype(np.float32) * a + bg.astype(np.float32) * (1 - a)).astype(np.uint8)
        print(f"  {name:12s} alpha min={alpha.min()} max={alpha.max()} "
              f"transparent={(alpha < 128).mean() * 100:.1f}%")
    return img


def main() -> int:
    groups = [("frontage", 3), ("back", 3)]
    rows = []
    for prefix, n in groups:
        tiles = []
        for i in range(1, n + 1):
            img = load(f"{prefix}_{i}")
            big = cv2.resize(img, (img.shape[1] * SCALE, img.shape[0] * SCALE),
                             interpolation=cv2.INTER_NEAREST)
            cv2.putText(big, f"{prefix}_{i}", (6, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)
            tiles.append(big)
        pad = 10
        w = sum(t.shape[1] for t in tiles) + pad * (len(tiles) + 1)
        h = max(t.shape[0] for t in tiles) + pad * 2
        row = np.full((h, w, 3), 30, np.uint8)
        x = pad
        for t in tiles:
            row[pad:pad + t.shape[0], x:x + t.shape[1]] = t
            x += t.shape[1] + pad
        rows.append(row)

    max_w = max(r.shape[1] for r in rows)
    total_h = sum(r.shape[0] for r in rows)
    sheet = np.full((total_h, max_w, 3), 30, np.uint8)
    y = 0
    for r in rows:
        sheet[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0]

    dst = os.path.join(OUT, "_chicken_poses.png")
    cv2.imwrite(dst, sheet)
    print(f"\nsaved {dst} ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
