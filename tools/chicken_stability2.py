"""先对齐再比较，区分「整体位移」和「真实动画」。

用相位相关估出帧间整体平移，对齐后再看残余差异。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter\screenshots\burst\pad_test2"


def load(i):
    return cv2.imread(os.path.join(ROOT, f"cell0_f{i:02d}.png"))


def phase_shift(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ga, gb, win)
    return dx, dy, resp


def shift_img(img, dx, dy):
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def main() -> int:
    f = [load(i) for i in range(3)]
    print("frames:", [x.shape for x in f])

    for i, j in ((0, 1), (1, 2), (0, 2)):
        dx, dy, resp = phase_shift(f[i], f[j])
        print(f"f{i:02d} -> f{j:02d}: shift=({dx:+.3f},{dy:+.3f}) response={resp:.4f}")

    dx, dy, _ = phase_shift(f[0], f[1])
    a = f[0]
    b = shift_img(f[1], dx, dy)
    c = f[2]
    dx2, dy2, _ = phase_shift(f[1], f[2])
    c = shift_img(f[2], dx + dx2, dy + dy2)

    d = np.maximum(cv2.absdiff(a, b), cv2.absdiff(a, c))
    g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    print(f"\nafter alignment: max={g.max()} mean={g.mean():.2f}")

    print("\n row range   max_diff  mean_diff   verdict")
    h = g.shape[0]
    for y0 in range(0, h, 20):
        y1 = min(y0 + 20, h)
        band = g[y0:y1]
        mx, mn = int(band.max()), float(band.mean())
        verdict = "static" if mx < 20 else ("small" if mx < 60 else "ANIMATED")
        print(f" {y0:4d}..{y1:4d}   {mx:4d}     {mn:7.2f}   {verdict}")

    heat = cv2.applyColorMap(g, cv2.COLORMAP_JET)
    out = np.hstack([a, b, c, heat])
    dst = os.path.join(ROOT, "_chicken_diff_aligned.png")
    cv2.imwrite(dst, out)
    print(f"\nsaved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
