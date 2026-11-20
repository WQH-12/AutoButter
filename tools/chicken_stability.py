"""量鸡妖各区域在连续帧之间是否稳定，用来判断模板该裁哪一块。

输出每块区域在 f00/f01/f02 之间的最大像素差。
差异小 = 该区域不动，适合做模板；差异大 = 有动画，不能直接当模板。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter\screenshots\burst\pad_test2"


def load(name):
    p = os.path.join(ROOT, name)
    img = cv2.imread(p)
    if img is None:
        raise SystemExit(f"missing {p}")
    return img


def main() -> int:
    frames = [load(f"cell0_f{i:02d}.png") for i in range(3)]
    print("frame sizes:", [f.shape for f in frames])

    h, w = frames[0].shape[:2]
    print(f"cell0 context: {w}x{h}\n")

    # 按行统计三帧之间的差异，找出「哪一段在动」
    print(" row range   max_diff  mean_diff   verdict")
    for y0 in range(0, h, 20):
        y1 = min(y0 + 20, h)
        a, b, c = (f[y0:y1] for f in frames)
        d1 = cv2.absdiff(a, b)
        d2 = cv2.absdiff(b, c)
        d = np.maximum(d1, d2)
        g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        mx, mn = int(g.max()), float(g.mean())
        verdict = "static" if mx < 12 else ("small" if mx < 30 else "ANIMATED")
        print(f" {y0:4d}..{y1:4d}   {mx:4d}     {mn:7.2f}   {verdict}")

    # 保存三帧的差异热图
    diff = np.maximum(cv2.absdiff(frames[0], frames[1]),
                      cv2.absdiff(frames[1], frames[2]))
    heat = cv2.applyColorMap(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    out = np.hstack([frames[0], frames[1], frames[2], heat])
    dst = os.path.join(ROOT, "_chicken_diff.png")
    cv2.imwrite(dst, out)
    print(f"\nsaved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
