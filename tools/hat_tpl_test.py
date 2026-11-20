"""量鸡妖斗笠区域：帧间稳定性 + 候选模板裁切范围。

输出：
  1. 带坐标网格的放大图，用来确定斗笠边界
  2. 几组候选裁切作为模板，在其它帧和另一张截图上的匹配分数
"""
from __future__ import annotations

import os
import sys

import cv2

sys.stdout.reconfigure(encoding="utf-8")

BURST = r"E:\work\AutoButter\screenshots\burst\pad_test2"
SHOTS = r"E:\work\AutoButter\screenshots"


def load(p):
    img = cv2.imread(p)
    if img is None:
        raise SystemExit(f"missing {p}")
    return img


def grid(img, scale=4, step=10):
    big = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale),
                     interpolation=cv2.INTER_NEAREST)
    h, w = big.shape[:2]
    for x in range(0, img.shape[1], step):
        c = (0, 255, 0) if x % 50 == 0 else (70, 70, 70)
        cv2.line(big, (x * scale, 0), (x * scale, h), c, 1)
        if x % 50 == 0:
            cv2.putText(big, str(x), (x * scale + 2, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    for y in range(0, img.shape[0], step):
        c = (0, 255, 0) if y % 50 == 0 else (70, 70, 70)
        cv2.line(big, (0, y * scale), (w, y * scale), c, 1)
        if y % 50 == 0:
            cv2.putText(big, str(y), (2, y * scale + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return big


def score(img, tpl, roi=None):
    if roi:
        x, y, w, h = roi
        img = img[y:y + h, x:x + w]
    if img.shape[0] < tpl.shape[0] or img.shape[1] < tpl.shape[1]:
        return -1.0, None
    res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, ml = cv2.minMaxLoc(res)
    return float(mx), ml


def main() -> int:
    frames = [load(os.path.join(BURST, f"cell0_f{i:02d}.png")) for i in range(3)]
    print("frame:", frames[0].shape)

    big = grid(frames[0][:130])
    cv2.imwrite(os.path.join(BURST, "_hat_grid.png"), big)
    print(f"saved _hat_grid.png {big.shape[1]}x{big.shape[0]}\n")

    cands = {
        "A_full":   (30, 5, 90, 75),
        "B_tight":  (35, 10, 80, 65),
        "C_wide":   (25, 0, 100, 85),
        "D_nocomb": (35, 22, 80, 55),
    }
    shot = load(os.path.join(SHOTS, "159_before_drain.png"))

    print(f"{'cand':10s} {'size':8s} {'f00':>6s} {'f01':>6s} {'f02':>6s} {'159':>6s}")
    for name, (x, y, w, h) in cands.items():
        tpl = frames[0][y:y + h, x:x + w]
        cv2.imwrite(os.path.join(BURST, f"_hat_{name}.png"), tpl)
        row = [f"{name:10s} {w}x{h:<3d}"]
        for f in frames:
            s, _ = score(f, tpl, (0, 0, f.shape[1], 170))
            row.append(f"{s:6.3f}")
        s, _ = score(shot, tpl, (0, 250, 720, 420))
        row.append(f"{s:6.3f}")
        print(" ".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
