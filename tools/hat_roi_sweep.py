"""穷举斗笠周围的小区域，找出跨帧稳定的候选模板。"""
from __future__ import annotations

import os
import sys

import cv2

sys.stdout.reconfigure(encoding="utf-8")

BURST = r"E:\work\AutoButter\screenshots\burst\pad_test2"
SHOT = r"E:\work\AutoButter\screenshots\159_before_drain.png"


def score(img, tpl, roi):
    x, y, w, h = roi
    crop = img[y:y + h, x:x + w]
    if crop.shape[0] < tpl.shape[0] or crop.shape[1] < tpl.shape[1]:
        return -1.0, None
    res = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, ml = cv2.minMaxLoc(res)
    return float(mx), ml


def main() -> int:
    frames = [cv2.imread(os.path.join(BURST, f"cell0_f{i:02d}.png")) for i in range(3)]
    shot = cv2.imread(SHOT)
    fr = (0, 0, 146, 200)
    sr = (0, 250, 720, 420)

    results = []
    # 只在鸡妖本体范围内滑窗：x 25..125, y 5..120
    # 排除左边缘 x<25（绿色边框），那里是纯色块，会给出假的 1.000
    for size in (24, 32, 40, 48):
        for y in range(5, 121 - size, 6):
            for x in range(25, 126 - size, 6):
                tpl = frames[0][y:y + size, x:x + size]
                g = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
                if g.std() < 20:      # 太素，没有特征
                    continue
                s1, _ = score(frames[1], tpl, fr)
                s2, _ = score(frames[2], tpl, fr)
                worst = min(s1, s2)
                if worst >= 0.80:
                    results.append((worst, size, x, y, s1, s2))

    results.sort(reverse=True)
    print(f"candidates with min(f01,f02) >= 0.80 : {len(results)}\n")
    print(f"{'worst':>6s} {'size':>5s} {'x':>4s} {'y':>4s} {'f01':>6s} {'f02':>6s}")
    for worst, size, x, y, s1, s2 in results[:15]:
        print(f"{worst:6.3f} {size:5d} {x:4d} {y:4d} {s1:6.3f} {s2:6.3f}")

    if results:
        worst, size, x, y, s1, s2 = results[0]
        tpl = frames[0][y:y + size, x:x + size]
        dst = os.path.join(BURST, "_best_hat_part.png")
        cv2.imwrite(dst, cv2.resize(tpl, (size * 6, size * 6),
                                    interpolation=cv2.INTER_NEAREST))
        s3, ml = score(shot, tpl, sr)
        print(f"\nbest: {size}x{size} @({x},{y}) f01={s1:.3f} f02={s2:.3f} 159={s3:.3f}")
        print(f"saved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
