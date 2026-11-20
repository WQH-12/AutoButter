"""测试鸡妖各个部位在帧间的稳定性，找出不随旋转变化的部分。"""
from __future__ import annotations

import os
import sys

import cv2

sys.stdout.reconfigure(encoding="utf-8")

BURST = r"E:\work\AutoButter\screenshots\burst\pad_test2"
SHOT = r"E:\work\AutoButter\screenshots\159_before_drain.png"


def score(img, tpl, roi=None):
    if roi:
        x, y, w, h = roi
        img = img[y:y + h, x:x + w]
    if img.shape[0] < tpl.shape[0] or img.shape[1] < tpl.shape[1]:
        return -1.0
    res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    return float(cv2.minMaxLoc(res)[1])


def main() -> int:
    frames = [cv2.imread(os.path.join(BURST, f"cell0_f{i:02d}.png")) for i in range(3)]
    shot = cv2.imread(SHOT)
    print("frame", frames[0].shape, " shot", shot.shape)

    # 网格图给出：斗笠 x20..115 y10..65；鸡冠 y0..20；黄身 y60..110
    parts = {
        "comb_only":     (55, 0, 55, 25),    # 红鸡冠
        "comb_hat_top":  (40, 0, 75, 40),    # 鸡冠+斗笠顶
        "body_yellow":   (35, 60, 80, 45),   # 黄色身体
        "hat_ear_left":  (18, 40, 30, 30),   # 左耳
        "hat_ear_right": (100, 40, 30, 30),  # 右耳
        "hat_full":      (20, 8, 95, 60),    # 整个斗笠
        "hat_center":    (45, 15, 55, 45),   # 斗笠中央图案
        "comb_body":     (35, 0, 80, 105),   # 鸡冠+身体（跨过斗笠）
    }

    print(f"\n{'part':16s} {'size':9s} {'f01':>6s} {'f02':>6s} {'159':>6s}  verdict")
    for name, (x, y, w, h) in parts.items():
        tpl = frames[0][y:y + h, x:x + w]
        cv2.imwrite(os.path.join(BURST, f"_part_{name}.png"), tpl)
        s1 = score(frames[1], tpl, (0, 0, 146, 170))
        s2 = score(frames[2], tpl, (0, 0, 146, 170))
        s3 = score(shot, tpl, (0, 250, 720, 420))
        worst = min(s1, s2)
        verdict = "GOOD" if worst >= 0.88 else ("marginal" if worst >= 0.75 else "rotates")
        print(f"{name:16s} {w}x{h:<6d} {s1:6.3f} {s2:6.3f} {s3:6.3f}  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
