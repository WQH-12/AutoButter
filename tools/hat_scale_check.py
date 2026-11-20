"""检查三帧的斗笠尺寸是否一致（缩放会毁掉模板匹配）。"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BURST = r"E:\work\AutoButter\screenshots\burst\pad_test2"


def hat_mask(img):
    """斗笠是灰褐色 (BGR 约 60,70,85)，用它做掩码。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 灰褐：低饱和，V 中等
    m = ((hsv[:, :, 1] > 25) & (hsv[:, :, 1] < 110) &
         (hsv[:, :, 2] > 60) & (hsv[:, :, 2] < 170) &
         (hsv[:, :, 0] > 5) & (hsv[:, :, 0] < 40))
    return (m.astype(np.uint8) * 255)


def main() -> int:
    for i in range(3):
        img = cv2.imread(os.path.join(BURST, f"cell0_f{i:02d}.png"))
        top = img[:130]
        m = hat_mask(top)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
        if n <= 1:
            print(f"f{i:02d}: no hat blob")
            continue
        # 最大连通域
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = stats[idx]
        print(f"f{i:02d}: hat bbox=({x},{y},{w},{h}) area={area} "
              f"center=({cents[idx][0]:.1f},{cents[idx][1]:.1f})")
        cv2.imwrite(os.path.join(BURST, f"_hatmask_f{i:02d}.png"), m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
