"""测旋转不变特征匹配（SIFT/ORB）能否稳定认出鸡妖斗笠。

Maa 的 FeatureMatch 就是这一类算法，如果 SIFT 能跨旋转稳定匹配，
就可以用 FeatureMatch 而不是 TemplateMatch。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BURST = r"E:\work\AutoButter\screenshots\burst\pad_test2"
SHOT = r"E:\work\AutoButter\screenshots\159_before_drain.png"
BOX = (20, 5, 100, 62)   # 斗笠整体


def main() -> int:
    frames = [cv2.imread(os.path.join(BURST, f"cell0_f{i:02d}.png")) for i in range(3)]
    shot = cv2.imread(SHOT)
    if any(f is None for f in frames) or shot is None:
        raise SystemExit("missing image")

    x, y, w, h = BOX
    tpl = frames[0][y:y + h, x:x + w]
    cv2.imwrite(os.path.join(BURST, "_hat_tpl_full.png"),
                cv2.resize(tpl, (w * 5, h * 5), interpolation=cv2.INTER_NEAREST))

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(tpl, None)
    print(f"template keypoints: {len(kp1)}")
    if des1 is None or len(kp1) < 4:
        print("[FAIL] 模板特征点太少，SIFT 不可用")
        return 1

    bf = cv2.BFMatcher(cv2.NORM_L2)

    def match(name, img, roi):
        rx, ry, rw, rh = roi
        crop = img[ry:ry + rh, rx:rx + rw]
        kp2, des2 = sift.detectAndCompute(crop, None)
        if des2 is None or len(kp2) < 4:
            print(f"  {name:12s} no keypoints in scene")
            return
        raw = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]
        # 单应性验证：好匹配点能否拟合出一致的几何变换
        inliers = 0
        if len(good) >= 4:
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(mask.sum())
        print(f"  {name:12s} kp={len(kp2):4d} good={len(good):3d} inliers={inliers:3d}")

    print("\n=== 同格子的其它帧（鸡妖在转） ===")
    match("f01", frames[1], (0, 0, 146, 200))
    match("f02", frames[2], (0, 0, 146, 200))

    print("\n=== 真实截图（159，两处鸡妖） ===")
    match("159 full", shot, (0, 250, 720, 460))

    return 0


if __name__ == "__main__":
    sys.exit(main())
