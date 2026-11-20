"""测 6 张鸡妖图（frontage 正面 / back 背面）能否互相匹配。

目的：判断需要几张模板，以及 TemplateMatch 是否可行。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

D = r"E:\work\AutoButter\User_screenshot"
NAMES = ["frontage_1", "frontage_2", "frontage_3", "back_1", "back_2", "back_3"]


def load(name):
    p = os.path.join(D, name + ".png")
    raw = np.fromfile(p, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read {p}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def main() -> int:
    imgs = {n: load(n) for n in NAMES}
    for n in NAMES:
        g = cv2.cvtColor(imgs[n], cv2.COLOR_BGR2GRAY)
        print(f"{n:12s} {imgs[n].shape[1]}x{imgs[n].shape[0]} std={g.std():5.1f}")

    print("\n=== TemplateMatch 交叉矩阵（行=模板, 列=搜索图）===")
    print(f"{'':12s}" + "".join(f"{n[:9]:>10s}" for n in NAMES))
    for a in NAMES:
        row = f"{a:12s}"
        for b in NAMES:
            tpl, scene = imgs[a], imgs[b]
            if scene.shape[0] < tpl.shape[0] or scene.shape[1] < tpl.shape[1]:
                row += f"{'n/a':>10s}"
                continue
            res = cv2.matchTemplate(scene, tpl, cv2.TM_CCOEFF_NORMED)
            row += f"{cv2.minMaxLoc(res)[1]:10.3f}"
        print(row)

    print("\n=== SIFT 交叉矩阵（inliers）===")
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    kps, dess = {}, {}
    for n in NAMES:
        kp, des = sift.detectAndCompute(imgs[n], None)
        kps[n], dess[n] = kp, des
        print(f"  {n:12s} keypoints={len(kp)}")

    print(f"\n{'':12s}" + "".join(f"{n[:9]:>10s}" for n in NAMES))
    for a in NAMES:
        row = f"{a:12s}"
        for b in NAMES:
            des1, des2 = dess[a], dess[b]
            if des1 is None or des2 is None:
                row += f"{'n/a':>10s}"
                continue
            raw = bf.knnMatch(des1, des2, k=2)
            good = [m for m, n2 in raw if n2 is not None and m.distance < 0.75 * n2.distance]
            inl = 0
            if len(good) >= 4:
                src = np.float32([kps[a][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kps[b][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if mask is not None:
                    inl = int(mask.sum())
            row += f"{inl:10d}"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
