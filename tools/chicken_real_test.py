"""把 6 张鸡妖模板（正面/背面）放到真实截图里测，看能否区分。

关键问题：正面和背面能否被分别识别，且不互相误报。
"""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

D = r"E:\work\AutoButter\User_screenshot"
SHOTS = r"E:\work\AutoButter\screenshots"
NAMES = ["frontage_1", "frontage_2", "frontage_3", "back_1", "back_2", "back_3"]
ROI = (0, 250, 720, 500)   # 秘境内容区


def load(name):
    raw = np.fromfile(os.path.join(D, name + ".png"), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read {name}")
    return img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img


def main() -> int:
    tpls = {n: load(n) for n in NAMES}
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    kp_t, des_t = {}, {}
    for n, t in tpls.items():
        kp, des = sift.detectAndCompute(t, None)
        kp_t[n], des_t[n] = kp, des

    shots = sorted(glob.glob(os.path.join(SHOTS, "*.png")))
    # 只保留实机截图（排除工具生成的中间图）
    skip = ("overlay", "marked", "target", "red_dot", "band", "pop_", "compare",
            "safepoint", "unsafe", "chicken", "item_names", "template_sheet",
            "offline", "trace", "dismiss", "best_hat", "hat_", "part_", "probe")
    shots = [s for s in shots
             if not any(k in os.path.basename(s) for k in skip)]
    print(f"testing {len(shots)} screenshots\n")

    for p in shots:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        if w != 720:
            img = cv2.resize(img, (720, int(round(h * 720 / w))),
                             interpolation=cv2.INTER_AREA)
        x, y, ww, hh = ROI
        crop = img[y:y + hh, x:x + ww]
        kp_s, des_s = sift.detectAndCompute(crop, None)
        if des_s is None:
            print(f"{os.path.basename(p):28s} no keypoints")
            continue

        parts = []
        for n in NAMES:
            raw = bf.knnMatch(des_t[n], des_s, k=2)
            good = [m for m, n2 in raw if n2 is not None and m.distance < 0.75 * n2.distance]
            inl = 0
            if len(good) >= 4:
                src = np.float32([kp_t[n][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if mask is not None:
                    inl = int(mask.sum())
            parts.append((n, inl))

        best = max(parts, key=lambda t: t[1])
        tag = f"{best[0]}({best[1]})" if best[1] >= 6 else "none"
        detail = " ".join(f"{n.replace('frontage_','F').replace('back_','B')}={v}"
                          for n, v in parts)
        print(f"{os.path.basename(p):28s} {tag:14s} {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
