"""收集所有真实截图里 6 张鸡妖模板的内点分布，定出可靠阈值。

关键：真鸡妖 vs 边缘残片/背景噪声 的内点差异。
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
ROI = (0, 250, 720, 500)


def load_tpl(n):
    raw = np.fromfile(os.path.join(D, n + ".png"), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    return img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img


def all_hits(img, tpl, kp_t, des_t, sift, bf, min_inl=4, max_hits=6):
    x, y, w, h = ROI
    crop = img[y:y + h, x:x + w].copy()
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return []
    out = []
    for _ in range(max_hits):
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            break
        kp_s, des_s = sift.detectAndCompute(crop, None)
        if des_s is None or len(kp_s) < 4:
            break
        raw = bf.knnMatch(des_t, des_s, k=2)
        good = [m for m, n2 in raw if n2 is not None and m.distance < 0.75 * n2.distance]
        if len(good) < 4:
            break
        src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask is None or int(mask.sum()) < min_inl:
            break
        inl = int(mask.sum())
        th, tw = tpl.shape[:2]
        corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        cx = float(proj[:, 0].mean()) + x
        cy = float(proj[:, 1].mean()) + y
        out.append((inl, cx, cy))
        xs, ys = proj[:, 0], proj[:, 1]
        x0 = int(max(0, xs.min())) - 5
        x1 = int(min(crop.shape[1], xs.max())) + 5
        y0 = int(max(0, ys.min())) - 5
        y1 = int(min(crop.shape[0], ys.max())) + 5
        crop[max(0, y0):y1, max(0, x0):x1] = 0
    return out


def main() -> int:
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    tpls = {}
    for n in NAMES:
        t = load_tpl(n)
        kp, des = sift.detectAndCompute(t, None)
        tpls[n] = (t, kp, des)

    files = sorted(glob.glob(os.path.join(SHOTS, "*.png")))
    skip = ("overlay", "marked", "target", "red_dot", "band", "pop_", "compare",
            "safepoint", "unsafe", "chicken", "item_names", "template_sheet",
            "offline", "trace", "dismiss", "best_hat", "hat_", "part_", "probe")
    files = [f for f in files if not any(k in os.path.basename(f) for k in skip)]

    buckets = {n: [] for n in NAMES}
    for p in files:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        if w != 720:
            img = cv2.resize(img, (720, int(round(h * 720 / w))),
                             interpolation=cv2.INTER_AREA)
        for n in NAMES:
            t, kp, des = tpls[n]
            for inl, cx, cy in all_hits(img, t, kp, des, sift, bf):
                buckets[n].append((inl, os.path.basename(p), round(cx), round(cy)))

    print("=== 每个模板命中的内点分布 ===")
    for n in NAMES:
        vals = sorted(v[0] for v in buckets[n])
        if not vals:
            print(f"{n:12s} no hits")
            continue
        print(f"\n{n:12s} n={len(vals)} values={vals}")
        for inl, f, cx, cy in sorted(buckets[n]):
            print(f"    {inl:3d}  {f:28s} ({cx},{cy})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
