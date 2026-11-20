"""用 user_test_1/2 验证 6 张鸡妖模板：多实例 + 边界剔除。

每个模板独立做「找到一只 -> 抹掉 -> 继续找」，要求：
  1. 内点数 >= min_inliers
  2. 投影四边形完整落在 ROI 内（四角都在，且离边界 >= margin）
输出每张图的命中列表，用于确认正面/背面能否分开、会不会互相误报。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

D = r"E:\work\AutoButter\User_screenshot"
NAMES = ["frontage_1", "frontage_2", "frontage_3", "back_1", "back_2", "back_3"]
SHOTS = ["user_test_1.png", "user_test_2.png"]
ROI = (0, 250, 720, 800)


def load(name):
    raw = np.fromfile(os.path.join(D, name), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read {name}")
    return img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img


def find_all(img, tpl, roi, min_inliers, margin=8, max_hits=8):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    kp_t, des_t = sift.detectAndCompute(tpl, None)
    x0, y0, w, h = roi
    crop = img[y0:y0 + h, x0:x0 + w].copy()
    th, tw = tpl.shape[:2]
    corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
    hits = []

    for _ in range(max_hits):
        kp_s, des_s = sift.detectAndCompute(crop, None)
        if des_s is None or len(kp_s) < 4:
            break
        raw = bf.knnMatch(des_t, des_s, k=2)
        good = [m for m, n in raw if n is not None and m.distance < 0.75 * n.distance]
        if len(good) < 4:
            break
        src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            break
        inl = int(mask.sum())
        proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        xs, ys = proj[:, 0], proj[:, 1]
        inside = (xs.min() >= margin and ys.min() >= margin
                  and xs.max() <= w - margin and ys.max() <= h - margin)
        # 尺寸合理性：投影面积应为模板面积的 0.4~3 倍
        area = cv2.contourArea(proj.astype(np.float32)) / float(tw * th)
        sane = 0.4 <= area <= 3.0
        if inl < min_inliers:
            break
        if inside and sane:
            hits.append((inl, float(xs.mean()) + x0, float(ys.mean()) + y0,
                         (int(xs.min()) + x0, int(ys.min()) + y0,
                          int(xs.max()) + x0, int(ys.max()) + y0),
                         round(area, 2)))
        # 无论是否采纳都抹掉，避免同一只被反复命中
        xa, xb = int(max(0, xs.min())) - 5, int(min(crop.shape[1], xs.max())) + 5
        ya, yb = int(max(0, ys.min())) - 5, int(min(crop.shape[0], ys.max())) + 5
        crop[max(0, ya):yb, max(0, xa):xb] = 0
    return hits


def main() -> int:
    tpls = {n: load(n + ".png") for n in NAMES}
    for n, t in tpls.items():
        print(f"template {n}: {t.shape[1]}x{t.shape[0]}")
    print()

    for shot in SHOTS:
        img = load(shot)
        h, w = img.shape[:2]
        if w != 720:
            img = cv2.resize(img, (720, int(round(h * 720 / w))),
                             interpolation=cv2.INTER_AREA)
        print(f"=== {shot} {img.shape[1]}x{img.shape[0]} ROI={ROI} ===")
        marked = img.copy()
        for name in NAMES:
            hits = find_all(img, tpls[name], ROI, min_inliers=12)
            tag = "F" if name.startswith("frontage") else "B"
            print(f"  {name:12s} {len(hits)} hit(s)")
            for inl, cx, cy, box, area in hits:
                print(f"      inliers={inl:3d} center=({cx:6.1f},{cy:6.1f}) "
                      f"box={box} area={area}")
                col = (0, 0, 255) if tag == "F" else (255, 0, 0)
                cv2.rectangle(marked, (box[0], box[1]), (box[2], box[3]), col, 2)
                cv2.putText(marked, f"{tag}{inl}", (box[0], box[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        out = os.path.join(r"E:\work\AutoButter\screenshots", f"_chicken_verify_{shot}")
        cv2.imwrite(out, marked)
        print(f"  -> {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
