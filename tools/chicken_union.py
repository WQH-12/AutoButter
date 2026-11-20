"""6 张模板并联检测 + 去重，看能否找回单模板漏掉的实例。"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

D = r"E:\work\AutoButter\User_screenshot"
NAMES = ["frontage_1", "frontage_2", "frontage_3", "back_1", "back_2", "back_3"]
ROI = (0, 250, 720, 800)


def load(name):
    raw = np.fromfile(os.path.join(D, name), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read {name}")
    return img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img


def find_all(img, tpl, roi, min_inliers=10, margin=8, max_hits=8):
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
        if H is None or mask is None or int(mask.sum()) < 4:
            break
        inl = int(mask.sum())
        proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        xs, ys = proj[:, 0], proj[:, 1]
        inside = (xs.min() >= margin and ys.min() >= margin
                  and xs.max() <= w - margin and ys.max() <= h - margin)
        area = cv2.contourArea(proj.astype(np.float32)) / float(tw * th)
        if inl >= min_inliers and inside and 0.4 <= area <= 3.0:
            hits.append((inl, float(xs.mean()) + x0, float(ys.mean()) + y0,
                         (int(xs.min()) + x0, int(ys.min()) + y0,
                          int(xs.max()) + x0, int(ys.max()) + y0)))
        xa, xb = int(max(0, xs.min())) - 5, int(min(crop.shape[1], xs.max())) + 5
        ya, yb = int(max(0, ys.min())) - 5, int(min(crop.shape[0], ys.max())) + 5
        crop[max(0, ya):yb, max(0, xa):xb] = 0
    return hits


def merge(hits, iou_thr=0.35):
    out = []
    for inl, cx, cy, box in sorted(hits, key=lambda t: -t[0]):
        dup = False
        for o in out:
            ix = max(0, min(box[2], o[3][2]) - max(box[0], o[3][0]))
            iy = max(0, min(box[3], o[3][3]) - max(box[1], o[3][1]))
            inter = ix * iy
            a1 = (box[2] - box[0]) * (box[3] - box[1])
            a2 = (o[3][2] - o[3][0]) * (o[3][3] - o[3][1])
            if inter / float(max(1, a1 + a2 - inter)) > iou_thr:
                dup = True
                break
        if not dup:
            out.append((inl, cx, cy, box))
    return out


def main() -> int:
    tpls = {n: load(n + ".png") for n in NAMES}
    for shot in ["user_test_1.png", "user_test_2.png"]:
        img = load(shot)
        h, w = img.shape[:2]
        if w != 720:
            img = cv2.resize(img, (720, int(round(h * 720 / w))),
                             interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        allh = []
        per = {}
        for n in NAMES:
            hs = find_all(img, tpls[n], ROI)
            per[n] = [(i, round(cx), round(cy)) for i, cx, cy, _ in hs]
            allh.extend(hs)
        dt = time.perf_counter() - t0
        m = merge(allh)
        print(f"=== {shot}  {dt*1000:.0f}ms for 6 templates ===")
        for n in NAMES:
            print(f"  {n:12s} {per[n]}")
        print(f"  merged {len(m)}:")
        for inl, cx, cy, box in m:
            print(f"      inliers={inl:3d} center=({cx:6.1f},{cy:6.1f}) box={box}")
        marked = img.copy()
        for inl, cx, cy, box in m:
            cv2.rectangle(marked, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(marked, str(inl), (box[0], box[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(r"E:\work\AutoButter\screenshots",
                                 f"_chicken_union_{shot}"), marked)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
