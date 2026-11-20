"""测能否在一张图里找到「多只」鸡妖，并量出它们与格子的 x 轴关系。"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

D = r"E:\work\AutoButter\User_screenshot"
SHOTS = r"E:\work\AutoButter\screenshots"
TPL = "back_2"
ROI = (0, 250, 720, 500)


def load_tpl():
    raw = np.fromfile(os.path.join(D, TPL + ".png"), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    return img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img


def find_all(img, tpl, roi, min_inliers=6, max_hits=8):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    kp_t, des_t = sift.detectAndCompute(tpl, None)
    x, y, w, h = roi
    crop = img[y:y + h, x:x + w].copy()
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
        if H is None or mask is None or int(mask.sum()) < min_inliers:
            break

        th, tw = tpl.shape[:2]
        corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        cx = float(proj[:, 0].mean()) + x
        cy = float(proj[:, 1].mean()) + y
        hits.append((int(mask.sum()), cx, cy, proj))

        # 抹掉这一只，继续找下一只
        m2 = np.float32([[1, 0, 0], [0, 1, 0]])
        cv2.fillConvexPoly(crop, proj.astype(np.int32), (0, 0, 0))
        # 用矩形抹除更彻底
        xs, ys = proj[:, 0], proj[:, 1]
        x0, x1 = int(max(0, xs.min())) - 5, int(min(crop.shape[1], xs.max())) + 5
        y0, y1 = int(max(0, ys.min())) - 5, int(min(crop.shape[0], ys.max())) + 5
        crop[max(0, y0):y1, max(0, x0):x1] = 0
    return hits


def main() -> int:
    tpl = load_tpl()
    print(f"template {TPL} {tpl.shape[1]}x{tpl.shape[0]}\n")

    for name in ["159_before_drain.png", "158_state.png", "157_popup_for_test.png",
                 "155_no_popup_fast.png", "154_after_fix.png", "132_live_popup.png"]:
        p = os.path.join(SHOTS, name)
        img = cv2.imread(p)
        if img is None:
            print(f"{name}: missing")
            continue
        h, w = img.shape[:2]
        if w != 720:
            img = cv2.resize(img, (720, int(round(h * 720 / w))),
                             interpolation=cv2.INTER_AREA)
        hits = find_all(img, tpl, ROI)
        xs = sorted(round(hh[1]) for hh in hits)
        print(f"{name:26s} {len(hits)} chicken(s)  x={xs}")
        marked = img.copy()
        for inl, cx, cy, _ in hits:
            print(f"    inliers={inl:3d} center=({cx:.0f},{cy:.0f})")
            cv2.drawMarker(marked, (int(cx), int(cy)), (0, 0, 255),
                           cv2.MARKER_CROSS, 50, 3)
            cv2.putText(marked, str(inl), (int(cx) + 12, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(SHOTS, f"_chicken_{name}"), marked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
