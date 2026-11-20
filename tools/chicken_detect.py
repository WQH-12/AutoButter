"""鸡妖检测核心：6 模板并联 SIFT + 抹除迭代 + 去重。

性能要点：每一轮只对画面做一次 SIFT，6 个模板共用这一轮的描述子，
避免 6 次重复特征提取。
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

D = r"E:\work\AutoButter\User_screenshot"
NAMES = ["frontage_1", "frontage_2", "frontage_3", "back_1", "back_2", "back_3"]
DEFAULT_ROI = (0, 250, 720, 800)


def load(name):
    raw = np.fromfile(os.path.join(D, name), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read {name}")
    return img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img


class ChickenDetector:
    def __init__(self, names=None, min_inliers=10, ratio=0.75, ransac=5.0,
                 margin=8, area_lo=0.4, area_hi=3.0):
        self.names = names or NAMES
        self.min_inliers = min_inliers
        self.ratio = ratio
        self.ransac = ransac
        self.margin = margin
        self.area_lo = area_lo
        self.area_hi = area_hi
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.tpl = {}
        for n in self.names:
            img = load(n + ".png")
            kp, des = self.sift.detectAndCompute(img, None)
            th, tw = img.shape[:2]
            corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
            self.tpl[n] = (kp, des, corners, float(tw * th))

    def detect(self, img, roi=DEFAULT_ROI, max_rounds=8):
        """返回 [(inliers, cx, cy, (x0,y0,x1,y1), tpl_name)]，已去重。"""
        x0, y0, w, h = roi
        crop = img[y0:y0 + h, x0:x0 + w].copy()
        if crop.size == 0:
            return []
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        elif crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
        if crop.dtype != np.uint8:
            crop = crop.astype(np.uint8)
        found = []
        for _ in range(max_rounds):
            kp_s, des_s = self.sift.detectAndCompute(crop, None)
            if des_s is None or len(kp_s) < 4:
                break
            round_hits = []
            for n in self.names:
                kp_t, des_t, corners, tpl_area = self.tpl[n]
                raw = self.bf.knnMatch(des_t, des_s, k=2)
                good = [m for m, nn in raw
                        if nn is not None and m.distance < self.ratio * nn.distance]
                if len(good) < 4:
                    continue
                src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac)
                if H is None or mask is None:
                    continue
                inl = int(mask.sum())
                if inl < self.min_inliers:
                    continue
                proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
                xs, ys = proj[:, 0], proj[:, 1]
                if not (xs.min() >= self.margin and ys.min() >= self.margin
                        and xs.max() <= w - self.margin
                        and ys.max() <= h - self.margin):
                    continue
                area = cv2.contourArea(proj.astype(np.float32)) / tpl_area
                if not (self.area_lo <= area <= self.area_hi):
                    continue
                round_hits.append((inl, float(xs.mean()) + x0, float(ys.mean()) + y0,
                                   (int(xs.min()) + x0, int(ys.min()) + y0,
                                    int(xs.max()) + x0, int(ys.max()) + y0), n))
            if not round_hits:
                break
            # 本轮按内点降序贪心去重：多个模板可能指向同一只，只保留最强的
            round_hits.sort(key=lambda t: -t[0])
            accepted = []
            for hit in round_hits:
                if all(_iou(hit[3], a[3]) <= 0.35 for a in accepted):
                    accepted.append(hit)
            if not accepted:
                break
            found.extend(accepted)
            # 只抹掉采纳的框，保留画面里其它鸡妖继续参与下一轮
            for inl, cx, cy, box, n in accepted:
                xa, xb = max(0, box[0] - x0 - 5), min(crop.shape[1], box[2] - x0 + 5)
                ya, yb = max(0, box[1] - y0 - 5), min(crop.shape[0], box[3] - y0 + 5)
                crop[ya:yb, xa:xb] = 0
        return found


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    a1 = (a[2] - a[0]) * (a[3] - a[1])
    a2 = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(max(1, a1 + a2 - inter))


def main() -> int:
    det = ChickenDetector()
    for shot in ["user_test_1.png", "user_test_2.png"]:
        img = load(shot)
        h, w = img.shape[:2]
        if w != 720:
            img = cv2.resize(img, (720, int(round(h * 720 / w))),
                             interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        hits = det.detect(img)
        dt = (time.perf_counter() - t0) * 1000
        print(f"=== {shot}  {len(hits)} chicken(s)  {dt:.0f}ms ===")
        for inl, cx, cy, box, n in hits:
            print(f"    {n:12s} inliers={inl:3d} center=({cx:6.1f},{cy:6.1f}) box={box}")
        marked = img.copy()
        for inl, cx, cy, box, n in hits:
            cv2.rectangle(marked, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(marked, f"{n.split('_')[0][:1].upper()}{inl}",
                        (box[0], box[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 255), 2)
        cv2.imwrite(os.path.join(r"E:\work\AutoButter\screenshots",
                                 f"_chicken_det_{shot}"), marked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
