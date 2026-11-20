"""SIFT 鸡妖模板的误报测试：在没有鸡妖的界面上必须匹配失败。"""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BURST = r"E:\work\AutoButter\screenshots\burst\pad_test2"
SHOTS = r"E:\work\AutoButter\screenshots"
BOX = (20, 5, 100, 62)
ROI = (0, 250, 720, 460)   # 秘境内容区


def main() -> int:
    tpl = cv2.imread(os.path.join(BURST, "cell0_f00.png"))[BOX[1]:BOX[1] + BOX[3],
                                                           BOX[0]:BOX[0] + BOX[2]]
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(tpl, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    print(f"template keypoints: {len(kp1)}\n")

    def count(img):
        rx, ry, rw, rh = ROI
        crop = img[ry:ry + rh, rx:rx + rw]
        kp2, des2 = sift.detectAndCompute(crop, None)
        if des2 is None or len(kp2) < 4:
            return 0, 0, 0
        raw = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]
        inl = 0
        if len(good) >= 4:
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if mask is not None:
                inl = int(mask.sum())
        return len(kp2), len(good), inl

    # 有鸡妖的
    positives = ["159_before_drain.png", "160_after_wait.png", "154_after_fix.png"]
    # 无鸡妖的（弹窗、纯面板等）
    negatives = ["153_before.png", "132_live_popup.png", "131_live_nav.png",
                 "150_after_close.png", "102_popup_gone.png", "113_final2.png"]

    print("=== 应当命中（有鸡妖） ===")
    for f in positives:
        p = os.path.join(SHOTS, f)
        img = cv2.imread(p)
        if img is None:
            print(f"  {f:24s} MISSING")
            continue
        h, w = img.shape[:2]
        if w != 720:
            img = cv2.resize(img, (720, int(round(h * 720 / w))), interpolation=cv2.INTER_AREA)
        kp, good, inl = count(img)
        print(f"  {f:24s} kp={kp:4d} good={good:3d} inliers={inl:3d} "
              f"{'HIT' if inl >= 6 else 'miss'}")

    print("\n=== 不应当命中（无鸡妖） ===")
    for f in negatives:
        p = os.path.join(SHOTS, f)
        img = cv2.imread(p)
        if img is None:
            print(f"  {f:24s} MISSING")
            continue
        h, w = img.shape[:2]
        if w != 720:
            img = cv2.resize(img, (720, int(round(h * 720 / w))), interpolation=cv2.INTER_AREA)
        kp, good, inl = count(img)
        flag = "FALSE POSITIVE" if inl >= 6 else "clean"
        print(f"  {f:24s} kp={kp:4d} good={good:3d} inliers={inl:3d} {flag}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
