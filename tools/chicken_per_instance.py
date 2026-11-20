"""逐只鸡妖单独裁剪，量 6 张模板各自的内点数。

目的：判断「漏检」是模板姿态覆盖不足，还是单应性一次只能出一个模型。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

D = r"E:\work\AutoButter\User_screenshot"
NAMES = ["frontage_1", "frontage_2", "frontage_3", "back_1", "back_2", "back_3"]

# 目测坐标（720x1560 画布），从 user_test_1.png 读出
CASES = [
    ("user_test_1.png", "back_23:10",   550, 335, 110, 100),
    ("user_test_1.png", "front_00:38",   40, 865, 110, 110),
    ("user_test_1.png", "front_00:06",  145, 885, 110, 110),
    ("user_test_1.png", "front_00:05a", 250, 890, 110, 110),
    ("user_test_1.png", "front_00:05b", 350, 890, 110, 110),
    ("user_test_2.png", "back_21:03",   550, 325, 110, 100),
    ("user_test_2.png", "front_00:08",  455, 910, 110, 110),
]


def load(name):
    raw = np.fromfile(os.path.join(D, name), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read {name}")
    return img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img


def inliers(tpl_kp, tpl_des, crop):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    kp, des = sift.detectAndCompute(crop, None)
    if des is None or len(kp) < 4:
        return 0
    raw = bf.knnMatch(tpl_des, des, k=2)
    good = [m for m, n in raw if n is not None and m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return 0
    src = np.float32([tpl_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return 0 if mask is None else int(mask.sum())


def main() -> int:
    sift = cv2.SIFT_create()
    tpls = {}
    for n in NAMES:
        t = load(n + ".png")
        kp, des = sift.detectAndCompute(t, None)
        tpls[n] = (kp, des)

    hdr = "instance".ljust(18) + "".join(n.replace("frontage_", "F").replace("back_", "B").rjust(6) for n in NAMES)
    print(hdr)
    imgs = {}
    for shot, label, x, y, w, h in CASES:
        if shot not in imgs:
            imgs[shot] = load(shot)
        crop = imgs[shot][y:y + h, x:x + w]
        row = label.ljust(18)
        for n in NAMES:
            row += str(inliers(tpls[n][0], tpls[n][1], crop)).rjust(6)
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
