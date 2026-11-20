"""量模板匹配本身的耗时，决定 ROI 要不要收紧。"""
from __future__ import annotations

import statistics
import sys
import time

import cv2

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter"
IMG = f"{ROOT}\\screenshots\\131_live_nav.png"
TPL = f"{ROOT}\\resource\\image\\obtain_resource.png"

img = cv2.imread(IMG)
tpl = cv2.imread(TPL)
print(f"image {img.shape}  template {tpl.shape}")

ROIS = {
    "current [0,400,720,400]": (0, 400, 720, 400),
    "tight   [180,470,360,160]": (180, 470, 360, 160),
    "tighter [190,480,340,140]": (190, 480, 340, 140),
    "full screen": (0, 0, 720, 1560),
}

for name, (x, y, w, h) in ROIS.items():
    crop = img[y:y + h, x:x + w]
    times = []
    best = 0.0
    for _ in range(30):
        t = time.perf_counter()
        res = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, _ = cv2.minMaxLoc(res)
        times.append((time.perf_counter() - t) * 1000)
        best = max(best, mx)
    print(f"{name:28s} median={statistics.median(times):6.2f}ms best={best:.4f}")
