"""扫描秘境面板上「弹窗面板下方、且没有 UI」的空白带。

同时给出每个候选点的邻域纹理强度：纹理越低越可能是纯色空白。
"""
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter\screenshots"


def analyse(path, label):
    img = cv2.imread(path)
    if img is None:
        print(f"missing {path}")
        return
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"\n=== {label} ({img.shape[1]}x{img.shape[0]}) ===")
    # 每 20px 一行，统计 x=280..420 这条竖带的边缘密度
    print("  y    edge_density(x280-420)  mean")
    for y in range(940, 1340, 20):
        band = g[y:y + 20, 280:420]
        if band.size == 0:
            continue
        edges = cv2.Canny(band, 60, 160)
        print(f" {y:4d}   {edges.mean() / 255 * 100:6.2f}%             {band.mean():6.1f}")


analyse(f"{ROOT}\\154_after_fix.png", "realm panel, no popup")
analyse(f"{ROOT}\\156_reenter2.png", "realm panel, current")
