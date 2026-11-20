"""量「恭喜獲得」弹窗面板的边界，确认候选点击点落在面板外。"""
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter\screenshots"


def panel_rows(path):
    img = cv2.imread(path)
    if img is None:
        print(f"  missing {path}")
        return None
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 面板是亮的，背景是暗的：逐行统计"明显亮"的像素比例
    bright = (g > 60).mean(axis=1)
    rows = np.where(bright > 0.15)[0]
    if rows.size == 0:
        print(f"  {path}: no bright rows")
        return None
    top, bot = int(rows.min()), int(rows.max())
    scale = 720 / w
    print(f"  {path.split(chr(92))[-1]:24s} {w}x{h} "
          f"bright rows {top}..{bot}  -> 720 canvas {top * scale:.0f}..{bot * scale:.0f}")
    return top * scale, bot * scale


print("panel vertical extent (720 canvas):")
for f in ["131_live_nav.png", "142_state.png", "140_now.png", "150_still_open.png"]:
    panel_rows(f"{ROOT}\\{f}")

print()
print("candidate click points (720 canvas):")
for name, y in [("原十字 y=851", 851), ("红点 y=1326", 1326), ("y=1450", 1450)]:
    print(f"  {name:16s} (350, {y})")
