"""用带「恭喜獲得」弹窗的截图核对：每个 item_*.png 到底对应哪个中文物品名。

弹窗里每个物品下面有名字，但这里不跑 OCR，而是把图标和名字一起裁出来，
让人工看图确认映射，避免再写错。
"""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter"
IMAGE_DIR = os.path.join(ROOT, "resource", "image")
SHOT_DIR = os.path.join(ROOT, "screenshots")
SCALE = 3


def main() -> int:
    # 这两张都有多物品弹窗，能看到名字
    shots = ["131_live_nav.png", "156_reenter2.png", "100_now.png",
             "101_after_tap1.png", "170_popup_before_test.png"]
    tpls = {}
    for p in sorted(glob.glob(os.path.join(IMAGE_DIR, "item_*.png"))):
        name = os.path.basename(p)[:-4].replace("item_", "")
        tpls[name] = cv2.imread(p)

    rows = []
    for shot in shots:
        img = cv2.imread(os.path.join(SHOT_DIR, shot))
        if img is None:
            print(f"missing {shot}")
            continue
        # 弹窗物品区
        roi = (40, 620, 640, 260)
        x, y, w, h = roi
        crop = img[y:y + h, x:x + w]
        big = cv2.resize(crop, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, shot, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        rows.append(big)

        # 同时报告每个模板在这块区域的最佳分数和位置
        print(f"\n=== {shot} 物品区 ({x},{y},{w},{h}) ===")
        for name, tpl in tpls.items():
            if tpl is None:
                continue
            if crop.shape[0] < tpl.shape[0] or crop.shape[1] < tpl.shape[1]:
                continue
            res = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, ml = cv2.minMaxLoc(res)
            if mx >= 0.85:
                cx = x + ml[0] + tpl.shape[1] // 2
                cy = y + ml[1] + tpl.shape[0] // 2
                print(f"  {name:8s} {mx:.4f} @({cx},{cy})")

    if not rows:
        return 1
    max_w = max(r.shape[1] for r in rows)
    total_h = sum(r.shape[0] for r in rows) + 10 * len(rows)
    sheet = np.full((total_h, max_w, 3), 30, np.uint8)
    yy = 0
    for r in rows:
        sheet[yy:yy + r.shape[0], :r.shape[1]] = r
        yy += r.shape[0] + 10

    dst = os.path.join(SHOT_DIR, "_item_names_check.png")
    cv2.imwrite(dst, sheet)
    print(f"\nsaved {dst} ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
