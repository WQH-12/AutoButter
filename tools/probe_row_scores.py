"""只读：打印列表页每行的采集中分数，定位底部/临界行。"""
from __future__ import annotations

import glob
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))

from row_reco import RowPicker  # noqa: E402


def main() -> int:
    picker = RowPicker()
    paths = (sorted(glob.glob(os.path.join(ROOT, "screenshots", "*.png")))
             + sorted(glob.glob(os.path.join(ROOT, "screenshots", "snap", "*.png")))
             + sorted(glob.glob(os.path.join(ROOT, "User_screenshot", "full", "*.png"))))
    for p in paths:
        if os.path.basename(p).startswith("_"):
            continue
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        if w != 720:
            img = cv2.resize(img, (720, int(round(h * 720 / w))),
                             interpolation=cv2.INTER_AREA)
        if img.shape[0] < 1200:
            continue
        res = picker.scan(img)
        if not res["rows"]:
            continue
        print(os.path.basename(p))
        for r in res["rows"]:
            print("  y=%4d collect=%5.3f C=%s R=%s s60=%.3f wanted=%d"
                  % (r["btn"][1], r["collect_score"], r["collecting"],
                     r["readable"], r["s60"], len(r["wanted"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
