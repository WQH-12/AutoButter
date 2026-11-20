"""全仓库扫一遍 PickRow，统计每页的行状态分布。

用来回答两个问题：
  1. 有没有「整页都不可用」的样本（用来做负例）
  2. 有没有「整页都是采集中」的样本（用户提到的边界场景）

用法：python tools\check_rows.py
"""
from __future__ import annotations

import glob
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))

from row_reco import RowPicker  # noqa: E402


def to720(im):
    h, w = im.shape[:2]
    if w == 720:
        return im
    return cv2.resize(im, (720, int(round(h * 720 / w))),
                      interpolation=cv2.INTER_AREA)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    picker = RowPicker()
    print("templates: items=%d collect=%d ready=%s"
          % (len(picker.items), len(picker.collect), picker.ready))

    paths = (sorted(glob.glob(os.path.join(ROOT, "screenshots", "*.png")))
             + sorted(glob.glob(os.path.join(ROOT, "screenshots", "snap", "*.png")))
             + sorted(glob.glob(os.path.join(ROOT, "User_screenshot", "full", "*.png"))))

    frames = []
    for p in paths:
        if os.path.basename(p).startswith("_"):
            continue
        im = cv2.imread(p)
        if im is None:
            continue
        im = to720(im)
        if im.shape[0] < 1200:
            continue
        res = picker.scan(im)
        if not res["rows"]:
            continue
        frames.append((os.path.basename(p), res))

    print("\n列表页样本 %d 张" % len(frames))

    no_cand = [f for f in frames if f[1]["no_candidate"]]
    all_col = [f for f in frames if f[1]["all_collecting"]]
    has_col = [f for f in frames if any(r["collecting"] for r in f[1]["rows"])]
    dim_last = [f for f in frames if f[1]["rows"] and not f[1]["rows"][-1]["readable"]]

    print("  有「采集中」徽标: %d" % len(has_col))
    print("  整页都是采集中:   %d" % len(all_col))
    print("  整页无可进入的行: %d" % len(no_cand))
    print("  末行被渐隐压暗:   %d" % len(dim_last))

    if no_cand:
        print("\n整页无可进入的行（负例候选）:")
        for name, res in no_cand[:10]:
            print("   %-32s rows=%d" % (name, len(res["rows"])))
            for i, r in enumerate(res["rows"]):
                print("      row%d readable=%s collecting=%s wanted=%d"
                      % (i, r["readable"], r["collecting"], len(r["wanted"])))

    print("\n每张的行状态矩阵（R=可读 C=采集中 W=有想要的）:")
    for name, res in frames[:12]:
        cells = []
        for r in res["rows"]:
            tag = ""
            tag += "R" if r["readable"] else "-"
            tag += "C" if r["collecting"] else "-"
            tag += "W" if r["wanted"] else "-"
            cells.append(tag)
        print("   %-32s %s" % (name, " ".join(cells)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
