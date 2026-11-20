"""量鸡妖检测框与物品格子的 x 轴对齐关系。

目的：用户要求「识别出鸡妖后屏蔽其 x 轴范围内整个 y 轴」，
必须先确认一只鸡妖对应哪一列格子。
"""
from __future__ import annotations

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from burst import detect_cells  # noqa: E402
from chicken_detect import ChickenDetector, load  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    det = ChickenDetector()
    for shot in ["user_test_1.png", "user_test_2.png"]:
        img = load(shot)
        h, w = img.shape[:2]
        if w != 720:
            img = cv2.resize(img, (720, int(round(h * 720 / w))),
                             interpolation=cv2.INTER_AREA)
        cells = detect_cells(img)
        hits = det.detect(img)
        print(f"=== {shot} ===")
        print(f"  cells ({len(cells)}):")
        for i, (x, y, cw, ch) in enumerate(cells):
            print(f"    cell{i}: x={x:3d}..{x+cw:3d} cx={x+cw/2:6.1f}  "
                  f"y={y:3d}..{y+ch:3d}  {cw}x{ch}")
        print(f"  chickens ({len(hits)}):")
        for inl, cx, cy, box, n in hits:
            pose = "FRONT" if n.startswith("frontage") else "BACK"
            near = min(cells, key=lambda c: abs((c[0] + c[2] / 2) - cx)) if cells else None
            d = abs((near[0] + near[2] / 2) - cx) if near else -1.0
            print(f"    {pose:5s} {n:12s} inl={inl:3d} cx={cx:6.1f} cy={cy:6.1f} "
                  f"box_x={box[0]}..{box[2]}  nearest_cell_dx={d:6.1f}")
        marked = img.copy()
        for x, y, cw, ch in cells:
            cv2.rectangle(marked, (x, y), (x + cw, y + ch), (0, 200, 0), 2)
            cv2.line(marked, (x + cw // 2, 250), (x + cw // 2, 1150), (0, 200, 0), 1)
        for inl, cx, cy, box, n in hits:
            col = (0, 0, 255) if n.startswith("frontage") else (255, 0, 0)
            cv2.rectangle(marked, (box[0], box[1]), (box[2], box[3]), col, 2)
            cv2.line(marked, (int(cx), 250), (int(cx), 1150), col, 1)
        cv2.imwrite(os.path.join(r"E:\work\AutoButter\screenshots",
                                 f"_chicken_cells_{shot}"), marked)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
