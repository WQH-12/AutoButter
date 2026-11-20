"""验证「检测格子 -> 排除被掠夺列 -> 选目标格子」的完整逻辑。

规则（用户指定）：
  识别到鸡妖后，屏蔽该鸡妖 x 轴范围内的整列格子。
  格子中心 x 落在任一鸡妖框的 x 区间内 -> 该格子在掠夺中 -> 跳过。
"""
from __future__ import annotations

import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from burst import detect_cells  # noqa: E402
from chicken_detect import ChickenDetector, load  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def pick(img, det, roi=(0, 250, 720, 810)):
    cells = detect_cells(img)
    hits = det.detect(img, roi=roi)
    xr = [(b[0], b[2]) for _, _, _, b, _ in hits]
    free, busy = [], []
    for x, y, cw, ch in cells:
        cx = x + cw / 2.0
        blocked = any(x0 <= cx <= x1 for x0, x1 in xr)
        (busy if blocked else free).append((x, y, cw, ch))
    return cells, hits, free, busy


def main() -> int:
    det = ChickenDetector()
    targets = sys.argv[1:] or [
        r"User_screenshot\user_test_1.png",
        r"User_screenshot\user_test_2.png",
        r"screenshots\snap\live_now_720.png",
        r"screenshots\159_before_drain.png",
        r"screenshots\157_popup_for_test.png",
    ]
    for p in targets:
        if not os.path.exists(p):
            print(f"{p}: missing")
            continue
        img = cv2.imread(p)
        if img is None:
            print(f"{p}: unreadable")
            continue
        h, w = img.shape[:2]
        if w != 720:
            img = cv2.resize(img, (720, int(round(h * 720 / w))),
                             interpolation=cv2.INTER_AREA)
        cells, hits, free, busy = pick(img, det)
        print(f"=== {os.path.basename(p)} ===")
        print(f"  cells={len(cells)} chickens={len(hits)} "
              f"free={len(free)} busy={len(busy)}")
        for x, y, cw, ch in busy:
            print(f"    BUSY cell cx={x+cw/2:6.1f} y={y:4d} ({cw}x{ch})")
        for x, y, cw, ch in free:
            print(f"    FREE cell cx={x+cw/2:6.1f} y={y:4d} ({cw}x{ch})")
        marked = img.copy()
        for x, y, cw, ch in busy:
            cv2.rectangle(marked, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
        for x, y, cw, ch in free:
            cv2.rectangle(marked, (x, y), (x + cw, y + ch), (0, 200, 0), 3)
        for _, _, _, b, _ in hits:
            cv2.rectangle(marked, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 1)
        out = os.path.join(r"E:\work\AutoButter\screenshots",
                           "_pick_" + os.path.basename(p))
        cv2.imwrite(out, marked)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
