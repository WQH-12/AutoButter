"""在历史截图集上跑鸡妖检测，统计命中数与耗时，用于查误报。"""
from __future__ import annotations

import glob
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chicken_detect import ChickenDetector, load  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

SHOTS = r"E:\work\AutoButter\screenshots"
SKIP = ("overlay", "marked", "target", "red_dot", "band", "pop_", "compare",
        "safepoint", "unsafe", "chicken", "item_names", "template_sheet",
        "offline", "trace", "dismiss", "best_hat", "hat_", "part_", "probe",
        "_grid", "_cells", "cell_")


def main() -> int:
    det = ChickenDetector()
    shots = sorted(glob.glob(os.path.join(SHOTS, "*.png")))
    shots = [s for s in shots if not any(k in os.path.basename(s) for k in SKIP)]
    print(f"scanning {len(shots)} screenshots\n")
    times, total_hits, nonzero = [], 0, 0
    rows = []
    for p in shots:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        if w != 720:
            img = cv2.resize(img, (720, int(round(h * 720 / w))),
                             interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        hits = det.detect(img)
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)
        total_hits += len(hits)
        if hits:
            nonzero += 1
        rows.append((os.path.basename(p), hits))

    times.sort()
    print(f"images={len(times)}  with_hits={nonzero}  total_hits={total_hits}")
    print(f"time median={times[len(times)//2]:.0f}ms  max={times[-1]:.0f}ms\n")
    for name, hits in rows:
        if not hits:
            continue
        desc = " ".join(f"{n.split('_')[0][:1].upper()}{i}@({cx:.0f},{cy:.0f})"
                        for i, cx, cy, _, n in hits)
        print(f"{name:32s} {len(hits)}  {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
