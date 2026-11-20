"""内点阈值扫描：在不同 min_inliers 下统计命中总数，并导出命中裁剪图供人工核对。"""
from __future__ import annotations

import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chicken_detect import ChickenDetector  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

SHOTS = r"E:\work\AutoButter\screenshots"
SKIP = ("overlay", "marked", "target", "red_dot", "band", "pop_", "compare",
        "safepoint", "unsafe", "chicken", "item_names", "template_sheet",
        "offline", "trace", "dismiss", "best_hat", "hat_", "part_", "probe",
        "_grid", "_cells", "cell_", "_chk", "_v_", "_tpl", "_hit", "_blue")


def load_all():
    shots = sorted(glob.glob(os.path.join(SHOTS, "*.png")))
    shots = [s for s in shots if not any(k in os.path.basename(s) for k in SKIP)]
    out = []
    for p in shots:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        if w != 720:
            img = cv2.resize(img, (720, int(round(h * 720 / w))),
                             interpolation=cv2.INTER_AREA)
        out.append((os.path.basename(p), img))
    # 两张用户新图也参与
    for n in ["user_test_1.png", "user_test_2.png"]:
        raw = np.fromfile(os.path.join(r"E:\work\AutoButter\User_screenshot", n),
                          dtype=np.uint8)
        out.append((n, cv2.imdecode(raw, cv2.IMREAD_COLOR)))
    return out


def main() -> int:
    data = load_all()
    print(f"{len(data)} images\n")
    for thr in [6, 8, 10, 12, 15, 18, 20]:
        det = ChickenDetector(min_inliers=thr)
        t0 = time.perf_counter()
        hits_all = []
        for name, img in data:
            for inl, cx, cy, box, t in det.detect(img):
                hits_all.append((name, inl, cx, cy, box, t))
        dt = (time.perf_counter() - t0) / len(data) * 1000
        imgs_with = len({h[0] for h in hits_all})
        low = [h for h in hits_all if h[1] < 15]
        print(f"min_inliers={thr:2d}  hits={len(hits_all):3d}  "
              f"images={imgs_with:3d}  weak(<15)={len(low):3d}  {dt:.0f}ms/img")
    return 0


if __name__ == "__main__":
    sys.exit(main())
