"""「采集中」徽标阈值扫描：一次扫描算完所有阈值。

关键点：每个 (图, 模板) 只做一次 matchTemplate，记录 best score，
然后按阈值分档统计。不要对每个阈值重扫全量。
"""
from __future__ import annotations

import glob
import os
import sys
import time

import cv2

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter"
USER = os.path.join(ROOT, "User_screenshot")
SHOTS = os.path.join(ROOT, "screenshots")
THRESHOLDS = [0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97]


def load_tpls():
    out = {}
    for i in range(1, 5):
        p = os.path.join(USER, f"collect_{i}.png")
        t = cv2.imread(p)
        if t is not None:
            out[f"collect_{i}"] = t
    return out


def iter_images():
    pats = [os.path.join(SHOTS, "*.png"),
            os.path.join(SHOTS, "snap", "*.png")]
    seen = set()
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            b = os.path.basename(p)
            if b.startswith("_"):
                continue
            if p in seen:
                continue
            seen.add(p)
            yield p


def main() -> int:
    tpls = load_tpls()
    print(f"templates: {sorted(tpls)}")
    imgs = list(iter_images())
    print(f"images: {len(imgs)}")
    t0 = time.perf_counter()

    scores = {}   # filename -> best score
    where = {}
    for k, p in enumerate(imgs, 1):
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        if w != 720:
            img = cv2.resize(img, (720, int(round(h * 720 / w))),
                             interpolation=cv2.INTER_AREA)
        best, best_at, best_name = 0.0, None, None
        for name, t in tpls.items():
            if img.shape[0] < t.shape[0] or img.shape[1] < t.shape[1]:
                continue
            r = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, mxl = cv2.minMaxLoc(r)
            if mx > best:
                best, best_at, best_name = float(mx), mxl, name
        scores[os.path.basename(p)] = best
        where[os.path.basename(p)] = (best_name, best_at)
        if k % 100 == 0:
            print(f"  ...{k}/{len(imgs)}  {(time.perf_counter()-t0):.0f}s",
                  flush=True)

    print(f"\nscanned {len(scores)} images in "
          f"{time.perf_counter()-t0:.0f}s\n")

    # 人工真值：snap/ 下这三张含真实「采集中」
    truth = {b for b in scores if b.startswith("20261108-1523")
             or b.startswith("20261108-1525") or b.startswith("20261108-1521")}
    print(f"ground truth positives ({len(truth)}): {sorted(truth)}\n")

    for thr in THRESHOLDS:
        tp = [b for b in truth if scores[b] >= thr]
        fp = [(b, round(scores[b], 3)) for b, s in scores.items()
              if s >= thr and b not in truth]
        fn = [b for b in truth if scores[b] < thr]
        fp.sort(key=lambda t: -t[1])
        print(f"thr={thr:.2f}  tp={len(tp)}/{len(truth)}  fp={len(fp)}  "
              f"fn={len(fn)}")
        if fp:
            print(f"    fp top: {fp[:4]}")

    print("\ntruth scores:")
    for b in sorted(truth):
        print(f"  {b:32s} {scores[b]:.3f}  {where[b]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
