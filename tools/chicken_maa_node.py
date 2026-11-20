"""验证「单个 FeatureMatch 节点 + 6 张模板」能否一次找出多只鸡妖。

MaaFramework 的 FeatureMatch raw_detail 返回 count（内点数）而不是 score，
所以判定阈值就是 count。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"E:\work\AutoButter")
USER = ROOT / "User_screenshot"
NAMES = ["frontage_1", "frontage_2", "frontage_3", "back_1", "back_2", "back_3"]


class OneShot:
    pass


def make_controller(frame):
    from maa.controller import CustomController

    class C(CustomController):
        def __init__(self):
            super().__init__()

        def connect(self):
            return True

        def request_uuid(self):
            return "oneshot"

        def start_app(self, intent):
            return True

        def stop_app(self, intent):
            return True

        def screencap(self):
            return frame.copy()

        def click(self, x, y):
            return True

        def swipe(self, *a):
            return True

        def touch_down(self, *a):
            return True

        def touch_move(self, *a):
            return True

        def touch_up(self, *a):
            return True

        def click_key(self, k):
            return True

        def input_text(self, t):
            return True

        def key_down(self, k):
            return True

        def key_up(self, k):
            return True

    return C()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", default="0,250,720,810")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--images", default="user_test_1.png,user_test_2.png")
    args = ap.parse_args()
    roi = [int(v) for v in args.roi.split(",")]

    tmp = ROOT / "screenshots" / "_tmp_res2"
    (tmp / "image").mkdir(parents=True, exist_ok=True)
    (tmp / "pipeline").mkdir(exist_ok=True)
    for n in NAMES:
        (tmp / "image" / f"{n}.png").write_bytes((USER / f"{n}.png").read_bytes())

    pipe = {
        "鸡妖": {
            "recognition": "FeatureMatch",
            "template": [f"{n}.png" for n in NAMES],
            "roi": roi,
            "detector": "SIFT",
            "ratio": args.ratio,
            "count": args.count,
            "action": "DoNothing",
        }
    }
    (tmp / "pipeline" / "main.json").write_text(
        json.dumps(pipe, ensure_ascii=False, indent=2), encoding="utf-8")

    from maa.resource import Resource
    from maa.tasker import Tasker

    res = Resource()
    if not res.post_bundle(str(tmp)).wait().succeeded:
        raise SystemExit("resource load failed")

    imgs = []
    for name in args.images.split(","):
        p = USER / name
        if not p.exists():
            p = ROOT / "screenshots" / name
        imgs.append((name, p))

    for name, p in imgs:
        frame = cv2.imread(str(p))
        h, w = frame.shape[:2]
        if w != 720:
            frame = cv2.resize(frame, (720, int(round(h * 720 / w))),
                               interpolation=cv2.INTER_AREA)
        ctrl = make_controller(frame)
        if not ctrl.post_connection().wait().succeeded:
            raise SystemExit("controller connect failed")
        tasker = Tasker()
        if not tasker.bind(res, ctrl):
            raise SystemExit("bind failed")
        t0 = time.perf_counter()
        job = tasker.post_task("鸡妖").wait()
        dt = (time.perf_counter() - t0) * 1000
        detail = tasker.get_task_detail(job.job_id)
        rec = detail.nodes[0].recognition if detail.nodes else None
        print(f"=== {name}  roi={roi} count={args.count} ratio={args.ratio} "
              f"-> {dt:.0f}ms ===")
        if rec is None:
            print("  no recognition detail")
            continue
        print(f"  hit={rec.hit} box={rec.box}")
        raw = rec.raw_detail or {}
        for k in ("all", "best", "filtered"):
            if k in raw:
                print(f"  {k}: {raw[k]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
