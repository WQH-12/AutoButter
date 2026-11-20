"""用真实 MaaFramework 引擎跑 FeatureMatch，验证鸡妖节点参数。

对比 6 张模板的命中情况，以及 count/index/ratio 的效果。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"E:\work\AutoButter")
USER = ROOT / "User_screenshot"
NAMES = ["frontage_1", "frontage_2", "frontage_3", "back_1", "back_2", "back_3"]


def build_pipeline(ratio, count, roi):
    nodes = {}
    for n in NAMES:
        nodes[f"chk_{n}"] = {
            "recognition": "FeatureMatch",
            "template": f"{n}.png",
            "roi": list(roi),
            "detector": "SIFT",
            "ratio": ratio,
            "count": count,
            "action": "DoNothing",
        }
    return nodes


def run(img_path, ratio, count, roi, tmp_res):
    from maa.controller import CustomController
    from maa.resource import Resource
    from maa.tasker import Tasker

    frame = cv2.imread(str(img_path))
    if frame is None:
        raise SystemExit(f"cannot read {img_path}")

    class OneShot(CustomController):
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

        def swipe(self, x1, y1, x2, y2, d):
            return True

        def touch_down(self, c, x, y, p):
            return True

        def touch_move(self, c, x, y, p):
            return True

        def touch_up(self, c):
            return True

        def click_key(self, k):
            return True

        def input_text(self, t):
            return True

        def key_down(self, k):
            return True

        def key_up(self, k):
            return True

    ctrl = OneShot()
    if not ctrl.post_connection().wait().succeeded:
        raise SystemExit("controller connect failed")
    res = Resource()
    if not res.post_bundle(str(tmp_res)).wait().succeeded:
        raise SystemExit("resource load failed")
    tasker = Tasker()
    if not tasker.bind(res, ctrl):
        raise SystemExit("bind failed")

    out = {}
    for n in NAMES:
        node = f"chk_{n}"
        task = tasker.post_task(node).wait()
        detail = tasker.get_task_detail(task.job_id)
        found = []
        for nd in getattr(detail, "nodes", []):
            if getattr(nd, "name", "") != node:
                continue
            rec = getattr(nd, "recognition", None)
            if rec is None:
                continue
            for rr in getattr(rec, "all_results", []) or []:
                box = getattr(rr, "box", None)
                found.append((round(float(getattr(rr, "score", 0.0)), 3),
                              tuple(int(v) for v in box) if box is not None else None))
        out[n] = (bool(task.succeeded), found)
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--roi", default="0,250,720,800")
    ap.add_argument("--images", default="user_test_1.png,user_test_2.png")
    args = ap.parse_args()
    roi = tuple(int(v) for v in args.roi.split(","))

    tmp_res = ROOT / "screenshots" / "_tmp_res"
    (tmp_res / "image").mkdir(parents=True, exist_ok=True)
    for n in NAMES:
        src = USER / f"{n}.png"
        dst = tmp_res / "image" / f"{n}.png"
        dst.write_bytes(src.read_bytes())
    pipe = build_pipeline(args.ratio, args.count, roi)
    (tmp_res / "pipeline").mkdir(exist_ok=True)
    (tmp_res / "pipeline" / "main.json").write_text(
        json.dumps(pipe, ensure_ascii=False, indent=2), encoding="utf-8")

    for img in args.images.split(","):
        print(f"=== {img} ratio={args.ratio} count={args.count} roi={roi} ===")
        out = run(USER / img, args.ratio, args.count, roi, tmp_res)
        for n in NAMES:
            ok, found = out[n]
            print(f"  {n:12s} ok={ok} n={len(found)} {found}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
