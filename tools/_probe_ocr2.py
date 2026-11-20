"""对两张新整帧做分区 OCR，确认界面身份。只读。

用法：python tools\_probe_ocr2.py <相对 User_screenshot 的路径>
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import cv2

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter"
USER = os.path.join(ROOT, "User_screenshot")
TMP = os.path.join(ROOT, "screenshots", "_tmp_ocr2")


def build_res(rois):
    if os.path.isdir(TMP):
        shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(os.path.join(TMP, "pipeline"), exist_ok=True)
    model = os.path.join(ROOT, "resource", "model")
    if os.path.isdir(model):
        shutil.copytree(model, os.path.join(TMP, "model"))
    pipe = {n: {"recognition": "OCR", "roi": r, "action": "DoNothing",
                "expected": []} for n, r in rois}
    with open(os.path.join(TMP, "pipeline", "main.json"), "w",
              encoding="utf-8") as f:
        json.dump(pipe, f, ensure_ascii=False, indent=2)


def ctrl_for(frame):
    from maa.controller import CustomController

    class C(CustomController):
        def connect(self):
            return True

        def request_uuid(self):
            return "ocr2"

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


def main():
    rel = sys.argv[1] if len(sys.argv) > 1 else "full/720_fullscreen_activity.png"
    p = os.path.join(USER, rel)
    frame = cv2.imread(p)
    if frame is None:
        raise SystemExit("cannot read " + p)
    h, w = frame.shape[:2]
    print("%s  %dx%d" % (rel, w, h))

    rois = [("band_%04d" % y, [0, y, 720, min(160, h - y)])
            for y in range(0, h, 160)]
    build_res(rois)

    from maa.resource import Resource
    from maa.tasker import Tasker

    ctrl = ctrl_for(frame)
    if not ctrl.post_connection().wait().succeeded:
        raise SystemExit("connect failed")
    res = Resource()
    if not res.post_bundle(TMP).wait().succeeded:
        raise SystemExit("load failed")
    tasker = Tasker()
    if not tasker.bind(res, ctrl):
        raise SystemExit("bind failed")

    for name, roi in rois:
        task = tasker.post_task(name).wait()
        detail = tasker.get_task_detail(task.job_id)
        texts = []
        for nd in getattr(detail, "nodes", []) or []:
            rec = getattr(nd, "recognition", None)
            if rec is None:
                continue
            for rr in (getattr(rec, "all_results", None) or []):
                t = getattr(rr, "text", None)
                if t:
                    texts.append(str(t))
        if texts:
            print("  y=%4d..%4d  %s" % (roi[1], roi[1] + roi[3], texts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
