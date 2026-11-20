"""精确定位底部「探尋」文字在三种界面上的 bbox。只读。

用 OCR 而不是模板：按钮有三种文案（探尋 / 探尋託管中 / 探尋掌櫃託管），
裁一个模板覆盖不了，OCR 只要判断「包含探尋」即可。
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import cv2

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter"
USER = os.path.join(ROOT, "User_screenshot", "full")
TMP = os.path.join(ROOT, "screenshots", "_tmp_ocr3")

# 底部右侧按钮区。三张图的按钮条 y 略有不同，ROI 放宽一点。
ROI = [440, 1250, 280, 90]


def build_res():
    if os.path.isdir(TMP):
        shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(os.path.join(TMP, "pipeline"), exist_ok=True)
    model = os.path.join(ROOT, "resource", "model")
    if os.path.isdir(model):
        shutil.copytree(model, os.path.join(TMP, "model"))
    pipe = {"t": {"recognition": "OCR", "roi": ROI, "action": "DoNothing",
                  "expected": []}}
    with open(os.path.join(TMP, "pipeline", "main.json"), "w",
              encoding="utf-8") as f:
        json.dump(pipe, f, ensure_ascii=False, indent=2)


def ctrl_for(frame):
    from maa.controller import CustomController

    class C(CustomController):
        def connect(self):
            return True

        def request_uuid(self):
            return "ocr3"

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
    build_res()
    from maa.resource import Resource
    from maa.tasker import Tasker

    for n in ["720_realm_panel.png", "720_fullscreen_others_home.png",
              "720_list_top.png", "720_dispatch_menu.png"]:
        p = os.path.join(USER, n)
        frame = cv2.imread(p)
        if frame is None:
            print("MISSING", n)
            continue
        ctrl = ctrl_for(frame)
        if not ctrl.post_connection().wait().succeeded:
            print(n, "connect failed")
            continue
        res = Resource()
        if not res.post_bundle(TMP).wait().succeeded:
            print(n, "load failed")
            continue
        tasker = Tasker()
        if not tasker.bind(res, ctrl):
            print(n, "bind failed")
            continue
        task = tasker.post_task("t").wait()
        detail = tasker.get_task_detail(task.job_id)
        out = []
        for nd in getattr(detail, "nodes", []) or []:
            rec = getattr(nd, "recognition", None)
            if rec is None:
                continue
            for rr in (getattr(rec, "all_results", None) or []):
                t = getattr(rr, "text", None)
                box = getattr(rr, "box", None)
                if t:
                    out.append((str(t), tuple(int(v) for v in box) if box else None))
        print("%-32s %s" % (n, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
