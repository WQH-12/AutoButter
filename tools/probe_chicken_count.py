r"""只读探针：OCR 秘境面板「管理」左边的鸡妖数量区域。

用法：
    python tools\probe_chicken_count.py
    python tools\probe_chicken_count.py --roi 0,950,298,230
    python tools\probe_chicken_count.py --frame User_screenshot\full\720_realm_panel.png

只读取本地帧，不截图、不点击。输出 ROI 内每个 OCR 文本、框和分数。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "screenshots" / "_tmp_ocr_chicken"


def build_res(roi: list[int]) -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / "pipeline").mkdir(exist_ok=True)
    model_src = ROOT / "resource" / "model"
    model_dst = TMP / "model"
    if model_src.is_dir() and not model_dst.exists():
        shutil.copytree(model_src, model_dst)
    pipe = {
        "t": {
            "recognition": "OCR",
            "roi": roi,
            "action": "DoNothing",
            "expected": [],
        }
    }
    (TMP / "pipeline" / "main.json").write_text(
        json.dumps(pipe, ensure_ascii=False, indent=2), encoding="utf-8")


def ctrl_for(frame):
    from maa.controller import CustomController

    class C(CustomController):
        def connect(self):
            return True

        def request_uuid(self):
            return "ocr-chicken-count"

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame",
                    default=str(ROOT / "User_screenshot" / "full" /
                                "720_realm_panel.png"))
    ap.add_argument("--roi", default="0,950,298,230")
    args = ap.parse_args()
    roi = [int(v) for v in args.roi.split(",")]
    frame = cv2.imread(args.frame)
    if frame is None:
        print("[FATAL] cannot read", args.frame)
        return 2
    print("frame", args.frame, "shape", frame.shape, "roi", roi)
    build_res(roi)

    from maa.resource import Resource
    from maa.tasker import Tasker
    from maa.define import LoggingLevelEnum

    ctrl = ctrl_for(frame)
    if not ctrl.post_connection().wait().succeeded:
        print("[FATAL] connect failed")
        return 2
    res = Resource()
    if not res.post_bundle(str(TMP)).wait().succeeded:
        print("[FATAL] resource load failed")
        return 2
    tasker = Tasker()
    if not tasker.bind(res, ctrl):
        print("[FATAL] bind failed")
        return 2
    tasker.set_stdout_level(LoggingLevelEnum.Error)
    task = tasker.post_task("t").wait()
    detail = tasker.get_task_detail(task.job_id)
    rows = []
    for nd in getattr(detail, "nodes", []) or []:
        rec = getattr(nd, "recognition", None)
        if rec is None:
            continue
        for rr in (getattr(rec, "all_results", None) or []):
            text = getattr(rr, "text", None)
            box = getattr(rr, "box", None)
            score = getattr(rr, "score", None)
            if text:
                rows.append({
                    "text": str(text),
                    "box": [int(v) for v in box] if box else None,
                    "score": float(score) if score is not None else None,
                })
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
