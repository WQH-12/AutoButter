"""量 Maa 原生 Click 的落点分布：它在识别框里是**随机取点**，不是取中心。

背景（HANDOVER §3.45）：`nav_home.png` 的识别框是 100x100，但框里只有
中间一块是图标，四边是底栏背景。Maa 原生 `Click` 会在整个框里随机选点，
于是有一部分点击落在图标外面，点了等于没点。

本脚本用 ReplayController 反复执行同一个节点，统计落点：

  - 不带 `target_offset`：落点铺满整个框，约 18% 在图标外
  - 带 `target_offset: [30,30,-50,-50]`：落点收进图标中心，100% 命中

用法（不需要设备）：

    python tools\\probe_click_spread.py
    python tools\\probe_click_spread.py --n 200
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from maa.controller import CustomController
from maa.resource import Resource
from maa.tasker import Tasker

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "screenshots" / "_probe_spread"
(OUT / "pipeline").mkdir(parents=True, exist_ok=True)
(OUT / "image").mkdir(parents=True, exist_ok=True)
shutil.copy(str(ROOT / "resource/image/nav_home.png"),
            str(OUT / "image/nav_home.png"))

FRAME = cv2.imread(str(ROOT / "User_screenshot" / "full" / "720_realm_panel.png"))

# nav_home.png 在 100x100 框里的图标范围（相对坐标，tools 里量过）
ICON = (14, 18, 84, 82)  # x, y, w, h


class Replay(CustomController):
    def __init__(self):
        super().__init__()
        self.pts = []

    def connect(self):
        return True

    def request_uuid(self):
        return "r"

    def screencap(self):
        return FRAME.copy()

    def click(self, x, y):
        self.pts.append((x, y))
        return True

    def swipe(self, *a):
        return True

    def touch_down(self, c, x, y, p):
        self.pts.append((x, y))
        return True

    def touch_move(self, *a):
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


def run(offset, n=60):
    node = {"recognition": "TemplateMatch", "template": "nav_home.png",
            "roi": [0, 600, 720, 1200], "threshold": 0.8, "action": "Click"}
    if offset is not None:
        node["target_offset"] = offset
    (OUT / "pipeline" / "main.json").write_text(
        json.dumps({"t": node}), encoding="utf-8")
    res = Resource()
    res.post_bundle(str(OUT)).wait()
    ctrl = Replay()
    tasker = Tasker()
    tasker.bind(res, ctrl)
    for _ in range(n):
        tasker.post_task("t").wait()

    box = (157, 1427)
    ix, iy, iw, ih = ICON
    inside = 0
    for x, y in ctrl.pts:
        rx, ry = x - box[0], y - box[1]
        if ix <= rx < ix + iw and iy <= ry < iy + ih:
            inside += 1
    xs = [p[0] - box[0] for p in ctrl.pts]
    ys = [p[1] - box[1] for p in ctrl.pts]
    print("offset=%-22s n=%d  x %3d..%3d  y %3d..%3d  落在图标内 %d/%d (%.0f%%)"
          % (str(offset), n, min(xs), max(xs), min(ys), max(ys),
             inside, n, 100.0 * inside / n), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="每种配置点几次")
    args = ap.parse_args()
    run(None, args.n)
    run([30, 30, -50, -50], args.n)
