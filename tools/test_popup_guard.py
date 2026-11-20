"""弹窗护栏离线单测（不连设备）。

用 `screenshots/100_now.png`（有「恭喜獲得」）和 `102_popup_gone.png`
（已清空）验证 `agent/popup_guard.py` 的三件事：

1. 有弹窗的帧能命中，没弹窗的帧不误报；
2. `clear_popups` 会点掉弹窗、重新截图，直到干净；
3. 没有 controller 的离线 context 不会抛异常。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "agent")

import popup_guard  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(ROOT, "screenshots")
ON_ERROR = os.path.join(
    ROOT, "install", "debug", "on_error",
    "2026.11.18-18.52.30.915_主循环选格.png",
)


def _load(name):
    p = os.path.join(SHOTS, name)
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("missing frame: %s" % p)
    return img


def _load_path(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("missing frame: %s" % path)
    return img


class _Job:
    def __init__(self, value=None, ok=True):
        self.value = value
        self.succeeded = ok

    def wait(self):
        return self

    def get(self):
        return self.value


class _Ctrl:
    def __init__(self, frames):
        self.frames = list(frames)
        self.clicks = []

    def post_screencap(self):
        return _Job(self.frames.pop(0))

    def post_click(self, x, y):
        self.clicks.append((x, y))
        return _Job(None, True)


class _Ctx:
    def __init__(self, ctrl):
        self.tasker = type("_Tasker", (), {"controller": ctrl})()


def main() -> int:
    popup = _load("100_now.png")
    clean = _load("102_popup_gone.png")

    box, score = popup_guard.find_popup(popup)
    assert box is not None and score >= 0.9, (box, score)

    box2, _score2 = popup_guard.find_popup(clean)
    assert box2 is None, box2

    # 18:52 实机失败帧：弹窗盖住秘境面板，必须能命中。
    # `install/debug/on_error` 是诊断产物，可能被清理；不存在时跳过。
    failed = None
    if os.path.exists(ON_ERROR):
        failed = _load_path(ON_ERROR)
        box3, score3 = popup_guard.find_popup(failed)
        assert box3 is not None and score3 >= 0.9, (box3, score3)
    else:
        print("[skip] on_error frame not found: %s" % ON_ERROR)

    # 一条弹窗：第一次用传入的 image，点掉后重新截图拿到 clean
    ctrl = _Ctrl([clean])
    out = popup_guard.clear_popups(_Ctx(ctrl), image=popup)
    assert len(ctrl.clicks) == 1, ctrl.clicks
    assert out is clean, "should return the clean re-screencap"

    # 实机失败帧：同一条路径，点一次后返回干净帧
    if failed is not None:
        ctrl3 = _Ctrl([clean])
        out4 = popup_guard.clear_popups(_Ctx(ctrl3), image=failed)
        assert len(ctrl3.clicks) == 1, ctrl3.clicks
        assert out4 is clean

    # 连续两条：重新截图先给 popup，再给 clean
    ctrl2 = _Ctrl([popup, clean])
    out2 = popup_guard.clear_popups(_Ctx(ctrl2), image=popup)
    assert len(ctrl2.clicks) == 2, ctrl2.clicks
    assert out2 is clean

    # 离线 context：没有 controller 时原样返回，不抛异常
    class _Offline:
        class tasker:
            @property
            def controller(self):
                raise RuntimeError("no controller")

    out3 = popup_guard.clear_popups(_Offline(), image=clean)
    assert out3 is clean

    print("PopupGuard tests: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
