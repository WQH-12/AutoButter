"""「该资源正在被他人掠夺」的离线单测（不需要设备）。

场景（需求方 11-14 补充）：
  点「掠奪」后，游戏弹「该资源正在被他人掠夺」，**浮窗不关**。
  正确行为：
    1. `ClickReset` 检测到 `btn_dispatch.png` 还在；
    2. 点一次右上角蓝点（`exit_anchor` 缓存点）退出浮窗；
    3. 该格记为「延迟」（`cell_reco.defer_last`），**不**记为已派遣；
    4. 循环计数清零，继续下一格。

本脚本用 ReplayController 造两种帧序列：
  - 正常：点「掠奪」→ 浮窗消失（秘境面板）
  - 冲突：点「掠奪」→ 浮窗还在 → 点蓝点 → 秘境面板
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))

import loop_ctl  # noqa: E402

POPUP = cv2.imread(str(ROOT / "User_screenshot" / "full" / "720_dispatch_menu.png"))
PANEL = cv2.imread(str(ROOT / "User_screenshot" / "full" / "720_realm_panel.png"))


class _Job:
    def __init__(self, value=None):
        self.value = value
        self.succeeded = True

    def wait(self):
        return self

    def get(self):
        return self.value


class _Ctrl:
    """按顺序返回帧；点一下前进一帧（模拟界面切换）。"""

    def __init__(self, frames):
        self.frames = frames
        self.idx = 0
        self.clicks = []

    def post_screencap(self):
        frame = self.frames[min(self.idx, len(self.frames) - 1)]
        return _Job(frame.copy())

    def post_click(self, x, y):
        self.clicks.append((x, y))
        if self.idx + 1 < len(self.frames):
            self.idx += 1
        return _Job(None)


def _ctx(ctrl):
    return SimpleNamespace(tasker=SimpleNamespace(controller=ctrl))


def _arg(box=(262, 985, 204, 54)):
    return SimpleNamespace(
        box=SimpleNamespace(x=box[0], y=box[1], w=box[2], h=box[3]),
        custom_action_param="{}",
    )


def _reset_cell_state():
    import cell_reco
    cell_reco.clear_deferred()
    cell_reco.remember_pick({
        "box": [100, 300, 90, 90],
        "item": "berry",
        "cx": 145,
    })


def case_normal() -> bool:
    """点「掠奪」→ 浮窗消失：记为已派遣。"""
    _reset_cell_state()
    ctrl = _Ctrl([PANEL])
    loop_ctl.reset_state()
    ok = loop_ctl.ClickReset().run(_ctx(ctrl), _arg())
    import cell_reco
    dispatched = list(cell_reco.DISPATCHED)
    deferred = list(cell_reco.DEFERRED)
    print("  normal: ok=%s clicks=%s dispatched=%s deferred=%s"
          % (ok, ctrl.clicks, dispatched, deferred), flush=True)
    assert ok
    assert len(ctrl.clicks) == 1, "正常路径只该点一次「掠奪」"
    assert dispatched and not deferred, "正常路径应记已派遣"
    return True


def case_blocked() -> bool:
    """点「掠奪」→ 浮窗还在 → 点蓝点 → 秘境面板：记为延迟。"""
    _reset_cell_state()
    ctrl = _Ctrl([POPUP, POPUP, PANEL])
    loop_ctl.reset_state()
    ok = loop_ctl.ClickReset().run(_ctx(ctrl), _arg())
    import cell_reco
    dispatched = list(cell_reco.DISPATCHED)
    deferred = list(cell_reco.DEFERRED)
    print("  blocked: ok=%s clicks=%s dispatched=%s deferred=%s"
          % (ok, ctrl.clicks, dispatched, deferred), flush=True)
    assert ok
    assert len(ctrl.clicks) == 2, "冲突路径应是「掠奪 + 蓝点」两次点击"
    # 第 2 次点击必须是蓝点（缓存点），不是「掠奪」
    assert ctrl.clicks[1] != ctrl.clicks[0], "第二次点击不该还是掠奪"
    assert deferred and not dispatched, "冲突路径应记延迟，不是已派遣"
    return True


def case_blocked_twice() -> bool:
    """点蓝点后浮窗还开着 → 再补一次蓝点。"""
    _reset_cell_state()
    ctrl = _Ctrl([POPUP, POPUP, POPUP, PANEL])
    loop_ctl.reset_state()
    ok = loop_ctl.ClickReset().run(_ctx(ctrl), _arg())
    print("  blocked x2: ok=%s clicks=%s" % (ok, ctrl.clicks), flush=True)
    assert ok
    assert len(ctrl.clicks) == 3, "应补第二次蓝点"
    return True


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    assert POPUP is not None and PANEL is not None, "缺测试帧"
    print("[start] rob-conflict cases", flush=True)
    case_normal()
    case_blocked()
    case_blocked_twice()
    print("rob conflict tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
