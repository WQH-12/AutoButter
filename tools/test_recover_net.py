"""RecoverNet（网络异常 / 掉线兜底）的离线单测，不需要设备。

需求方 11-14 19:13：**不要添加校验，任何异常都返回主页面**。
所以 RecoverNet 不带任何文字识别，只有固定三步：

  1. 已在冒险主页面（nav_home 命中、无 btn_manage、无列表行）→ 命中，放行；
  2. 否则点蓝点（exit_anchor 中心）；
  3. 等 WAIT 秒，进入下一轮；最多 max_rounds 轮。

轮次在动作内部循环 —— 不走 `max_hit` + 自引用。原因见类 docstring：
RecoverNet 只有 `run()`，写成 custom_recognition 会让 Maa 报
`recognition is null`，节点永远失败（恢复门此前从未执行过）。

注意：**不用 BACK**。这个游戏不响应系统返回键。
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

PANEL = cv2.imread(str(ROOT / "User_screenshot" / "full" / "720_realm_panel.png"))
HOME = np.zeros((1560, 720, 3), np.uint8)
_nav = cv2.imread(str(ROOT / "resource" / "image" / "nav_home.png"))
HOME[1427:1527, 157:257] = _nav
BLANK = np.zeros((1560, 720, 3), np.uint8)


class _Job:
    def __init__(self, value=None):
        self.value = value
        self.succeeded = True

    def wait(self):
        return self

    def get(self):
        return self.value


class _Ctrl:
    def __init__(self, frames):
        self.frames = frames
        self.idx = 0
        self.taps = []

    def post_screencap(self):
        return _Job(self.frames[min(self.idx, len(self.frames) - 1)].copy())

    def post_click(self, x, y):
        self.taps.append((x, y))
        if self.idx + 1 < len(self.frames):
            self.idx += 1
        return _Job(None)


def _ctx(ctrl):
    """只有 controller：RecoverNet 不该再调 OCR，一调就 AttributeError。"""
    return SimpleNamespace(tasker=SimpleNamespace(controller=ctrl))


def _arg(max_rounds=None):
    if max_rounds is None:
        return SimpleNamespace(custom_action_param="{}")
    return SimpleNamespace(
        custom_action_param='{"max_rounds": %d}' % max_rounds)


def _fast(act):
    """把每轮等待压到 0，单测不用真睡。"""
    act.WAIT = 0.0
    return act


def case_home() -> None:
    """已在主页面：命中，不点任何东西。"""
    loop_ctl.reset_state()
    ctrl = _Ctrl([HOME])
    ok = _fast(loop_ctl.RecoverNet()).run(_ctx(ctrl), _arg(2))
    print("  home: ok=%s taps=%s" % (ok, ctrl.taps), flush=True)
    assert ok is True
    assert ctrl.taps == []


def case_panel() -> None:
    """秘境面板：点蓝点后重试；第三帧回主页即命中。"""
    loop_ctl.reset_state()
    ctrl = _Ctrl([PANEL, PANEL, HOME])
    ok = _fast(loop_ctl.RecoverNet()).run(_ctx(ctrl), _arg(8))
    print("  panel: ok=%s taps=%s" % (ok, ctrl.taps), flush=True)
    assert ok is True
    assert ctrl.taps == [(674, 256)] * 2, ctrl.taps


def case_unknown() -> None:
    """完全不认识的界面：轮次用完返回 False，每轮点一次蓝点。"""
    loop_ctl.reset_state()
    ctrl = _Ctrl([BLANK])
    ok = _fast(loop_ctl.RecoverNet()).run(_ctx(ctrl), _arg(3))
    print("  unknown: ok=%s taps=%s" % (ok, ctrl.taps), flush=True)
    assert ok is False
    assert ctrl.taps == [(674, 256)] * 3


def case_max_rounds_default() -> None:
    """没传 max_rounds 时用类默认值 8；传 0 / 负数按 1 轮处理。"""
    loop_ctl.reset_state()
    ctrl = _Ctrl([BLANK])
    ok = _fast(loop_ctl.RecoverNet()).run(_ctx(ctrl), _arg())
    print("  default rounds: ok=%s taps=%d" % (ok, len(ctrl.taps)), flush=True)
    assert ok is False
    assert len(ctrl.taps) == loop_ctl.RecoverNet.MAX_ROUNDS

    loop_ctl.reset_state()
    ctrl = _Ctrl([BLANK])
    ok = _fast(loop_ctl.RecoverNet()).run(_ctx(ctrl), _arg(0))
    assert ok is False
    assert len(ctrl.taps) == 1, ctrl.taps


def case_no_ocr() -> None:
    """回归：RecoverNet 不许再碰 OCR。上下文里没有 run_recognition_direct，
    真去调会 AttributeError，用例会直接炸。"""
    loop_ctl.reset_state()
    ctrl = _Ctrl([BLANK])
    ctx = _ctx(ctrl)
    assert not hasattr(ctx, "run_recognition_direct")
    ok = _fast(loop_ctl.RecoverNet()).run(ctx, _arg(1))
    assert ok is False


def case_wait_value() -> None:
    """每轮固定 3s（8 轮 ≈ 25s）。改这个数要同步 §3.47 和 main.json 注释。"""
    assert loop_ctl.RecoverNet.WAIT == 3.0, loop_ctl.RecoverNet.WAIT
    assert loop_ctl.RecoverNet.MAX_ROUNDS == 8


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("[start] recover-net cases", flush=True)
    case_home()
    case_panel()
    case_unknown()
    case_max_rounds_default()
    case_no_ocr()
    case_wait_value()
    print("recover net tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
