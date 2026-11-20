"""「刷新」按钮判定 + 刷新链路闸门的离线单测（不连设备）。

覆盖三层：

1. `loop_ctl.refresh_state()` 的判据 —— 用真机素材（橙/灰/非列表帧）
2. `RefreshGate` 的三个闸门条件 —— 可点 / 有配额 / 有空闲鸡妖
3. `ClickRefresh` / `CloseList` 的动作语义

素材（`User_screenshot/`）：

    720_fullscreen_activity.png  刷新可点（橙）S=124.5  R-B=113.5
    720_list_top.png             刷新不可点（灰）S=0.0  R-B=0.0
    720_realm_panel.png          不在探寻列表（定位分数 0.17）

用法：python tools\\test_refresh_gate.py
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
import registry  # noqa: E402

USER = ROOT / "User_screenshot"
SHOTS = ROOT / "screenshots"

LIST_ON = cv2.imread(str(USER / "full" / "720_fullscreen_activity.png"))
LIST_OFF = cv2.imread(str(USER / "full" / "720_list_top.png"))
PANEL = cv2.imread(str(USER / "full" / "720_realm_panel.png"))


class _Job:
    def __init__(self, value=None):
        self.value = value
        self.succeeded = True

    def wait(self):
        return self

    def get(self):
        return self.value


class _Ctrl:
    """按点击次数推进剧本；每次 screencap 返回当前帧。"""

    def __init__(self, frames):
        self.frames = frames
        self.idx = 0
        self.clicks = []

    def post_screencap(self):
        return _Job(self.frames[min(self.idx, len(self.frames) - 1)].copy())

    def post_click(self, x, y):
        self.clicks.append((x, y))
        if self.idx + 1 < len(self.frames):
            self.idx += 1
        return _Job(None)


def _ctx(ctrl):
    return SimpleNamespace(tasker=SimpleNamespace(controller=ctrl))


def _recog_arg(image, param=None):
    return SimpleNamespace(
        image=image,
        roi=None,
        custom_recognition_param=param or "{}")


def _action_arg(param=None, box=(0, 0, 1, 1)):
    bx, by, bw, bh = box
    return SimpleNamespace(
        box=SimpleNamespace(x=bx, y=by, w=bw, h=bh),
        custom_action_param=param or "{}")


# ---------- 1. 判据 ----------

def case_state_orange() -> None:
    """橙按钮：found + clickable。"""
    st = loop_ctl.refresh_state(LIST_ON)
    print("  orange: %s" % st, flush=True)
    assert st["found"] is True
    assert st["clickable"] is True
    assert st["S"] >= loop_ctl.REFRESH_SAT
    assert st["RB"] >= loop_ctl.REFRESH_RB
    # 位置要在 720 基线的 (265,1101) 附近，允许 ±8
    x, y, _w, _h = st["box"]
    assert abs(x - 265) <= 8 and abs(y - 1101) <= 8, st["box"]


def case_state_grey() -> None:
    """灰按钮：found 但不可点。"""
    st = loop_ctl.refresh_state(LIST_OFF)
    print("  grey: %s" % st, flush=True)
    assert st["found"] is True
    assert st["clickable"] is False
    assert st["S"] < loop_ctl.REFRESH_SAT


def case_state_not_list() -> None:
    """秘境面板 / 主页面：定位不到按钮。"""
    for name, img in (("panel", PANEL),):
        st = loop_ctl.refresh_state(img)
        print("  %s: %s" % (name, st), flush=True)
        assert st["found"] is False, st
        assert st["clickable"] is False
    home = cv2.imread(str(SHOTS / "21_main_720.png"))
    if home is not None:
        st = loop_ctl.refresh_state(home)
        print("  home: %s" % st, flush=True)
        assert st["found"] is False, st


def case_state_missing_template() -> None:
    """模板缺失时不能抛异常，要安全返回不可点。"""
    saved = dict(loop_ctl._TPL_CACHE)
    try:
        loop_ctl._TPL_CACHE["refresh_off.png"] = None
        st = loop_ctl.refresh_state(LIST_ON)
        print("  no tpl: %s" % st, flush=True)
        assert st["found"] is False and st["clickable"] is False
    finally:
        loop_ctl._TPL_CACHE.clear()
        loop_ctl._TPL_CACHE.update(saved)


def case_state_other_canvas() -> None:
    """720x1600 / 720x1680：按钮像素尺寸不变，只挪位置。

    11-17 实跑（720x1600）`RefreshGate` 全程 `found=False, score=0.694`，
    根因是 `refresh_state()` 把模板按 `h/1560` 放大再匹配。游戏的 UI
    控件像素尺寸不随长边变（`btn_manage` 在 1560/1600 上都是 120x121），
    只有 y 位置变。这里合成帧钉住这一点：把 1560 帧拉高，按钮区贴回原尺寸。
    """
    bx, by, bw, bh = 265, 1101, 210, 56
    for h, nav_y in ((1600, 1457), (1680, 1536)):
        out = cv2.resize(LIST_OFF, (720, h), interpolation=cv2.INTER_AREA)
        btn = LIST_OFF[by:by + bh, bx:bx + bw].copy()
        ny = int(round(by * h / 1560.0))
        out[ny:ny + bh, bx:bx + bw] = btn
        s = h / 1560.0
        list_top = max(0, nav_y - int(round(1070 * s)))
        list_h = max(120, min(int(round(822 * s)),
                              nav_y - int(round(248 * s)) - list_top))
        roi = [216, max(0, list_top + list_h - int(round(110 * s))),
               302, int(round(110 * s))]
        st = loop_ctl.refresh_state(out, roi)
        print("  canvas 720x%d: %s" % (h, st), flush=True)
        assert st["found"] is True, (h, st)
        assert st["clickable"] is False, (h, st)
        assert st["score"] >= loop_ctl.REFRESH_SCORE, (h, st)


# ---------- 2. RefreshGate 的三个条件 ----------

def _gate(image, param=None):
    return loop_ctl.RefreshGate().analyze(_ctx(_Ctrl([image])),
                                          _recog_arg(image, param))


def case_gate_pass() -> None:
    """可点 + 有配额 + 有鸡妖 → 命中。"""
    loop_ctl.reset_state()
    loop_ctl._STATE["stock"] = 3
    res = _gate(LIST_ON)
    print("  gate pass: %s" % res.detail, flush=True)
    assert res.box is not None
    assert res.detail["clickable"] is True
    assert res.detail["has_chicken"] is True


def case_gate_no_stock() -> None:
    """没有空闲鸡妖 → 不刷新（刷出来也派不出去）。"""
    loop_ctl.reset_state()
    loop_ctl._STATE["stock"] = 0
    res = _gate(LIST_ON)
    print("  gate stock=0: %s" % res.detail, flush=True)
    assert res.box is None
    assert res.detail["has_chicken"] is False

    loop_ctl.reset_state()
    loop_ctl._STATE["stock"] = None
    res = _gate(LIST_ON)
    assert res.box is None, res.detail


def case_gate_quota() -> None:
    """配额用满 → 不命中（避免在刷新按钮上空转）。"""
    loop_ctl.reset_state()
    loop_ctl._STATE["stock"] = 3
    loop_ctl._STATE["refresh"] = loop_ctl.MAX_REFRESH
    res = _gate(LIST_ON)
    print("  gate quota: %s" % res.detail, flush=True)
    assert res.box is None
    assert res.detail["used"] == loop_ctl.MAX_REFRESH


def case_gate_grey() -> None:
    """按钮灰 → 不命中（调用方落到重进秘境）。"""
    loop_ctl.reset_state()
    loop_ctl._STATE["stock"] = 3
    res = _gate(LIST_OFF)
    print("  gate grey: %s" % res.detail, flush=True)
    assert res.box is None
    assert res.detail["clickable"] is False


def case_gate_custom_limit() -> None:
    """param 里的 max_refresh 覆盖默认值。"""
    loop_ctl.reset_state()
    loop_ctl._STATE["stock"] = 3
    loop_ctl._STATE["refresh"] = 2
    res = _gate(LIST_ON, '{"max_refresh": 2}')
    assert res.box is None, res.detail
    res = _gate(LIST_ON, '{"max_refresh": 3}')
    assert res.box is not None, res.detail


# ---------- 3. 动作 ----------

def case_click_refresh() -> None:
    """点刷新：落在按钮中心、配额 +1。"""
    loop_ctl.reset_state()
    ctrl = _Ctrl([LIST_ON])
    act = loop_ctl.ClickRefresh()
    act.SETTLE = 0.0
    ok = act.run(_ctx(ctrl), _action_arg())
    print("  click: ok=%s clicks=%s quota=%d"
          % (ok, ctrl.clicks, loop_ctl._STATE["refresh"]), flush=True)
    assert ok is True
    assert len(ctrl.clicks) == 1
    x, y = ctrl.clicks[0]
    # 按钮 (265,1101,210,56) 中心 ≈ (370,1129)
    assert abs(x - 370) <= 8 and abs(y - 1129) <= 8, ctrl.clicks
    assert loop_ctl._STATE["refresh"] == 1


def case_click_refresh_grey() -> None:
    """按钮灰时不能点，返回 False 让调用方去重进。"""
    loop_ctl.reset_state()
    ctrl = _Ctrl([LIST_OFF])
    act = loop_ctl.ClickRefresh()
    act.SETTLE = 0.0
    ok = act.run(_ctx(ctrl), _action_arg())
    print("  click grey: ok=%s clicks=%s" % (ok, ctrl.clicks), flush=True)
    assert ok is False
    assert ctrl.clicks == []
    assert loop_ctl._STATE["refresh"] == 0


def case_close_list() -> None:
    """列表 → 点一次蓝点 → 面板（btn_manage 命中）就算关掉。"""
    loop_ctl.reset_state()
    ctrl = _Ctrl([LIST_OFF, PANEL])
    act = loop_ctl.CloseList()
    act.GAP = 0.0
    ok = act.run(_ctx(ctrl), _action_arg())
    print("  close: ok=%s clicks=%s" % (ok, ctrl.clicks), flush=True)
    assert ok is True
    assert len(ctrl.clicks) == 1


def case_close_list_already_closed() -> None:
    """已经在面板上（btn_manage 命中）：一次都不点。"""
    loop_ctl.reset_state()
    ctrl = _Ctrl([PANEL])
    act = loop_ctl.CloseList()
    act.GAP = 0.0
    ok = act.run(_ctx(ctrl), _action_arg())
    print("  close noop: ok=%s clicks=%s" % (ok, ctrl.clicks), flush=True)
    assert ok is True
    assert ctrl.clicks == []


def case_close_list_panel_with_fake_rows() -> None:
    """秘境面板上 find_rows 会误检 4 行 —— 不能因此继续点。

    11-15 实跑日志：`[BackToHome] state #1 panel=False(0.234) rows=4`。
    面板命中 btn_manage 时，即使 rows>0 也必须判「已关」。
    """
    loop_ctl.reset_state()
    ctrl = _Ctrl([PANEL])
    act = loop_ctl.CloseList()
    act.GAP = 0.0
    # 打桩：让 _list_rows 在面板帧上也返回 4 行
    orig = loop_ctl._list_rows
    try:
        loop_ctl._list_rows = lambda img: [1, 2, 3, 4]
        ok = act.run(_ctx(ctrl), _action_arg())
    finally:
        loop_ctl._list_rows = orig
    print("  close panel+fake rows: ok=%s clicks=%s" % (ok, ctrl.clicks),
          flush=True)
    assert ok is True, "面板命中 btn_manage 就该判已关"
    assert ctrl.clicks == [], ctrl.clicks


def case_setrealm_resets_quota() -> None:
    """进入秘境时刷新配额归零（游戏自己的 CD 才是真上限）。"""
    loop_ctl.reset_state()
    loop_ctl._STATE["refresh"] = 2
    act = loop_ctl.SetRealm()
    act.run(_ctx(_Ctrl([PANEL])), _action_arg('{"realm": "own"}'))
    print("  setrealm: quota=%d" % loop_ctl._STATE["refresh"], flush=True)
    assert loop_ctl._STATE["refresh"] == 0


def case_thresholds_sane() -> None:
    """阈值必须是「灰(0) 与橙(124) 之间」，改坏了这个用例会炸。"""
    assert 10 <= loop_ctl.REFRESH_SAT <= 100, loop_ctl.REFRESH_SAT
    assert 10 <= loop_ctl.REFRESH_RB <= 100, loop_ctl.REFRESH_RB
    assert 0.6 <= loop_ctl.REFRESH_SCORE <= 0.95, loop_ctl.REFRESH_SCORE
    assert loop_ctl.MAX_REFRESH >= 1


def case_registered() -> None:
    """三个新名字必须注册在**正确的那一侧**（§3.47 的教训）。"""
    recs = {n for n, _m, _c in registry.RECOGNITIONS}
    acts = {n for n, _m in registry.ACTIONS}
    assert "RefreshGate" in recs, recs
    assert "ClickRefresh" in acts, acts
    assert "CloseList" in acts, acts
    # 反向：RefreshGate 不是动作，ClickRefresh / CloseList 不是识别
    assert "RefreshGate" not in acts
    assert "ClickRefresh" not in recs
    assert "CloseList" not in recs


def case_pipeline_wiring() -> None:
    """pipeline 里三个节点的 recognition/action 类型必须对。"""
    import json
    import re

    p = ROOT / "resource" / "pipeline" / "main.json"
    data = json.loads(re.sub(r"//[^\n]*", "", p.read_text(encoding="utf-8")))
    gate = data["刷新闸门"]
    assert gate["recognition"] == "Custom"
    assert gate["custom_recognition"] == "RefreshGate"
    assert "custom_action" not in gate
    for name, action in (("点刷新", "ClickRefresh"), ("关列表", "CloseList")):
        node = data[name]
        assert node["recognition"] == "DirectHit", name
        assert node["action"] == "Custom", name
        assert node["custom_action"] == action, name


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    for name, img in (("LIST_ON", LIST_ON), ("LIST_OFF", LIST_OFF),
                      ("PANEL", PANEL)):
        assert img is not None, "缺素材 %s" % name

    print("[start] refresh gate cases", flush=True)
    case_state_orange()
    case_state_grey()
    case_state_not_list()
    case_state_missing_template()
    case_state_other_canvas()
    case_gate_pass()
    case_gate_no_stock()
    case_gate_quota()
    case_gate_grey()
    case_gate_custom_limit()
    case_click_refresh()
    case_click_refresh_grey()
    case_close_list()
    case_close_list_already_closed()
    case_close_list_panel_with_fake_rows()
    case_setrealm_resets_quota()
    case_thresholds_sane()
    case_registered()
    case_pipeline_wiring()
    print("refresh gate tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
