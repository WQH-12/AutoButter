"""验证 layout.py 的 ROI 推导在不同长边下仍然框得住控件（纯几何，不连设备）。

背景
----
Maa 把截图缩到 720 短边，所以「不同分辨率」实际只表现为**长边不同**：
1080x2340 → 720x1560，1080x2400 → 720x1600，720x1280 → 720x1280。

HANDOVER §3.43 记录的新设备实测 `btn_manage @(298,1065)`（720x1600），
而 1040 * 1600/1560 = 1066.7 —— 说明 UI 的 y **随长边等比缩放**，
不是贴底对齐（贴底会是 1040+40=1080）。

如果推导公式用固定的像素偏移（`count = [mx+90, my-260, 240, 100]`），
控件按比例下移而窗口不跟着长，长边越大余量越小，最后控件被裁到 ROI 外，
`读派遣数量` 会永远读不到。这个测试就是钉住这一点。

基线（720x1560）实测控件位置：

    雞妖数量 n/3    y 795..823   （OCR 框 [403,795,135,28]）
    btn_plus        y 836..876   （40x40 @(521,836)）
    btn_dispatch    y 985..1039  （204x54 @(262,985)）
    manage          y 1040..1161

用法：python tools\\test_layout_scale.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))

import layout  # noqa: E402

PANEL = os.path.join(ROOT, "User_screenshot", "full", "720_realm_panel.png")
BASE_H = layout.BASE_H

# 基线上的控件上下边（实测）
WIDGETS = {
    "count_roi": (795, 823),
    "plus_roi": (836, 876),
    "detail_roi": (985, 1039),
}

HEIGHTS = (1280, 1440, 1520, 1560, 1600, 1680, 1800, 1920)


class _Job:
    def __init__(self, value):
        self.value = value

    def wait(self):
        return self

    def get(self):
        return self.value


class _Ctrl:
    def __init__(self, image):
        self.image = image

    def post_screencap(self):
        return _Job(self.image)


class _Ctx:
    def __init__(self, image):
        self.image = image
        self.overrides = None
        self.tasker = type("T", (), {"controller": _Ctrl(image)})()

    def override_pipeline(self, ov):
        self.overrides = ov
        return True


def _frame(h):
    """把 720x1560 的秘境面板重采样到 720xh。

    只用于让 `layout.ensure()` 有图可测；本测试的断言全部基于几何，
    不依赖模板分数（重采样会让分数偏低，不能当真实匹配率用）。
    """
    src = cv2.imread(PANEL)
    if src is None:
        raise SystemExit("cannot read " + PANEL)
    return cv2.resize(src, (720, h), interpolation=cv2.INTER_AREA)


def margins_for(h):
    """返回 {roi: (above, below)}，单位像素，负值 = 控件被裁出 ROI。"""
    img = _frame(h)
    layout.reset()
    ctx = _Ctx(img)
    layout.ensure(ctx, img)
    lay = dict(layout._LAYOUT)
    s = h / BASE_H
    out = {}
    for roi_name, (top, bot) in WIDGETS.items():
        wtop, wbot = top * s, bot * s
        rx, ry, rw, rh = lay[roi_name]
        out[roi_name] = (wtop - ry, (ry + rh) - wbot)
    return out, lay


def case_margins() -> None:
    print("\n[1] 各长边下控件到 ROI 边缘的余量（负值 = 被裁）")
    print("    %5s %-16s %-16s %-16s" % ("H", "count_roi", "plus_roi",
                                         "detail_roi"))
    for h in HEIGHTS:
        m, _ = margins_for(h)
        cells = []
        bad = []
        for name in ("count_roi", "plus_roi", "detail_roi"):
            above, below = m[name]
            cells.append("%+.0f/%+.0f" % (above, below))
            if above < 0 or below < 0:
                bad.append(name)
        print("    %5d %-16s %-16s %-16s%s"
              % (h, cells[0], cells[1], cells[2],
                 "  **CLIP** " + ",".join(bad) if bad else ""))
        assert not bad, (
            "长边 %d 时 %s 把控件裁出了 ROI：%s。"
            "layout.py 的 y 偏移必须按 h/%d 缩放。"
            % (h, bad, m, BASE_H))


def case_baseline_unchanged() -> None:
    """s=1.0 时必须逐像素等于原来的硬编码值，不能因为缩放引入漂移。"""
    print("\n[2] 720x1560 基线与旧硬编码值一致")
    _, lay = margins_for(BASE_H)
    expect = {
        "count_roi": [388, 780, 240, 100],
        "plus_roi": [468, 790, 260, 220],
        "detail_roi": [0, 880, 720, 300],
        "idle_roi": [213, 1084, 120, 80],
        "total_roi": [431, 1084, 120, 80],
        "content_roi": [0, 250, 720, 810],
        "confirm_roi": [248, 950, 260, 260],
        "tanxun_roi": [440, 1250, 280, 90],
    }
    for name, want in expect.items():
        got = list(lay[name])
        print("    %-12s %s" % (name, got))
        assert got == want, "%s 变了：%s != %s" % (name, got, want)


def case_margin_stable() -> None:
    """缩放后余量应当基本恒定（不随长边塌陷）。"""
    print("\n[3] 缩放后 count_roi 的上余量不随长边塌陷")
    prev = None
    for h in HEIGHTS:
        m, _ = margins_for(h)
        above = m["count_roi"][0]
        print("    H=%-5d above=%+.0f" % (h, above))
        if prev is not None:
            assert abs(above - prev) <= 3, (
                "H=%d 的 count_roi 上余量 %.0f 与上一档 %.0f 差太多，"
                "说明偏移没有按比例缩放" % (h, above, prev))
        prev = above


def case_dismiss_offset() -> None:
    """「关闭获得弹窗」的 target_offset 也要跟着缩放。"""
    print("\n[4] 关闭获得弹窗的 target_offset 随长边缩放")
    for h in HEIGHTS:
        _, lay = margins_for(h)
        img = _frame(h)
        ctx = _Ctx(img)
        layout.reset()
        layout.ensure(ctx, img)
        ov = ctx.overrides or {}
        dy = ov["关闭获得弹窗"]["target_offset"][1]
        want = round(775 * h / BASE_H)
        print("    H=%-5d target_offset[1]=%-4d (期望 %d)" % (h, dy, want))
        assert abs(dy - want) <= 1, (h, dy, want)


def case_anchor_fallback() -> None:
    """manage 匹配失败时用比例回退，回退值应与等比模型一致。"""
    print("\n[5] manage 未命中时的比例回退")
    blank = np.zeros((1600, 720, 3), np.uint8)
    layout.reset()
    ctx = _Ctx(blank)
    layout.ensure(ctx, blank)
    lay = dict(layout._LAYOUT)
    my = lay["manage"][1]
    want = round(0.667 * 1600)
    print("    manage.y=%d (0.667*h=%d, 等比=%d)"
          % (my, want, round(1040 * 1600 / BASE_H)))
    assert my == want, (my, want)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("[start] layout scale cases")
    case_margins()
    case_baseline_unchanged()
    case_margin_stable()
    case_dismiss_offset()
    case_anchor_fallback()
    print("\nlayout scale tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
