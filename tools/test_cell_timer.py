"""「格子正在被采集」的倒计时判据离线单测（不连设备）。

覆盖：

1. `cell_reco.find_timers()` 在真机素材上的检出/不检出
2. 倒计时相对格子的归属：下方 = 在采，上方 = 被抢
3. `scan()` 在 **SIFT 全部失效** 时仍能靠倒计时判 busy（§3.55 的核心诉求）
4. 换长边（720x1600 / 720x1680）不失效
5. 误检负例：主界面 / 探寻列表 / 派遣浮窗

素材都在 `User_screenshot/` 与 `screenshots/`。

用法：python tools\\test_cell_timer.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))

import cell_reco  # noqa: E402
import priority  # noqa: E402

USER = ROOT / "User_screenshot"
SHOTS = ROOT / "screenshots"

# 全物品优先级：让 scan() 有机会真的选出 target，验证 busy 真的挡住了它
ALL_ITEMS = {
    "honey": 7, "jade": 6, "silver": 5, "berry": 4,
    "alcohol": 3, "fabric": 2, "stone": 1,
}


def _load(*parts) -> "cv2.Mat":
    p = ROOT.joinpath(*parts)
    img = cv2.imread(str(p))
    assert img is not None, "缺素材 %s" % p
    return img


class _NoSift:
    """假装 SIFT 一只鸡都没检到 —— 真机 §3.55 就是这种帧。"""

    ready = True

    def find(self, img, roi):
        return []


def _scan_no_sift(img, priority_levels=None):
    """只用倒计时判 busy 的 scan()。"""
    picker = cell_reco.CellPicker()
    picker._chicken = _NoSift()
    return picker.scan(img, priority=priority_levels or ALL_ITEMS)


# ---------- 1. 检出 ----------

def case_find_user_test_1() -> None:
    """自家秘境：4 个格子下方 + 右上 1 个格子上方，共 5 条倒计时。"""
    img = _load("User_screenshot", "user_test_1.png")
    timers = cell_reco.find_timers(img)
    print("  user_test_1 timers=%s" % timers, flush=True)
    assert len(timers) == 5, timers
    # 四条在下排（y>800），一条是右上的 23:10（y<500）
    low = [t for t in timers if t[1] > 700]
    high = [t for t in timers if t[1] < 700]
    assert len(low) == 4, timers
    assert len(high) == 1, timers
    # 宽度必须落在实测的 46~51 之间（4 个数字 + 冒号）
    for t in timers:
        assert 40 <= t[2] <= 70, t


def case_find_others_home() -> None:
    """别人秘境：1 条。"""
    img = _load("User_screenshot", "full", "720_fullscreen_others_home.png")
    timers = cell_reco.find_timers(img)
    print("  others_home timers=%s" % timers, flush=True)
    assert len(timers) == 1, timers


# ---------- 2. 归属 ----------

def case_below_vs_above() -> None:
    """同一个格子：下方倒计时算 below，上方倒计时算 above。"""
    img = _load("User_screenshot", "user_test_1.png")
    timers = cell_reco.find_timers(img)
    # (570,570,89,92) 是右上格，23:10 在它上方
    top_right = (570, 570, 89, 92)
    assert cell_reco._timer_near(top_right, timers, "below") is None
    assert cell_reco._timer_near(top_right, timers, "above") is not None
    # (67,733,81,85) 是左下的格子，倒计时在它下方
    left = (67, 733, 81, 85)
    assert cell_reco._timer_near(left, timers, "below") is not None
    assert cell_reco._timer_near(left, timers, "above") is None


def case_timer_is_exclusive() -> None:
    """一条倒计时不能被两个格子同时认领（否则相邻格会互相污染）。"""
    img = _load("User_screenshot", "user_test_1.png")
    cells = cell_reco.detect_cells(img)
    timers = cell_reco.find_timers(img)
    claimed, used = cell_reco._claim_below(cells, timers)
    print("  claimed=%s" % claimed, flush=True)
    assert len(set(used)) == len(used)
    for t in claimed.values():
        assert t in timers


# ---------- 3. SIFT 全失效时倒计时兜底 ----------

def case_busy_without_sift() -> None:
    """核心：SIFT 一只都没检到，靠倒计时仍然判 busy。"""
    img = _load("User_screenshot", "user_test_1.png")
    result = _scan_no_sift(img)
    print("  no-sift cells:", flush=True)
    for c in result["cells"]:
        print("    %s busy=%s below=%s above=%s"
              % (c["box"], c["busy"], c["timer_below"], c["timer_above"]),
              flush=True)
    assert result["cells"], "一格都没检出来"
    assert all(c["busy_sift"] is False for c in result["cells"])
    assert all(c["busy"] for c in result["cells"]), "倒计时没兜住"
    # 4 格下方 + 1 格上方
    below = [c for c in result["cells"] if c["timer_below"]]
    above = [c for c in result["cells"] if c["timer_above"]]
    assert len(below) == 4, below
    assert len(above) == 1, above
    # 全 busy → 没有 target
    assert result["target"] is None, result["target"]


def case_free_cells_still_selectable() -> None:
    """有空闲格时仍然能选出 target（不能因为加了倒计时就全屏蔽）。"""
    img = _load("screenshots", "159_before_drain.png")
    result = _scan_no_sift(img)
    free = [c for c in result["cells"] if not c["busy"]]
    print("  free=%d busy=%d" % (len(free), len(result["cells"]) - len(free)),
          flush=True)
    assert len(free) == 4, [c["box"] for c in result["cells"]]
    assert result["target"] is not None, "有空格却选不出 target"
    assert result["target"]["busy"] is False


def case_contested_alcohol_selectable() -> None:
    """自家秘境被抢的酒米：`allow_contested={"alcohol"}` 时要能选中。

    `user_test_1` 的右上格 (570,570,89,92) 是「自家 + 酒米 + 上方倒计时」。
    默认全 busy → target=None；放行后应该正好选它，且 `contested=True`。
    """
    img = _load("User_screenshot", "user_test_1.png")
    picker = cell_reco.CellPicker()
    picker._chicken = _NoSift()

    strict = picker.scan(img, priority=ALL_ITEMS)
    assert strict["target"] is None, strict["target"]

    loose = picker.scan(img, priority=ALL_ITEMS,
                        allow_contested={"alcohol"})
    target = loose["target"]
    print("  contested target=%s" % (None if target is None else target["box"]),
          flush=True)
    assert target is not None, "放行后仍选不出被抢的酒米"
    assert target["item"] == "alcohol", target
    assert target["box"] == [570, 570, 89, 92], target["box"]
    assert target["contested"] is True
    assert target["contested_ok"] is True
    # 其它资源不因为「放行酒米」而被误选（它们仍是 busy）
    others = [c for c in loose["cells"]
              if c["item"] != "alcohol" and not c["busy"]]
    assert others == [], others


def case_contested_gate_by_realm_and_stock() -> None:
    """只有「自家 + 鸡妖够 3 只」才放行；别的秘境 / 鸡妖不够都不放行。"""
    rec = cell_reco.CellRecognition()
    assert rec is not None, "maa 不可用时跳过"
    import loop_ctl
    for realm, stock, total, want in (
            ("own", 10, 10, {"alcohol"}),
            ("own", 3, 10, {"alcohol"}),
            ("own", 2, 10, None),        # 鸡妖不够 3 只
            ("own", 0, 10, None),
            ("own", None, 10, None),
            ("own", 10, 2, {"alcohol"}),  # 总数只有 2：派 2 只也算夺
            ("other", 10, 10, None),
            (None, 10, 10, None),
    ):
        loop_ctl._STATE.update({"realm": realm, "stock": stock,
                                "total": total})
        got = rec._allow_contested()
        print("  realm=%-5s stock=%-4s total=%-3s -> %s"
              % (realm, stock, total, got), flush=True)
        assert got == want, (realm, stock, total, got, want)
    loop_ctl.reset_state()


def case_no_false_busy() -> None:
    """没有倒计时、没有鸡的帧：一格都不能判 busy。"""
    img = _load("screenshots", "113_final2.png")
    result = _scan_no_sift(img)
    print("  clean cells=%d timers=%s"
          % (len(result["cells"]), result["timers"]), flush=True)
    assert result["timers"] == [], result["timers"]
    assert all(not c["busy"] for c in result["cells"]), [
        c["box"] for c in result["cells"] if c["busy"]]
    assert result["target"] is not None


# ---------- 4. 换长边 ----------

def case_other_canvas() -> None:
    """720x1600 / 720x1680 也要检出同样的条数。"""
    img = _load("User_screenshot", "user_test_1.png")
    for h, want in ((1600, 5), (1680, 5)):
        resized = cv2.resize(img, (720, h), interpolation=cv2.INTER_AREA)
        timers = cell_reco.find_timers(resized)
        print("  canvas 720x%d timers=%d" % (h, len(timers)), flush=True)
        assert len(timers) == want, (h, timers)


# ---------- 5. 误检负例 ----------

def case_no_false_positive() -> None:
    """主界面 / 探寻列表 / 派遣浮窗 / 干净面板：0 条。"""
    negatives = [
        ("screenshots", "21_main_720.png"),
        ("User_screenshot", "full", "720_list_top.png"),
        ("User_screenshot", "full", "720_fullscreen_activity.png"),
        ("User_screenshot", "full", "720_dispatch_menu.png"),
        ("screenshots", "113_final2.png"),
    ]
    for parts in negatives:
        img = _load(*parts)
        timers = cell_reco.find_timers(img)
        print("  %-32s timers=%s" % (parts[-1], timers), flush=True)
        assert timers == [], (parts, timers)


def case_thresholds_sane() -> None:
    """阈值区间自检：改坏了这个用例会炸。"""
    assert 150 <= cell_reco.TIMER_V_BRIGHT <= 255
    assert 0 <= cell_reco.TIMER_S_MAX <= 120
    assert 1.5 <= cell_reco.TIMER_MIN_ASPECT <= 3.0
    assert 0.15 <= cell_reco.TIMER_MIN_DARK <= 0.5
    assert cell_reco.TIMER_MIN_DIGITS >= 3
    assert cell_reco.TIMER_BELOW_GAP[1] > cell_reco.TIMER_BELOW_GAP[0]
    assert cell_reco.TIMER_ABOVE_GAP[1] < 0


def case_scan_fields() -> None:
    """scan() 的返回结构必须有新字段，否则 pipeline 日志没法定位。"""
    img = _load("screenshots", "113_final2.png")
    result = _scan_no_sift(img)
    for key in ("timers", "cells", "chickens", "blocked_x", "target"):
        assert key in result, key
    cell = result["cells"][0]
    for key in ("box", "cx", "item", "score", "busy", "busy_sift",
                "timer_below", "timer_above", "contested", "contested_ok"):
        assert key in cell, key


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    priority.set_levels(ALL_ITEMS)

    print("[start] cell timer cases", flush=True)
    case_thresholds_sane()
    case_scan_fields()
    case_find_user_test_1()
    case_find_others_home()
    case_below_vs_above()
    case_timer_is_exclusive()
    case_busy_without_sift()
    case_free_cells_still_selectable()
    case_contested_alcohol_selectable()
    case_contested_gate_by_realm_and_stock()
    case_no_false_busy()
    case_other_canvas()
    case_no_false_positive()
    print("cell timer tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
