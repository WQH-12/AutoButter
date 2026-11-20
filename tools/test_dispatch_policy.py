"""DispatchPolicy 的离线单测（不连设备）。"""
from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import cv2

sys.path.insert(0, "agent")

import cell_reco  # noqa: E402
import loop_ctl  # noqa: E402
import priority  # noqa: E402
import row_reco  # noqa: E402

TEST_LEVELS = {
    "honey": 7,
    "jade": 6,
    "silver": 5,
    "berry": 4,
    "alcohol": 3,
    "fabric": 2,
    "stone": 1,
}
DEFAULT_ORDER = priority.order()
assert DEFAULT_ORDER == ["alcohol"]
priority.set_levels(TEST_LEVELS)


class _RR:
    def __init__(self, text):
        self.text = text
        self.box = [184, 794, 116, 26]


class _Detail:
    def __init__(self, text):
        self.all_results = [_RR(text)]


class _Ctx:
    def __init__(self, text):
        self.text = text

    def run_recognition_direct(self, *args):
        return _Detail(self.text)


class _StockCtx:
    """左 ROI 返回空闲数，右 ROI 返回总数。"""

    def run_recognition_direct(self, _typ, jocr, _image):
        roi = list(getattr(jocr, "roi", []) or [])
        text = "閒【2】" if roi and roi[0] < 300 else "【2】總"
        return _Detail(text)


class _NoOcrCtx:
    """target=1 时不应该调用 OCR；一旦调用就判失败。"""

    def run_recognition_direct(self, *args):
        raise AssertionError("target=1 should skip OCR")


class _AddCtrl:
    """点「+」的假 controller；`eat` 表示前 N 次点击被弹窗吃掉。"""

    def __init__(self, n, eat=0):
        self.n = n
        self.eat = eat
        self.clicks = 0

    def post_screencap(self):
        return _Job()

    def post_click(self, _x, _y):
        self.clicks += 1
        if self.eat > 0:
            self.eat -= 1
        else:
            self.n += 1
        return _Job()


class _Job:
    def __init__(self, value=None):
        self.value = value

    def wait(self):
        return self

    def get(self):
        return self.value


class _ExitCtrl:
    def __init__(self, image):
        self.image = image
        self.clicks = []

    def post_click(self, x, y):
        self.clicks.append((x, y))
        return _Job()

    def post_screencap(self):
        return _Job(self.image)


def _arg(mode="full"):
    return SimpleNamespace(
        custom_recognition_param='{"dispatch_mode":"%s"}' % mode,
        image=None)


def _case(realm, item, level, stock, total=3, mode="full", contested=False):
    loop_ctl._STATE.update({
        "realm": realm, "stock": stock, "total": total, "target": 3,
        "qty": None,
    })
    cell_reco.LAST_PICK = {
        "item": item, "level": level, "level_text": "%d级" % level,
        "box": [0, 0, 1, 1], "contested": contested,
    }
    return loop_ctl.DispatchPolicy().analyze(_Ctx(""), _arg(mode)).detail


def main() -> int:
    levels = {
        "stone": 1, "fabric": 2, "jade": 3, "honey": 4,
        "silver": 5, "berry": 6, "alcohol": 7,
    }
    assert priority.order(levels) == [
        "alcohol", "berry", "silver", "honey", "jade", "fabric", "stone"]
    assert priority.order({"alcohol": 0, "berry": 7}) == ["berry"]
    priority.reset_defaults()
    assert priority.order({"stone": "{stone_priority}"}) == ["alcohol"]
    priority.set_levels(TEST_LEVELS)

    assert _case("own", "berry", 3, 1)["target"] == 1
    # 自家秘境被抢的酒米：派 3 只夺回（需求方 11-17）。
    d = _case("own", "alcohol", 1, 10, total=10, contested=True)
    assert d["target"] == 3 and d["defer"] is False, d
    # 鸡妖不够 3 只 → defer，不去点一个派不满的资源。
    d = _case("own", "alcohol", 1, 2, total=10, contested=True)
    assert d["target"] == 3 and d["defer"] is True, d
    # 总数上限同样生效：total=2 时最多 2 只。
    d = _case("own", "alcohol", 1, 5, total=2, contested=True)
    assert d["target"] == 2 and d["defer"] is False, d
    # 未被抢的自家酒米仍然只派 1 只。
    d = _case("own", "alcohol", 1, 10, total=10, contested=False)
    assert d["target"] == 1 and d["defer"] is False, d
    # 自家其它资源即使被抢也不夺（需求方：只处理酒米）。
    d = _case("own", "berry", 3, 10, total=10, contested=True)
    assert d["target"] == 1, d
    assert _case("other", "berry", 3, 2)["defer"] is True
    d = _case("other", "berry", 2, 2)
    assert d["target"] == 2 and d["defer"] is False
    d = _case("other", "berry", 4, 3)
    assert d["target"] == 3 and d["defer"] is False
    assert _case("other", "alcohol", 1, 2)["defer"] is True
    d = _case("other", "alcohol", 1, 3)
    assert d["target"] == 3 and d["defer"] is False
    d = _case("other", "berry", 1, 1)
    assert d["target"] == 1 and d["defer"] is False

    # §3.66：target=1 时 `读派遣数量` 直接命中，不再读 `n/3`。
    loop_ctl._STATE["target"] = 1
    count_arg = SimpleNamespace(
        custom_recognition_param="{}", image=None,
        roi=SimpleNamespace(x=395, y=790, w=165, h=45))
    res = loop_ctl.DispatchCountReached().analyze(_NoOcrCtx(), count_arg)
    assert res.box is not None and res.detail["skip"] is True, res.detail
    # target>1 仍然读计数；1/3 没到 2 只 → 失败。
    loop_ctl._STATE["target"] = 2
    res = loop_ctl.DispatchCountReached().analyze(_Ctx("1/3"), count_arg)
    assert res.box is None and res.detail["reached"] is False, res.detail

    # §3.67：+ 点击被「恭喜獲得」吃掉一次时，AddChicken 必须回读并补点，
    # 不能把没生效的点击算进 plus 配额后直接去掠奪。
    add_ctrl = _AddCtrl(1, eat=1)
    add_ctx = SimpleNamespace(
        tasker=SimpleNamespace(controller=add_ctrl))
    add_arg = SimpleNamespace(
        box=SimpleNamespace(x=0, y=0, w=10, h=10),
        custom_action_param='{"max_plus": 4, "gap": 0.01}')
    orig_ocr = loop_ctl._ocr_texts
    orig_clear = loop_ctl.clear_popups
    loop_ctl._ocr_texts = lambda _ctx, _img, _roi: (
        "雞妖數量：%d/3" % add_ctrl.n, [])
    loop_ctl.clear_popups = lambda _ctx, image=None, ctrl=None, **kw: image
    loop_ctl._STATE["target"] = 3
    loop_ctl._STATE["plus"] = 0
    try:
        ok = loop_ctl.AddChicken().run(add_ctx, add_arg)
    finally:
        loop_ctl._ocr_texts = orig_ocr
        loop_ctl.clear_popups = orig_clear
    assert ok and add_ctrl.n == 3 and add_ctrl.clicks == 3, (
        ok, add_ctrl.n, add_ctrl.clicks)
    assert loop_ctl._STATE["plus"] == 2, loop_ctl._STATE["plus"]

    # 总数上限：total=1/2 时酒米和 3 级资源不能硬要 3 只。
    d = _case("other", "alcohol", 1, 1, total=1)
    assert d["target"] == 1 and d["defer"] is False
    d = _case("other", "alcohol", 1, 2, total=2)
    assert d["target"] == 2 and d["defer"] is False
    d = _case("other", "berry", 3, 2, total=2)
    assert d["target"] == 2 and d["defer"] is False

    # 两种模式在 idle < 目标数时行为不同。
    d = _case("other", "alcohol", 1, 1, total=2, mode="full")
    assert d["target"] == 2 and d["defer"] is True
    d = _case("other", "alcohol", 1, 1, total=2, mode="utilize")
    assert d["target"] == 1 and d["defer"] is False

    # ChickenStock 一次读左空闲、一次读右总数。
    loop_ctl._STATE["total"] = None
    stock_arg = SimpleNamespace(
        custom_recognition_param='{"total_roi":[420,1065,120,80]}',
        image=None,
        roi=SimpleNamespace(x=205, y=1065, w=120, h=80))
    d = loop_ctl.ChickenStock().analyze(_StockCtx(), stock_arg).detail
    assert d["stock"] == 2 and d["total"] == 2
    assert loop_ctl._STATE["total"] == 2

    # 等级 OCR：优先右下角；数量与等级粘连时取紧挨「级」的最后一位。
    assert cell_reco._read_level(_Ctx("3501级"), None, [0, 0, 100, 100])[0] == 1
    assert cell_reco._read_level(_Ctx("5級"), None, [0, 0, 100, 100])[0] == 5
    assert cell_reco._read_level(_Ctx("350 2级"), None, [0, 0, 100, 100])[0] == 2
    assert cell_reco._read_level(_Ctx("350级"), None, [0, 0, 100, 100])[0] is None

    # 只剩一行不在采集中、且该行 `N级` 没读出来时，应退到 soft target。
    rows = [
        {"btn": [556, 454, 94, 26], "readable": True, "collecting": True,
         "wanted": [{"item": "berry", "x": 334}],
         "items": [{"item": "berry", "x": 334}], "level_visible": True},
        {"btn": [556, 757, 94, 26], "readable": True, "collecting": False,
         "wanted": [{"item": "fabric", "x": 411}],
         "items": [{"item": "fabric", "x": 411}], "level_visible": False},
    ]
    soft = row_reco.RowPicker.finalize(rows)
    assert soft["target"] is rows[1] and soft["target_kind"] == "soft"

    # 只剩底部渐隐行不在采集中：readable=False 也要兜底尝试一次。
    fallback_rows = [
        {"btn": [556, 454, 94, 26], "readable": True, "collecting": True,
         "wanted": [{"item": "berry", "x": 334}],
         "items": [{"item": "berry", "x": 334}], "level_visible": True},
        {"btn": [556, 967, 94, 25], "readable": False, "collecting": False,
         "wanted": [{"item": "alcohol", "x": 564}],
         "items": [{"item": "alcohol", "x": 564}], "level_visible": False},
    ]
    fallback = row_reco.RowPicker.finalize(fallback_rows)
    assert (fallback["target"] is fallback_rows[1]
            and fallback["target_kind"] == "fallback")

    last_rows = [
        {"btn": [556, 454, 94, 26], "readable": True, "collecting": True,
         "wanted": [{"item": "berry", "x": 334}],
         "items": [{"item": "berry", "x": 334}], "level_visible": True},
        {"btn": [556, 967, 94, 25], "readable": False, "collecting": False,
         "wanted": [], "items": [{"item": "berry", "x": 334}],
         "level_visible": False},
    ]
    last = row_reco.RowPicker.finalize(last_rows)
    assert (last["target"] is last_rows[1]
            and last["target_kind"] == "last_resort")

    # 真实帧交叉验证：720_list_top 里前 3 行采集中/无 alcohol，
    # 只有底部第 4 行有 alcohol 且 readable=False，应走 fallback。
    real_img = cv2.imread("User_screenshot/full/720_list_top.png")
    assert real_img is not None
    real = row_reco.RowPicker().scan(real_img, priority=["alcohol"])
    assert (real["target"] is not None
            and real["target"]["btn"][1] == 967
            and real["target_kind"] == "fallback")
    row_reco.remember_row(real["target"])
    row_reco.mark_last_row_bad()
    again = row_reco.RowPicker().scan(real_img, priority=["alcohol"])
    assert (again["target"] is None
            or again["target"]["btn"][1] != 967)
    row_reco.clear_bad_rows()

    # 空秘境行走时间冷却；到期后重新成为候选，不再永久坏行。
    row_reco.remember_row(rows[1])
    row_reco.mark_last_row_bad()
    assert row_reco.is_bad_row(rows[1])
    assert not row_reco.is_bad_row(
        rows[1], now=time.time() + row_reco.EMPTY_COOLDOWN_SECONDS + 1)
    row_reco.clear_bad_rows()
    assert not row_reco.is_bad_row(rows[1])

    # §3.61 修法 1：同名玩家的不同秘境不能互相锁死（物品串是 sig）。
    row_reco.clear_bad_rows()
    a = {"name": "刀霸郭破曉", "avatar": "aa",
         "items": [{"item": "berry", "x": 256}], "btn": [556, 479, 94, 26]}
    b = {"name": "刀霸郭破曉", "avatar": "aa",
         "items": [{"item": "alcohol", "x": 564}], "btn": [556, 783, 94, 26]}
    row_reco.remember_row(a)
    row_reco.mark_last_row_done()
    assert row_reco.is_bad_row(a)
    assert not row_reco.is_bad_row(b)
    row_reco.clear_bad_rows()

    # 头像 hash 碰撞（同名 avatar 但不同玩家）也不能锁无关行。
    x = {"name": "甲", "avatar": "COLLIDE",
         "items": [{"item": "berry", "x": 256}], "btn": [556, 479, 94, 26]}
    y = {"name": "乙", "avatar": "COLLIDE",
         "items": [{"item": "honey", "x": 411}], "btn": [556, 631, 94, 26]}
    row_reco.remember_row(x)
    row_reco.mark_last_row_done()
    assert row_reco.is_bad_row(x)
    assert not row_reco.is_bad_row(y)
    row_reco.clear_bad_rows()

    # 同名同物品串 = 同一行；物品串变了 = 不同秘境。
    same_a = {"name": "玩家A", "items": [{"item": "alcohol", "x": 256}]}
    same_b = {"name": "玩家A", "items": [{"item": "berry", "x": 256}]}
    row_reco.remember_row(same_a)
    row_reco.mark_last_row_bad()
    assert row_reco.is_bad_row(same_a)
    assert not row_reco.is_bad_row(same_b)
    row_reco.clear_bad_rows()

    # §3.61 修法 3：done 用独立表，不再给 bad 叠 strikes（330 → 660）。
    row_reco.clear_bad_rows()
    r = {"name": "丙", "avatar": "h3",
         "items": [{"item": "berry", "x": 256}], "btn": [556, 479, 94, 26]}
    row_reco.remember_row(r)
    row_reco.mark_last_row_done()
    done_recs = list(row_reco.DONE_COOLDOWN.values())
    assert len(done_recs) == 1 and done_recs[0]["delay"] == 330.0, done_recs
    assert row_reco.COOLDOWN == {}
    row_reco.remember_row(r)
    row_reco.mark_last_row_bad()
    bad_recs = list(row_reco.COOLDOWN.values())
    assert bad_recs and bad_recs[0]["delay"] == 330.0, bad_recs
    row_reco.remember_row(r)
    row_reco.mark_last_row_bad()
    assert bad_recs[0]["delay"] == 660.0, bad_recs
    # done 再标一次仍是固定 330，不乘 2。
    row_reco.remember_row(r)
    row_reco.mark_last_row_done()
    assert len(row_reco.DONE_COOLDOWN) == 1
    assert list(row_reco.DONE_COOLDOWN.values())[0]["delay"] == 330.0
    row_reco.clear_bad_rows()
    assert row_reco.DONE_COOLDOWN == {}

    # §3.61 修法 4：DispatchOkGate 只在 ClickReset 真派出去时命中。
    loop_ctl._STATE["last_dispatch_ok"] = None
    assert loop_ctl.DispatchOkGate().analyze(None, None).box is None
    loop_ctl._STATE["last_dispatch_ok"] = False
    assert loop_ctl.DispatchOkGate().analyze(None, None).box is None
    loop_ctl._STATE["last_dispatch_ok"] = True
    assert loop_ctl.DispatchOkGate().analyze(None, None).box is not None
    loop_ctl._STATE["last_dispatch_ok"] = None

    # §3.61 修法 5：last_resort 的 wanted 为空时，detail 用 items 兜底。
    assert row_reco._target_item(last["target"]) == "berry"
    assert row_reco._target_item({"wanted": [], "items": []}) is None

    # 全页都在冷却：finalize 应给出 all_cooling + next_ready，
    # 让 SwipeRows 等待而不是误判成列表到底。
    c1 = {"name": "冷却A", "btn": [556, 454, 94, 26],
          "items": [{"item": "alcohol", "x": 256}],
          "wanted": [], "collecting": False, "readable": True}
    c2 = {"name": "冷却B", "btn": [556, 606, 94, 26],
          "items": [{"item": "berry", "x": 256}],
          "wanted": [], "collecting": False, "readable": True}
    for c in (c1, c2):
        row_reco.remember_row(c)
        row_reco.mark_last_row_bad()
    cooled = row_reco.RowPicker.finalize([c1, c2])
    assert cooled["all_cooling"] is True
    assert cooled["next_ready"] > 0
    row_reco.clear_bad_rows()

    # 鸡妖不足延迟：按时间冷却，不再按列表轮次。
    row_reco.clear_bad_rows()
    defer_row = {
        "btn": [556, 606, 94, 26],
        "items": [{"item": "alcohol", "x": 256}],
    }
    row_reco.remember_row(defer_row)
    row_reco.skip_last_row(seconds=1)
    assert row_reco.is_bad_row(defer_row)
    assert not row_reco.is_bad_row(defer_row, now=time.time() + 2)
    row_reco.clear_bad_rows()

    # 别人秘境退出：必须点两次空处，然后看到列表才算成功。
    list_img = cv2.imread("User_screenshot/full/720_list_top.png")
    assert list_img is not None
    exit_ctrl = _ExitCtrl(list_img)
    exit_ctx = SimpleNamespace(
        tasker=SimpleNamespace(controller=exit_ctrl))
    assert loop_ctl.ExitOtherRealm().run(
        exit_ctx, SimpleNamespace(custom_action_param="{}"))
    assert exit_ctrl.clicks == [(350, 1326), (350, 1326)]

    # 延迟格排除：同一帧第一次选中的格子，第二次 scan 不应再选它
    img = cv2.imread("screenshots/159_before_drain.png")
    assert img is not None
    picker = cell_reco.CellPicker()
    first = picker.scan(img, exclude=[])
    assert first["target"] is not None
    deferred = [first["target"]["box"]]
    second = picker.scan(img, exclude=deferred)
    if second["target"] is not None:
        a = first["target"]["box"]
        b = second["target"]["box"]
        assert abs(a[0] - b[0]) > 24 or abs(a[1] - b[1]) > 24

    # 滑动前清弹窗必须复用调用方已取的 controller（§3.66）。
    # 若这里让 clear_popups 自己再取一次，agent 模式下会重建
    # RemoteController，调用方手里的指针随即悬垂。
    seen = {}
    orig_clear = loop_ctl.clear_popups

    def fake_clear(ctx, image=None, ctrl=None, **kwargs):
        seen["ctx"] = ctx
        seen["ctrl"] = ctrl
        return image

    loop_ctl.clear_popups = fake_clear
    try:
        marker_ctx = object()
        marker_ctrl = object()
        loop_ctl.SwipeRows._grab_without_popup(marker_ctx, marker_ctrl)
        assert seen["ctx"] is marker_ctx, seen
        assert seen["ctrl"] is marker_ctrl, seen
    finally:
        loop_ctl.clear_popups = orig_clear

    print("DispatchPolicy tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
