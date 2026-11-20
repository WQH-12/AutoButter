"""等待鸡妖回补的离线单测（不连设备）。"""
from __future__ import annotations

import json
import os
import re
import sys
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, "agent")

import loop_ctl  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Job:
    def __init__(self, value=None):
        self.value = value
        self.succeeded = True

    def wait(self):
        return self

    def get(self):
        return self.value


class _Ctrl:
    def __init__(self, popup_after=1):
        self.popup_clicks = 0
        self.popup_after = popup_after
        self.calls = 0

    def post_screencap(self):
        self.calls += 1
        if self.popup_after is not None and self.calls >= self.popup_after:
            self.popup_clicks += 1
        return _Job(None)


def _ctx(ctrl):
    return SimpleNamespace(tasker=SimpleNamespace(controller=ctrl))


def _run_arg(timeout=20):
    return SimpleNamespace(
        custom_action_param='{"timeout": %d, "interval": 0.01}' % timeout)


def main() -> int:
    # 重进后的顺序必须是「读鸡妖 → 0 才等弹窗」，不能反过来先等弹窗。
    pipe_path = os.path.join(ROOT, "resource", "pipeline", "main.json")
    pipe_text = open(pipe_path, encoding="utf-8").read()
    pipe_body = re.sub(r"^\s*//.*$", "", pipe_text, flags=re.M)
    nodes = json.loads(pipe_body)
    assert nodes["重进-确认"]["next"] == ["读空闲鸡妖", "等待获得弹窗"]
    assert nodes["等待获得弹窗"]["on_error"] == ["确认已进入秘境"]
    assert nodes["点探尋并下翻"]["next"] == ["主循环选行"]
    assert "post_wait_freezes" not in nodes["点探尋并下翻"]
    assert "post_wait_freezes" not in nodes["主循环探尋"]
    # §3.50：三个「选行失败 → 去滑」的入口都改成先过 `滑动闸门`
    assert nodes["点探尋并下翻"]["on_error"][0] == "滑动闸门"
    assert nodes["主循环选行"]["on_error"][0] == "滑动闸门"
    assert nodes["主循环探尋"]["on_error"][0] == "滑动闸门"
    assert nodes["确认他人秘境"]["next"] == ["读空闲鸡妖", "没有空闲鸡妖"]
    assert nodes["没有空闲鸡妖"]["next"] == ["退到主页面"]
    assert nodes["退到主页面"]["custom_action"] == "BackToHome"
    assert nodes["循环闸门"]["next"] == ["退到主页面"]
    # §3.45：回主页面不再点底栏「冒险」 tab，改成只点蓝点（exit_anchor）。
    # DirectHit 是因为「是否已在主页面」由 ClickAdventure 自己判。
    assert nodes["回主界面"]["recognition"] == "DirectHit"
    assert nodes["回主界面"]["custom_action"] == "ClickAdventure"
    assert nodes["回主界面"]["next"] == ["重进-找菜单"]
    # §3.49：划不动时先看刷新按钮能不能点，能点就刷一遍再看
    assert nodes["滑不动了"]["next"] == [
        "冷却闸门", "刷新闸门", "循环闸门", "异常收尾"]
    assert nodes["刷新闸门"]["custom_recognition"] == "RefreshGate"
    assert nodes["刷新闸门"]["next"] == ["点刷新"]
    assert nodes["点刷新"]["custom_action"] == "ClickRefresh"
    assert nodes["点刷新"]["next"] == ["关列表"]
    # 刷新只重刷最上面 3 个，必须从列表顶部重看 → 关列表再点探尋
    assert nodes["关列表"]["custom_action"] == "CloseList"
    assert nodes["关列表"]["next"] == ["主循环探尋"]
    # 刷新链任何一环失败都要回到重进，不能直接收尾
    for n in ("刷新闸门", "点刷新", "关列表"):
        assert nodes[n]["on_error"] == ["循环闸门"], n

    # §3.50：滑之前必须先过闸门，否则「滑→选行失败→再滑」这条最热路径
    # 完全不判划不划得动（11-15 14:47 实跑：连滑 4 次 icon_y 不变，
    # 一次闸门都没过，空转到 Ctrl+C）。
    for n in ("主循环选行", "点探尋并下翻"):
        assert nodes[n]["on_error"][0] == "滑动闸门", \
            "%s 的 on_error 第一候选必须是滑动闸门（§3.50）" % n
        # 第二候选必须是 `滑不动了`，不能是 `列表需要滑动`：
        # 否则闸门判「划不动」之后照样会滑一次，闸门白做
        assert nodes[n]["on_error"] == ["滑动闸门", "滑不动了"], n
        assert "列表需要滑动" not in nodes[n]["on_error"], n

    # §3.50：`异常收尾` 判失败（需求方 11-15 明确要求）
    end = nodes["异常收尾"]
    assert end["recognition"] == "DirectHit"
    assert end["action"] == "Custom"
    assert end["custom_action"] == "AbortTask"
    assert end.get("on_error") is None, "收尾节点不能再有 on_error"
    assert end.get("focus"), "失败收尾必须带 focus 提示"
    # 调试入口的正常终点仍然是绿的
    assert nodes["选行结束"]["action"] == "DoNothing"
    assert nodes["选行并点击"]["on_error"] == ["选行结束"]
    assert nodes["点探尋"]["on_error"] == ["探尋失败"]
    assert nodes["探尋失败"]["custom_action"] == "AbortTask"

    # §3.52：别人秘境里没格子 → 直接标记 + 退出，不再绕
    # `主循环无目标` → `他人无目标分支` 两步。
    assert nodes["主循环选格"]["on_error"] == ["他人空秘境退出", "主循环无目标"], \
        "别人秘境空 → 直接退出；自家秘境仍走主循环无目标（§3.52）"
    # **但真正的判断必须在 `主循环无目标.next`**：`主循环选格` 是
    # `读空闲鸡妖.next` 的候选节点，候选失败时它自己的 on_error 不生效
    # （§3.21）。11-15 17:07 实机 trace 证实 `他人空秘境退出` 全程未出现。
    assert nodes["主循环无目标"]["next"] == ["他人空秘境退出", "主循环探尋"], \
        "判断必须放在调用方 next 里，不能只放候选的 on_error（§3.21/§3.52）"
    empty = nodes["他人空秘境退出"]
    assert empty["recognition"] == "Custom"
    assert empty["custom_recognition"] == "RealmOtherGate"
    assert empty["custom_action"] == "MarkBadRow"
    # §3.54：回列表 = 点「探尋」，不再走 `退出他人秘境`（点面板空白处，6/6 失败）
    assert empty["next"] == ["点探尋并下翻"], empty["next"]
    assert "退出他人秘境" not in nodes, "`退出他人秘境` 节点已删除（§3.54）"
    # 掠夺成功 / 延迟 / 标记空秘境 三条回列表路径也都走「点探尋」
    for n in ("标记已访问", "延迟后分支", "标记空秘境"):
        assert nodes[n]["next"] == ["点探尋并下翻"], (n, nodes[n]["next"])

    # §3.53：`打开家园菜单` / `重进-开菜单` 的 max_hit 是**全局配额**，
    # 不能只有 3（长任务重进 4 次就用光，之后家園再也点不开、任务无声收尾）。
    for n in ("打开家园菜单", "重进-开菜单"):
        hit = nodes[n].get("max_hit")
        assert hit is None or hit >= 15, \
            "%s 的 max_hit=%s 太小，长任务必然用光（§3.53）" % (n, hit)
    assert nodes["冷却闸门"]["custom_recognition"] == "CooldownGate"
    assert nodes["等待冷却"]["custom_action"] == "WaitCooldown"

    # §3.44：Maa 的 `timeout` 是**候选扫描窗口**，不是动作超时；候选自己的
    # timeout 不生效。凡是 next 里有「可能失败」的候选、又不写 timeout 的节点，
    # 一次没命中就白等默认的 20000ms。11-14 实机因此每滑一页多花 20s。
    # 这四个节点必须显式给短 timeout，别再退化回默认值。
    for node_name, limit in (("列表需要滑动", 5000),
                             ("打开家园菜单", 5000),
                             ("重进-开菜单", 5000),
                             ("等待获得弹窗", 5000)):
        got = nodes[node_name].get("timeout")
        assert got is not None, "%s 缺 timeout，会白等 20s（§3.44）" % node_name
        assert got <= limit, "%s timeout=%s 太长（§3.44）" % (node_name, got)

    # §3.45：Maa 原生 Click 在识别框里随机取点，`nav_home.png` 的 100x100
    # 框里图标只占 5%，不加 target_offset 会有约 18% 点到图标外。
    # `max_hit` 也必须靠「节点把自己放进 next」才生效。
    for node_name in ("打开家园菜单", "重进-开菜单", "进入家园"):
        assert nodes[node_name].get("target_offset"), \
            "%s 缺 target_offset，点击会随机落到图标外（§3.45）" % node_name
    # §3.53：`打开家园菜单` / `重进-开菜单` 都**不再自引用**。
    # 原因：自引用必须配 `max_hit` 才有刹车，而 `max_hit` 是整趟任务共享的
    # 全局配额 —— 长任务里必然用光，之后家園再也点不开、任务无声收尾。
    # 只去 max_hit 而保留自引用 = 死循环（11-15 实测连点 20+ 次停不下来）。
    # 现在重试由 `重进开菜单闸门`（每轮配额）负责。
    for n in ("打开家园菜单", "重进-开菜单"):
        assert n not in (nodes[n].get("next") or []), \
            "%s 不能再自引用（§3.53 的死循环）" % n
        assert nodes[n].get("max_hit") is None, \
            "%s 不该再有 max_hit（全局配额，长任务必然用光）" % n
    assert nodes["打开家园菜单"]["next"] == ["进入家园"]
    assert nodes["打开家园菜单"]["on_error"] == ["开菜单闸门", "重进失败"]
    assert nodes["开菜单闸门"]["custom_recognition"] == "ReopenGate"
    assert nodes["开菜单闸门"]["next"] == ["计数并重开菜单"]
    assert nodes["计数并重开菜单"]["custom_action"] == "CountReopen"
    assert nodes["计数并重开菜单"]["next"] == ["打开家园菜单"]
    assert nodes["重进-开菜单"]["on_error"] == ["重进开菜单闸门", "重进失败"]

    # §3.47：重进秘境失败后不再直接收尾，先过 `重进失败重试门`
    # （每轮点蓝点 + 等 3s，不做任何 OCR 校验 —— 需求方 11-14 19:13）。
    # 11-15 改名两次：`网络恢复门`（名字像有网络检测，其实没有）
    # → `重进重试门` → `重进失败重试门`（需求方指定，明确它接的是
    # `重进失败` 这条边）。旧的两个名字都必须不存在。
    assert nodes["重进失败"]["next"] == ["重进失败重试门"]
    gate = nodes["重进失败重试门"]
    assert "网络恢复门" not in nodes, "旧名字 `网络恢复门` 应该已经改掉"
    assert "重进重试门" not in nodes, "旧名字 `重进重试门` 应该已经改掉"
    # 必须是 DirectHit + Custom action：RecoverNet 只有 run()，没有 analyze()，
    # 写成 custom_recognition 会让 Maa 报 `recognition is null`，节点永远失败
    # （11-14 修的就是这个，恢复门此前从未真正执行过）。
    assert gate["recognition"] == "DirectHit"
    assert gate["action"] == "Custom"
    assert gate["custom_action"] == "RecoverNet"
    assert "custom_recognition" not in gate
    assert gate.get("max_hit") is None, \
        "轮次由 RecoverNet 自己数（max_rounds），不该再用 max_hit"
    assert "重进失败重试门" not in (gate.get("next") or []), \
        "轮次在动作内部循环，next 里不该再有自己"
    assert gate["next"] == ["重进-找菜单"]
    assert gate["on_error"] == ["异常收尾"]
    assert gate["custom_action_param"]["max_rounds"] == 8
    assert nodes["主任务"]["custom_action_param"]["重进失败"] == ["重进失败重试门"]
    assert loop_ctl.RecoverNet.WAIT == 3.0, \
        "§3.47 的每轮等待改成别的值要同步 HANDOVER 和 main.json 注释"
    assert loop_ctl.RecoverNet.MAX_ROUNDS == 8

    # §3.48：带 max_hit 的节点不能是别人 next/on_error 的**末位**候选。
    # 配额用光后 Maa 静默跳过它（不报错、不走它自己的 on_error），
    # 调用方没别的候选就 `invalid node id, handle error` 直接收尾。
    # 11-15 11:53 的实跑就是这么无声退出的：`回主界面.on_error` 只有
    # `重进-开菜单`，而它已经在 209/275/389s 命中过 3 次。
    quota = {n for n, d in nodes.items()
             if isinstance(d, dict) and "max_hit" in d}
    for name, node in nodes.items():
        for field in ("next", "on_error"):
            refs = node.get(field) or []
            if isinstance(refs, str):
                refs = [refs]
            if not refs:
                continue
            last = refs[-1]
            if isinstance(last, dict):
                last = last.get("name")
            # 自引用是 §3.45 的重试写法，不算风险（耗尽后走自己的 on_error）
            assert not (last in quota and last != name), \
                "%s.%s 末位是 max_hit 节点 %s，配额用光会静默硬退（§3.48）" \
                % (name, field, last)
    # 这两个出口必须各有兜底
    assert nodes["进入淫乱秘境"]["on_error"] == ["打开家园菜单", "重进失败"]
    # §3.53：先过 `重进开菜单闸门`（每轮配额），再点家園
    assert nodes["回主界面"]["on_error"] == ["重进开菜单闸门", "重进失败"]
    gate_reopen = nodes["重进开菜单闸门"]
    assert gate_reopen["recognition"] == "Custom"
    assert gate_reopen["custom_recognition"] == "ReopenGate"
    assert gate_reopen["next"] == ["计数并开菜单"]
    assert nodes["计数并开菜单"]["custom_action"] == "CountReopen"
    assert nodes["计数并开菜单"]["next"] == ["重进-开菜单"]

    # 滑动量 = 首行顶 → 末行底（完整行跨度），不是旧的末行顶-首行顶。
    img_a = cv2.imread("User_screenshot/full/720_list_top.png")
    assert img_a is not None
    dy_a, rows_a = loop_ctl.page_dy(img_a, [40, 357, 640, 822])
    assert (dy_a, rows_a) == (448, [488, 640, 791])
    img_b = cv2.imread("User_screenshot/full/720_fullscreen_activity.png")
    assert img_b is not None
    dy_b, rows_b = loop_ctl.page_dy(img_b, [40, 357, 640, 822])
    assert (dy_b, rows_b) == (599, [431, 582, 734, 885])

    loop_ctl._STATE["need_wait"] = False
    assert loop_ctl.MarkWait().run(None, SimpleNamespace(custom_action_param="{}"))
    gate = loop_ctl.WaitReturnGate().analyze(None, SimpleNamespace(
        custom_recognition_param="{}"))
    assert gate.box is not None

    ctrl = _Ctrl(popup_after=2)
    assert loop_ctl.WaitPopup().run(_ctx(ctrl), _run_arg(2))
    ctrl = _Ctrl(popup_after=None)
    assert not loop_ctl.WaitPopup().run(_ctx(ctrl), _run_arg(0.2))

    # 滑动期间 popup_clicks 增加时，SwipeGate 不能把“没动”判成到底，
    # 应允许下一轮清弹窗后重滑。
    img = np.zeros((1560, 720, 3), dtype=np.uint8)
    roi = SimpleNamespace(x=40, y=357, w=640, h=822)
    loop_ctl.remember(img, [40, 357, 640, 822])
    loop_ctl._STATE.update({"swipes": 1, "stuck": False, "popup_at_swipe": 0})
    gate = loop_ctl.SwipeGate().analyze(
        SimpleNamespace(tasker=SimpleNamespace(
            controller=SimpleNamespace(popup_clicks=1))),
        SimpleNamespace(image=img, roi=roi, custom_recognition_param="{}"))
    assert gate.box is not None
    assert gate.detail["popup_event"] is True

    # 退到主页面：BackToHome 只负责清掉挡点击的弹窗；
    # 回主页面由 `回主界面` 点底部「冒险」tab 完成。
    clicks = []

    class _HomeCtrl:
        def post_click(self, x, y):
            clicks.append((x, y))
            return _Job(None)

        def post_screencap(self):
            return _Job(np.zeros((1560, 720, 3), dtype=np.uint8))

    state = {"n": 0}

    def fake_popup(*_a, **_k):
        state["n"] += 1
        return (state["n"] == 1, 0.99 if state["n"] == 1 else 0.10)

    real_popup = loop_ctl._popup_match
    real_match = loop_ctl._match_template
    real_rows = loop_ctl._list_rows
    real_sleep = loop_ctl.time.sleep
    loop_ctl._popup_match = fake_popup
    loop_ctl._match_template = lambda *_a, **_k: (False, 0.10)
    loop_ctl._list_rows = lambda *_a, **_k: []
    loop_ctl.time.sleep = lambda *_a, **_k: None
    try:
        assert loop_ctl.BackToHome().run(
            _ctx(_HomeCtrl()), SimpleNamespace(custom_action_param="{}"))
    finally:
        loop_ctl._popup_match = real_popup
        loop_ctl._match_template = real_match
        loop_ctl._list_rows = real_rows
        loop_ctl.time.sleep = real_sleep
    assert clicks == [(350, 1326)]

    print("Wait loop tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
