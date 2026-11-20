"""运行时布局校准：按当前设备截图测量一次，再改写 pipeline 的 ROI/落点。

Maa 会把截图缩到 720 短边，但不同设备的长边可能是 1560/1600/...，
所以面板、底部导航、探尋、弹窗的 y 不能再用旧设备的绝对值。

校准锚点：
  - `btn_manage.png`：秘境面板中下方「管理」按钮，任意秘境都有；
  - `nav_home.png` / `nav_adventure.png`：底部导航，用于确定 nav_y。

测量一次后缓存，并通过 Context.override_pipeline 改写当前任务的节点。
"""
from __future__ import annotations

import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_LAYOUT = {}
_TEMPLATES = {}

# 布局推导的基线画布：720x1560（旧设备实测）。
#
# 所有写成绝对像素的 **y 偏移和高度** 都按 s = h / BASE_H 缩放。依据：
# HANDOVER §3.43 记录的新设备（720x1600）实测 `btn_manage @(298,1065)`，
# 而 1040 * 1600/1560 = 1066.7 —— 说明 UI 的 y 随长边等比缩放，
# 不是贴底对齐（贴底会是 1040 + 40 = 1080）。
#
# 不缩放的话，`count_roi = [mx+90, my-260, 240, 100]` 这类窗口在
# 1680 以上的长边会把 `雞妖数量 n/3` 那行文字裁到 ROI 外面，
# 导致 `读派遣数量` 永远读不到（见 tools/test_layout_scale.py）。
BASE_H = 1560


def _sy(value, s):
    """把基线（720x1560）上的 y 偏移/高度换算到当前画布。"""
    return int(round(value * s))


def _img_dir():
    for p in (os.path.join(ROOT, "resource", "image"),):
        if os.path.isdir(p):
            return p
    return ""


def _template(name):
    if name not in _TEMPLATES:
        p = os.path.join(_img_dir(), name)
        _TEMPLATES[name] = cv2.imread(p) if os.path.exists(p) else None
    return _TEMPLATES[name]


def _match(img, name, roi, threshold):
    tpl = _template(name)
    if tpl is None or img is None:
        return None, 0.0
    x, y, w, h = [int(v) for v in roi]
    sub = img[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
    if sub.shape[0] < tpl.shape[0] or sub.shape[1] < tpl.shape[1]:
        return None, 0.0
    r = cv2.matchTemplate(sub, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, ml = cv2.minMaxLoc(r)
    if mx < threshold:
        return None, round(float(mx), 3)
    return (x + ml[0], y + ml[1], tpl.shape[1], tpl.shape[0]), round(float(mx), 3)


def reset():
    _LAYOUT.clear()


def ready():
    return bool(_LAYOUT)


def get(name, default=None):
    return _LAYOUT.get(name, default)


def safe_point(default=(350, 1326)):
    return tuple(_LAYOUT.get("safe_point") or default)


def exit_point(default=(674, 256)):
    """退出秘境用的点击点：右上角固定 HUD 锚点（`exit_anchor.png`）中心。

    实测该锚点在秘境面板 / 别人秘境 / 列表 / 活动页 / 派遣浮窗上
    都命中 0.987~0.988，位置恒为 (637,221) 74x70，中心 (674,256)。
    点它不会触发格子、「前往」或其它操作；比屏幕中右空白区更靠上，
    不会压到资源格。"""
    return tuple(_LAYOUT.get("exit_point") or default)


def exit_anchor_roi(default=(500, 120, 220, 220)):
    return list(_LAYOUT.get("exit_anchor_roi") or default)


def manage_roi(default=(250, 950, 250, 250)):
    return list(_LAYOUT.get("manage_roi") or default)


def detail_roi(default=(200, 930, 320, 160)):
    return list(_LAYOUT.get("detail_roi") or default)


def popup_roi(default=(0, 400, 720, 400)):
    return list(_LAYOUT.get("popup_roi") or default)


def tanxun_roi(default=(440, 1250, 280, 90)):
    return list(_LAYOUT.get("tanxun_roi") or default)


def swipe_from_y(default=1000):
    return int(_LAYOUT.get("swipe_from_y") or default)


def _pipeline_overrides(lay):
    nav = lay["nav_roi"]
    menu = lay["menu_roi"]
    popup = lay["popup_roi"]
    full = lay["full_roi"]
    # 「恭喜獲得」标题中心 → 面板外安全点的竖直偏移，基线上是 775px
    # （§3.27：标题中心 y=551，551+775=1326）。同样要按长边缩放，
    # 否则高屏上会点到标题本身或点到面板里的按钮。
    s = lay["screen"][1] / BASE_H
    dismiss_dy = _sy(775, s)
    return {
        "进入家园": {"roi": menu},
        "打开家园菜单": {"roi": nav},
        "关闭获得弹窗": {"roi": popup, "target_offset": [0, dismiss_dy, 0, 0]},
        "清弹窗后继续": {"roi": popup, "target_offset": [0, dismiss_dy, 0, 0]},
        "确认已进入秘境": {"roi": lay["confirm_roi"]},
        "读空闲鸡妖": {
            "roi": lay["idle_roi"],
            "custom_recognition_param": {"total_roi": lay["total_roi"]},
        },
        "选格并点击": {"roi": lay["content_roi"]},
        "确认资源详情": {"roi": lay["detail_roi"]},
        "派遣并掠奪": {"roi": lay["detail_roi"]},
        "延迟该资源": {"roi": full},
        "读派遣数量": {"roi": lay["count_roi"]},
        "加一只鸡妖": {"roi": lay["plus_roi"]},
        "掠奪": {"roi": lay["detail_roi"]},
        "确认他人秘境": {"roi": lay["confirm_roi"]},
        "他人秘境选格": {"roi": lay["content_roi"]},
        "点探尋": {"roi": lay["tanxun_roi"]},
        "主循环探尋": {"roi": lay["tanxun_roi"]},
        "点探尋并下翻": {"roi": lay["tanxun_roi"]},
        "列表需要滑动": {
            "roi": lay["list_roi"],
            "custom_action_param": {"roi": lay["list_roi"], "duration": 400},
        },
        "滑动闸门": {
            "roi": lay["list_roi"],
            "custom_recognition_param": {"max_swipes": 8, "threshold": 4.0},
        },
        "刷新闸门": {"roi": lay["refresh_roi"]},
        "回主界面": {"roi": nav},
        "重进-找菜单": {"roi": menu},
        "重进-开菜单": {"roi": nav},
        "重进-确认": {"roi": lay["confirm_roi"]},
        "等待获得弹窗": {"roi": full},
        "主循环选格": {"roi": lay["content_roi"]},
    }


def ensure(context, image=None):
    """测量一次并改写当前任务 pipeline；成功/已有缓存返回 True。"""
    if _LAYOUT:
        return True
    if image is None:
        try:
            image = context.tasker.controller.post_screencap().wait().get()
        except Exception:  # noqa: BLE001
            image = None
    if image is None or not hasattr(image, "shape"):
        return False

    h, w = int(image.shape[0]), int(image.shape[1])
    manage, manage_score = _match(
        image, "btn_manage.png", [0, int(0.42 * h), w, int(0.50 * h)], 0.90)
    nav, nav_score = _match(
        image, "nav_home.png", [0, max(0, h - 200), w, min(200, h)], 0.90)
    anchor, anchor_score = _match(
        image, "exit_anchor.png",
        [int(0.60 * w), 0, int(0.40 * w), int(0.30 * h)], 0.90)

    if manage:
        mx, my, mw, mh = manage
    else:
        mx, my, mw, mh = int(0.414 * w), int(0.667 * h), 120, 121
    nav_y = nav[1] if nav else h - 135

    # 所有偏移以旧设备实测的 manage=(298,1040)、nav_y=1427 为基线。
    # y 方向一律按 s = h/BASE_H 缩放（依据见 BASE_H 注释）；x 方向不动，
    # 因为短边恒为 720。
    s = h / BASE_H
    idle = [mx - 85, my + _sy(44, s), 120, _sy(80, s)]
    total = [mx + 133, my + _sy(44, s), 120, _sy(80, s)]
    content = [0, my - _sy(790, s), w, _sy(810, s)]
    confirm = [mx - 50, my - _sy(90, s), 260, _sy(260, s)]
    detail = [0, max(0, my - _sy(160, s)), w, _sy(300, s)]
    count = [mx + 90, my - _sy(260, s), 240, _sy(100, s)]
    plus = [mx + 170, my - _sy(250, s), 260, _sy(220, s)]
    tanxun = [mx + 142, my + _sy(210, s), 280, _sy(90, s)]
    nav_roi = [0, max(0, nav_y - _sy(30, s)), w,
               min(_sy(200, s), h - max(0, nav_y - _sy(30, s)))]
    menu = [0, max(0, nav_y - _sy(427, s)), 400, _sy(400, s)]
    list_top = max(0, nav_y - _sy(1070, s))
    list_h = max(120, min(_sy(822, s), nav_y - _sy(248, s) - list_top))
    list_roi = [40, list_top, max(120, w - 80), list_h]
    # 探寻列表底部的「刷新」按钮（720 基线实测 @(265,1101,210,56)）。
    # 贴着 list_roi 下沿往上 110px，同样按长边缩放。
    refresh = [int(w * 0.30),
               max(0, list_top + list_h - _sy(110, s)),
               int(w * 0.42), _sy(110, s)]
    popup = [0, max(0, my - _sy(640, s)), w, _sy(400, s)]
    safe = (mx + 52, my + _sy(286, s))
    swipe_y = nav_y - _sy(427, s)
    if anchor:
        ax, ay, aw, ah = anchor
        exit_pt = (ax + aw // 2, ay + ah // 2)
        anchor_roi = [ax - 80, ay - 60, 220, 220]
    else:
        exit_pt = (int(0.936 * w), int(0.164 * h))
        anchor_roi = [int(0.60 * w), 0, int(0.40 * w), int(0.30 * h)]

    lay = {
        "screen": [w, h],
        "manage": [mx, my, mw, mh],
        "manage_score": manage_score,
        "nav_y": nav_y,
        "nav_score": nav_score,
        "idle_roi": idle,
        "total_roi": total,
        "content_roi": content,
        "confirm_roi": confirm,
        "detail_roi": detail,
        "count_roi": count,
        "plus_roi": plus,
        "tanxun_roi": tanxun,
        "nav_roi": nav_roi,
        "menu_roi": menu,
        "list_roi": list_roi,
        "refresh_roi": refresh,
        "popup_roi": popup,
        "safe_point": safe,
        # 右上角固定 HUD 锚点（exit_anchor.png）中心；实测跨页面恒定。
        "exit_point": exit_pt,
        "exit_anchor_roi": anchor_roi,
        "exit_anchor_score": anchor_score,
        "swipe_from_y": swipe_y,
        "full_roi": [0, 0, w, h],
    }
    _LAYOUT.update(lay)
    try:
        ok = context.override_pipeline(_pipeline_overrides(lay))
        print("[Layout] measured screen=%dx%d scale=%.3f manage=%s(%.3f) "
              "nav_y=%d(%.3f) exit=%s(%.3f) safe=%s override=%s"
              % (w, h, s, manage, manage_score, nav_y, nav_score,
                 exit_pt, anchor_score, safe, ok),
              flush=True)
        if not ok:
            print("[Layout] warning: override_pipeline returned False", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("[Layout] warning: override_pipeline failed: %s" % exc, flush=True)
    return True
