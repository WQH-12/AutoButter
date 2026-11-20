"""GUI 路径的「恭喜獲得」弹窗护栏。

CLI 走 `tools/cli_popup_guard.py` 的 `PopupGuardController`：每次 screencap
之前先检查并清掉弹窗，所以弹窗刚出来就会被点掉。

MFAAvalonia 不能注入 Python controller，pipeline 里的 `清弹窗后继续`
又只是候选节点：候选失败时 Maa 只试一次就落到下一个候选，弹窗若还在
入场动画里（模板分只有 0.2~0.3）就会被漏过去。18:52 那轮就是这样：
`清弹窗后继续` 在 18:51:50 只拿到 0.261，随后 `读空闲鸡妖` 白跑 23s，
`主循环选格` 被弹窗挡住，任务判失败。

本模块只做一件事：**在自定义识别拿到截图后，先把弹窗清干净，再识别**。
它不额外开截图流：没弹窗时只做一次模板匹配（约 6~20ms），命中时才
点掉、等 0.25s、重新截图，最多 50 条，和 CLI 的护栏一致。
"""
from __future__ import annotations

import os
import time

try:
    import cv2
except ImportError:  # Android 的 agent core 不带 opencv，用 numpy/Pillow 子集
    import cv2lite as cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

POPUP_ROI = (0, 400, 720, 400)
POPUP_THRESHOLD = 0.9
MAX_POPUPS = 50
POST_DELAY = 0.25

_TPL = None
_WARNED = False


def _resolve_img_dir() -> str:
    for p in (os.path.join(ROOT, "resource", "image"),):
        if os.path.isdir(p):
            return p
    return ""


def _template():
    global _TPL
    if _TPL is None:
        p = os.path.join(_resolve_img_dir(), "obtain_resource.png")
        if os.path.exists(p):
            _TPL = cv2.imdecode(np.fromfile(p, dtype=np.uint8),
                                cv2.IMREAD_COLOR)
    return _TPL


def _roi():
    try:
        import layout
        return [int(v) for v in layout.popup_roi(POPUP_ROI)]
    except Exception:  # noqa: BLE001
        return list(POPUP_ROI)


def find_popup(img):
    """返回 (box, score)；box 是图像坐标 [x, y, w, h]，未命中为 None。"""
    tpl = _template()
    if tpl is None or img is None or not hasattr(img, "shape"):
        return None, 0.0
    x, y, w, h = _roi()
    roi = img[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
    if (roi.shape[0] < tpl.shape[0] or roi.shape[1] < tpl.shape[1]):
        return None, 0.0
    r = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, ml = cv2.minMaxLoc(r)
    score = round(float(mx), 3)
    if mx < POPUP_THRESHOLD:
        return None, score
    return [x + ml[0], y + ml[1], tpl.shape[1], tpl.shape[0]], score


def _dismiss_point():
    """弹窗面板外的安全点；默认 (350, 1326)，布局校准后按长边缩放。"""
    try:
        import layout
        return layout.safe_point((350, 1326))
    except Exception:  # noqa: BLE001
        return (350, 1326)


def clear_popups(context, image=None, ctrl=None, max_popups=MAX_POPUPS,
                 post_delay=POST_DELAY):
    """清掉当前屏幕上的「恭喜獲得」弹窗，返回干净的截图。

    - `image` 给了就先查这张；没给就现截一张。
    - 没弹窗时原样返回，不产生额外截图。
    - 控件不可用（离线测试 / 没有 controller）时安静地返回原图。

    ⚠️ **调用方如果已经持有 controller，必须通过 `ctrl=` 传进来。**
    Maa 5.8.0 的 agent 模式下 `context.tasker.controller` 每次访问都会
    重建底层 `RemoteController`，旧指针立即失效；本函数若自己再取一次，
    调用方手里的旧 controller 就会在 C++ 侧 access violation（§3.66）。
    """
    global _WARNED
    if ctrl is None:
        try:
            ctrl = context.tasker.controller
        except Exception as exc:  # noqa: BLE001
            if not _WARNED:
                print("[PopupGuard] controller unavailable: %s" % exc,
                      flush=True)
                _WARNED = True
            return image

    img = image
    if img is None:
        try:
            img = ctrl.post_screencap().wait().get()
        except Exception as exc:  # noqa: BLE001
            if not _WARNED:
                print("[PopupGuard] screencap failed: %s" % exc, flush=True)
                _WARNED = True
            return image

    px, py = _dismiss_point()
    for n in range(1, max_popups + 1):
        box, score = find_popup(img)
        if box is None:
            return img
        try:
            ok = bool(ctrl.post_click(px, py).wait().succeeded)
        except Exception as exc:  # noqa: BLE001
            print("[PopupGuard] click failed: %s" % exc, flush=True)
            return img
        print("[PopupGuard] popup #%d score=%.3f click=(%d,%d) ok=%s"
              % (n, score, px, py, ok), flush=True)
        if not ok:
            return img
        time.sleep(post_delay)
        try:
            img = ctrl.post_screencap().wait().get()
        except Exception as exc:  # noqa: BLE001
            print("[PopupGuard] re-screencap failed: %s" % exc, flush=True)
            return img
    print("[PopupGuard] %d popups cleared, still present" % max_popups,
          flush=True)
    return img
