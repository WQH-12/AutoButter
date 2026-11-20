"""全局「恭喜獲得」弹窗护栏（CLI 用）。

需求方 11-11 补充：弹窗不是只在进秘境时出现，只要之前采集的资源完成，
任何一步都可能弹。因此不能只在 `进入家园` 后清一次，而要在**每一次截图**
之前检查并清掉弹窗，清完再让原来的节点继续。

实现方式：包一层 `CustomController`，把真正的 AdbController 藏在里面。
Maa 每次调用 `screencap()`，本类先截当前帧；如果识别到
`obtain_resource.png`，就点弹窗面板外部，再截一张，直到没有弹窗为止，
最后才把「没有弹窗的截图」返回给 Maa。这样原生 TemplateMatch / OCR /
Custom 节点都会自动经过这一步。

注意：这是 CLI 路径（`run_main_trace.py` / `run_task.py`）的全局护栏。
MFAAvalonia 走自己的 Adb controller，不能注入这层 Python 包装；GUI 路径
要么用 CLI 跑，要么后续在 pipeline 层给每个节点套弹窗门。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

from maa.controller import AdbController, CustomController

ROOT = Path(__file__).resolve().parent.parent

# 与 pipeline 的 `关闭获得弹窗` 保持一致
POPUP_ROI = (0, 400, 720, 400)      # x, y, w, h（720 画布）
POPUP_THRESHOLD = 0.9
POPUP_OFFSET_Y = 775                # 标题框中心 → 面板外 y≈1326
POPUP_POST_DELAY = 0.25
MAX_POPUPS = 50


def _resolve_img_dir() -> str:
    candidates = [
        ROOT / "resource" / "image",
    ]
    for p in candidates:
        if p.is_dir():
            return str(p)
    return str(candidates[0])


def _imread_rgb(path: str) -> np.ndarray | None:
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return img


class PopupGuardController(CustomController):
    """代理 AdbController，并在每次 screencap 前清弹窗。"""

    def __init__(self, adb_path: str, serial: str, short_side: int = 720,
                 inner: AdbController | None = None):
        super().__init__()
        self.inner = inner or AdbController(adb_path, serial)
        self.inner.set_screenshot_target_short_side(short_side)
        self._connected = False
        self.popup_clicks = 0
        self.last_popup_score = 0.0
        tpl_path = os.path.join(_resolve_img_dir(), "obtain_resource.png")
        self.tpl = _imread_rgb(tpl_path)
        if self.tpl is None:
            raise FileNotFoundError(tpl_path)

    # --- Controller options ---
    def set_screenshot_target_short_side(self, short_side: int) -> bool:
        return self.inner.set_screenshot_target_short_side(short_side)

    @property
    def resolution(self):
        return self.inner.resolution

    @property
    def cached_image(self):
        return self.inner.cached_image

    @property
    def uuid(self) -> str:
        try:
            return self.inner.uuid
        except Exception:  # noqa: BLE001
            return "popup-guard"

    def get_features(self) -> int:
        # 0 = 普通 click / swipe 回调；本类没有额外的 mouse/key 语义
        return 0

    def get_custom_info(self) -> dict:
        return {
            "type": "popup_guard",
            "popup_clicks": self.popup_clicks,
            "last_popup_score": round(self.last_popup_score, 4),
        }

    # --- connection / app ---
    def connect(self) -> bool:
        self._connected = bool(self.inner.post_connection().wait().succeeded)
        return self._connected

    def connected(self) -> bool:
        return self._connected

    def request_uuid(self) -> str:
        return self.uuid

    def start_app(self, intent: str) -> bool:
        return bool(self.inner.post_start_app(intent).wait().succeeded)

    def stop_app(self, intent: str) -> bool:
        return bool(self.inner.post_stop_app(intent).wait().succeeded)

    # --- 核心：每次截图前清弹窗 ---
    def _find_popup(self, img: np.ndarray):
        h, w = img.shape[:2]
        if w != 720 or h != 1560:
            # 正常会走 720x1560；尺寸不对时缩放检测，点击坐标仍按 720 画布
            img720 = cv2.resize(img, (720, 1560), interpolation=cv2.INTER_AREA)
        else:
            img720 = img
        x, y, rw, rh = POPUP_ROI
        roi = img720[y:y + rh, x:x + rw]
        th, tw = self.tpl.shape[:2]
        if roi.shape[0] < th or roi.shape[1] < tw:
            return None, 0.0
        res = cv2.matchTemplate(roi, self.tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < POPUP_THRESHOLD:
            return None, float(max_val)
        return (x + max_loc[0], y + max_loc[1], tw, th), float(max_val)

    def screencap(self) -> np.ndarray:
        img = self.inner.post_screencap().wait().get()
        for _ in range(MAX_POPUPS):
            box, score = self._find_popup(img)
            if box is None:
                self.last_popup_score = score
                break
            bx, by, bw, bh = box
            cx = bx + bw // 2
            cy = by + bh // 2 + POPUP_OFFSET_Y
            ok = bool(self.inner.post_click(cx, cy).wait().succeeded)
            self.popup_clicks += 1
            self.last_popup_score = score
            print("[popup-guard] popup #%d score=%.3f click=(%d,%d) ok=%s"
                  % (self.popup_clicks, score, cx, cy, ok), flush=True)
            if not ok:
                break
            time.sleep(POPUP_POST_DELAY)
            img = self.inner.post_screencap().wait().get()
        return img

    # --- 普通输入，全部代理给内层 AdbController ---
    def click(self, x: int, y: int) -> bool:
        return bool(self.inner.post_click(x, y).wait().succeeded)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int) -> bool:
        return bool(self.inner.post_swipe(x1, y1, x2, y2, duration).wait().succeeded)

    def touch_down(self, contact: int, x: int, y: int, pressure: int) -> bool:
        return bool(self.inner.post_touch_down(x, y, contact, pressure).wait().succeeded)

    def touch_move(self, contact: int, x: int, y: int, pressure: int) -> bool:
        return bool(self.inner.post_touch_move(x, y, contact, pressure).wait().succeeded)

    def touch_up(self, contact: int) -> bool:
        return bool(self.inner.post_touch_up(contact).wait().succeeded)

    def click_key(self, keycode: int) -> bool:
        return bool(self.inner.post_click_key(keycode).wait().succeeded)

    def input_text(self, text: str) -> bool:
        return bool(self.inner.post_input_text(text).wait().succeeded)

    def key_down(self, keycode: int) -> bool:
        return bool(self.inner.post_key_down(keycode).wait().succeeded)

    def key_up(self, keycode: int) -> bool:
        return bool(self.inner.post_key_up(keycode).wait().succeeded)

    def scroll(self, dx: int, dy: int) -> bool:
        return bool(self.inner.post_scroll(dx, dy).wait().succeeded)

    def relative_move(self, dx: int, dy: int) -> bool:
        return bool(self.inner.post_relative_move(dx, dy).wait().succeeded)

    def shell(self, cmd: str, timeout: int):
        job = self.inner.post_shell(cmd, timeout).wait()
        if not job.succeeded:
            return None
        return job.get()

    def inactive(self) -> bool:
        return bool(self.inner.post_inactive().wait().succeeded)

