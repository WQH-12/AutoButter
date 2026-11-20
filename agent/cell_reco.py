"""秘境格子识别 + 按物品优先级选目标格（自定义识别）。

规则（用户指定）：
  1. 先检测鸡妖，识别到的鸡妖 x 轴区间内、整个 y 轴的格子都算「正在被掠夺」，跳过。
  2. 其余格子里按优先级挑第一个命中的物品。

为什么不用原生 TemplateMatch：
  原生节点只能返回一个框，无法「先排除若干列再排序」。而且 7 个物品要 7 个节点、
  无法在一次识别里比较优先级。
"""
from __future__ import annotations

import json
import os
import re
import sys

import cv2
import numpy as np

import priority as priority_cfg
from popup_guard import clear_popups

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _resolve_img_dir():
    """找模板目录。

    工作区与打包目录都是 <root>/resource/image（agent/ 与 resource/ 同级）。
    """
    candidates = [
        os.path.join(ROOT, "resource", "image"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[0]


IMG_DIR = _resolve_img_dir()

# 720 画布上的秘境内容区。y 上限 1060 是实测边界：
#   - 格子最高到 y≈967
#   - 格子下方的正面鸡妖最高到 y≈1031
#   - 「管理」按钮上的鸡妖在 y≈1097，必须排除，否则会被当成掠夺中的鸡妖
DEFAULT_ROI = [0, 250, 720, 810]

# 物品模板 -> 内部短名（短名只是键，不是中文名翻译）
ITEM_FILES = {
    "alcohol": "item_alcohol.png",   # 酒米
    "berry": "item_berry.png",       # 枸杞
    "fabric": "item_fabric.png",     # 無字密卷
    "honey": "item_honey.png",       # 蜂蜜
    "jade": "item_jade.png",         # 玉璧
    "silver": "item_silver.png",     # 餉銀
    "stone": "item_stone.png",       # 雲英
}
# 兼容旧调用；实际排序由 priority.py 的 0..7 等级计算。
DEFAULT_PRIORITY = priority_cfg.order()
ITEM_THRESHOLD = 0.88

# 倒计时（`00:38` / `23:10`）的定位参数。
#
# 为什么用「白色数字」当锚点而不是「黑色底条」：黑条经常和鸡妖的身体/扬尘
# 连成一片（实测 user_test_1 的 23:10 黑条与鸡身合并成 84x162 的连通块），
# 而数字是白字，与周围的深色底对比极强，连通域稳定是 8x14 的字形。
#
# 实测（720x1560）：4 个数字 + 冒号共 4~5 个 blob，横向跨度 46~69，
# 纵向 13~21；`V>200 & S<60` 能在白字上取到 148~288 个像素。
TIMER_V_BRIGHT = 200
TIMER_S_MAX = 60
TIMER_BLOB_MIN_W = 3
TIMER_BLOB_MAX_W = 26
TIMER_BLOB_MIN_H = 8
TIMER_BLOB_MAX_H = 30
TIMER_MAX_GAP = 20        # 相邻字形之间允许的最大水平空隙（含冒号）
TIMER_MAX_DY = 10         # 同一行数字的垂直抖动
TIMER_MIN_DIGITS = 3      # 至少 3 个字形才算倒计时
TIMER_MIN_SPAN_W = 40
TIMER_MAX_SPAN_W = 175
TIMER_MIN_SPAN_H = 12
TIMER_MAX_SPAN_H = 32
TIMER_MIN_ASPECT = 2.0    # 横向长条；主界面角色腰带 65x39（1.67）要排除
# 数字周围一圈必须够黑（否则是白底文字）。实测被鸡妖扬尘半遮的 `23:10`
# 只有 0.35，取 0.30 留一点余量；0.25 以下开始把深色 UI 描边算进来。
TIMER_MIN_DARK = 0.30

# 倒计时相对格子的位置：
#   below = 自己的鸡在格子下方正面采集（最常见）
#   above = 鸡在格子上方背面采集 / 别人在抢
# 两者都表示「这一格正在被采集」，都算 busy。
TIMER_BELOW_GAP = (-8, 58)
TIMER_ABOVE_GAP = (-190, -2)


def _imread(path):
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def detect_cells(img, top_ratio=0.13, bot_ratio=0.62):
    """用格子的高饱和描边定位物品格。返回 [(x, y, w, h)]，按行再按列排序。

    与 tools/burst.py 的 detect_cells 同一套参数；这里内联一份，
    避免 agent 进程依赖 tools/。
    """
    h, w = img.shape[:2]
    top, bot = int(h * top_ratio), int(h * bot_ratio)
    roi = img[top:bot]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    mask = ((S > 90) & (V > 90)).astype(np.uint8) * 255
    purple = ((H >= 110) & (H <= 170) & (S > 45) & (V > 80)).astype(np.uint8) * 255
    blue = ((H >= 92) & (H <= 112) & (S > 55) & (V > 150)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(cv2.bitwise_or(mask, purple), blue)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if not (70 <= cw <= 115 and 70 <= ch <= 115):
            continue
        if not (0.7 <= cw / ch <= 1.45):
            continue
        if cv2.contourArea(c) < cw * ch * 0.40:
            continue
        boxes.append((x, y + top, cw, ch))
    boxes.sort(key=lambda b: -b[2] * b[3])
    kept = []
    for b in boxes:
        if all(abs(b[0] - k[0]) > 30 or abs(b[1] - k[1]) > 30 for k in kept):
            kept.append(b)
    kept.sort(key=lambda b: (b[1] // 45, b[0]))
    return kept


def classify_cell(img, box, tpls, threshold=ITEM_THRESHOLD):
    """返回 (item, score)；没有模板达标时 item 为 None。"""
    x, y, w, h = box
    crop = img[y:y + h, x:x + w]
    best_item, best_score = None, 0.0
    for name, tpl in tpls.items():
        if crop.shape[0] < tpl.shape[0] or crop.shape[1] < tpl.shape[1]:
            continue
        r = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, _ = cv2.minMaxLoc(r)
        if mx > best_score:
            best_item, best_score = name, float(mx)
    if best_score < threshold:
        return None, best_score
    return best_item, best_score


def _digit_blobs(area, scale=1.0):
    """找白色数字字形。返回 [(x, y, w, h)]（相对 area）。

    `scale` 是当前画布 / 720x1560 的比值：换长边时字形会等比变大，
    尺寸阈值不跟着放大会整片漏检（实测 2400 长边只剩 2/5）。
    """
    hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    mask = ((v > TIMER_V_BRIGHT) & (s < TIMER_S_MAX)).astype(np.uint8)
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    min_w = max(2, int(round(TIMER_BLOB_MIN_W * scale)))
    max_w = max(min_w + 1, int(round(TIMER_BLOB_MAX_W * scale)))
    min_h = max(4, int(round(TIMER_BLOB_MIN_H * scale)))
    max_h = max(min_h + 1, int(round(TIMER_BLOB_MAX_H * scale)))
    blobs = []
    for i in range(1, n):
        x, y, w, h = (int(stats[i, 0]), int(stats[i, 1]),
                      int(stats[i, 2]), int(stats[i, 3]))
        if min_w <= w <= max_w and min_h <= h <= max_h:
            blobs.append((x, y, w, h))
    blobs.sort(key=lambda b: b[0])
    return blobs


def _group_digits(blobs, scale=1.0):
    """把同一行的相邻字形聚成一组（倒计时）。"""
    max_gap = max(4, int(round(TIMER_MAX_GAP * scale)))
    max_dy = max(4, int(round(TIMER_MAX_DY * scale)))
    groups = []
    for b in blobs:
        placed = False
        for g in groups:
            lx, ly, lw, lh = g[-1]
            gap = b[0] - (lx + lw)
            dy = abs((ly + lh / 2.0) - (b[1] + b[3] / 2.0))
            if -max_gap <= gap <= max_gap and dy <= max_dy:
                g.append(b)
                placed = True
                break
        if not placed:
            groups.append([b])
    return groups


def find_timers(img, roi=DEFAULT_ROI):
    """找秘境面板上的倒计时（`00:38` / `23:10`）。返回绝对坐标 [(x, y, w, h)]。

    做法：白色数字字形 → 同一行的相邻字形聚成一组 → 组的外接框就是倒计时。
    不用「黑色底条」当锚点，因为黑条会和鸡妖身体/扬尘连成一片
    （实测 user_test_1 的 23:10 黑条并进 84x162 的大连通块）。

    为什么不用 OCR：数字每秒都在变，模板匹配不可用；OCR 一次 20~60ms，
    而这里只需要「有没有」，连通域 4~10ms 就够（见 §3.55）。
    """
    x0, y0, rw, rh = [int(v) for v in roi]
    h, w = img.shape[:2]
    y1 = min(h, y0 + rh)
    if y1 - y0 < 16 or rw <= 0:
        return []

    # 基线 720x1560；换分辨率时 UI 的 y 等比缩放（同 layout.BASE_H）
    s = h / 1560.0
    min_span_w = int(round(TIMER_MIN_SPAN_W * s))
    max_span_w = int(round(TIMER_MAX_SPAN_W * s))
    min_span_h = max(8, int(round(TIMER_MIN_SPAN_H * s)))
    max_span_h = int(round(TIMER_MAX_SPAN_H * s))
    pad = max(4, int(round(8 * s)))

    area = img[y0:y1, x0:x0 + rw]
    v = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)[:, :, 2]
    out = []
    for g in _group_digits(_digit_blobs(area, s), s):
        if len(g) < TIMER_MIN_DIGITS:
            continue
        gx0 = min(b[0] for b in g)
        gy0 = min(b[1] for b in g)
        gx1 = max(b[0] + b[2] for b in g)
        gy1 = max(b[1] + b[3] for b in g)
        # 贴着 ROI 上下边的组是「被裁断的 UI」（换长边时头部文字会顶到
        # y0），不是倒计时。实测 1680 长边会在 (46,250) 误检一条。
        if gy0 <= 1 or gy1 >= area.shape[0] - 1:
            continue
        bw, bh = gx1 - gx0, gy1 - gy0
        if not (min_span_w <= bw <= max_span_w and min_span_h <= bh <= max_span_h):
            continue
        if bw / float(bh) < TIMER_MIN_ASPECT:
            continue
        # 数字周围必须是深色底：否则是白底上的白字（如主界面的浅色 UI）
        xa, xb = max(0, gx0 - pad), min(area.shape[1], gx1 + pad)
        ya, yb = max(0, gy0 - pad), min(area.shape[0], gy1 + pad)
        dark = float((v[ya:yb, xa:xb] < 70).mean())
        if dark < TIMER_MIN_DARK:
            continue
        out.append((gx0 + x0, gy0 + y0, bw, bh))
    out.sort(key=lambda b: b[1])
    return out


def _timer_near(cell_box, timers, side, scale=1.0):
    """格子上/下方有没有倒计时。返回命中的倒计时框或 None。

    side="below"：倒计时**顶边**落在格子底边下方 gap 内（在采集中）
    side="above"：倒计时**底边**落在格子顶边上方 gap 内（被抢）
    """
    bx, by, bw, bh = cell_box
    if side == "below":
        lo, hi = TIMER_BELOW_GAP
        edge = by + bh
    else:
        lo, hi = TIMER_ABOVE_GAP
        edge = by
    lo *= scale
    hi *= scale
    for tx, ty, tw, th in timers:
        # 水平：倒计时中心要落在格子的 x 区间内（左右各放 10px）
        tcx = tx + tw / 2.0
        if not (bx - 10 <= tcx <= bx + bw + 10):
            continue
        probe = ty if side == "below" else ty + th
        if edge + lo <= probe <= edge + hi:
            return (tx, ty, tw, th)
    return None


def _claim_below(cells, timers, scale=1.0):
    """给每个格子找「自己下方」的倒计时，返回 {格子下标: 倒计时框}。

    一个倒计时只能属于一个格子：先到先得（cells 已按 y 升序）。
    """
    claimed = {}
    used = set()
    for i, box in enumerate(cells):
        hit = None
        for t in timers:
            if t in used:
                continue
            if _timer_near(box, [t], "below", scale):
                hit = t
                break
        if hit is not None:
            claimed[i] = hit
            used.add(hit)
    return claimed, used


def _claim_above(cells, timers, skip, scale=1.0):
    """给每个格子找「自己上方」的倒计时，返回 {格子下标: 倒计时框}。

    `skip` 是已经被别的格子认领为「下方」的倒计时 —— 一个倒计时不能
    同时是 A 格的下方和 B 格的上方（相邻两行格子间距只有 ~10px，
    A 的倒计时会落进 B 的上方窗口）。
    """
    claimed = {}
    used = set(skip)
    for i, box in enumerate(cells):
        for t in timers:
            if t in used:
                continue
            if _timer_near(box, [t], "above", scale):
                claimed[i] = t
                used.add(t)
                break
    return claimed


class CellPicker:
    def __init__(self, img_dir=IMG_DIR):
        self.tpls = {}
        for name, fn in ITEM_FILES.items():
            p = os.path.join(img_dir, fn)
            if os.path.exists(p):
                self.tpls[name] = _imread(p)
        self._chicken = None

    @property
    def chicken(self):
        if self._chicken is None:
            from chicken_reco import get_finder
            self._chicken = get_finder()
        return self._chicken

    def scan(self, img, roi=DEFAULT_ROI, priority=None,
             threshold=ITEM_THRESHOLD, exclude=None,
             allow_contested=None):
        """扫描一屏格子，返回目标格。

        `allow_contested`：允许「格子上方有倒计时」（= 别人正在抢）的物品种类
        集合。只在**自家秘境、且鸡妖够派 3 只**时由调用方传入
        `{"alcohol"}`；其它情况保持默认 None，行为不变。
        """
        x0, y0, rw, rh = [int(v) for v in roi]
        area = img[y0:y0 + rh, x0:x0 + rw]
        if area.size == 0:
            return {"cells": [], "chickens": [], "timers": [], "target": None}

        chickens = self.chicken.find(img, (x0, y0, rw, rh)) if self.chicken.ready else []
        blocked = [(c[3][0], c[3][2]) for c in chickens]
        timers = find_timers(img, roi)
        scale = img.shape[0] / 1560.0

        # 只处理完全落在 ROI 内的格子
        cells = [box for box in detect_cells(img)
                 if (x0 <= box[0] and box[1] >= y0
                     and box[0] + box[2] <= x0 + rw
                     and box[1] + box[3] <= y0 + rh)]

        # 一条倒计时只能属于一个格子：先按「下方」分配，剩下的才轮到「上方」。
        # 不这么做的话，相邻两行格子（间距 ~10px）里 A 的下方倒计时会同时
        # 落进 B 的上方窗口，B 会被误判成 contested。
        below_of = _claim_below(cells, timers, scale)[0]
        above_of = _claim_above(cells, timers,
                                set(below_of.values()), scale)

        rows = []
        for i, box in enumerate(cells):
            bx, by, bw, bh = box
            cx = bx + bw / 2.0
            item, score = classify_cell(img, box, self.tpls, threshold)
            # 「正在被采集」= SIFT 检到鸡 **或** 格子上/下方有倒计时。
            #
            # 倒计时是 UI 层固定元素，不受鸡妖动画帧影响 —— 真机 trace
            # 出现过 `blocked=[]` 但主人明明在采集（§3.55）。
            #   下方倒计时 = 正面鸡（frontage）在自己格子里采
            #   上方倒计时 = 背面鸡（back）在格子上面采 / 别人在抢
            # 两种都说明这一格现在点进去没用，先都算 busy。
            #
            # `contested`（上方倒计时）单独留一个标记：自家秘境里它表示
            # 「别人在抢我」，可以派 3 只夺回（需求方 11-17）。是否放行由
            # `allow_contested` 决定 —— 只有调用方确认「自家 + 鸡妖够」时才传。
            # 放行时忽略 `busy_sift`：抢夺者的鸡本来就会在格子上方被 SIFT
            # 检到，那正是 `timer_above` 的来源，不是「我自己在采」。
            busy_sift = any(b0 <= cx <= b1 for b0, b1 in blocked)
            timer_below = i in below_of
            timer_above = i in above_of
            contested_ok = bool(timer_above and allow_contested
                                and item in allow_contested)
            if contested_ok:
                busy = timer_below
            else:
                busy = busy_sift or timer_below or timer_above
            rows.append({
                "box": [bx, by, bw, bh],
                "cx": round(cx, 1),
                "item": item,
                "score": round(score, 3),
                "busy": busy,
                "busy_sift": busy_sift,
                "timer_below": timer_below,
                "timer_above": timer_above,
                "contested": timer_above,
                "contested_ok": contested_ok,
            })

        prio = priority_cfg.order(priority)
        order = {n: i for i, n in enumerate(prio)}
        exclude = exclude or []
        def _excluded(row):
            cx = row["cx"]
            cy = row["box"][1] + row["box"][3] / 2.0
            for b in exclude:
                ex = b[0] + b[2] / 2.0
                ey = b[1] + b[3] / 2.0
                if abs(cx - ex) <= 24 and abs(cy - ey) <= 24:
                    return True
            return False

        candidates = [r for r in rows
                      if not r["busy"] and r["item"] in order
                      and not _excluded(r)]
        # 同样优先级下先选「没人抢」的格子：抢回来的代价是 3 只鸡，
        # 有白捡的就别先打。
        candidates.sort(key=lambda r: (order[r["item"]],
                                       not r["contested_ok"], -r["score"]))
        target = candidates[0] if candidates else None
        return {
            "cells": rows,
            "chickens": [
                {"box": list(c[3]), "cx": round(c[1], 1), "cy": round(c[2], 1),
                 "pose": c[5], "inliers": c[0]} for c in chickens
            ],
            "blocked_x": [[b0, b1] for b0, b1 in blocked],
            "timers": [list(t) for t in timers],
            "priority": prio,
            "target": target,
        }


_PICKER = None

# 最近一次 PickCell 选中的格子；DispatchPolicy 用它判断酒米，
# DeferCell 用它把「数量不足 3」的格子加入延迟列表。
LAST_PICK = {}
DEFERRED = []
DISPATCHED = []


def remember_pick(target):
    global LAST_PICK
    LAST_PICK = {
        "box": list(target["box"]),
        "item": target["item"],
        "cx": target["cx"],
        # 自家秘境被抢的酒米：派遣策略据此把 target 从 1 提到 3
        "contested": bool(target.get("contested")),
    }


def defer_last():
    global LAST_PICK
    box = LAST_PICK.get("box")
    if box and not any(
            abs(box[0] - b[0]) <= 24 and abs(box[1] - b[1]) <= 24
            for b in DEFERRED):
        DEFERRED.append(list(box))
    LAST_PICK = {}
    return list(DEFERRED)


def mark_dispatched_last():
    """``掠奪`` 成功后记住刚派出的格子，下一轮选格强制跳过它。"""
    global LAST_PICK
    box = LAST_PICK.get("box")
    if box and not any(
            abs(box[0] - b[0]) <= 24 and abs(box[1] - b[1]) <= 24
            for b in DISPATCHED):
        DISPATCHED.append(list(box))
    LAST_PICK = {}
    return list(DISPATCHED)


def clear_deferred():
    DEFERRED.clear()
    DISPATCHED.clear()
    LAST_PICK.clear()


def have_deferred():
    return bool(DEFERRED)


def get_picker():
    global _PICKER
    if _PICKER is None:
        _PICKER = CellPicker()
    return _PICKER


try:
    from maa.custom_recognition import CustomRecognition
    from maa.context import Context
    from maa.pipeline import JRecognitionType, JOCR

    def _read_level(context: Context, image, box):
        """读物品图标右下角的 `N级`；返回 (int|None, OCR 原文)。

        先读格子右下角，读不到再退回整格。若 OCR 把数量与等级连成
        `3501级`（1 级云英 350 个），等级会先被判成 3501；此时取紧挨
        「级」的最后一位数字，避免把 1 级资源误派 3 只。
        """
        x, y, w, h = [int(v) for v in box]
        bottom_right = [
            x + int(w * 0.40),
            y + int(h * 0.55),
            w - int(w * 0.40),
            h - int(h * 0.55),
        ]
        last_text = ""
        for roi in (bottom_right, [x, y, w, h]):
            detail = context.run_recognition_direct(
                JRecognitionType.OCR,
                JOCR(roi=[int(v) for v in roi], expected=[]),
                image,
            )
            texts = []
            for rr in (getattr(detail, "all_results", None) or []):
                text = getattr(rr, "text", None)
                if text:
                    texts.append(str(text))
            joined = "".join(texts)
            last_text = joined or last_text
            m = re.search(r"(\d+)\s*[级級]", joined)
            if not m:
                continue
            level = int(m.group(1))
            if not (1 <= level <= 9):
                tail = re.search(r"(\d)\s*[级級]", joined)
                level = int(tail.group(1)) if tail else None
            if level is not None and 1 <= level <= 9:
                return level, joined
        return None, last_text

    class CellRecognition(CustomRecognition):
        """pipeline: recognition=Custom, custom_recognition=PickCell。"""

        # 自家秘境里允许「抢回来」的物品（需求方 11-17：只处理酒米）。
        # 条件：自家秘境 + 空闲鸡妖够派 3 只。鸡妖不够就仍当 busy，
        # 免得点进去又因为 target > stock 走 `延迟该资源` 白跑一趟。
        CONTESTED_ITEMS = {"alcohol"}
        CONTESTED_TARGET = 3

        def _allow_contested(self):
            """返回 `{"alcohol"}` 或 None。状态全部读 `loop_ctl._STATE`。"""
            try:
                import loop_ctl
                st = loop_ctl._STATE
            except Exception:  # noqa: BLE001
                return None
            if st.get("realm") != "own":
                return None
            stock = st.get("stock")
            if not isinstance(stock, int) or stock <= 0:
                return None
            total = st.get("total")
            need = self.CONTESTED_TARGET
            if isinstance(total, int) and total > 0:
                need = min(need, total)
            if stock < need:
                return None
            return set(self.CONTESTED_ITEMS)

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            # GUI 没有 PopupGuardController：先清弹窗再识别（§3.65）。
            # 没弹窗时只做一次模板匹配，不产生额外截图。
            argv.image = clear_popups(context, argv.image)
            param = json.loads(argv.custom_recognition_param or "{}") or {}
            # 优先用节点 roi（Maa 会算好传进来），其次 param，最后默认值
            node_roi = getattr(argv, "roi", None)
            if node_roi is not None and node_roi.w > 0 and node_roi.h > 0:
                roi = [node_roi.x, node_roi.y, node_roi.w, node_roi.h]
            else:
                roi = param.get("roi") or DEFAULT_ROI
            allow_contested = self._allow_contested()
            result = get_picker().scan(
                argv.image, roi=roi,
                priority=param.get("priority_levels", param.get("priority")),
                threshold=float(param.get("threshold", ITEM_THRESHOLD)),
                exclude=DEFERRED + DISPATCHED,
                allow_contested=allow_contested,
            )
            target = result["target"]
            if target is None:
                # cells= 带上 item / contested：真机 trace 里曾出现
                # 「自家秘境、上方有倒计时、stock 够」却判 no target，
                # 但旧日志只有 busy 框，看不出那一格到底识别成了什么物品
                # （§3.58 验证小节）。
                print("[PickCell] no target blocked=%s timers=%s busy=%s"
                      " cells=%s allow_contested=%s"
                      % (result.get("blocked_x"),
                         result.get("timers"),
                         [c["box"] for c in result["cells"] if c["busy"]],
                         [(c["box"], c["item"], c["contested"])
                          for c in result["cells"]],
                         sorted(allow_contested) if allow_contested else None),
                      flush=True)
                return CustomRecognition.AnalyzeResult(
                    box=None, detail={"error": "no target cell", **result})
            x, y, w, h = target["box"]
            remember_pick(target)
            level, level_text = _read_level(
                context, argv.image, target["box"])
            LAST_PICK["level"] = level
            LAST_PICK["level_text"] = level_text
            print("[PickCell] target=%s box=%s contested=%s blocked=%s"
                  " timers=%s level=%s text=%r priority=%s"
                  % (target["item"], target["box"], target.get("contested"),
                     result.get("blocked_x"), result.get("timers"),
                     level, level_text, result.get("priority")),
                  flush=True)
            return CustomRecognition.AnalyzeResult(
                box=(x, y, w, h),
                detail={"item": target["item"], "score": target["score"], **result},
            )

    def register(name="PickCell"):
        """注册到 AgentServer（agent/main.py 里调用）。"""
        from maa.agent.agent_server import AgentServer
        return AgentServer.register_custom_recognition(name, CellRecognition())

except Exception:  # pragma: no cover
    CellRecognition = None

    def register(name="PickCell"):
        raise RuntimeError("maa import failed")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    picker = CellPicker()
    print(f"item templates: {sorted(picker.tpls)}")
    for p in sys.argv[1:] or [
        r"User_screenshot\user_test_1.png",
        r"User_screenshot\user_test_2.png",
        r"screenshots\snap\live_now_720.png",
        r"screenshots\159_before_drain.png",
        r"screenshots\157_popup_for_test.png",
    ]:
        if not os.path.isabs(p):
            p = os.path.join(ROOT, p)
        img = _imread(p)
        h, w = img.shape[:2]
        if w != 720:
            img = cv2.resize(img, (720, int(round(h * 720 / w))),
                             interpolation=cv2.INTER_AREA)
        r = picker.scan(img)
        print(f"\n=== {os.path.basename(p)} ===")
        if r["timers"]:
            print(f"  timers={r['timers']}")
        for c in r["cells"]:
            flag = "BUSY" if c["busy"] else "free"
            why = "+".join(k for k, on in
                           (("sift", c["busy_sift"]),
                            ("below", c["timer_below"]),
                            ("above", c["timer_above"])) if on) or "-"
            print(f"  {flag}  cx={c['cx']:6.1f} y={c['box'][1]:4d} "
                  f"item={str(c['item']):8s} score={c['score']:.3f}  {why}")
        print(f"  chickens={len(r['chickens'])}  target={r['target']}")
