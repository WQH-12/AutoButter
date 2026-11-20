"""探寻列表的按行识别（自定义识别，注册名 PickRow）。

列表页的几何（720 画布，实测见 HANDOVER §3.15）：

    「前往」按钮 y = 454 / 606 / 757 / 967
    图标带   y = 按钮 y + 34，高 66
    「采集中」徽标 y = 按钮 y + 82，78x41，x 61..139
    相邻行差 151~152；跨分区（附近秘境 → 同行秘境）是 210

所以不能按固定行距推，必须用「前往」按钮当行锚点。

一行要被选中，需要同时满足：
  1. 不是「采集中」（用户规则：正在采集的跳过）
  2. 该行**没有被面板底部渐隐压暗**（否则物品分数会在 0.88 附近摆动，不可靠）
  3. 该行的物品串里有用户想要的物品（阈值 0.88）

返回值是**该行「前往」按钮的框**，pipeline 直接 Click 它。
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np

import priority as priority_cfg
from popup_guard import clear_popups

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _resolve_img_dir():
    """模板目录：<root>/resource/image（同 chicken_reco / cell_reco）。"""
    candidates = [
        os.path.join(ROOT, "resource", "image"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[0]


IMG_DIR = _resolve_img_dir()

ITEM_FILES = {
    "alcohol": "item_alcohol.png",
    "berry": "item_berry.png",
    "fabric": "item_fabric.png",
    "honey": "item_honey.png",
    "jade": "item_jade.png",
    "silver": "item_silver.png",
    "stone": "item_stone.png",
}
# 兼容旧调用；实际排序由 priority.py 的 0..7 等级计算。
DEFAULT_PRIORITY = priority_cfg.order()
ITEM_THRESHOLD = 0.88

# 「采集中」阈值 0.92：比物品模板高，因为 0.88 档有 2 张误报（HANDOVER §3.13）
COLLECT_THRESHOLD = 0.92

# 行内偏移（实测，见模块注释）
ICON_DY = 34
ICON_H = 66
COLLECT_DY = 82
COLLECT_W = 160      # 徽标 78 宽，留余量到 x≈220
COLLECT_H = 55
ROW_TOP_PAD = 10
ROW_BOTTOM_PAD = 135

# 冷却时间（秒）。资源采集完大约 5 分钟后刷新：
# 空秘境/已访问用 330s（5.5 分钟），留 30s 余量；
# 鸡妖不足的延迟普通资源 90s，酒米 330s。
EMPTY_COOLDOWN_SECONDS = 330
DONE_COOLDOWN_SECONDS = 330
DEFER_COOLDOWN_SECONDS = 90
ALCOHOL_DEFER_COOLDOWN_SECONDS = 330
MAX_COOLDOWN_SECONDS = 1800

# 行是否「可读」：图标带的高饱和像素占比。
# 实测正常行 0.448~0.676，被底部渐隐压暗的行 0.001~0.222（§3.15 / §3.17）。
# 0.30 落在两者中间。
READABLE_S60 = 0.30


def _imread(path):
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def find_rows(img, x0=540, x1=670, min_w=90, max_w=100,
              min_h=23, max_h=29):
    """用绿色「前往」按钮做行锚点。返回 [(btn_y, btn_x, w, h)]，按 y 升序。

    几何约束故意收得很紧：实测「前往」按钮恒为 94x26（高有 25/26 两档）。
    放宽到 h>=3 时，非列表界面上大片绿色会误检成「行」
    （实测 01_current 误检 58x41 / 89x43 / 130x5，113_final2 误检 55x6），
    进而让 `选行并点击` 在错误界面上走 on_error 去滑动。
    收到 80~110 宽还会把「資源詳情」浮窗里的绿块算进来
    （33_manage_t0 误检 84x25 / 85x26），所以宽度也卡到 90~100。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85)
            & (hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 90)).astype(np.uint8) * 255
    mask[:, :x0] = 0
    mask[:, x1:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    rows = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if min_w <= w <= max_w and min_h <= h <= max_h:
            rows.append((y, x, w, h))
    return sorted(rows)


class RowPicker:
    """按行扫列表，返回第一个「可进入且含目标物品」的行。"""

    def __init__(self, img_dir=IMG_DIR):
        self.items = {}
        for name, fn in ITEM_FILES.items():
            p = os.path.join(img_dir, fn)
            if os.path.exists(p):
                self.items[name] = _imread(p)
        self.collect = []
        for i in range(1, 5):
            p = os.path.join(img_dir, "collect_%d.png" % i)
            if os.path.exists(p):
                self.collect.append(_imread(p))

    @property
    def ready(self):
        return bool(self.items) and bool(self.collect)

    def _collecting(self, img, btn_y):
        """该行是不是「采集中」。"""
        y = btn_y + COLLECT_DY
        if y + COLLECT_H > img.shape[0]:
            return False, 0.0
        sub = img[y:y + COLLECT_H, 0:COLLECT_W]
        best = 0.0
        for t in self.collect:
            if sub.shape[0] < t.shape[0] or sub.shape[1] < t.shape[1]:
                continue
            r = cv2.matchTemplate(sub, t, cv2.TM_CCOEFF_NORMED)
            best = max(best, float(r.max()))
        return best >= COLLECT_THRESHOLD, round(best, 3)

    def _items(self, img, btn_y, threshold=ITEM_THRESHOLD):
        """该行图标带里命中的物品，返回 [(name, x, score)]。"""
        y = btn_y + ICON_DY
        if y + ICON_H > img.shape[0]:
            return []
        sub = img[y:y + ICON_H, 150:660]
        out = []
        for name, t in self.items.items():
            if sub.shape[0] < t.shape[0] or sub.shape[1] < t.shape[1]:
                continue
            r = cv2.matchTemplate(sub, t, cv2.TM_CCOEFF_NORMED)
            rr = r.copy()
            while True:
                _, mx, _, ml = cv2.minMaxLoc(rr)
                if mx < threshold:
                    break
                out.append((name, ml[0] + 150, round(float(mx), 3)))
                rr[max(0, ml[1] - 20):ml[1] + 20,
                   max(0, ml[0] - 20):ml[0] + 20] = -1
        return sorted(out, key=lambda t: t[1])

    @staticmethod
    def _readable(img, btn_y):
        """该行是否没被底部渐隐压暗。压暗行的物品分数不可靠，不该选。"""
        y = btn_y + ICON_DY
        if y + ICON_H > img.shape[0]:
            return False, 0.0
        band = cv2.cvtColor(img[y:y + ICON_H, 150:660], cv2.COLOR_BGR2HSV)
        frac = float((band[:, :, 1] > 60).mean())
        return frac >= READABLE_S60, round(frac, 3)

    def scan(self, img, priority=None, threshold=ITEM_THRESHOLD):
        global NEXT_READY
        NEXT_READY = 0.0
        order = {n: i for i, n in enumerate(priority_cfg.order(priority))}
        rows = []
        for btn_y, btn_x, w, h in find_rows(img):
            collecting, collect_score = self._collecting(img, btn_y)
            readable, s60 = self._readable(img, btn_y)
            items = self._items(img, btn_y, threshold)
            wanted = [t for t in items if t[0] in order]
            wanted.sort(key=lambda t: (order[t[0]], -t[2]))
            rows.append({
                "btn": [btn_x, btn_y, w, h],
                "collecting": collecting,
                "collect_score": collect_score,
                "readable": readable,
                "s60": s60,
                "items": [{"item": n, "x": x, "score": s} for n, x, s in items],
                "wanted": [{"item": n, "x": x, "score": s} for n, x, s in wanted],
            })

        result = self.finalize(rows)
        result["priority"] = list(order)
        return result

    @staticmethod
    def finalize(rows):
        """按当前 rows 重算 target / all_collecting（OCR 兜底后可重算）。"""
        global NEXT_READY
        strict = [r for r in rows
                  if r["readable"] and not r["collecting"] and r["wanted"]
                  and r.get("level_visible", True)
                  and not r.get("bad", False) and not is_bad_row(r)]
        # 只剩一行不在采集中、但该行的 `N级` 被底边/遮挡吃掉时，不能让整页
        # 永远没有 target。严格候选为空时退到 soft：仍要求可读、非采集中。
        soft = [r for r in rows
                if r["readable"] and not r["collecting"] and r["wanted"]
                and not r.get("bad", False) and not is_bad_row(r)]
        # 最后兜底：只剩底部渐隐行不在采集中时，`readable=False` 但仍可能是
        # 真实可进入的行；先让它进一次，进去后若没有资源会被标记坏行。
        fallback = [r for r in rows
                    if not r["collecting"] and r["wanted"]
                    and not r.get("bad", False) and not is_bad_row(r)]
        # 最后一道：整页只剩唯一一行不在采集中，但物品模板也因底部渐隐
        # 没命中时，仍允许点进去一次；进去后没资源会被标记坏行。
        last_resort = [r for r in rows
                       if not r["collecting"]
                       and r["items"]
                       and not r.get("bad", False) and not is_bad_row(r)]
        fallback_kind = "fallback" if fallback else None
        if not fallback and len(last_resort) == 1:
            fallback = last_resort
            fallback_kind = "last_resort"
        target = (strict[0] if strict else
                  soft[0] if soft else
                  fallback[0] if fallback else None)
        target_kind = ("strict" if strict else
                       "soft" if soft else
                       fallback_kind if fallback else None)

        # 「整页都是采集中」只看**可读**的行。
        # 末行通常落在底部渐隐区（readable=False），它的 collecting 判定不可信
        # （实测 720_fullscreen_activity 末行 collect_score 只有 0.104），
        # 把它算进来会让 all_collecting 永远为 False。
        readable = [r for r in rows if r["readable"]]
        non_collecting = [r for r in rows if not r.get("collecting")]
        all_cooling = (bool(rows)
                       and len(non_collecting) == len(rows)
                       and all(r.get("bad", False) or is_bad_row(r)
                               for r in rows))
        if all_cooling or not rows:
            NEXT_READY = cooldown_earliest()
        else:
            NEXT_READY = 0.0
        return {
            "rows": rows,
            "target": target,
            "target_kind": target_kind,
            "fallback_count": len(fallback),
            "bad_count": sum(1 for r in rows
                             if r.get("bad", False) or is_bad_row(r)),
            "all_collecting": bool(readable) and all(r["collecting"] for r in readable),
            "all_cooling": all_cooling,
            "next_ready": round(NEXT_READY, 1),
            "none_wanted": bool(rows) and not any(r["wanted"] for r in rows),
            "no_candidate": bool(rows) and target is None,
        }


_PICKER = None
LAST_ROW = {}
# 坏行 / 延迟行冷却：key -> {"until", "strikes", "reason", <身份字段>}
COOLDOWN = {}
# 已派遣行冷却单独一张表：done 固定 330s，不与 empty/defer 的
# 指数退避共享 `until`（§3.61 修法 3）。
DONE_COOLDOWN = {}
NAME_CACHE = {}
LIST_ROUND = 0
# 全页都在冷却时，PickRow 写入“最早可再访问”的秒数；
# SwipeRows 看到它就先等待，不把冷却误判成列表到底。
NEXT_READY = 0.0


def _row_signature(row):
    """行身份：物品 + 横坐标。同一行的物品串在滚动后 x 不变。"""
    return tuple((str(t.get("item")), int(t.get("x", 0)))
                 for t in (row.get("items") or []))


def _target_item(target):
    """PickRow 要写进 detail 的物品名；last_resort 时 wanted 为空。"""
    wanted = target.get("wanted") or []
    items = target.get("items") or []
    if wanted:
        return wanted[0]["item"]
    if items:
        return items[0]["item"]
    return None


def _row_key(row):
    """稳定行身份：玩家名优先，头像 hash 兜底，最后才用物品签名。"""
    name = str(row.get("name") or "").strip()
    if name:
        return "name:%s" % name
    avatar = str(row.get("avatar") or "")
    if avatar:
        return "avatar:%s" % avatar
    sig = _row_signature(row)
    if sig:
        return "sig:%r" % (sig,)
    btn = list(row.get("btn") or [])
    return "pos:%s" % (tuple(btn[:2]),)


def _row_identity(row):
    """一行的身份字段。冷却表里存这几个字段，比对时逐项判断。"""
    return {
        "name": str(row.get("name") or "").strip(),
        "avatar": str(row.get("avatar") or ""),
        "sig": _row_signature(row),
        "pos": tuple(list(row.get("btn") or [])[:2]),
    }


def _identity_key(ident):
    return (ident.get("name", ""), ident.get("avatar", ""),
            ident.get("sig", ()), ident.get("pos", ()))


def _same_row(a, b):
    """同一秘境的判据（§3.61 修法 1）。

    同名玩家的**不同秘境**物品串不同，不能因为 name 相同就锁死：

      - name 或 avatar 命中，**且**物品签名 sig 命中 → 同一行；
      - sig 任一侧缺失（空秘境 / 物品没识别出来）→ 退回 name + avatar；
      - 两侧都没有 name/avatar 时，只能靠 sig + 按钮 x（滚动只改 y）。
    """
    a_name = str(a.get("name") or "").strip()
    b_name = str(b.get("name") or "").strip()
    a_avatar = str(a.get("avatar") or "")
    b_avatar = str(b.get("avatar") or "")
    a_sig = a.get("sig") or ()
    b_sig = b.get("sig") or ()

    name_match = bool(a_name and b_name and a_name == b_name)
    avatar_match = bool(a_avatar and b_avatar and a_avatar == b_avatar)
    sig_match = bool(a_sig and b_sig and tuple(a_sig) == tuple(b_sig))

    if sig_match:
        if name_match or avatar_match:
            return True
        if not a_name and not b_name and not a_avatar and not b_avatar:
            a_pos = tuple(a.get("pos") or ())
            b_pos = tuple(b.get("pos") or ())
            return bool(a_pos and b_pos and a_pos[0] == b_pos[0])
        return False
    if not a_sig or not b_sig:
        return name_match and avatar_match
    return False


def _find_record(table, ident):
    for rec in table.values():
        if _same_row(rec, ident):
            return rec
    return None


def remember_row(row):
    """记下 PickRow 刚返回的行，供「进去发现没资源」时标记坏行。"""
    global LAST_ROW
    ident = _row_identity(row)
    LAST_ROW = {
        "btn": list(row.get("btn") or []),
        "items": list(row.get("items") or []),
        "sig": ident["sig"],
        "name": ident["name"],
        "avatar": ident["avatar"],
        "pos": ident["pos"],
        "key": _row_key(row),
        "ident": ident,
    }


def is_bad_row(row, now=None):
    """该行是否还在冷却期。冷却到期后自动重新成为候选。"""
    now = time.time() if now is None else now
    ident = _row_identity(row)
    for table in (COOLDOWN, DONE_COOLDOWN):
        for rec in table.values():
            if (now < float(rec.get("until", 0))
                    and _same_row(rec, ident)):
                return True
    return False


def begin_list_round():
    """保留列表轮次计数，供日志/旧逻辑使用；坏行已改为按时间冷却。"""
    global LIST_ROUND
    LIST_ROUND += 1
    return LIST_ROUND


def _mark_cooldown(seconds, reason):
    """给 LAST_ROW 设置指数退避冷却（只用于 empty/defer）。"""
    global LAST_ROW
    ident = LAST_ROW.get("ident") or _row_identity(LAST_ROW)
    if not any(ident.get(k) for k in ("name", "avatar", "sig", "pos")):
        return list(COOLDOWN.items())
    now = time.time()
    rec = _find_record(COOLDOWN, ident)
    if rec is None:
        rec = {"strikes": 0, "until": 0.0}
        COOLDOWN[_identity_key(ident)] = rec
    rec.update(ident)
    delay = min(MAX_COOLDOWN_SECONDS,
                float(seconds) * (2 ** int(rec.get("strikes", 0))))
    rec["until"] = max(float(rec.get("until", 0)), now + delay)
    rec["strikes"] = int(rec.get("strikes", 0)) + 1
    rec["reason"] = reason
    rec["delay"] = round(delay, 1)
    LAST_ROW = {}
    return list(COOLDOWN.items())


def skip_last_row(seconds=DEFER_COOLDOWN_SECONDS, reason="defer"):
    """鸡妖不足延迟：按时间冷却，重复命中同一条目时指数退避。"""
    return _mark_cooldown(seconds, reason)


def mark_last_row_bad(seconds=EMPTY_COOLDOWN_SECONDS):
    """进去发现空秘境：冷却约 5.5 分钟，而不是任务内永久坏行。"""
    return _mark_cooldown(seconds, "empty")


def mark_last_row_done(seconds=DONE_COOLDOWN_SECONDS):
    """成功派遣过：资源约 5 分钟后刷新。

    写独立的 DONE_COOLDOWN，固定 `now + seconds`，不再指数叠加；
    与 empty/defer 的 until 互不污染（§3.61 修法 3）。
    """
    global LAST_ROW
    ident = LAST_ROW.get("ident") or _row_identity(LAST_ROW)
    if not any(ident.get(k) for k in ("name", "avatar", "sig", "pos")):
        return list(DONE_COOLDOWN.items())
    rec = _find_record(DONE_COOLDOWN, ident)
    if rec is None:
        rec = {}
        DONE_COOLDOWN[_identity_key(ident)] = rec
    rec.update(ident)
    rec["until"] = time.time() + float(seconds)
    rec["reason"] = "done"
    rec["delay"] = round(float(seconds), 1)
    rec.pop("strikes", None)
    LAST_ROW = {}
    return list(DONE_COOLDOWN.items())


def cooldown_earliest(now=None):
    """返回距今最近的冷却剩余秒数；没有有效冷却返回 0。"""
    now = time.time() if now is None else now
    waits = [float(rec.get("until", 0)) - now
             for table in (COOLDOWN, DONE_COOLDOWN)
             for rec in table.values()
             if float(rec.get("until", 0)) > now]
    return max(0.0, min(waits)) if waits else 0.0


def clear_bad_rows():
    """任务开始时清空；同一任务内跨列表保留，避免重复进同一个空秘境。"""
    global LIST_ROUND, NEXT_READY
    COOLDOWN.clear()
    DONE_COOLDOWN.clear()
    NAME_CACHE.clear()
    LIST_ROUND = 0
    NEXT_READY = 0.0
    LAST_ROW.clear()


def get_picker():
    global _PICKER
    if _PICKER is None:
        _PICKER = RowPicker()
    return _PICKER


try:
    from maa.custom_recognition import CustomRecognition
    from maa.context import Context
    from maa.pipeline import JRecognitionType, JOCR

    def _ocr_collecting(context, image, btn_y):
        """模板没判中时的文字兜底：读徽标区域是否出现「采/採…集」。"""
        roi = [0, int(btn_y) + COLLECT_DY, 180, COLLECT_H]
        detail = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(roi=roi, expected=[]),
            image,
        )
        texts = []
        for rr in (getattr(detail, "all_results", None) or []):
            text = getattr(rr, "text", None)
            if text:
                texts.append(str(text))
        joined = "".join(texts)
        hit = ("采" in joined or "採" in joined) and "集" in joined
        return joined, hit

    def _ocr_levels(context, image, btn_y):
        """读资源图标下方是否露出 `N级`；没有级字说明该行中间被裁掉了。"""
        roi = [150, int(btn_y) + 70, 510, 40]
        detail = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(roi=roi, expected=[]),
            image,
        )
        texts = []
        for rr in (getattr(detail, "all_results", None) or []):
            text = getattr(rr, "text", None)
            if text:
                texts.append(str(text))
        joined = "".join(texts)
        return joined, ("级" in joined or "級" in joined)

    def _ocr_name(context, image, btn_y):
        """读列表行左侧的玩家名，作为稳定行身份。"""
        roi = [150, int(btn_y) - 10, 260, 45]
        detail = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(roi=roi, expected=[]),
            image,
        )
        texts = []
        for rr in (getattr(detail, "all_results", None) or []):
            text = getattr(rr, "text", None)
            if text:
                texts.append(str(text).strip())
        return "".join(texts)

    def _avatar_hash(image, btn_y):
        """行头像的 256-bit 感知 hash；名称 OCR 失败时兜底。"""
        # 旧框 (55, btn_y-10, 100, 80) 基本落在背景上：16x16 感知 hash
        # 前 80 bit 全饱和，同帧不同玩家的 hamming 只有 2（§3.61）。
        # 新框收进头像本体，同帧最小 hamming 21，跨帧同玩家仍稳定。
        x, y, w, h = 86, int(btn_y) + 2, 130, 68
        crop = image[max(0, y):y + h, x:x + w]
        if crop is None or crop.size == 0:
            return ""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        bits = (gray > float(gray.mean())).astype(np.uint8).flatten()
        return np.packbits(bits).tobytes().hex()

    class RowRecognition(CustomRecognition):
        """pipeline: recognition=Custom, custom_recognition=PickRow。

        命中时返回「前往」按钮的框，配合 action=Click 直接进入该秘境。
        """

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            # GUI 没有 PopupGuardController：先清弹窗再识别（§3.65）。
            # 弹窗盖住列表时，PickRow 会把整页判成「没有可进入的行」。
            argv.image = clear_popups(context, argv.image)
            param = json.loads(argv.custom_recognition_param or "{}") or {}
            picker = get_picker()
            if not picker.ready:
                return CustomRecognition.AnalyzeResult(
                    box=None, detail={"error": "row templates not found"})

            result = picker.scan(
                argv.image,
                priority=param.get("priority_levels", param.get("priority")),
                threshold=float(param.get("threshold", ITEM_THRESHOLD)),
            )
            # 模板只覆盖 4 种已知徽标；某些列表/底部渐隐行会出现
            # 不同样式的「采集中」，用徽标区 OCR 兜底一次。
            for row in result["rows"]:
                row.setdefault("collect_text", "")
                row["sig"] = _row_signature(row)
                row["avatar"] = _avatar_hash(argv.image, row["btn"][1])
                name = NAME_CACHE.get(row["avatar"]) if row["avatar"] else None
                if name is None:
                    name = _ocr_name(context, argv.image, row["btn"][1])
                    if row["avatar"]:
                        NAME_CACHE[row["avatar"]] = name
                row["name"] = name
                row["key"] = _row_key(row)
                row["bad"] = is_bad_row(row)
                if row["bad"]:
                    print("[PickRow] skip cooldown y=%d key=%s"
                          % (row["btn"][1], row["key"]),
                          flush=True)
                    continue
                if row["readable"] and row["wanted"]:
                    level_text, level_visible = _ocr_levels(
                        context, argv.image, row["btn"][1])
                    row["level_text"] = level_text
                    row["level_visible"] = level_visible
                    if not level_visible:
                        print("[PickRow] hidden-level row y=%d text=%r "
                              "(keep as soft candidate)"
                              % (row["btn"][1], level_text), flush=True)
                if (not row["collecting"] and row["readable"]
                        and row["wanted"]):
                    text, hit = _ocr_collecting(
                        context, argv.image, row["btn"][1])
                    row["collect_text"] = text
                    if hit:
                        row["collecting"] = True
                        row["collect_source"] = "ocr"
                        print("[PickRow] OCR collected row y=%d text=%r"
                              % (row["btn"][1], text), flush=True)
            result = picker.finalize(result["rows"])
            print("[PickRow] priority=%s %s"
                  % (result.get("priority"),
                     [(r["btn"][1], r["collect_score"], r["collecting"],
                       r.get("collect_source", "template"), r["readable"],
                       r.get("level_visible", True), r.get("bad", False),
                       len(r["wanted"]))
                      for r in result["rows"]]),
                  flush=True)
            target = result["target"]
            if target is None:
                return CustomRecognition.AnalyzeResult(
                    box=None, detail={"error": "no row", **result})
            item = _target_item(target)
            if item is None:
                return CustomRecognition.AnalyzeResult(
                    box=None, detail={"error": "target has no item", **result})
            x, y, w, h = target["btn"]
            remember_row(target)
            return CustomRecognition.AnalyzeResult(
                box=(x, y, w, h),
                detail={"item": item,
                        "target_kind": result.get("target_kind"), **result},
            )

    def register(name="PickRow"):
        from maa.agent.agent_server import AgentServer
        return AgentServer.register_custom_recognition(name, RowRecognition())

except Exception:  # pragma: no cover
    RowRecognition = None

    def register(name="PickRow"):
        raise RuntimeError("maa import failed")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    picker = RowPicker()
    print("items=%d collect=%d ready=%s"
          % (len(picker.items), len(picker.collect), picker.ready))
    for rel in ["User_screenshot/full/720_list_top.png",
                "screenshots/45_tanxun_list.png",
                "screenshots/49_list2.png",
                "screenshots/snap/20261108-152332_720.png"]:
        p = os.path.join(ROOT, rel)
        img = cv2.imread(p)
        if img is None:
            print("MISSING", rel)
            continue
        res = picker.scan(img)
        print("\n== %s" % rel.split("/")[-1])
        for i, r in enumerate(res["rows"]):
            print("   row%d btn_y=%4d readable=%-5s(%.3f) collecting=%-5s %.3f  wanted=%s"
                  % (i, r["btn"][1], r["readable"], r["s60"],
                     r["collecting"], r["collect_score"],
                     [t["item"] for t in r["wanted"]]))
        print("   target=%s all_collecting=%s none_wanted=%s no_candidate=%s"
              % ((res["target"] or {}).get("wanted", [{}])[0].get("item")
                 if res["target"] else None,
                 res["all_collecting"], res["none_wanted"], res["no_candidate"]))
