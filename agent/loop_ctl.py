"""列表循环的控制件（自定义动作 / 识别），注册名见 registry.py。

需求方 11-10 的规则：

    每次滑动当前页面的像素；当上一张测量和下一张测量相同时就是划不动了。

本模块提供 6 个件：

    SwipeRows      Custom Action    运行时算 dy，先记帧再滑
    ChickenStock   Custom Recog     读「管理」左边 `閒【N】`，定目标派遣数
    DispatchCountReached Custom Recog 读浮窗 `雞妖数量 n/3`，达到目标才放行
    ListMoved      Custom Recog     和记住的帧比，命中=动了，不命中=划不动
    LoopGuard      Custom Recog     回主界面重来的次数闸门
    DispatchReset  Custom Action    浮窗确认时清零「+」计数
    AddChicken     Custom Action    点「+」，超次数返回 False
    ClickReset     Custom Action    点识别框中心，并把循环计数清零

为什么滑动量要运行时算：列表是分区的（附近秘境 / 同行秘境），跨分区行距 210
而不是 151.5，写死 454 会在有分区标题的页面上少滑一截（§3.16）。

为什么「+」不用 Maa 的 max_hit：max_hit 是**整个任务**共享的，用完不恢复。
长循环里第二次派遣时加号就点不动了。计数器放在 Python 侧，浮窗每次确认时清零。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

try:
    import cv2
except ImportError:  # Android 的 agent core 不带 opencv，用 numpy/Pillow 子集
    import cv2lite as cv2
import numpy as np

from popup_guard import clear_popups

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 判据阈值：同一页不同帧最大 3.19，换页最小 8.13（§3.16）
DIFF_THRESHOLD = 4.0

# 模板与 ROI 全部按 720 短边、1560 长边的画布裁；长边变化时 y 等比缩放。
# 和 layout.BASE_H 是同一个数，这里单独定义是为了 loop_ctl 不硬依赖 layout。
BASE_H = 1560

# 距上次记帧超过这个秒数就当「没有基准」，避免跨任务残留
RESET_SECONDS = 30.0

# 列表可视区（§3.15：y 357..1179）
DEFAULT_ROI = [40, 357, 640, 822]

# 滑动起点：列表中部，实测不会碰到行内按钮
SWIPE_X = 360
SWIPE_FROM_Y = 1000
SWIPE_DURATION = 400

# 连续「回主界面重来」的上限（中间没有一次成功掠奪）
MAX_LOOPS = 5

# 一次派遣最多点几次「+」（1/3 → 3/3 只要 2 次）
MAX_PLUS = 4

# 连续滑动多少次都没选中行就认输（防止列表无限长时空转）
MAX_SWIPES = 8

# 一趟列表最多点几次「刷新」（刷新按钮自带 CD，用满后闸门不再命中）
MAX_REFRESH = 3

# 每次「回主界面」成功后，最多点几次「家園」把菜单展开。
# 注意这是**每轮重进**的配额（`ClickAdventure` 成功即清零），
# 不是整趟任务共享 —— 后者会让长任务第 4 次重进必然失败（§3.53）。
MAX_REOPEN = 3

# 「刷新」按钮判定（720x1560 基线实测，见 probe_refresh_orange.py）
#   可点（橙）：S=124.5  R-B=113.5
#   不可点（灰）：S=0.0  R-B=0.0
# 阈值取中间偏保守：灰按钮的 S 恒为 0（纯灰），橙按钮 120+，区分度极大。
REFRESH_SCORE = 0.85      # 按钮形状匹配（灰模板，橙灰形状一致）
REFRESH_SAT = 60.0        # 按钮内缩区平均饱和度
REFRESH_RB = 40.0         # 平均 R-B（橙色通道差）

_TPL_CACHE = {}


def _popup_match(img):
    """GUI 兜底：直接在截图里找 `obtain_resource.png`。"""
    try:
        from popup_guard import find_popup
        box, score = find_popup(img)
        return box is not None, score
    except Exception:  # noqa: BLE001
        return False, 0.0


def _load_template(name):
    """从 `resource/image` 加载模板。"""
    if name in _TPL_CACHE:
        return _TPL_CACHE[name]
    tpl = None
    for base in (os.path.join(ROOT, "resource", "image"),):
        p = os.path.join(base, name)
        if os.path.exists(p):
            tpl = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            break
    _TPL_CACHE[name] = tpl
    return tpl


def _match_template(img, name, roi, threshold):
    """在 ROI 里匹配模板，返回 (是否命中, 最高分)。"""
    tpl = _load_template(name)
    if tpl is None or img is None:
        return False, 0.0
    x, y, w, h = [int(v) for v in roi]
    sub = img[y:y + h, x:x + w]
    if sub.shape[0] < tpl.shape[0] or sub.shape[1] < tpl.shape[1]:
        return False, 0.0
    r = cv2.matchTemplate(sub, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, _ = cv2.minMaxLoc(r)
    return mx >= threshold, round(float(mx), 3)


# 秘境面板「管理」按钮的判定：
#   hit  (>=0.94)：面板完整显示；
#   gray (>=0.70)：面板正在关闭 / 被压暗的过渡帧。
# 19:48:14 实测面板关闭动画中 `btn_manage` 只有 0.909，旧代码只认 0.94，
# 于是 `BackToHome` / `ClickAdventure` 把它当成「已经回到主页面」，
# 不再点右上角退出锚点；接着 `重进-开菜单` 在秘境面板上点「家園」只切
# tab，菜单永远不展开，重进链 3 次后失败（§3.66）。
# 0.70 的依据：87 张 on_error 实机帧里，主页面/列表页最高 0.433；
# 資源詳情浮窗（`btn_dispatch` 可见）历史测量 0.658（§4.3c）；
# 面板过渡帧最低 0.771（19:48:19 的 `回主界面`）。0.70 落在
# 0.658 与 0.771 之间：不会把浮窗误判成面板，也接得住过渡帧；
# 0.75 离 0.771 只有 0.021 余量，动画再低一点就会漏判。
PANEL_HIT_THRESHOLD = 0.94
PANEL_GRAY_THRESHOLD = 0.70


def _panel_state(img, roi):
    """返回 (严格命中, 灰区, 分数)。灰区也算「面板还在」。"""
    hit, score = _match_template(
        img, "btn_manage.png", roi, PANEL_HIT_THRESHOLD)
    return hit, (score >= PANEL_GRAY_THRESHOLD), score


def _match_box(img, name, roi, threshold):
    """在 ROI 里匹配模板，返回 (命中, 分数, 框[绝对坐标])。"""
    tpl = _load_template(name)
    if tpl is None or img is None:
        return False, 0.0, None
    x, y, w, h = [int(v) for v in roi]
    sub = img[y:y + h, x:x + w]
    if sub.shape[0] < tpl.shape[0] or sub.shape[1] < tpl.shape[1]:
        return False, 0.0, None
    r = cv2.matchTemplate(sub, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, ml = cv2.minMaxLoc(r)
    if mx < threshold:
        return False, round(float(mx), 3), None
    return True, round(float(mx), 3), (
        x + ml[0], y + ml[1], tpl.shape[1], tpl.shape[0])


def _refresh_roi(param_roi=None):
    """「刷新」按钮的搜索区域。优先 layout 推导值，退回基线比例。"""
    if param_roi:
        return [int(v) for v in param_roi]
    try:
        import layout
        got = layout.get("refresh_roi")
        if got:
            return [int(v) for v in got]
    except Exception:  # noqa: BLE001
        pass
    return [216, 1069, 302, 110]


def refresh_state(img, roi=None):
    """判断「刷新」按钮当前是否可点。

    为什么不用「橙模板 / 灰模板谁分高」：两张模板形状一致，互相都会给出
    0.88 的分数（`720_fullscreen_activity` 上灰模板 0.880），谁高谁低
    取决于截图压缩噪声，不够稳。

    改成分两步：
      1. **定位**：用灰模板（形状中性）在 ROI 里多尺度找按钮，拿框；
      2. **量色**：在框内缩 12% 的区域量 HSV 饱和度 S 和 R-B 通道差。

    实测（11-15，720x1560）：
        可点（橙）S=124.5  R-B=113.5
        不可点（灰）S=  0.0  R-B=  0.0
        秘境面板/主页面等非列表帧：定位分数 0.28~0.69，直接判「没有按钮」

    ROI 内定位分数不到 `REFRESH_SCORE` 时返回 `None`（= 当前不在探寻列表），
    调用方据此区分「不可点」和「根本不在列表」。

    返回 dict：{"found": bool, "clickable": bool, "score": float,
                "S": float, "RB": float, "box": (x,y,w,h) | None}
    """
    tpl = _load_template("refresh_off.png")
    if tpl is None or img is None:
        return {"found": False, "clickable": False, "score": 0.0,
                "S": 0.0, "RB": 0.0, "box": None}
    x, y, w, h = _refresh_roi(roi)
    sub = img[y:y + h, x:x + w]
    if sub.size == 0:
        return {"found": False, "clickable": False, "score": 0.0,
                "S": 0.0, "RB": 0.0, "box": None}

    # 按钮的**像素尺寸不随长边变**，只有位置变（§3.43：btn_manage 在
    # 720x1560 / 720x1600 上都是 120x121）。所以模板固定按 1.0 附近扫，
    # 不能乘 base_s：11-17 在 720x1600 上乘 1.026 后，匹配分从 0.979
    # 掉到 0.694，`RefreshGate` 全程 found=False，刷新一次都没触发。
    best_score, best_box = 0.0, None
    for sc in (0.94, 1.0, 1.06):
        tw = int(tpl.shape[1] * sc)
        th = int(tpl.shape[0] * sc)
        if tw < 20 or th < 10 or tw > sub.shape[1] or th > sub.shape[0]:
            continue
        scaled = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)
        r = cv2.matchTemplate(sub, scaled, cv2.TM_CCOEFF_NORMED)
        _mn, mx, _ml, loc = cv2.minMaxLoc(r)
        if mx > best_score:
            best_score = float(mx)
            best_box = (x + loc[0], y + loc[1], tw, th)

    if best_box is None or best_score < REFRESH_SCORE:
        return {"found": False, "clickable": False,
                "score": round(best_score, 3), "S": 0.0, "RB": 0.0,
                "box": best_box}

    bx, by, bw, bh = best_box
    pad_x, pad_y = int(bw * 0.12), int(bh * 0.12)
    patch = img[by + pad_y:by + bh - pad_y, bx + pad_x:bx + bw - pad_x]
    if patch.size == 0:
        return {"found": True, "clickable": False,
                "score": round(best_score, 3), "S": 0.0, "RB": 0.0,
                "box": best_box}
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sat = float(hsv[:, :, 1].mean())
    rb = float((patch[:, :, 2].astype(np.int16)
                - patch[:, :, 0].astype(np.int16)).mean())
    clickable = sat >= REFRESH_SAT and rb >= REFRESH_RB
    return {"found": True, "clickable": clickable,
            "score": round(best_score, 3), "S": round(sat, 1),
            "RB": round(rb, 1), "box": best_box}

_STATE = {"gray": None, "ts": 0.0, "loops": 0, "plus": 0, "swipes": 0,
          "stuck": False, "stock": None, "total": None, "target": 3,
          "realm": None,
          "qty": None, "tanxun_points": {}, "need_wait": False,
          "popup_at_swipe": None, "recover": 0,
          # 「刷新」按钮：本趟已点次数 + 上次判定（诊断用）
          "refresh": 0, "refresh_state": None,
          # 滑动前行锚点（「前往」按钮的 y 列表）。用来判「列表有没有真的动」，
          # 比灰度帧差可靠得多（§3.51）。
          "rows_before": None,
          # 本轮重进里已点过几次「家園」。`ClickAdventure` 成功即清零，
          # 所以是「每轮重进」的配额而不是整趟任务的（§3.53）。
          "reopen": 0,
          # 上一次 `ClickReset` 是否真的把「掠奪」派出去了（浮窗已关闭）。
          # `掠夺后分支` 用它决定要不要 MarkRowDone（§3.61 修法 4）。
          "last_dispatch_ok": None}


def reset_state():
    _STATE.update({"gray": None, "ts": 0.0, "loops": 0, "plus": 0, "swipes": 0,
                   "stuck": False, "stock": None, "total": None, "target": 3,
                   "realm": None, "qty": None, "tanxun_points": {},
                   "need_wait": False, "popup_at_swipe": None, "recover": 0,
                   "refresh": 0, "refresh_state": None, "rows_before": None,
                   "reopen": 0, "last_dispatch_ok": None})


def _gray(img, roi):
    x, y, w, h = [int(v) for v in roi]
    sub = img[y:y + h, x:x + w]
    if sub.size == 0:
        return None
    return cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)


def remember(img, roi):
    """记下「滑动前」的帧，供下一次 compare 用。"""
    gray = _gray(img, roi)
    if gray is None:
        return False
    _STATE["gray"] = gray
    _STATE["ts"] = time.time()
    return True


def compare(img, roi, threshold=DIFF_THRESHOLD):
    """和记住的那一帧比。返回 (moved, diff, first)。"""
    gray = _gray(img, roi)
    if gray is None:
        return True, None, True
    prev = _STATE["gray"]
    now = time.time()
    first = (prev is None or prev.shape != gray.shape
             or now - _STATE["ts"] > RESET_SECONDS)
    _STATE["gray"] = gray
    _STATE["ts"] = now
    if first:
        # 没有基准：不能凭空说划不动，判「动了」让流程继续
        return True, None, True
    diff = float(np.mean(cv2.absdiff(prev, gray)))
    return diff >= threshold, round(diff, 2), False


def page_dy(img, roi):
    """按当前帧算「滑过所有完整行」的距离。算不出来退回 480。

    以前用 `last_icon_y - first_icon_y`，只前进 (N-1) 个行距，
    末行会留在新页顶部。现在用完整行跨度：

        首行 btn_y - ROW_TOP_PAD  →  末行 btn_y + ROW_BOTTOM_PAD

    即首行顶到末行底，滑完这一页的完整行全部越出。
    """
    try:
        import row_reco
    except Exception:  # noqa: BLE001
        return 480, []

    rows = row_reco.find_rows(img)
    btn_ys = []
    readable = []
    for btn_y, _x, _w, _h in rows:
        ok, _s60 = row_reco.RowPicker._readable(img, btn_y)
        if ok:
            btn_ys.append(btn_y)
            readable.append(btn_y + row_reco.ICON_DY)
    if len(btn_ys) < 2:
        return 480, readable
    dy = int((btn_ys[-1] + row_reco.ROW_BOTTOM_PAD)
             - (btn_ys[0] - row_reco.ROW_TOP_PAD))
    # 太小说明只识别到一行（分区标题把行截断了），给一个下限
    if dy < 120:
        dy = 480
    return dy, readable


def _list_rows(img):
    """当前帧是否像探寻列表（有「前往」行）。供退出前判断页面状态。"""
    try:
        import row_reco
        return row_reco.find_rows(img)
    except Exception:  # noqa: BLE001
        return []


def rows_moved(img, before):
    """列表真的滚动了没：比「前往」按钮的 y 序列。

    为什么不用灰度帧差（`compare()`）：列表页底部有循环动画 + 数字角标，
    即使一像素都没滚，整帧灰度差也能到 20~30，远高于阈值 4.0。
    11-15 14:47 实跑：`icon_y` 连续 3 次都是 `[514,969]`（纹丝不动），
    `SwipeGate` 的 diff 却是 21.33 / 32.15 / 21.48 —— 闸门一直放行，
    连滑 4 次一次都没走，空转到用户 Ctrl+C（§3.51）。

    「前往」按钮是行锚点（`row_reco.find_rows` 用颜色 + 尺寸双重约束，
    非列表界面误检率很低）。行锚点 y 序列变了 = 真的滚了。

    只比 y 坐标（`find_rows` 返回 (y, x, w, h)，w/h 会因抗锯齿抖动 1px）。

    返回 (moved, before_rows, now_rows)，三者都是 int 列表。
      - `before` 为空（第一次滑 / 刚换页）→ moved=True，别凭空判死；
      - 两次都识别不到行 → moved=True（可能是过渡帧，交给下一轮再判）。
    """
    # 只比 y 坐标（`find_rows` 返回 (y, x, w, h) 元组，宽度可能因抗锯齿抖动）
    now = [int(r[0]) for r in _list_rows(img)]
    if not before:
        return True, before, now
    if not now:
        return True, before, now
    return (list(before) != list(now)), before, now


try:
    from maa.context import Context
    from maa.custom_action import CustomAction
    from maa.custom_recognition import CustomRecognition
    from maa.pipeline import JRecognitionType, JOCR

    def _ocr_texts(context: Context, image, roi, want=None) -> tuple[str, list]:
        """在给定 ROI 上跑一次 Maa OCR，返回拼接文本和每个结果的框。

        `want(text)` 可选：只保留满足条件的文字/框，避免把 ROI 里的杂字
        （例如「探尋」旁边的数字角标）当成按钮框。
        """
        detail = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(roi=[int(v) for v in roi], expected=[]),
            image,
        )
        if detail is None:
            return "", []
        texts = []
        boxes = []
        for rr in (getattr(detail, "all_results", None) or []):
            text = getattr(rr, "text", None)
            if text and (want is None or want(str(text))):
                texts.append(str(text))
                box = getattr(rr, "box", None)
                if box is not None:
                    boxes.append(box)
        return "".join(texts), boxes

    def _box_tuple(box) -> tuple[int, int, int, int]:
        """OCRResult.box 在绑定层是 list；Replay/其他实现可能是 Rect。"""
        if isinstance(box, (list, tuple)) and len(box) == 4:
            return tuple(int(v) for v in box)
        return (int(box.x), int(box.y), int(box.w), int(box.h))

    class ChickenStock(CustomRecognition):
        """读秘境面板「管理」两侧的鸡妖数。

        实测（720 画布）：
          720_realm_panel       閒【0】 @(213,1084,111,35)
          720_fullscreen_others_home 閒【1】 @(213,1083,111,36)
          720_dispatch_menu     閒【8】 @(213,1081,90,41)
          720_realm_panel        【8】總 @(431,1084,84,37)

        左 `閒【N】` = 当前空闲 `stock`；右 `總【M】` = 总数 `total`。
        `total` 每次进入秘境时读一次并缓存；`stock` 每次读。

        stock>0：成功；具体派遣数由 DispatchPolicy 决定；
        stock=0：失败，由调用方的第二候选落到 `没有空闲鸡妖`
              → `退到主页面` → `回主界面` 等待回补（§3.40）。
        OCR 读不到时也失败，不猜。
        """

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            param = json.loads(argv.custom_recognition_param or "{}") or {}
            node_roi = getattr(argv, "roi", None)
            layout_roi = None
            try:
                import layout
                layout_roi = layout.get("idle_roi")
            except Exception:  # noqa: BLE001
                layout_roi = None
            if layout_roi:
                roi = list(layout_roi)
            elif node_roi is not None and node_roi.w > 0 and node_roi.h > 0:
                roi = [node_roi.x, node_roi.y, node_roi.w, node_roi.h]
            else:
                roi = param.get("roi") or [205, 1065, 120, 80]
            # 弹窗会盖住「閒【N】」那一行，先清掉再 OCR（§3.65）。
            argv.image = clear_popups(context, argv.image)
            text, boxes = _ocr_texts(context, argv.image, roi)
            m = re.search(r"(\d+)", text)
            stock = int(m.group(1)) if m else None

            total = _STATE.get("total")
            total_text = ""
            if total is None:
                try:
                    import layout
                    total_roi = (param.get("total_roi")
                                 or layout.get("total_roi")
                                 or [420, 1065, 120, 80])
                except Exception:  # noqa: BLE001
                    total_roi = param.get("total_roi") or [420, 1065, 120, 80]
                total_text, _ = _ocr_texts(context, argv.image, total_roi)
                mt = re.search(r"(\d+)", total_text)
                total = int(mt.group(1)) if mt else None
                _STATE["total"] = total

            _STATE["stock"] = stock
            _STATE["target"] = min(3, stock) if stock and stock > 0 else 3
            detail = {"text": text, "stock": stock,
                      "total_text": total_text,
                      "total": total,
                      "target": _STATE["target"], "roi": roi}
            print("[ChickenStock] %s" % detail, flush=True)
            if stock is None or stock <= 0:
                return CustomRecognition.AnalyzeResult(box=None, detail=detail)
            _STATE["need_wait"] = False
            box = _box_tuple(boxes[0]) if boxes else tuple(roi)
            return CustomRecognition.AnalyzeResult(
                box=box, detail=detail)

    class DispatchCountReached(CustomRecognition):
        """读浮窗里的 `雞妖数量 n/3`，达到 DispatchPolicy 的 target 才成功。"""

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            param = json.loads(argv.custom_recognition_param or "{}") or {}
            node_roi = getattr(argv, "roi", None)
            if node_roi is not None and node_roi.w > 0 and node_roi.h > 0:
                roi = [node_roi.x, node_roi.y, node_roi.w, node_roi.h]
            else:
                roi = param.get("roi") or [395, 790, 165, 45]
            target = int(_STATE.get("target") or 3)
            # target=1 时浮窗默认就是 1/3，不用再 OCR `n/3`（§3.66）。
            # 需求方 11-18：只派一只的场次直接去 `掠奪`，省掉每格
            # 0.8~1.0s 的 OCR。target>1 仍走原来的读计数逻辑。
            if target == 1:
                detail = {"text": "", "n": None, "target": target,
                          "reached": True, "skip": True, "roi": roi}
                print("[DispatchCountReached] %s" % detail, flush=True)
                return CustomRecognition.AnalyzeResult(
                    box=tuple(roi), detail=detail)
            # 浮窗也可能被「恭喜獲得」盖住，先清弹窗再读 n/3（§3.65）。
            argv.image = clear_popups(context, argv.image)
            text, boxes = _ocr_texts(context, argv.image, roi)
            m = re.search(r"(\d+)\s*/\s*3", text)
            n = int(m.group(1)) if m else None
            reached = n is not None and n >= target
            detail = {"text": text, "n": n, "target": target,
                      "reached": reached, "roi": roi}
            print("[DispatchCountReached] %s" % detail, flush=True)
            if not reached:
                return CustomRecognition.AnalyzeResult(box=None, detail=detail)
            box = _box_tuple(boxes[0]) if boxes else tuple(roi)
            return CustomRecognition.AnalyzeResult(
                box=box, detail=detail)

    class SetRealm(CustomAction):
        """进入自家/别人秘境时记录 realm，并清空上一轮的延迟格。

        `tanxun_points` 按 realm 分开缓存，不在这里清空：
        别人秘境共用同一个「探尋」位置；自家秘境虽然可能因秘境管家有无而
        左右移动，但点击时会用当前 OCR 框校验缓存点，失效后自动重测。
        """

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            param = json.loads(argv.custom_action_param or "{}") or {}
            realm = str(param.get("realm", "own"))
            _STATE["realm"] = realm
            _STATE["qty"] = None
            # 总数在每次进入秘境时重新读；同一秘境内的循环复用缓存。
            _STATE["total"] = None
            # 刷新配额按「每次进入秘境」重置：游戏自己的刷新 CD 才是真上限，
            # 这个计数只是防同一轮里在刷新按钮上空转。
            _STATE["refresh"] = 0
            try:
                import cell_reco
                cell_reco.clear_deferred()
            except Exception:  # noqa: BLE001
                pass
            cached = (_STATE.get("tanxun_points") or {}).get(realm)
            print("[SetRealm] realm=%s tanxun_cache=%s" % (realm, cached),
                  flush=True)
            # 第一次进入任意秘境面板时测量本设备布局，并改写本次任务的 ROI。
            try:
                import layout
                layout.ensure(context, getattr(argv, "image", None))
            except Exception as exc:  # noqa: BLE001
                print("[SetRealm] layout calibration failed: %s" % exc,
                      flush=True)
            return True

    class MarkWait(CustomAction):
        """空闲鸡妖为 0：标记需要等待回补，然后走回主页/重进链。"""

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            _STATE["need_wait"] = True
            print("[MarkWait] need_wait=True", flush=True)
            return True

    class WaitReturnGate(CustomRecognition):
        """只有 need_wait 时才允许进入等待弹窗，普通重进直接去读库存。"""

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            ok = bool(_STATE.get("need_wait"))
            detail = {"need_wait": ok}
            print("[WaitReturnGate] %s" % detail, flush=True)
            if not ok:
                return CustomRecognition.AnalyzeResult(box=None, detail=detail)
            return CustomRecognition.AnalyzeResult(box=(0, 0, 1, 1), detail=detail)

    class ReopenGate(CustomRecognition):
        """命中 = 本轮重进还可以再点一次「家園」展开菜单。

        §3.53：原来靠 pipeline 的 `max_hit: 3` 当刹车，但那是**整趟任务
        共享**的全局配额，长任务重进 4 次就用光，之后家園再也点不开、
        任务无声收尾（11-15 15:25 那轮 583s 的 `重进失败` 就是这个）。

        改成动作侧计数：`ClickAdventure` 每成功一次（= 真的回到了主页面）
        就把 `reopen` 清零，所以配额是**每轮重进**的，不是全局的。

        为什么还要上限：`家園` 是 toggle，如果识别到按钮但菜单始终不展开
        （界面卡死 / 被别的浮层挡住），没有上限会无限点。3 次够正常重试。
        """

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            param = json.loads(argv.custom_recognition_param or "{}") or {}
            limit = int(param.get("max_reopen", MAX_REOPEN))
            used = int(_STATE.get("reopen") or 0)
            # `used` 是**已经重试过**的次数（不含第一次点击）。
            # `max_reopen` 语义 = 重试次数上限，所以总点击数 = 1 + limit。
            ok = used < limit
            detail = {"used": used, "max_reopen": limit, "ok": ok}
            print("[ReopenGate] %s" % detail, flush=True)
            if not ok:
                return CustomRecognition.AnalyzeResult(box=None, detail=detail)
            return CustomRecognition.AnalyzeResult(box=(0, 0, 1, 1),
                                                   detail=detail)

    class CountReopen(CustomAction):
        """点「家園」之前把本轮的 reopen 计数 +1。

        为什么单独一个动作而不是塞进 `重进-开菜单`：那个节点是 Maa 原生
        `Click`，没法跑 Python。所以在它前面插一个 DirectHit + Custom。
        """

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            _STATE["reopen"] = int(_STATE.get("reopen") or 0) + 1
            print("[CountReopen] reopen=%d" % _STATE["reopen"], flush=True)
            return True

    class ConfirmDispatch(CustomRecognition):
        """「确认资源详情」：先清「恭喜獲得」弹窗，再找派遣浮窗按钮。

        弹窗会在两种时机盖住屏幕：上一轮派遣的资源收完了（奖励弹窗），
        或者点的格子刚好被别人抢走。pipeline 里原来只有纯 TemplateMatch，
        弹窗一盖就判失败，主链直接断（11-20 17:52 实机：点了格子 →
        恭喜獲得 玉璧 弹窗 → 确认资源详情 连续 4 次失败 → 主循环选格 失败）。

        这里复用 PickCell 用的 clear_popups：点面板外、点完复查、最多 50 条。
        没弹窗时只多一次模板匹配（约 6~20ms）。
        """

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            param = json.loads(argv.custom_recognition_param or "{}") or {}
            node_roi = getattr(argv, "roi", None)
            if node_roi is not None and node_roi.w > 0 and node_roi.h > 0:
                roi = [node_roi.x, node_roi.y, node_roi.w, node_roi.h]
            else:
                roi = param.get("roi") or [200, 930, 320, 160]
            threshold = float(param.get("threshold", 0.94))

            img = clear_popups(context, argv.image)
            detail = context.run_recognition("__confirm_dispatch", img, {
                "__confirm_dispatch": {
                    "recognition": "TemplateMatch",
                    "template": "btn_dispatch.png",
                    "roi": [int(v) for v in roi],
                    "threshold": threshold,
                },
            })
            if detail is None or not detail.hit:
                print("[ConfirmDispatch] btn_dispatch not found roi=%s threshold=%.2f"
                      % (roi, threshold), flush=True)
                return CustomRecognition.AnalyzeResult(
                    box=None, detail={"hit": False, "roi": roi})

            b = detail.best_result.box
            box = (int(b.x), int(b.y), int(b.w), int(b.h))
            print("[ConfirmDispatch] hit box=%s" % (list(box),), flush=True)
            return CustomRecognition.AnalyzeResult(
                box=box, detail={"hit": True, "box": list(box)})

    class RealmOtherGate(CustomRecognition):
        """命中 = 当前是别人秘境；用于掠奪后直接回列表找下一家。"""

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            ok = _STATE.get("realm") == "other"
            detail = {"realm": _STATE.get("realm"), "other": ok}
            print("[RealmOtherGate] %s" % detail, flush=True)
            if not ok:
                return CustomRecognition.AnalyzeResult(box=None, detail=detail)
            return CustomRecognition.AnalyzeResult(box=(0, 0, 1, 1), detail=detail)

    class DispatchOkGate(CustomRecognition):
        """命中 = 上一次 `ClickReset` 真的派出去了（浮窗已关闭）。

        `ClickReset` 判 blocked=True（点完「掠奪」浮窗还在 = 被抢/点击丢失）
        时只把格子记为延迟，没有产生派遣。pipeline 不能在这种情况下把列表行
        标成「已访问」，否则这一行要等 330s 才会再被考虑（§3.61 修法 4）。
        """

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            value = _STATE.get("last_dispatch_ok")
            ok = value is True
            detail = {"last_dispatch_ok": value, "ok": ok}
            print("[DispatchOkGate] %s" % detail, flush=True)
            if not ok:
                return CustomRecognition.AnalyzeResult(box=None, detail=detail)
            return CustomRecognition.AnalyzeResult(box=(0, 0, 1, 1), detail=detail)

    class RefreshGate(CustomRecognition):
        """命中 = 探寻列表的「刷新」按钮**可点**，且本趟还没点满。

        需求方 11-15：列表划不动、又还有空闲鸡妖时，先试着刷新列表
        （刷新只重刷最上面 3 个秘境），刷新也点不动（带倒计时）才退出
        主页面重进秘境。

        判据见 `refresh_state()`：先用灰模板定位按钮，再量饱和度。
        可点（橙）S≈124、R-B≈113；不可点（灰）S≈0、R-B≈0。

        与 `CooldownGate` 的分工：

          - `CooldownGate` 管**单个秘境**的时间冷却（进过/空/延迟的行）；
          - `RefreshGate` 管**整个列表**的重刷次数。
          两者的触发条件互斥：冷却闸门只看「有没有行在读」，刷新闸门
          只看「按钮颜色」。

        配额 `max_refresh` 是**整个任务共享**的（和 `max_hit` 一样），
        每成功点一次 +1；用满后闸门恒不命中，直接落到「不可点」分支
        去重进秘境，避免在刷新按钮上空转。
        """

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            param = json.loads(argv.custom_recognition_param or "{}") or {}
            limit = int(param.get("max_refresh", MAX_REFRESH))
            used = int(_STATE.get("refresh") or 0)
            stock = _STATE.get("stock")
            roi = param.get("roi")
            st = refresh_state(argv.image, roi)
            _STATE["refresh_state"] = st
            # 「没有空闲鸡妖」时不刷新：刷出来也派不出去，直接重进等回补。
            has_chicken = isinstance(stock, int) and stock > 0
            detail = dict(st, used=used, max_refresh=limit, stock=stock,
                          has_chicken=has_chicken)
            detail.pop("box", None)
            ok = bool(st["found"] and st["clickable"]
                      and used < limit and has_chicken)
            print("[RefreshGate] %s used=%d/%d has_chicken=%s"
                  % (detail, used, limit, has_chicken), flush=True)
            if not ok:
                return CustomRecognition.AnalyzeResult(box=None, detail=detail)
            return CustomRecognition.AnalyzeResult(box=(0, 0, 1, 1),
                                                   detail=detail)

    class ClickRefresh(CustomAction):
        """点一次「刷新」，并把配额 +1。

        为什么自己重新截图定位，而不是用 `argv.box`：`RefreshGate` 返回的
        是 (0,0,1,1) 的哨兵框（和 `LoopGuard` 等闸门一致），真正的按钮
        位置在那之后可能已经变了（列表动画 / 弹窗）。这里再测一次，
        测不到就返回 False，让调用方回落到「不可点」分支。

        点完**不校验**列表是否真的变了：刷新只重刷最上面 3 个秘境，
        下面的行原样保留，用「有没有变化」判成败会误判。真正的验证是
        下一轮 `主循环选行` 能不能选到新行。
        """

        SETTLE = 1.5

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            param = json.loads(argv.custom_action_param or "{}") or {}
            ctrl = context.tasker.controller
            img = ctrl.post_screencap().wait().get()
            st = refresh_state(img, param.get("roi"))
            _STATE["refresh_state"] = st
            if not (st["found"] and st["clickable"]):
                print("[ClickRefresh] not clickable %s" % st, flush=True)
                return False
            bx, by, bw, bh = st["box"]
            x, y = bx + bw // 2, by + bh // 2
            ctrl.post_click(x, y).wait()
            _STATE["refresh"] = int(_STATE.get("refresh") or 0) + 1
            print("[ClickRefresh] click (%d,%d) score=%.3f S=%.1f RB=%.1f "
                  "used=%d" % (x, y, st["score"], st["S"], st["RB"],
                               _STATE["refresh"]), flush=True)
            time.sleep(self.SETTLE)
            return True

    class AbortTask(CustomAction):
        """收尾节点：**主动让任务判失败**，并弹一条提示。

        需求方 11-15：`列表已到底` 现在要判失败（红色），不能静默绿着结束。

        为什么不用 `StopTask`：实测 `StopTask` 之后 `task.succeeded == True`
        （§3.47 的实验），会静默成功、掩盖异常。

        为什么不用「TemplateMatch 必不中」：那要靠一张不存在的模板 +
        识别超时窗口，节点会反复重试到 timeout（默认 20s）才失败，
        收尾白白拖 20s。

        这里直接返回 False 且**不写 on_error**，Maa 立刻把任务判失败。

        附带 `focus`：失败时弹 toast（`Node.PipelineNode.Failed`）。
        focus 挂在 pipeline 节点上，Maa 会按 detail.focus 透传给回调；
        MFAAvalonia 侧负责真正显示。
        """

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            param = json.loads(argv.custom_action_param or "{}") or {}
            reason = str(param.get("reason") or "任务提前结束")
            print("[AbortTask] %s -> 判失败（succeeded=False）" % reason,
                  flush=True)
            return False

    class CloseList(CustomAction):
        """只关掉探寻列表，**不**继续退秘境面板。

        为什么不能用 `ClickAdventure`：那个动作的目标是「回到冒险主页面」，
        会一直点蓝点直到 `btn_manage` 也消失。这里只想关掉列表回到秘境
        面板，好重新点「探尋」从列表顶部再看一遍（刷新只重刷最上面 3 个，
        当前滚动位置看不到它们）。

        **判据不能只看 rows**：秘境面板上 `row_reco.find_rows` 会误检
        4 行（11-15 实跑 `[BackToHome] state #1 panel=False(0.234) rows=4`）。
        所以「已关」= 命中 `btn_manage.png`（在秘境面板上）**或** 一行都没有。

        真正的「还在列表」= 有行 **且** 没有管理按钮 —— 与 `ClickAdventure`
        / `ExitOtherRealm` 用的是同一个判据。
        """

        GAP = 1.2
        MAX_TRIES = 3

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            ctrl = context.tasker.controller
            try:
                import layout
                exit_pt = layout.exit_point()
                manage_roi = layout.manage_roi([250, 950, 250, 250])
            except Exception:  # noqa: BLE001
                exit_pt = (674, 256)
                manage_roi = [250, 950, 250, 250]

            for i in range(1, self.MAX_TRIES + 1):
                img = ctrl.post_screencap().wait().get()
                rows = _list_rows(img)
                panel, panel_gray, panel_score = _panel_state(img, manage_roi)
                in_list = bool(rows) and not panel and not panel_gray
                print("[CloseList] state #%d rows=%d panel=%s(%.3f) "
                      "gray=%s in_list=%s"
                      % (i, len(rows), panel, panel_score, panel_gray,
                         in_list),
                      flush=True)
                if not in_list:
                    print("[CloseList] list closed after %d clicks" % (i - 1),
                          flush=True)
                    return True
                ctrl.post_click(exit_pt[0], exit_pt[1]).wait()
                print("[CloseList] click exit (%d,%d) #%d rows=%d"
                      % (exit_pt[0], exit_pt[1], i, len(rows)), flush=True)
                time.sleep(self.GAP)

            img = ctrl.post_screencap().wait().get()
            rows = _list_rows(img)
            panel, panel_gray, panel_score = _panel_state(img, manage_roi)
            ok = not (rows and not panel and not panel_gray)
            print("[CloseList] after tries rows=%d panel=%s(%.3f) "
                  "gray=%s ok=%s"
                  % (len(rows), panel, panel_score, panel_gray, ok),
                  flush=True)
            return ok

    class CooldownGate(CustomRecognition):
        """命中 = 当前没有可读行，但还有秘境在时间冷却中。"""

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            try:
                import row_reco
                wait = float(getattr(row_reco, "NEXT_READY", 0.0) or 0.0)
            except Exception:  # noqa: BLE001
                wait = 0.0
            ok = wait > 0
            detail = {"next_ready": round(wait, 1), "ok": ok}
            print("[CooldownGate] %s" % detail, flush=True)
            if not ok:
                return CustomRecognition.AnalyzeResult(box=None, detail=detail)
            return CustomRecognition.AnalyzeResult(box=(0, 0, 1, 1), detail=detail)

    class WaitCooldown(CustomAction):
        """全页都在冷却：等到最早一条到期，再回主页面/重进。"""

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            try:
                import row_reco
                wait = float(row_reco.cooldown_earliest() or 0.0)
            except Exception:  # noqa: BLE001
                wait = 0.0
            if wait <= 0:
                print("[WaitCooldown] cooldown already expired", flush=True)
                return True
            sleep_s = min(30.0, max(1.0, wait))
            print("[WaitCooldown] wait %.1fs (next_ready=%.1fs)"
                  % (sleep_s, wait), flush=True)
            time.sleep(sleep_s)
            return True

    class WaitPopup(CustomAction):
        """等待“恭喜獲得”弹窗事件。

        CLI 走 `PopupGuardController.popup_clicks`：每次 screencap 若清掉弹窗，
        计数会 +1，本动作看到计数变化就返回成功。
        GUI 没有护栏，直接在新截图上匹配 `obtain_resource.png`，命中后返回成功，
        由后续 `清弹窗后继续` 点掉弹窗。
        """

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            param = json.loads(argv.custom_action_param or "{}") or {}
            # 默认从 180s 降到 30s（§3.66）：库存为 0 且没有派兵在途时，
            # 等 120s 也不会凭空出现「恭喜獲得」，只会让 GUI 看起来卡死。
            # 30s 足够覆盖「刚好有一条派遣完成」的情况；超时后
            # `等待获得弹窗.on_error` 会回到落地校验重新读库存。
            timeout = float(param.get("timeout", 30))
            interval = float(param.get("interval", 2.0))
            ctrl = context.tasker.controller
            start = getattr(ctrl, "popup_clicks", None)
            deadline = time.time() + timeout
            attempts = 0
            while time.time() < deadline:
                attempts += 1
                img = ctrl.post_screencap().wait().get()
                if start is not None:
                    now = getattr(ctrl, "popup_clicks", 0)
                    if now > start:
                        print("[WaitPopup] popup event clicks %d->%d"
                              % (start, now), flush=True)
                        return True
                else:
                    hit, score = _popup_match(img)
                    if hit:
                        print("[WaitPopup] popup template score=%.3f" % score,
                              flush=True)
                        return True
                time.sleep(max(0.2, interval))
            print("[WaitPopup] timeout %.0fs attempts=%d" % (timeout, attempts),
                  flush=True)
            return False

    class DispatchPolicy(CustomRecognition):
        """按秘境归属/物资等级 + 总数/空闲数决定本次派遣只数。

        自己秘境：默认所有资源 target=1；
          例外：`PickCell` 标了 `contested`（格子上方有倒计时 = 别人在抢）
          的酒米 → target=3，鸡妖不够 3 只时 defer（需求方 11-17）。
        别人秘境：
          - 酒米（alcohol）不论等级 target=3；
          - 普通资源按 `N级`：1→1、2→2、>=3→3；
          - 不使用道具数量（1级枸杞可能是3个、云英可能是350个）；
          - `cap = min(3, 鸡妖总数)`，target 不超过 cap；
          - 模式 full（默认）：target > 当前空闲鸡妖数时 defer，
            先处理鸡妖够用的资源；
          - 模式 utilize：有几只空闲就派几只，target=min(target, idle)，
            不再因为凑不满目标而 defer（鸡妖少时最大化利用率）。
        """

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            param = json.loads(argv.custom_recognition_param or "{}") or {}
            mode = str(param.get("dispatch_mode", "full"))
            try:
                import cell_reco
                last = cell_reco.LAST_PICK or {}
                item = last.get("item")
                level = last.get("level")
                level_text = last.get("level_text", "")
            except Exception:  # noqa: BLE001
                item = None
                level = None
                level_text = ""
            realm = _STATE.get("realm")
            stock = _STATE.get("stock")
            total = _STATE.get("total")
            cap = 3
            if isinstance(total, int) and total > 0:
                cap = max(1, min(3, total))
            defer = False
            if realm == "own":
                # 自家秘境默认 1 只；「别人正在抢我」的酒米派 3 只夺回
                # （需求方 11-17）。`contested` 由 `PickCell` 写进 LAST_PICK：
                # 格子上方有倒计时 = 有人在采/抢，不是我自己在采。
                target = (3 if last.get("contested") and item == "alcohol"
                          else 1)
            elif realm == "other" and item == "alcohol":
                target = 3
            elif realm == "other":
                # 按物品等级：1级→1、2级→2、>=3级→3。
                # 等级 OCR 失败时保守按 1 只，不按道具数量（350/450会误判）。
                target = max(1, min(3, int(level))) if level else 1
            else:
                # 调试入口直接跑 `派遣并掠奪` 时没有 realm，保持旧行为 3 只
                target = 3

            if realm in ("own", "other"):
                # 总数是单次派遣的硬上限；total=1/2 时酒米/3级资源也随之降级。
                target = min(target, cap)
                if mode == "utilize" and isinstance(stock, int):
                    if stock <= 0:
                        target = 0
                        defer = True
                    else:
                        target = max(1, min(target, stock))
                        defer = False
                elif isinstance(stock, int) and target > stock:
                    defer = True

            _STATE["target"] = target
            detail = {"realm": realm, "item": item, "level": level,
                      "level_text": level_text,
                      "contested": bool(last.get("contested")),
                      "stock": stock, "total": total, "cap": cap,
                      "mode": mode, "target": target,
                      "defer": defer}
            print("[DispatchPolicy] %s" % detail, flush=True)
            if defer:
                return CustomRecognition.AnalyzeResult(box=None, detail=detail)
            return CustomRecognition.AnalyzeResult(box=(0, 0, 1, 1), detail=detail)

    class DeferCell(CustomAction):
        """数量不足 3 的非酒米：关闭資源詳情，记住该格，回选格。"""

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            item = None
            try:
                import cell_reco
                item = (cell_reco.LAST_PICK or {}).get("item")
                pending = cell_reco.defer_last()
            except Exception:  # noqa: BLE001
                pending = []
            skipped = []
            if _STATE.get("realm") == "other":
                try:
                    import row_reco
                    seconds = (row_reco.ALCOHOL_DEFER_COOLDOWN_SECONDS
                               if item == "alcohol"
                               else row_reco.DEFER_COOLDOWN_SECONDS)
                    skipped = row_reco.skip_last_row(
                        seconds=seconds, reason="defer:%s" % (item or "?"))
                except Exception:  # noqa: BLE001
                    skipped = []
            # 与「关闭获得弹窗」相同的安全落点：面板外 y≈1326
            try:
                import layout
                sx, sy = layout.safe_point()
            except Exception:  # noqa: BLE001
                sx, sy = 350, 1326
            context.tasker.controller.post_click(sx, sy).wait()
            print("[DeferCell] item=%s deferred=%s cooldown=%s"
                  % (item, pending, skipped), flush=True)
            return True

    class MarkBadRow(CustomAction):
        """别人秘境里没有任何可抢资源：给该行设置约 5.5 分钟冷却。"""

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            try:
                import row_reco
                bad = row_reco.mark_last_row_bad()
            except Exception:  # noqa: BLE001
                bad = []
            print("[MarkBadRow] cooldown_entries=%d" % len(bad), flush=True)
            return True

    class MarkRowDone(CustomAction):
        """刚成功派遣过的列表行：设置 5.5 分钟冷却，等资源刷新后再考虑。"""

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            try:
                import row_reco
                done = row_reco.mark_last_row_done()
            except Exception:  # noqa: BLE001
                done = []
            print("[MarkRowDone] cooldown_entries=%d" % len(done), flush=True)
            return True

    class SwipeRows(CustomAction):
        """滑动一页：运行时算 dy，先记帧再滑。只负责「滑」，不判划不动。

        判据放在 `SwipeGate` 里（它在本节点之后的下一轮被调用），
        这样 pipeline 里不会出现 on_error 自引用——实测那会无限重跑（§3.21）。
        返回 False 只有一种情况：滑动预算用完。
        """

        @staticmethod
        def _grab_without_popup(context, ctrl=None):
            """GUI 没有 PopupGuardController 时，滑动前先把弹窗清干净。

            **必须把已经取到的 ctrl 传进来**：Maa 5.8.0 的 agent 模式下
            `RemoteTasker::controller()` 每次调用都会重建 RemoteController，
            旧指针立即失效。这里若让 `clear_popups` 自己再取一次，
            调用方手里的 ctrl 就变成悬垂指针，`post_swipe` 会在 C++ 里
            access violation（§3.66，19:46~20:03 实测 9 次）。
            """
            return clear_popups(context, ctrl=ctrl)

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            param = json.loads(argv.custom_action_param or "{}") or {}
            try:
                import layout
                roi = param.get("roi") or layout.get("list_roi") or DEFAULT_ROI
            except Exception:  # noqa: BLE001
                roi = param.get("roi") or DEFAULT_ROI
            if _STATE["swipes"] > int(param.get("max_swipes", MAX_SWIPES)):
                print("[SwipeRows] swipe budget %d exhausted"
                      % int(param.get("max_swipes", MAX_SWIPES)), flush=True)
                _STATE["stuck"] = True
                return False
            ctrl = context.tasker.controller
            img = self._grab_without_popup(context, ctrl)
            if img is None:
                return False
            dy, readable = page_dy(img, roi)
            # 记下「滑动前」的行锚点，供 `SwipeGate` 判有没有真的滚（§3.51）
            _STATE["rows_before"] = [int(r[0]) for r in _list_rows(img)]
            remember(img, roi)
            _STATE["popup_at_swipe"] = getattr(ctrl, "popup_clicks", None)
            _STATE["swipes"] += 1
            try:
                import layout
                y1 = layout.swipe_from_y(SWIPE_FROM_Y)
            except Exception:  # noqa: BLE001
                y1 = SWIPE_FROM_Y
            y2 = max(120, y1 - dy)
            duration = int(param.get("duration", SWIPE_DURATION))
            try:
                ctrl.post_swipe(SWIPE_X, y1, SWIPE_X, y2, duration).wait()
            except OSError as exc:
                # 兜底：即使还有别处把 controller 换掉，重新取一次就能
                # 拿到新的 RemoteController，再试一次（§3.66）。
                print("[SwipeRows] post_swipe crashed: %s -> refresh "
                      "controller" % exc, flush=True)
                time.sleep(0.3)
                ctrl = context.tasker.controller
                try:
                    ctrl.post_swipe(SWIPE_X, y1, SWIPE_X, y2,
                                    duration).wait()
                except OSError as exc2:
                    print("[SwipeRows] post_swipe retry failed: %s"
                          % exc2, flush=True)
                    return False
            print("[SwipeRows] n=%d dy=%d icon_y=%s popup_clicks=%s "
                  "swipe=(%d,%d)->(%d,%d)"
                  % (_STATE["swipes"], dy, readable,
                     _STATE["popup_at_swipe"], SWIPE_X, y1, SWIPE_X, y2),
                  flush=True)
            return True

    class ListMoved(CustomRecognition):
        """命中 = 这一页确实滑动了；不命中 = 划不动了。"""

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            param = json.loads(argv.custom_recognition_param or "{}") or {}
            node_roi = getattr(argv, "roi", None)
            try:
                import layout
                layout_roi = layout.get("list_roi")
            except Exception:  # noqa: BLE001
                layout_roi = None
            if layout_roi:
                roi = list(layout_roi)
            elif node_roi is not None and getattr(node_roi, "w", 0) > 0 and node_roi.h > 0:
                roi = [node_roi.x, node_roi.y, node_roi.w, node_roi.h]
            else:
                roi = param.get("roi") or DEFAULT_ROI
            threshold = float(param.get("threshold", DIFF_THRESHOLD))
            moved, diff, first = compare(argv.image, roi, threshold)
            over_budget = _STATE["swipes"] > MAX_SWIPES
            if over_budget:
                moved = False
            detail = {"moved": moved, "stuck": not moved, "diff": diff,
                      "first": first, "roi": list(roi), "threshold": threshold,
                      "swipes": _STATE["swipes"], "max_swipes": MAX_SWIPES}
            if not moved:
                return CustomRecognition.AnalyzeResult(box=None, detail=detail)
            x, y, w, h = [int(v) for v in roi]
            return CustomRecognition.AnalyzeResult(box=(x, y, w, h), detail=detail)

    class LoopGuard(CustomRecognition):
        """命中 = 还可以再回主界面重来；不命中 = 该收尾了。"""

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            param = json.loads(argv.custom_recognition_param or "{}") or {}
            limit = int(param.get("max_loops", MAX_LOOPS))
            _STATE["loops"] += 1
            ok = _STATE["loops"] <= limit
            detail = {"loops": _STATE["loops"], "max_loops": limit, "ok": ok}
            print("[LoopGuard] loops=%d/%d ok=%s"
                  % (_STATE["loops"], limit, ok), flush=True)
            if not ok:
                return CustomRecognition.AnalyzeResult(box=None, detail=detail)
            return CustomRecognition.AnalyzeResult(box=(0, 0, 1, 1), detail=detail)

    class SwipeGate(CustomRecognition):
        """命中 = 还可以再滑一页；不命中 = 划不动了 / 预算用完。

        **主判据是行锚点**（§3.51）：比「前往」按钮的 y 序列有没有变。
        变了 = 真的滚了；没变 = 划不动。

        为什么不再用灰度帧差：列表页底部有循环动画 + 倒计时角标，
        即使一像素没滚，整帧灰度差也能到 20~30（阈值只有 4.0）。
        11-15 14:47 实跑：`icon_y` 连续 3 次 `[514,969]`（纹丝不动），
        `diff` 却是 21.33 / 32.15 / 21.48 —— 闸门一直放行，连滑 4 次
        一次都没走，空转到用户 Ctrl+C。

        灰度差现在只作为**兜底**：行锚点识别不出来时（过渡帧 / 弹窗遮挡）
        才用它。这样「行读不到但确实动了」不会被误判成划不动。

        为什么单独做成一个识别节点：Maa 的 on_error 里写节点自己会**无条件重跑**
        同一候选（实测刷 240 多次动作失败、一点没往前走），所以循环必须由
        「选行 → 闸门 → 滑 → 选行」这种链式结构表达，不能自引用（§3.21）。
        """

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            param = json.loads(argv.custom_recognition_param or "{}") or {}
            node_roi = getattr(argv, "roi", None)
            if node_roi is not None and getattr(node_roi, "w", 0) > 0 and node_roi.h > 0:
                roi = [node_roi.x, node_roi.y, node_roi.w, node_roi.h]
            else:
                roi = param.get("roi") or DEFAULT_ROI
            threshold = float(param.get("threshold", DIFF_THRESHOLD))
            budget = int(param.get("max_swipes", MAX_SWIPES))

            before = _STATE.get("rows_before")
            # 只取一次 controller：agent 模式下重复取会重建 RemoteController，
            # 让刚清弹窗用的那个指针失效（§3.66）。
            ctrl = context.tasker.controller
            # 弹窗遮挡会让行锚点/帧差都失真，先清弹窗（§3.65）。
            argv.image = clear_popups(context, argv.image, ctrl=ctrl)
            moved, before_rows, now_rows = rows_moved(argv.image, before)
            anchor_used = bool(before) and bool(now_rows)
            diff, first = None, not before
            if not anchor_used:
                # 行锚点不可用（第一次滑 / 过渡帧）→ 退回灰度差
                moved, diff, first = compare(argv.image, roi, threshold)
            else:
                # 仍然更新灰度基准，避免下一次 compare 用到过期帧
                remember(argv.image, roi)
            popup_at = _STATE.get("popup_at_swipe")
            popup_now = getattr(ctrl, "popup_clicks", None)
            popup_event = (popup_at is not None and popup_now is not None
                           and popup_now > popup_at)
            popup_visible, popup_score = _popup_match(argv.image)
            popup = bool(popup_event or popup_visible)
            if popup:
                # 弹窗帧/被弹窗吞掉的那一滑不能判成列表到底：
                # 允许下一轮 SwipeRows 清弹窗后重滑。
                moved = True
                _STATE["stuck"] = False
                if popup_now is not None:
                    _STATE["popup_at_swipe"] = popup_now
            elif not moved:
                _STATE["stuck"] = True
            ok = (not _STATE["stuck"]) and _STATE["swipes"] < budget
            detail = {"swipes": _STATE["swipes"], "max_swipes": budget,
                      "stuck": _STATE["stuck"], "diff": diff, "first": first,
                      "threshold": threshold, "anchor": anchor_used,
                      "before": before_rows, "now": now_rows,
                      "popup": popup,
                      "popup_event": popup_event,
                      "popup_visible": popup_visible,
                      "popup_score": popup_score, "ok": ok}
            print("[SwipeGate] %s" % detail, flush=True)
            if not ok:
                return CustomRecognition.AnalyzeResult(box=None, detail=detail)
            x, y, w, h = [int(v) for v in roi]
            return CustomRecognition.AnalyzeResult(box=(x, y, w, h), detail=detail)

    class DispatchReset(CustomAction):
        """浮窗确认时调用：清零「+」计数，不点任何东西。"""

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            _STATE["plus"] = 0
            stock = _STATE.get("stock")
            _STATE["target"] = (min(3, stock)
                                if isinstance(stock, int) and stock > 0 else 3)
            print("[DispatchReset] plus cleared, stock=%s target=%d"
                  % (stock, _STATE["target"]), flush=True)
            return True

    class AddChicken(CustomAction):
        """把「+」点到 target：先读 `雞妖數量 n/3`，连点补足，再回读复核。

        旧写法是「点一次 + → 回读 n/3 → 没到 target 就再点一次」，而回读
        由 `读派遣数量` 节点做，识别失败时 Maa 会在它的 `timeout` 里反复
        OCR 同一帧（11-17 trace：`reached:False` 连打 3 次，每次白烧
        ~2.5s）。7 次派遣实测 6.15 / 14.72 / 15.17 / 16.11 / 16.96 /
        17.01 / 20.11s，绝大部分耗在这段空转上。

        现在把「读 → 算 → 点 → 复核」收进一个动作里：

          1. 截图，OCR 一次 `雞妖數量 n/3`（ROI 与 `读派遣数量` 同一个）；
          2. `need = target - n`，夹到 `[0, max_plus]`；
          3. 连点 `need` 次，每次间隔 `gap`（实测点击后 0.1s 内计数就变，
             0.5s 是给动画留的余量）；
          4. **回读复核**：点完再读一次。弹窗可能吃掉其中一次点击
             （11-18 21:32 实测：点 2 次只到 2/3，随后直接掠奪），
             所以按实际增量更新 `_STATE["plus"]`，没到 target 就用
             剩余配额补点差额；连续 3 轮没有增量才放弃。

        读不到 n 时退回旧行为（点一次），由 `读派遣数量` 兜底。
        `_STATE["plus"]` 只累计**实际生效**的增量，`PlusGate` 仍可用。
        """

        GAP = 0.5
        VERIFY_GAP = 0.5
        MAX_STALL = 3
        COUNT_ROI = [395, 790, 165, 45]

        def _count_roi(self, param):
            roi = param.get("count_roi")
            if roi:
                return [int(v) for v in roi]
            try:
                import layout
                got = layout.get("count_roi")
                if got:
                    return [int(v) for v in got]
            except Exception:  # noqa: BLE001
                pass
            return list(self.COUNT_ROI)

        def _read_count(self, context, roi, ctrl=None):
            """返回 `n`（读不到返回 None）。先清弹窗再 OCR。"""
            try:
                ctrl = ctrl or context.tasker.controller
                img = ctrl.post_screencap().wait().get()
                # 弹窗盖住计数时 OCR 会读到旧数字/花屏；先清掉（§3.66）。
                img = clear_popups(context, img, ctrl=ctrl)
                text, _boxes = _ocr_texts(context, img, roi)
            except Exception as exc:  # noqa: BLE001
                print("[AddChicken] read failed: %s" % exc, flush=True)
                return None, ""
            m = re.search(r"(\d+)\s*/\s*3", text)
            return (int(m.group(1)) if m else None), text

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            param = json.loads(argv.custom_action_param or "{}") or {}
            target = int(_STATE.get("target") or 3)
            max_plus = max(0, int(param.get("max_plus", MAX_PLUS)))
            gap = float(param.get("gap", self.GAP))
            roi = self._count_roi(param)

            box = argv.box
            if box is None or box.w <= 0 or box.h <= 0:
                try:
                    import layout
                    px, py, pw, ph = layout.get("plus_roi")
                    x, y = px + pw // 2, py + ph // 2
                except Exception:  # noqa: BLE001
                    print("[AddChicken] no box and no plus_roi", flush=True)
                    return False
            else:
                x = box.x + box.w // 2
                y = box.y + box.h // 2

            # 本浮窗里累计点过几次「+」（`DispatchReset` 每次开窗清零）。
            # 上限沿用旧语义 `min(max_plus, target-1)`：target=3 最多点 2 次，
            # 不会在 3/3 之后继续点（游戏可能弹「已达上限」提示）。
            # 它同时是**防跑飞**的硬上限：用完还不到 target 时返回 False，
            # 否则「读派遣数量 miss → 加一只鸡妖 成功但没点」会无限循环。
            used = int(_STATE.get("plus") or 0)
            click_cap = max(0, min(max_plus, target - 1))
            remaining = max(0, click_cap - used)
            if remaining <= 0:
                print("[AddChicken] plus quota used %d/%d -> stop"
                      % (used, click_cap), flush=True)
                return False

            ctrl = context.tasker.controller
            n, text = self._read_count(context, roi, ctrl)
            if n is None:
                # 读不到计数：按旧行为点一次，交给 `读派遣数量` 兜底
                print("[AddChicken] read failed text=%r target=%d -> click 1"
                      % (text, target), flush=True)
                ctrl.post_click(x, y).wait()
                _STATE["plus"] = int(_STATE.get("plus") or 0) + 1
                return True

            stalls = 0
            while n < target and used < click_cap and stalls < self.MAX_STALL:
                need = max(0, min(target - n, click_cap - used))
                if need <= 0:
                    break
                print("[AddChicken] read=%s text=%r target=%d need=%d "
                      "(plus %d/%d)"
                      % (n, text, target, need, used, click_cap), flush=True)
                for i in range(need):
                    ctrl.post_click(x, y).wait()
                    print("[AddChicken] click (%d,%d) n=%d/%d"
                          % (x, y, i + 1, need), flush=True)
                    if i + 1 < need:
                        time.sleep(gap)
                time.sleep(self.VERIFY_GAP)
                n2, text2 = self._read_count(context, roi, ctrl)
                if n2 is None:
                    # 读不到就按旧行为认为这一批点击成功，交给
                    # `读派遣数量` 兜底，避免在这里死循环。
                    used += need
                    n = None
                    break
                gained = max(0, n2 - n)
                used += gained
                print("[AddChicken] verify n=%s->%s gained=%d plus=%d/%d "
                      "text=%r"
                      % (n, n2, gained, used, click_cap, text2), flush=True)
                if n2 <= n:
                    # 弹窗/动画吃掉点击：不计配额，稍后补点差额。
                    stalls += 1
                    time.sleep(0.4)
                else:
                    stalls = 0
                n = n2
                text = text2
            _STATE["plus"] = used
            if n is None or n >= target:
                return True
            print("[AddChicken] still below target n=%s target=%d "
                  "plus=%d/%d" % (n, target, used, click_cap), flush=True)
            return False

    class PlusGate(CustomRecognition):
        """命中 = 还可以再点一次「+」；不命中 = 到上限，该去掠奪了。

        为什么要闸门：`加一只鸡妖` 的 on_error 里写自己会**无条件重跑**
        （动作返回 False 也照跑），实测空转几百次（§3.21）。
        计数在 Python 侧，浮窗每次确认时由 `DispatchReset` 清零。
        """

        def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
            param = json.loads(argv.custom_recognition_param or "{}") or {}
            target = int(_STATE.get("target") or 3)
            limit = max(0, min(int(param.get("max_plus", MAX_PLUS)),
                               target - 1))
            ok = _STATE["plus"] < limit
            detail = {"plus": _STATE["plus"], "target": target,
                      "max_plus": limit, "ok": ok}
            print("[PlusGate] %s" % detail, flush=True)
            if not ok:
                return CustomRecognition.AnalyzeResult(box=None, detail=detail)
            return CustomRecognition.AnalyzeResult(box=(0, 0, 1, 1), detail=detail)

    class ClickReset(CustomAction):
        """点「掠奪」→ 校验是否真的离开了派遣浮窗。

        两种失败会**留在原浮窗**（需求方 11-14 补充）：

          1. 游戏提示「该资源正在被他人掠夺」——只弹提示，不关面板；
          2. 点击被退场动画 / 遮挡吃掉。

        两者表现一样：点完 `btn_dispatch.png` 还在。此时按需求方的说明，
        点一次右上角蓝点（`exit_anchor` 缓存点）就能退回当前秘境面板。

        退出后把该格记为「延迟」（`cell_reco.defer_last`），**不**记为
        「已派遣」——资源没抢到，下一轮仍要重新考虑它。
        """

        # 点完「掠奪」后等面板退场。真机截图 ~0.7s，这里再留 0.5s，
        # 合计约 1.2s 才判定，正常路径不会误判成「被抢」。
        SETTLE = 0.5
        GAP = 0.8
        DETAIL_ROI = [200, 930, 320, 160]
        MANAGE_ROI = [250, 950, 250, 250]

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            box = argv.box
            x = box.x + box.w // 2
            y = box.y + box.h // 2
            ctrl = context.tasker.controller
            ctrl.post_click(x, y).wait()
            print("[ClickReset] click (%d,%d)" % (x, y), flush=True)
            time.sleep(self.SETTLE)

            try:
                import layout
                detail_roi = layout.detail_roi(self.DETAIL_ROI)
                manage_roi = layout.manage_roi(self.MANAGE_ROI)
                exit_pt = layout.exit_point()
            except Exception:  # noqa: BLE001
                detail_roi = list(self.DETAIL_ROI)
                manage_roi = list(self.MANAGE_ROI)
                exit_pt = (674, 256)

            img = ctrl.post_screencap().wait().get()
            # 掠奪后可能先弹「恭喜獲得」，弹窗会盖住 `btn_dispatch.png`。
            # 先清掉再判浮窗是否真的关了，否则会把「点击被弹窗吃掉」
            # 误判成「已经派出去了」（§3.67）。
            img = clear_popups(context, img, ctrl=ctrl)
            still, score = _match_template(
                img, "btn_dispatch.png", detail_roi, 0.94)
            blocked = bool(still)
            # 给 `掠夺后分支` 用：blocked=True 表示这一行没有真的派出鸡妖，
            # 不能标成「已访问」（§3.61 修法 4）。
            _STATE["last_dispatch_ok"] = not blocked
            if blocked:
                # 「该资源正在被他人掠夺」/ 点击被动画吃掉：两者都留在浮窗。
                # 需求方 11-14：点一次右上角蓝点即可退回当前秘境面板。
                print("[ClickReset] dispatch still open (%.3f) "
                      "-> other player robbing / click lost, "
                      "tap anchor (%d,%d)" % (score, exit_pt[0], exit_pt[1]),
                      flush=True)
                ctrl.post_click(*exit_pt).wait()
                time.sleep(self.GAP)
                img = ctrl.post_screencap().wait().get()
                img = clear_popups(context, img, ctrl=ctrl)
                still2, score2 = _match_template(
                    img, "btn_dispatch.png", detail_roi, 0.94)
                panel, panel_score = _match_template(
                    img, "btn_manage.png", manage_roi, 0.94)
                print("[ClickReset] after dismiss dispatch=%s(%.3f) "
                      "panel=%s(%.3f)"
                      % (still2, score2, panel, panel_score), flush=True)
                if still2:
                    # 还开着就再补一次（最多两次，别把循环卡死）
                    ctrl.post_click(*exit_pt).wait()
                    time.sleep(self.GAP)

            try:
                import cell_reco
                if blocked:
                    # 没抢到：记为「延迟」，下一轮仍会重新考虑这一格
                    pending = cell_reco.defer_last()
                    dispatched = list(cell_reco.DISPATCHED)
                else:
                    dispatched = cell_reco.mark_dispatched_last()
                    pending = list(cell_reco.DEFERRED)
            except Exception:  # noqa: BLE001
                dispatched, pending = [], []

            _STATE["loops"] = 0
            _STATE["swipes"] = 0
            _STATE["stuck"] = False
            _STATE["rows_before"] = None
            print("[ClickReset] blocked=%s dispatched=%s deferred=%s "
                  "loops+swipes reset" % (blocked, dispatched, pending),
                  flush=True)
            return True

    class ListReset(CustomAction):
        """旧版「探尋」点击器，保留给旧 pipeline/手工测试；主链已改用 ClickTanxun。"""

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            box = argv.box
            # `探尋` 是“上方圆形图标 + 下方文字”；OCR 框是文字，
            # 按钮图标中心在文字上方约 25px（按用户裁切 only_tanxun_bar）。
            y = box.y - 25
            realm = _STATE.get("realm") or "default"
            points = _STATE.setdefault("tanxun_points", {})
            point = points.get(realm)
            if point is not None:
                px, py = point
                inside = (box.x <= px <= box.x + box.w
                          and box.y - 40 <= py <= box.y + box.h)
                if not inside:
                    point = None
            if point is None:
                # 首次进入某类秘境时测量：合并框「探尋託管中」点左侧
                # 30px（探尋文字中心）；独立「探尋」框点正中心。
                x = box.x + (30 if box.w > 100 else box.w // 2)
                point = (x, y)
                points[realm] = point
                print("[ListReset] measure tanxun=(%d,%d) box=%dx%d"
                      % (x, y, box.w, box.h), flush=True)
            else:
                x, y = point
            context.tasker.controller.post_click(x, y).wait()
            _STATE["swipes"] = 0
            _STATE["stuck"] = False
            _STATE["gray"] = None
            _STATE["rows_before"] = None
            print("[ListReset] click (%d,%d), swipes/stuck reset" % (x, y),
                  flush=True)
            return True

    class ClickTanxun(CustomAction):
        """点「探尋」，并校验是否真的进入了列表。

        缓存按 realm 分开：
          - other：所有别人秘境的「探尋」位置一致，首次测到后直接复用；
          - own：位置可能随「秘境管家」有无变化，当前 OCR 框与缓存不符时重测。

        每次点击后最多等 `LIST_WAIT` 秒确认列表出现。没出现就重新 OCR
        当前帧里的「探尋」，用新框再点一次；成功后更新该 realm 的缓存点。
        """

        LIST_WAIT = 3.0
        POLL = 0.35
        SETTLE = 2.0

        @staticmethod
        def _point(box):
            x = box.x + (30 if box.w > 100 else box.w // 2)
            y = box.y - 25
            return x, y

        @staticmethod
        def _rows(img):
            try:
                import row_reco
                return row_reco.find_rows(img)
            except Exception:  # noqa: BLE001
                return []

        def _wait_list(self, ctrl):
            """点击后轮询，避免列表动画未完成时误判失败。"""
            deadline = time.time() + self.LIST_WAIT
            last = None
            while True:
                img = ctrl.post_screencap().wait().get()
                last = img
                if self._rows(img):
                    return True, img
                if time.time() >= deadline:
                    return False, last
                time.sleep(self.POLL)

        def _click_and_wait(self, ctrl, point):
            ctrl.post_click(point[0], point[1]).wait()
            return self._wait_list(ctrl)

        def _box_from_cache(self, box, point):
            """缓存点是否仍落在当前 OCR 框对应的按钮区域内。"""
            if point is None:
                return False
            px, py = point
            return (box.x <= px <= box.x + box.w
                    and box.y - 40 <= py <= box.y + box.h)

        @staticmethod
        def _ocr_box(context, img):
            try:
                import layout
                roi = layout.tanxun_roi([440, 1250, 280, 90])
            except Exception:  # noqa: BLE001
                roi = [440, 1250, 280, 90]
            _text, boxes = _ocr_texts(
                context, img, roi,
                want=lambda t: "探" in t and ("尋" in t or "寻" in t))
            if not boxes:
                return None
            box = boxes[0]
            if hasattr(box, "x"):
                return box
            bx, by, bw, bh = _box_tuple(box)
            return type("_Box", (), {"x": bx, "y": by, "w": bw, "h": bh})()

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            ctrl = context.tasker.controller
            # 弹窗盖住面板时，「探尋」的缓存点/OCR 都会失准，先清弹窗。
            clear_popups(context, ctrl=ctrl)
            realm = _STATE.get("realm") or "default"
            points = _STATE.setdefault("tanxun_points", {})
            measured = self._point(argv.box)
            cached = points.get(realm)

            attempts = []
            if self._box_from_cache(argv.box, cached):
                attempts.append(("cache", cached))
            if measured not in [p for _src, p in attempts]:
                attempts.append(("ocr", measured))

            ok = False
            point = measured
            img = None
            for source, point in attempts:
                ok, img = self._click_and_wait(ctrl, point)
                print("[ClickTanxun] %s realm=%s point=(%d,%d) ok=%s"
                      % (source, realm, point[0], point[1], ok), flush=True)
                if ok:
                    break

            if not ok and img is not None:
                box = self._ocr_box(context, img)
                if box is not None:
                    point = self._point(box)
                    ok, img = self._click_and_wait(ctrl, point)
                    print("[ClickTanxun] remeasure realm=%s box=%dx%d "
                          "point=(%d,%d) ok=%s"
                          % (realm, box.w, box.h, point[0], point[1], ok),
                          flush=True)
            if not ok:
                print("[ClickTanxun] failed to reach list", flush=True)
                return False
            try:
                import row_reco
                row_reco.begin_list_round()
            except Exception:  # noqa: BLE001
                pass
            points[realm] = point
            _STATE["swipes"] = 0
            _STATE["stuck"] = False
            _STATE["gray"] = None
            _STATE["rows_before"] = None
            # 刚切到探寻页时列表可能还在卡顿/淡入，先等稳定再让 PickRow
            # 读「采集中」，否则会把过渡帧误判成采集中。
            time.sleep(self.SETTLE)
            print("[ClickTanxun] success realm=%s point=%s settle=%.1fs cache=%s"
                  % (realm, point, self.SETTLE, points), flush=True)
            return True

    class ExitOtherRealm(CustomAction):
        """退出别人秘境：按两次空处，随后校验列表是否出现。

        需求方确认：从别人秘境退出应连点两次「X / 空处」，而不是只点一次
        底部「探尋」。这里沿用 `清弹窗后继续` 的安全空处落点。
        """

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            ctrl = context.tasker.controller
            try:
                import layout
                sx, sy = layout.safe_point()
            except Exception:  # noqa: BLE001
                sx, sy = 350, 1326
            for attempt in range(2):
                ctrl.post_click(sx, sy).wait()
                print("[ExitOtherRealm] click #%d (%d,%d)"
                      % (attempt + 1, sx, sy), flush=True)
                # 间隔太短时第二次点击会被退场动画吞掉，实测只退到自家秘境。
                time.sleep(1.0)
            # 退场动画/列表淡入期间多轮截图，避免只取一帧就误判没回列表。
            deadline = time.time() + 2.0
            rows = []
            panel = False
            panel_score = 0.0
            while True:
                img = ctrl.post_screencap().wait().get()
                try:
                    import row_reco
                    rows = row_reco.find_rows(img)
                except Exception:  # noqa: BLE001
                    rows = []
                try:
                    import layout
                    manage_roi = layout.manage_roi()
                except Exception:  # noqa: BLE001
                    manage_roi = [250, 950, 250, 250]
                panel, panel_gray, panel_score = _panel_state(img, manage_roi)
                # 退别人秘境有时先落到「自家秘境」面板：即使绿色块误检出
                # 伪行，也不能当成列表，否则会继续滑秘境面板直到收尾。
                if rows and not panel and not panel_gray:
                    break
                if time.time() >= deadline:
                    break
                time.sleep(0.3)
            ok = bool(rows) and not panel and not panel_gray
            if ok:
                try:
                    import row_reco
                    row_reco.begin_list_round()
                except Exception:  # noqa: BLE001
                    pass
                # 回到的是新一页列表：清掉上一页的滑动预算/基准，
                # 否则上一页用满 8 次后，这一页第一次“没动”就直接收尾。
                _STATE["swipes"] = 0
                _STATE["stuck"] = False
                _STATE["gray"] = None
                _STATE["rows_before"] = None
            print("[ExitOtherRealm] list_rows=%d panel=%s(%.3f) gray=%s ok=%s"
                  % (len(rows), panel, panel_score, panel_gray, ok),
                  flush=True)
            return ok

    class BackToHome(CustomAction):
        """退出秘境面板回主页面：点右上角固定 HUD 锚点。

        实测底栏「冒险 / 家園」在秘境面板上也能命中（0.997），点它们
        只是切 tab。**不要用系统 BACK**——需求方 11-14 明确这个游戏
        不响应返回键，按了没有任何反应。
        点击点取右上角固定 HUD 锚点（`exit_anchor.png`）中心：
        实测在秘境面板 / 别人秘境 / 列表 / 活动页 / 派遣浮窗上
        都命中 0.987~0.988，跨页面位置恒定；比内容区空白点更安全。

        顺序：
          1. 有「恭喜獲得」/「資源詳情」→ 点安全空处关掉；
          2. 仍有 `btn_manage.png`（秘境面板）→ 点右上角锚点，最多 4 次；
          3. 面板消失即返回，交给 `回主界面` 继续重进流程。
        """

        SAFE = (350, 1326)
        CLICK_GAP = 1.0
        RETRY_GAP = 1.2
        MAX_POPS = 4
        MAX_TRIES = 4

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            ctrl = context.tasker.controller
            try:
                import layout
                safe = layout.safe_point(self.SAFE)
                exit_pt = layout.exit_point()
                detail_roi = layout.detail_roi([200, 930, 320, 160])
                manage_roi = layout.manage_roi([250, 950, 250, 250])
            except Exception:  # noqa: BLE001
                safe = self.SAFE
                exit_pt = (674, 256)
                detail_roi = [200, 930, 320, 160]
                manage_roi = [250, 950, 250, 250]
            img = None
            for idx in range(self.MAX_POPS):
                img = ctrl.post_screencap().wait().get()
                got, got_score = _popup_match(img)
                detail, detail_score = _match_template(
                    img, "btn_dispatch.png", detail_roi, 0.94)
                print("[BackToHome] check #%d obtain=%s(%.3f) "
                      "detail=%s(%.3f)"
                      % (idx + 1, got, got_score, detail, detail_score),
                      flush=True)
                if not got and not detail:
                    break
                ctrl.post_click(*safe).wait()
                print("[BackToHome] dismiss popup #%d (%d,%d)"
                      % (idx + 1, safe[0], safe[1]), flush=True)
                time.sleep(self.CLICK_GAP)
            img = ctrl.post_screencap().wait().get()
            got, got_score = _popup_match(img)
            detail, detail_score = _match_template(
                img, "btn_dispatch.png", detail_roi, 0.94)
            if got or detail:
                print("[BackToHome] popup still present obtain=%s(%.3f) "
                      "detail=%s(%.3f)" % (got, got_score, detail, detail_score),
                      flush=True)
                return False

            for attempt in range(self.MAX_TRIES):
                panel, panel_gray, panel_score = _panel_state(img, manage_roi)
                rows = _list_rows(img)
                print("[BackToHome] state #%d panel=%s(%.3f) gray=%s rows=%d"
                      % (attempt + 1, panel, panel_score, panel_gray,
                         len(rows)),
                      flush=True)
                if not panel and not panel_gray and not rows:
                    # 既不是秘境面板也不是探寻列表：已经在主页面附近，
                    # 不要再点，避免误触。
                    return True
                # 退出点用进入秘境时测量并缓存的值，不再每轮重新匹配。
                px, py = exit_pt
                ctrl.post_click(px, py).wait()
                print("[BackToHome] click cached exit (%d,%d) #%d"
                      % (px, py, attempt + 1), flush=True)
                time.sleep(self.RETRY_GAP)
                img = ctrl.post_screencap().wait().get()
                got, _ = _popup_match(img)
                if got:
                    ctrl.post_click(*safe).wait()
                    time.sleep(self.CLICK_GAP)
                    img = ctrl.post_screencap().wait().get()

            panel, panel_gray, panel_score = _panel_state(img, manage_roi)
            rows = _list_rows(img)
            print("[BackToHome] after backs panel=%s(%.3f) gray=%s rows=%d"
                  % (panel, panel_score, panel_gray, len(rows)), flush=True)
            # 面板/列表还在也返回 True：交给 `回主界面` 再兜底，
            # 避免直接落 `重进失败` 收尾。
            return True

    class ClickAdventure(CustomAction):
        """退出秘境 / 列表回主页面：只点右上角固定 HUD 锚点（蓝点）。

        需求方 11-14 明确：不要再点底部「冒险」tab。两个原因：
          1. 那个 tab 在秘境面板上也能匹配到，点它只切 tab、不关面板；
          2. 原生 Click 在 100x100 的 tab 框里随机取点，18~25% 落在图标外
             （§3.45），点没点中全看运气。

        统一改成点 `exit_anchor.png` 在进秘境时测出的缓存点（§3.40）。
        每轮先截图判断：既不是秘境面板也不是探寻列表 → 已在主页面，直接返回。
        """

        GAP = 1.2
        MAX_TRIES = 4

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            ctrl = context.tasker.controller
            try:
                import layout
                manage_roi = layout.manage_roi([250, 950, 250, 250])
                exit_pt = layout.exit_point()
                safe = layout.safe_point((350, 1326))
            except Exception:  # noqa: BLE001
                manage_roi = [250, 950, 250, 250]
                exit_pt = (674, 256)
                safe = (350, 1326)

            img = None
            for attempt in range(self.MAX_TRIES):
                img = ctrl.post_screencap().wait().get()
                got, got_score = _popup_match(img)
                panel, panel_gray, panel_score = _panel_state(img, manage_roi)
                rows = _list_rows(img)
                print("[ClickAdventure] state #%d popup=%s(%.3f) "
                      "panel=%s(%.3f) gray=%s rows=%d"
                      % (attempt + 1, got, got_score, panel, panel_score,
                         panel_gray, len(rows)), flush=True)
                if got:
                    ctrl.post_click(*safe).wait()
                    print("[ClickAdventure] dismiss popup (%d,%d)"
                          % (safe[0], safe[1]), flush=True)
                    time.sleep(self.GAP)
                    continue
                if not panel and not panel_gray and not rows:
                    # 真的回到主页面了 → 本轮重进的「家園」配额重置（§3.53）
                    _STATE["reopen"] = 0
                    return True
                px, py = exit_pt
                ctrl.post_click(px, py).wait()
                print("[ClickAdventure] click cached exit (%d,%d) #%d"
                      % (px, py, attempt + 1), flush=True)
                time.sleep(self.GAP)

            img = ctrl.post_screencap().wait().get()
            panel, panel_gray, panel_score = _panel_state(img, manage_roi)
            rows = _list_rows(img)
            print("[ClickAdventure] after tries panel=%s(%.3f) gray=%s rows=%d"
                  % (panel, panel_score, panel_gray, len(rows)), flush=True)
            ok = not panel and not panel_gray and not rows
            if ok:
                _STATE["reopen"] = 0
            return ok

    class RecoverNet(CustomAction):
        """网络异常 / 掉线的兜底门：**不读文字，自己循环重试**。

        需求方 11-14 19:13 明确要求：**不要添加校验，任何异常都返回主页面**。
        早先那版 OCR 校验（`NET_PATTERN` / `RETRY_PATTERN`）已删除，原因是它
        在识别到网络文案后会去点 OCR 框中心，而 `RETRY_PATTERN` 里含
        「确定 / 确认」，在陌生界面上可能点掉有副作用的弹窗；并且那段代码
        从未在真实网络弹窗上验证过（仓库里没有一张对应截图）。

        每轮只做三件事，最多 `max_rounds` 轮：

          1. 截图，判是不是已经在冒险主页面（`nav_home.png` 命中、
             没有 `btn_manage.png`、没有探寻列表行）→ 立即返回 True；
          2. 否则点蓝点（`exit_anchor` 中心）退出当前浮层；
          3. 等 `WAIT` 秒进入下一轮。

        轮次用完仍未回到主页面 → 返回 False，节点落到 `on_error`
        （`列表已到底` 收尾）。

        **不用 BACK 键**：这个游戏不响应系统返回键。

        返回值语义：

          - True → 已经在冒险主页面，pipeline 走 `next`（`重进-找菜单`）；
          - False → 轮次用完，pipeline 走 `on_error`（`列表已到底`）。

        **为什么自己循环而不是靠 `max_hit` + 自引用**：这个节点以前写的是
        `"recognition": "Custom"` + `"custom_recognition": "RecoverNet"`，
        但 RecoverNet 是 CustomAction（只有 `run`，没有 `analyze`），Maa 报
        `recognition is null`，节点每次都直接失败，恢复门**从未真正执行过**
        （11-14 18:22 的 trace 里 `重进失败` 之后直接就是 `列表已到底`）。
        现在改成 `DirectHit` + `Custom action`，动作内部自己数轮次。
        """

        # 每轮固定等待：8 轮 ≈ 25s，够网络抖动恢复，卡死时也不至于拖两分钟。
        WAIT = 3.0
        MAX_ROUNDS = 8

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            ctrl = context.tasker.controller
            param = json.loads(argv.custom_action_param or "{}") or {}
            raw_rounds = param.get("max_rounds")
            max_rounds = (self.MAX_ROUNDS if raw_rounds is None
                          else int(raw_rounds))
            # 0 / 负数按 1 轮处理：至少点一次，别变成空转
            max_rounds = max(1, max_rounds)
            _STATE["recover"] = int(_STATE.get("recover") or 0) + 1
            try:
                import layout
                manage_roi = layout.manage_roi([250, 950, 250, 250])
                nav_roi = layout.get("nav_roi") or [0, 1200, 720, 400]
                exit_pt = layout.exit_point()
            except Exception:  # noqa: BLE001
                manage_roi = [250, 950, 250, 250]
                nav_roi = [0, 1200, 720, 400]
                exit_pt = (674, 256)

            for i in range(1, max_rounds + 1):
                img = ctrl.post_screencap().wait().get()
                home, home_score = _match_template(
                    img, "nav_home.png", nav_roi, 0.8)
                panel, panel_gray, panel_score = _panel_state(img, manage_roi)
                rows = _list_rows(img)
                print("[RecoverNet] round=%d/%d home=%s(%.3f) "
                      "panel=%s(%.3f) gray=%s rows=%d"
                      % (i, max_rounds, home, home_score,
                         panel, panel_score, panel_gray, len(rows)),
                      flush=True)
                if home and not panel and not panel_gray and not rows:
                    print("[RecoverNet] already on adventure home page",
                          flush=True)
                    return True

                ctrl.post_click(exit_pt[0], exit_pt[1]).wait()
                if i < max_rounds:
                    print("[RecoverNet] tap exit anchor (%d,%d) wait %.1fs"
                          % (exit_pt[0], exit_pt[1], self.WAIT), flush=True)
                    time.sleep(self.WAIT)
                else:
                    print("[RecoverNet] tap exit anchor (%d,%d) (last round,"
                          " no wait)" % (exit_pt[0], exit_pt[1]), flush=True)

            print("[RecoverNet] *** gave up after %d rounds ***" % max_rounds,
                  flush=True)
            return False

    class RewireMain(CustomAction):
        """主任务入口的「接线」动作：把若干节点的 next 改写成主循环落点。

        为什么不在 JSON 里直接写死：同一个节点（`确认资源详情` / `掠奪` / `点探尋`）
        既是**调试入口**的终点，又是**主链**的中转。写死 next 会让调试入口
        一跑就冲进整条循环，没法单独验证某一步。

        Maa 的 `Context.override_next` 只影响当前任务，所以：

        - 单跑调试入口 → 节点保持「到此为止」
        - 跑 `主任务` → 先把它们接到主循环上，再开始

        param 形如 {"确认已进入秘境": ["主循环选格"], ...}。
        """

        def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
            param = json.loads(argv.custom_action_param or "{}") or {}
            try:
                import layout
                layout.reset()
            except Exception:  # noqa: BLE001
                pass
            try:
                import row_reco
                row_reco.clear_bad_rows()
            except Exception:  # noqa: BLE001
                pass
            for node, next_list in param.items():
                ok = context.override_next(node, list(next_list))
                print("[RewireMain] %s -> %s : %s"
                      % (node, next_list, ok), flush=True)
                if not ok:
                    return False
            return True

except Exception:  # pragma: no cover
    ChickenStock = DispatchCountReached = DispatchPolicy = WaitReturnGate = None
    RealmOtherGate = DispatchOkGate = None
    SwipeRows = ListMoved = LoopGuard = SwipeGate = PlusGate = None
    DispatchReset = AddChicken = ClickReset = ListReset = RewireMain = None
    SetRealm = DeferCell = MarkBadRow = MarkRowDone = ClickTanxun = None
    ExitOtherRealm = BackToHome = CooldownGate = WaitCooldown = None
    MarkWait = WaitPopup = None
    RefreshGate = ClickRefresh = CloseList = None
    AbortTask = ReopenGate = CountReopen = None


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    base = os.path.join(ROOT, "User_screenshot", "full", "720_list_top.png")
    other = os.path.join(ROOT, "User_screenshot", "full",
                         "720_fullscreen_activity.png")
    a = cv2.imread(base)
    b = cv2.imread(other)
    print("A=%s  B=%s" % (os.path.basename(base), os.path.basename(other)))
    reset_state()
    print("first      ", compare(a, DEFAULT_ROI))
    print("same frame ", compare(a, DEFAULT_ROI))
    print("other frame", compare(b, DEFAULT_ROI))
    print("same again ", compare(b, DEFAULT_ROI))
    print("dy(A) =", page_dy(a, DEFAULT_ROI))
    print("dy(B) =", page_dy(b, DEFAULT_ROI))
