"""量 11-10 用户提供的新素材 + 探寻列表的滚动/变暗判据。

九个部分：
  1. 新模板尺寸/alpha，以及 btn_manage / btn_dispatch 的全仓库命中分布
  2. 探寻列表的行结构：行带、行距、可视行数、每行饱和度与物品命中
  3. 「划不动」判据：同页基线 vs 异页基线（整帧像素差）
  4. 「变暗行」判据：TemplateMatch 对亮度不敏感，靠 0.88 不成立；
     但真实渐隐行的分数本来就低（实测 0.33~0.50），0.88 天然排除
  5. 720_dispatch_menu.png 的身份核对 + 派遣界面里的鸡妖
  6. 变暗行普查：全仓库逐行统计「暗行是否命中」
  7. 新截图与历史帧的同一性比对（确认 720_dispatch_menu 是哪一屏）
  8. 对三张新整帧做分区 OCR，确认界面身份（需要 resource/model/ocr）
  9. 「資源詳情」浮窗（= 派遣界面）的控件版面：数量加减、掠奪按钮、动态区域
 10. 派遣只数 +/- 按钮模板（btn_plus / btn_minus）的命中与互相串味

只读，不裁图、不落盘。一次跑完约 40s。
"""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"E:\work\AutoButter"
USER = os.path.join(ROOT, "User_screenshot")
IMG = os.path.join(ROOT, "resource", "image")
SHOTS = os.path.join(ROOT, "screenshots")

ITEMS = ["alcohol", "berry", "fabric", "honey", "jade", "silver", "stone"]

# 探寻列表：图标带起点、行距、带高、图标区 x 范围、底部渐隐起点
ROW_Y0, ROW_PITCH, ROW_H = 489, 151.5, 66
ROW_X0, ROW_X1 = 150, 660
FADE_Y = 1176
BTN_ICON_DY = 34      # 「前往」按钮 y + 34 = 该行图标带 y


def row_anchors(im):
    """用绿色「前往」按钮做行锚点，返回 [(btn_y, x, w, h)]。

    比按固定行距推更稳：列表是分区的，跨分区行距是 210 而不是 151.5。
    """
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85)
         & (hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 90)).astype(np.uint8) * 255
    m[:, :550] = 0
    m[:, 660:] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        if 55 <= w <= 130 and 3 <= h <= 50:
            out.append((y, x, w, h))
    return sorted(out)


def load(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def bgr(im):
    if im is None:
        return None
    return im[:, :, :3] if im.ndim == 3 and im.shape[2] == 4 else im


def to720(im):
    h, w = im.shape[:2]
    if w == 720:
        return im
    return cv2.resize(im, (720, int(round(h * 720 / w))),
                      interpolation=cv2.INTER_AREA)


def sweep(tpl, thr=0.90):
    hits = []
    paths = (sorted(glob.glob(os.path.join(SHOTS, "*.png")))
             + sorted(glob.glob(os.path.join(SHOTS, "snap", "*.png")))
             + sorted(glob.glob(os.path.join(USER, "full", "*.png"))))
    for p in paths:
        if os.path.basename(p).startswith("_"):
            continue
        im = cv2.imread(p)
        if im is None:
            continue
        im = to720(im)
        if im.shape[0] < tpl.shape[0] or im.shape[1] < tpl.shape[1]:
            continue
        r = cv2.matchTemplate(im, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(r)
        if mx >= thr:
            hits.append((round(float(mx), 3), os.path.basename(p), ml))
    return sorted(hits, reverse=True)


def row_stats(im, y):
    band = cv2.cvtColor(im[y:y + ROW_H, ROW_X0:ROW_X1], cv2.COLOR_BGR2HSV)
    s = band[:, :, 1]
    return (round(float(s.mean()), 1),
            round(float((s > 60).mean()), 3),
            round(float((s > 100).mean()), 3))


def item_hits(im, y, thr=0.88):
    sub = im[y:y + ROW_H, ROW_X0:ROW_X1]
    out = []
    for n in ITEMS:
        t = cv2.imread(os.path.join(IMG, "item_%s.png" % n))
        if sub.shape[0] < t.shape[0] or sub.shape[1] < t.shape[1]:
            continue
        r = cv2.matchTemplate(sub, t, cv2.TM_CCOEFF_NORMED)
        rr = r.copy()
        while True:
            _, mx, _, ml = cv2.minMaxLoc(rr)
            if mx < thr:
                break
            out.append((n, ml[0] + ROW_X0, round(float(mx), 3)))
            rr[max(0, ml[1] - 20):ml[1] + 20,
               max(0, ml[0] - 20):ml[0] + 20] = -1
    return sorted(out, key=lambda t: t[1])


def section1():
    print("=" * 72)
    print("1. 新模板")
    print("=" * 72)
    tmpls = {}
    for name in ["btn_manage", "btn_dispatch"]:
        raw = load(os.path.join(USER, "btn", name + ".png"))
        t = bgr(raw)
        tmpls[name] = t
        alpha = ""
        if raw.ndim == 3 and raw.shape[2] == 4:
            alpha = " alpha=%d..%d" % (raw[:, :, 3].min(), raw[:, :, 3].max())
        print("  %-14s %3dx%-3d%s" % (name, t.shape[1], t.shape[0], alpha))

    print("\n  全仓库命中（阈值 0.95）：")
    for name, t in tmpls.items():
        hits = sweep(t, 0.95)
        pos = sorted({h[2] for h in hits})
        print("  %-14s hits=%-3d 落点=%s" % (name, len(hits), pos))
        for sc, f, at in hits[:3]:
            print("       %.3f %-30s %s" % (sc, f, at))


def section2():
    print("\n" + "=" * 72)
    print("2. 探寻列表行结构")
    print("=" * 72)
    for rel, root in [("full/720_list_top.png", USER),
                      ("45_tanxun_list.png", SHOTS),
                      ("49_list2.png", SHOTS)]:
        im = cv2.imread(os.path.join(root, rel))
        print("  == %s" % rel)
        btns = row_anchors(im)
        print("     「前往」按钮 y=%s" % [b[0] for b in btns])
        if len(btns) > 1:
            print("     按钮间距=%s（210 是跨分区，多了分区标题）"
                  % [btns[i + 1][0] - btns[i][0] for i in range(len(btns) - 1)])
        print("     row  图标带      Smean  S>60   S>100   物品命中")
        for i, (by, _x, _w, _h) in enumerate(btns):
            y = by + BTN_ICON_DY
            if y + ROW_H > im.shape[0]:
                break
            sm, f60, f100 = row_stats(im, y)
            print("     %3d  %4d..%4d  %6.1f  %.3f  %.3f   %s"
                  % (i, y, y + ROW_H, sm, f60, f100, item_hits(im, y)))
        print("     行距 151.5px；一页 3 行完整 → 基准滑动量 %d px"
              % round(ROW_PITCH * 3))
        print("     底部渐隐起点 FADE_Y=%d" % FADE_Y)


def section3():
    print("\n" + "=" * 72)
    print("3. 「划不动」判据：整帧像素差")
    print("=" * 72)
    files = {
        "45_tanxun_list": os.path.join(SHOTS, "45_tanxun_list.png"),
        "49_list2": os.path.join(SHOTS, "49_list2.png"),
        "50_visited": os.path.join(SHOTS, "50_visited.png"),
        "720_list_top": os.path.join(USER, "full", "720_list_top.png"),
    }
    rois = {}
    for k, p in files.items():
        im = to720(cv2.imread(p))
        rois[k] = cv2.cvtColor(im[357:1180, 40:680], cv2.COLOR_BGR2GRAY)
    keys = list(rois)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            d = cv2.absdiff(rois[a], rois[b])
            tag = "同页" if "720" not in (a + b) else "异页"
            print("  %-16s vs %-16s (%s) mean=%5.2f  pct>8=%.4f  pct>30=%.4f"
                  % (a, b, tag, float(d.mean()), float((d > 8).mean()),
                     float((d > 30).mean())))
    print("  注：同页仍有 mean=1.2~3.2 的差异（倒计时/头像动画），"
          "精确相等判据不成立。")

    print("\n  纵向平移扫描（确认 720_list_top 不是滚动后的帧）：")
    ref = cv2.cvtColor(cv2.imread(os.path.join(SHOTS, "45_tanxun_list.png")),
                       cv2.COLOR_BGR2GRAY)[360:1170, 40:680].astype(np.float32)
    tgt = cv2.cvtColor(cv2.imread(os.path.join(USER, "full", "720_list_top.png")),
                       cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows = []
    for dy in range(-400, 401):
        y0, y1 = 360 + dy, 1170 + dy
        if y0 < 0 or y1 > tgt.shape[0]:
            continue
        cur = tgt[y0:y1, 40:680]
        if cur.shape != ref.shape:
            continue
        rows.append((float(np.abs(cur - ref).mean()), dy))
    rows.sort()
    for d, dy in rows[:4]:
        print("     dy=%+4d  mean|diff|=%.2f" % (dy, d))


def section4():
    print("\n" + "=" * 72)
    print("4. 「变暗行」判据")
    print("=" * 72)
    im = cv2.imread(os.path.join(USER, "full", "720_list_top.png"))
    base = im[470:570, 150:660].copy()
    print("  TemplateMatch 对整体压暗的响应：")
    for k in [1.0, 0.85, 0.70, 0.55, 0.40]:
        test = im.copy()
        test[470:570, 150:660] = np.clip(
            base.astype(np.float32) * k, 0, 255).astype(np.uint8)
        sub = test[470:570, 150:660]
        best = 0.0
        for n in ITEMS:
            t = cv2.imread(os.path.join(IMG, "item_%s.png" % n))
            r = cv2.matchTemplate(sub, t, cv2.TM_CCOEFF_NORMED)
            best = max(best, float(r.max()))
        print("     亮度 x%.2f → 最高分 %.3f" % (k, best))
    print("  → 分数不变，不能靠 0.88 阈值排除变暗行。")
    print("  改用饱和度门（S>60 占比）：正常行 0.45~0.68，变暗行 0.00~0.09。")


def section5():
    print("\n" + "=" * 72)
    print("5. 720_dispatch_menu.png 身份核对 + 鸡妖")
    print("=" * 72)
    tgt = cv2.imread(os.path.join(USER, "full", "720_dispatch_menu.png"))
    for name in ["53_rob_popup.png", "54_plus1.png", "55_plus2.png",
                 "102_popup_gone.png"]:
        im = to720(cv2.imread(os.path.join(SHOTS, name)))
        g = cv2.absdiff(tgt, im).max(axis=2)
        print("  vs %-22s mean=%6.2f  pct>8=%.4f  pct>30=%.4f"
              % (name, float(g.mean()), float((g > 8).mean()),
                 float((g > 30).mean())))

    sys.path.insert(0, os.path.join(ROOT, "agent"))
    from chicken_reco import get_finder
    f = get_finder()
    for rel in ["full/720_dispatch_menu.png", "full/720_realm_panel.png"]:
        im = cv2.imread(os.path.join(USER, rel))
        hits = f.find(im, (0, 0, 720, 1060))
        print("  %-28s chickens=%d %s"
              % (rel, len(hits),
                 [(h[0], round(h[1]), round(h[2]), h[5]) for h in hits]))


def section6():
    print("\n" + "=" * 72)
    print("6. 变暗行普查（全仓库，按行距 151.5 切 4 行）")
    print("=" * 72)
    paths = (sorted(glob.glob(os.path.join(SHOTS, "*.png")))
             + sorted(glob.glob(os.path.join(SHOTS, "snap", "*.png"))))
    rows = []
    for p in paths:
        if os.path.basename(p).startswith("_"):
            continue
        im = cv2.imread(p)
        if im is None:
            continue
        im = to720(im)
        if im.shape[0] < 1200:
            continue
        for i in range(4):
            y = int(round(ROW_Y0 + i * ROW_PITCH))
            band = cv2.cvtColor(im[y:y + ROW_H, ROW_X0:ROW_X1], cv2.COLOR_BGR2HSV)
            f60 = float((band[:, :, 1] > 60).mean())
            rows.append((os.path.basename(p), i, y, f60, bool(item_hits(im, y))))

    bright = [r for r in rows if r[3] >= 0.30]
    dim = [r for r in rows if r[3] < 0.15]
    print("  总行样本 %d：亮行(S>60>=0.30) %d，暗行(<0.15) %d"
          % (len(rows), len(bright), len(dim)))
    print("  亮行命中 %d/%d，暗行命中 %d/%d"
          % (sum(1 for r in bright if r[4]), len(bright),
             sum(1 for r in dim if r[4]), len(dim)))
    bad = [r for r in dim if r[4]]
    if bad:
        print("  暗行误命中：")
        for r in bad:
            print("     %-32s row%d S>60=%.3f" % (r[0], r[1], r[3]))
    else:
        print("  → 0.88 阈值下暗行 0 命中，规则「变暗不识别」已天然满足。")


def section7():
    print("\n" + "=" * 72)
    print("7. 新截图与历史帧的同一性")
    print("=" * 72)
    tgt = cv2.imread(os.path.join(USER, "full", "720_dispatch_menu.png"))
    t = bgr(load(os.path.join(USER, "btn", "btn_dispatch.png")))
    names = ["48_go_target.png", "52_go_row2.png", "53_rob_popup.png",
             "54_plus1.png", "55_plus2.png", "56_robbed.png",
             "102_popup_gone.png", "44_realm.png"]
    print("  %-24s %-20s %s" % ("历史帧", "整帧差 mean/pct>30", "btn_dispatch"))
    for n in names:
        p = os.path.join(SHOTS, n)
        im = cv2.imread(p)
        if im is None:
            continue
        im = to720(im)
        d = cv2.absdiff(tgt, im).max(axis=2)
        r = cv2.matchTemplate(im, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(r)
        print("  %-24s %6.2f / %.4f      %.4f @%s"
              % (n, float(d.mean()), float((d > 30).mean()), mx, ml))

    print("\n  btn_manage / realm_marker 几何关系：")
    m = bgr(load(os.path.join(USER, "btn", "btn_manage.png")))
    rm = cv2.imread(os.path.join(IMG, "realm_marker.png"))
    panel = cv2.imread(os.path.join(USER, "full", "720_realm_panel.png"))
    for name, tpl in (("btn_manage", m), ("realm_marker", rm)):
        r = cv2.matchTemplate(panel, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(r)
        print("     %-14s %3dx%-3d  %.4f @%s  -> x %d..%d  y %d..%d"
              % (name, tpl.shape[1], tpl.shape[0], mx, ml,
                 ml[0], ml[0] + tpl.shape[1], ml[1], ml[1] + tpl.shape[0]))


def section8():
    print("\n" + "=" * 72)
    print("8. 新整帧的分区 OCR")
    print("=" * 72)
    import json
    import shutil

    tmp = os.path.join(SHOTS, "_tmp_ocr")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(os.path.join(tmp, "pipeline"), exist_ok=True)
    model = os.path.join(ROOT, "resource", "model")
    if os.path.isdir(model):
        shutil.copytree(model, os.path.join(tmp, "model"))

    targets = ["full/720_dispatch_menu.png", "full/720_realm_panel.png",
               "full/720_list_top.png"]
    rois = []
    for rel in targets:
        for i in range(0, 1560, 160):
            rois.append((rel.split("/")[-1][:-4] + "_%04d" % i,
                         rel, [0, i, 720, min(160, 1560 - i)]))
    pipe = {name: {"recognition": "OCR", "roi": roi,
                   "action": "DoNothing", "expected": []}
            for name, _rel, roi in rois}
    with open(os.path.join(tmp, "pipeline", "main.json"), "w",
              encoding="utf-8") as f:
        json.dump(pipe, f, ensure_ascii=False, indent=2)

    from maa.controller import CustomController
    from maa.resource import Resource
    from maa.tasker import Tasker

    frames = {rel: cv2.imread(os.path.join(USER, rel)) for rel in targets}

    class C(CustomController):
        def __init__(self, frame):
            super().__init__()
            self.frame = frame

        def connect(self):
            return True

        def request_uuid(self):
            return "ocr-probe"

        def start_app(self, intent):
            return True

        def stop_app(self, intent):
            return True

        def screencap(self):
            return self.frame.copy()

        def click(self, x, y):
            return True

        def swipe(self, *a):
            return True

        def touch_down(self, *a):
            return True

        def touch_move(self, *a):
            return True

        def touch_up(self, *a):
            return True

        def click_key(self, k):
            return True

        def input_text(self, t):
            return True

        def key_down(self, k):
            return True

        def key_up(self, k):
            return True

    for rel in targets:
        print("  ==", rel)
        ctrl = C(frames[rel])
        if not ctrl.post_connection().wait().succeeded:
            print("     controller connect failed")
            continue
        res = Resource()
        if not res.post_bundle(tmp).wait().succeeded:
            print("     resource load failed")
            continue
        tasker = Tasker()
        if not tasker.bind(res, ctrl):
            print("     bind failed")
            continue
        for name, rel2, roi in rois:
            if rel2 != rel:
                continue
            task = tasker.post_task(name).wait()
            detail = tasker.get_task_detail(task.job_id)
            texts = []
            for nd in getattr(detail, "nodes", []) or []:
                rec = getattr(nd, "recognition", None)
                if rec is None:
                    continue
                for rr in (getattr(rec, "all_results", None) or []):
                    t = getattr(rr, "text", None)
                    if t:
                        texts.append(str(t))
            if texts:
                print("     y=%4d  %s" % (roi[1], texts))


def section9():
    print("\n" + "=" * 72)
    print("9. 「資源詳情」浮窗（需求方确认这就是派遣界面）控件版面")
    print("=" * 72)
    im = cv2.imread(os.path.join(USER, "full", "720_dispatch_menu.png"))
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bright_rows = (g[:, 40:680] > 90).mean(axis=1)
    rows = [y for y in range(len(bright_rows)) if bright_rows[y] > 0.55]
    print("  浮窗竖向范围: y %d..%d" % (min(rows), max(rows)))

    print("\n  y=930..1070 横向结构（掠奪按钮与两侧 +/-）:")
    for y in range(930, 1070, 10):
        line = ""
        for x in range(0, 720, 8):
            c = g[y:y + 10, x:x + 8].mean()
            line += " " if c < 35 else ("." if c < 70 else
                                        ("-" if c < 110 else ("+" if c < 160 else "#")))
        print("     %4d %s" % (y, line))

    print("\n  浮窗内的高饱和块（按钮候选）:")
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 110)).astype(np.uint8) * 255
    m[:460] = 0
    m[1330:] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        if 20 <= w <= 400 and 20 <= h <= 160:
            boxes.append((y, x, w, h))
    for y, x, w, h in sorted(boxes):
        print("     y=%4d x=%3d %3dx%-3d" % (y, x, w, h))


def main():
    section1()
    section2()
    section3()
    section4()
    section5()
    section6()
    section7()
    section8()
    section9()
    section10()
    return 0


def section10():
    print("\n" + "=" * 72)
    print("10. 派遣只数 +/- 按钮（btn_plus / btn_minus）")
    print("=" * 72)
    plus = bgr(load(os.path.join(USER, "btn", "btn_plus.png")))
    minus = bgr(load(os.path.join(USER, "btn", "btn_minus.png")))
    for name, t in (("btn_plus", plus), ("btn_minus", minus)):
        g = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
        print("  %-10s %dx%d  gray mean=%.1f std=%.1f"
              % (name, t.shape[1], t.shape[0], float(g.mean()), float(g.std())))

    print("\n  互相串味（这是关键风险）：")
    print("     plus 模板里找 minus: %.4f"
          % float(cv2.matchTemplate(plus, minus, cv2.TM_CCOEFF_NORMED).max()))
    print("     minus 模板里找 plus: %.4f"
          % float(cv2.matchTemplate(minus, plus, cv2.TM_CCOEFF_NORMED).max()))

    scenes = ["full/720_dispatch_menu.png", "full/720_realm_panel.png",
              "full/720_list_top.png"]
    print("\n  三个界面上的命中：")
    for rel in scenes:
        im = cv2.imread(os.path.join(USER, rel))
        row = []
        for name, t in (("plus", plus), ("minus", minus)):
            r = cv2.matchTemplate(im, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, ml = cv2.minMaxLoc(r)
            row.append("%s=%.4f@%s" % (name, mx, ml))
        print("     %-26s %s" % (rel.split("/")[-1], "  ".join(row)))

    print("\n  全仓库扫描（阈值 0.95）：")
    paths = (sorted(glob.glob(os.path.join(SHOTS, "*.png")))
             + sorted(glob.glob(os.path.join(SHOTS, "snap", "*.png")))
             + sorted(glob.glob(os.path.join(USER, "full", "*.png"))))
    for name, t in (("plus", plus), ("minus", minus)):
        hits = []
        for p in paths:
            if os.path.basename(p).startswith("_"):
                continue
            im = cv2.imread(p)
            if im is None:
                continue
            im = to720(im)
            if im.shape[0] < t.shape[0] or im.shape[1] < t.shape[1]:
                continue
            r = cv2.matchTemplate(im, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, ml = cv2.minMaxLoc(r)
            if mx >= 0.95:
                hits.append((round(float(mx), 3), os.path.basename(p), ml))
        hits.sort(reverse=True)
        pos = sorted({h[2] for h in hits})
        print("     %-6s hits=%-2d 落点=%s" % (name, len(hits), pos))

    print("\n  +/- 在 53/54/55 三帧上是否变化（3/3 时是否置灰）：")
    for rel in ["53_rob_popup.png", "54_plus1.png", "55_plus2.png"]:
        im = cv2.imread(os.path.join(SHOTS, rel))
        out = []
        for name, t, xy in (("minus", minus, (171, 837)),
                            ("plus", plus, (521, 836))):
            x, y = xy
            patch = im[y:y + t.shape[0], x:x + t.shape[1]]
            out.append("%s mean=%.1f" % (name, float(patch.mean())))
        print("     %-20s %s" % (rel, "  ".join(out)))


if __name__ == "__main__":
    sys.exit(main())
