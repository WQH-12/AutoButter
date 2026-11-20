"""诊断「回主界面 → 点家园 → 子菜单没展开」。

背景（11-14 实机 trace）：

    198.4s  重进-开菜单  hit box=[157,1457,100,100] Click ok
    219.3s  重进-开菜单  miss          <- 20.9s 后，重进-找菜单 一直没命中
    219.7s  重进失败

要区分三种可能：

  A. 点击没生效（点在 tab 边缘 / 被弹窗吃掉）
  B. 菜单开了，但 home_menu_first.png 在这台设备上位置/缩放对不上
  C. 点之前菜单就是开的，这一点反而把它关掉了（「家園」是 toggle）

做法：连点两轮同一个坐标，每轮点完连拍 8 帧、每帧给 home_menu_first
打分。第 1 轮分数起来、第 2 轮掉下去 → C；两轮都不起 → A；
分数 0.6~0.8 起不来 → B。

用法（手机上先停在**冒险主页面**）：

    python tools\\probe_reentry_menu.py
    python tools\\probe_reentry_menu.py --rounds 3 --interval 0.4
    python tools\\probe_reentry_menu.py --serial b1729535

截图存 `screenshots/_probe_reentry/r{n}_f{k}.png`，跑完把控制台输出发回来。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

IMG = ROOT / "resource" / "image"
OUT = ROOT / "screenshots" / "_probe_reentry"


def _imread(path: Path):
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def _best(img, tpl, roi):
    """返回 (score, (x, y))，坐标为整帧绝对坐标。"""
    x, y, w, h = [int(v) for v in roi]
    sub = img[y:y + h, x:x + w]
    if sub.shape[0] < tpl.shape[0] or sub.shape[1] < tpl.shape[1]:
        return 0.0, None
    r = cv2.matchTemplate(sub, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, ml = cv2.minMaxLoc(r)
    return float(mx), (x + ml[0], y + ml[1])


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None,
                    help="ADB serial；不填则用 ANDROID_SERIAL / 唯一在线设备")
    ap.add_argument("--rounds", type=int, default=2,
                    help="点几次「家園」（默认 2：一次开、一次关，用来验证 toggle）")
    ap.add_argument("--frames", type=int, default=8,
                    help="每次点击后连拍几帧（默认 8）")
    ap.add_argument("--interval", type=float, default=0.4,
                    help="连拍间隔秒（默认 0.4）")
    ap.add_argument("--short-side", type=int, default=720)
    args = ap.parse_args()

    from maa.controller import AdbController

    adb = os.environ.get("ADB_PATH") or "adb"
    if not args.serial:
        from adb_util import pick_serial
        args.serial = pick_serial(adb)
    print("[ok] adb serial: %s" % args.serial)

    ctrl = AdbController(adb, args.serial)
    ctrl.set_screenshot_target_short_side(args.short_side)
    job = ctrl.post_connection().wait()
    if not job.succeeded:
        print("[FATAL] 连不上 %s：%s" % (args.serial, job))
        return 2

    def shot():
        return ctrl.post_screencap().wait().get()

    nav = _imread(IMG / "nav_home.png")
    menu = _imread(IMG / "home_menu_first.png")
    popup = _imread(IMG / "obtain_resource.png")
    if nav is None or menu is None:
        print("[FATAL] 缺模板 nav_home.png / home_menu_first.png")
        return 2

    img = shot()
    h, w = img.shape[:2]
    print("[ok] 截图 %dx%d" % (w, h))

    # 底栏：nav_home 只可能在下 1/4
    nav_roi = [0, int(h * 0.75), w, h - int(h * 0.75)]
    nav_score, nav_xy = _best(img, nav, nav_roi)
    if nav_xy is None or nav_score < 0.8:
        print("[FATAL] 找不到底部「家園」tab：score=%.3f roi=%s"
              % (nav_score, nav_roi))
        print("        先退到冒险主页面再跑本脚本")
        return 2
    cx = nav_xy[0] + nav.shape[1] // 2
    cy = nav_xy[1] + nav.shape[0] // 2
    print("[ok] nav_home %.3f @%s -> 点击 (%d,%d)"
          % (nav_score, nav_xy, cx, cy))

    # 菜单项只可能出现在左下半屏
    menu_roi = [0, int(h * 0.55), int(w * 0.6), int(h * 0.45)]
    popup_roi = [0, int(h * 0.25), w, int(h * 0.35)]

    before_score, before_xy = _best(img, menu, menu_roi)
    print("[state] 点击前 home_menu_first=%.3f @%s（>=0.8 说明菜单已经开着）"
          % (before_score, before_xy))

    OUT.mkdir(parents=True, exist_ok=True)
    opened = []
    for r in range(1, args.rounds + 1):
        ctrl.post_click(cx, cy).wait()
        print("\n[round %d] click (%d,%d)" % (r, cx, cy))
        best = (0.0, None)
        for k in range(1, args.frames + 1):
            time.sleep(args.interval)
            img = shot()
            p_score, _ = _best(img, popup, popup_roi)
            m_score, m_xy = _best(img, menu, menu_roi)
            if m_score > best[0]:
                best = (m_score, m_xy)
            print("  f%d  menu=%.3f @%-14s popup=%.3f"
                  % (k, m_score, str(m_xy), p_score))
            cv2.imwrite(str(OUT / ("r%d_f%d.png" % (r, k))), img)
        opened.append(best[0] >= 0.8)
        print("  => 本轮最高 menu=%.3f @%s，判定菜单%s"
              % (best[0], best[1], "已展开" if opened[-1] else "没展开"))

    print("\n=== 结论 ===")
    if before_score >= 0.8 and not opened[0]:
        print("C：点之前菜单就是开的，这一点把它关掉了（「家園」是 toggle）")
    elif opened and opened[0] and len(opened) > 1 and not opened[1]:
        print("C：第 1 次打开、第 2 次关闭 —— 确认是 toggle，重复点击会关掉")
    elif not any(opened):
        print("A/B：两轮都没展开。看 r*_f* 里菜单有没有出现：")
        print("     - 出现了但分数 <0.8 → B（模板/位置对不上，需要重裁）")
        print("     - 完全没出现       → A（点击没生效）")
    else:
        print("正常：点击能打开菜单")
    print("截图目录：%s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
