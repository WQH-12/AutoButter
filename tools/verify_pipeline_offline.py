"""离线跑 main.json 的流程校验（不需要手机）。

用 CustomController 喂预先录好的截图：每次 click 之后切到下一张，
相当于把「点一下会到什么界面」固定成剧本。
识别、next/on_error、超时这些都走真实的 MaaFramework 引擎。

用法：
    python tools/verify_pipeline_offline.py
    python tools/verify_pipeline_offline.py dispatch     # 只跑一个用例
    python tools/verify_pipeline_offline.py --missing    # 只列出缺失的帧
    python tools/verify_pipeline_offline.py --canvas 1600   # 换长边重跑全部
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "screenshots"
USER_SHOTS = ROOT / "User_screenshot"

# 离线剧本继续使用旧的全物品优先级，避免只收集酒米的新默认值
# 把 pick/row/main_head 等回归用例的目标改掉。
sys.path.insert(0, str(ROOT / "agent"))
import priority  # noqa: E402

priority.set_levels({
    "honey": 7,
    "jade": 6,
    "silver": 5,
    "berry": 4,
    "alcohol": 3,
    "fabric": 2,
    "stone": 1,
})


def load(name: str, canvas_h: int | None = None) -> np.ndarray:
    # 剧本帧主要来自 screenshots/，用户提供的整帧在 User_screenshot/
    # （含 full/ 子目录）。两处都找，避免为了一个用例来回挪文件。
    for base in (SHOTS, USER_SHOTS, USER_SHOTS / "full"):
        p = base / name
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return _to_canvas(img, canvas_h)
    img = cv2.imread(str(SHOTS / name))
    if img is None:
        raise FileNotFoundError(SHOTS / name)
    return _to_canvas(img, canvas_h)


def _to_canvas(img: np.ndarray, canvas_h: int | None) -> np.ndarray:
    """把帧重采样到指定长边（短边固定 720，与 Maa 的缩放一致）。

    只用于「换分辨率」回归：模板是按 720 短边裁的，长边变了 UI 的 y
    会等比缩放，正好用来验证 layout.py 的推导是否还框得住控件。
    """
    if canvas_h is None:
        return img
    if img.shape[0] == canvas_h:
        return img
    return cv2.resize(img, (img.shape[1], canvas_h),
                      interpolation=cv2.INTER_AREA)


def frame_exists(name: str) -> bool:
    return any((b / name).exists() for b in (SHOTS, USER_SHOTS, USER_SHOTS / "full"))


def reset_agent_state() -> None:
    """清掉上一个用例留下的进程内状态。

    `loop_ctl._STATE` / `cell_reco.LAST_PICK` / `row_reco` 的坏行缓存都是
    模块级全局量，同一进程里跑多个用例会互相污染。实测症状：
    `dispatch` 单独跑要加 2 次「+」，放进全套却直接掠奪 —— 因为前面的
    用例把 `_STATE["target"]` 改成了 1，`读派遣数量` 一次就命中。
    """
    try:
        import loop_ctl
        loop_ctl.reset_state()
    except Exception as exc:  # noqa: BLE001
        print(f"    [WARN] loop_ctl.reset_state failed: {exc}")
    for mod_name, fn in (("cell_reco", "clear_deferred"),
                         ("row_reco", "clear_bad_rows")):
        try:
            mod = __import__(mod_name)
            getattr(mod, fn)()
        except Exception:  # noqa: BLE001
            pass
    try:
        import cell_reco
        cell_reco.LAST_PICK = {}
    except Exception:  # noqa: BLE001
        pass


def build_controller(frames: list[np.ndarray]):
    from maa.controller import CustomController

    class ReplayController(CustomController):
        def __init__(self) -> None:
            super().__init__()
            self.frames = frames
            self.idx = 0
            self.clicks: list[tuple[int, int]] = []

        def _advance(self) -> None:
            if self.idx + 1 < len(self.frames):
                self.idx += 1

        def connect(self) -> bool:
            return True

        def request_uuid(self) -> str:
            return "replay-controller"

        def start_app(self, intent: str) -> bool:
            return True

        def stop_app(self, intent: str) -> bool:
            return True

        def screencap(self) -> np.ndarray:
            self.caps = getattr(self, "caps", 0) + 1
            if self.caps <= 12:
                print(f"      screencap#{self.caps} idx={self.idx}", flush=True)
            # 只有点击/触摸才推进剧本：Maa 在连接、加载资源时也会截图，
            # 让截图推进会导致剧本错位。
            return self.frames[min(self.idx, len(self.frames) - 1)].copy()

        def click(self, x: int, y: int) -> bool:
            self.clicks.append((x, y))
            if len(self.clicks) <= 12:
                print(f"      click#{len(self.clicks)} at ({x},{y}) idx={self.idx}", flush=True)
            self._advance()
            return True

        def swipe(self, x1, y1, x2, y2, duration) -> bool:
            self.clicks.append((x1, y1))
            self._advance()
            return True

        def touch_down(self, contact, x, y, pressure) -> bool:
            self._down = (x, y)
            return True

        def touch_move(self, contact, x, y, pressure) -> bool:
            return True

        def touch_up(self, contact) -> bool:
            pos = getattr(self, "_down", None)
            if pos is not None:
                self.clicks.append(pos)
                if len(self.clicks) <= 12:
                    print(f"      tap#{len(self.clicks)} at {pos} idx={self.idx}", flush=True)
                self._advance()
            return True

        def click_key(self, keycode: int) -> bool:
            return True

        def input_text(self, text: str) -> bool:
            return True

        def key_down(self, keycode: int) -> bool:
            return True

        def key_up(self, keycode: int) -> bool:
            return True

    return ReplayController()


def run_case(name: str, entry: str, frame_names: list[str], expect_ok: bool = True,
             expect_clicks: int | None = None,
             expect_first_click_in: tuple[int, int, int, int] | str | None = None,
             expect_nodes_include: list[str] | None = None,
             canvas_h: int | None = None,
             count_plan: list[int] | None = None) -> bool:
    """跑一个离线用例。

    `count_plan` 可选：按「当前已点击数」覆盖「雞妖數量 n/3」的 OCR 读数
    （下标 = min(点击数, len-1)）。仅用于 `dispatch` 这类需要「点 + 之后
    计数会变」的用例：仓库里没有 2/3、3/3 的真实帧，而 `ReplayController`
    只在点击时切帧，回读永远拿到 1/3。其它 ROI 的 OCR 仍走真实 Maa 引擎。
    """
    from maa.resource import Resource
    from maa.tasker import Tasker

    print(f"[start] {name}", flush=True)
    reset_agent_state()
    frames = [load(n, canvas_h) for n in frame_names]
    ctrl = build_controller(frames)
    print("    controller built", flush=True)
    job = ctrl.post_connection().wait()
    if not job.succeeded:
        print(f"  [FATAL] custom controller connect failed: {job}")
        return False
    print("    controller connected", flush=True)

    res = Resource()
    if not res.post_bundle(str(ROOT / "resource")).wait().succeeded:
        print("  [FATAL] resource load failed")
        return False
    print("    resource loaded", flush=True)

    # 进程内注册自定义识别，否则 Custom 节点会判任务失败
    sys.path.insert(0, str(ROOT / "agent"))
    import registry
    reg = registry.register_all(res)
    if not all(reg.values()):
        print(f"  [WARN] custom recognition not fully registered: {reg}")

    tasker = Tasker()
    if not tasker.bind(res, ctrl):
        print("  [FATAL] bind failed")
        return False
    from maa.define import LoggingLevelEnum
    tasker.set_stdout_level(LoggingLevelEnum.Info)

    real_ocr = None
    if count_plan:
        import loop_ctl
        real_ocr = loop_ctl._ocr_texts
        plan = [int(v) for v in count_plan]
        count_roi = (395, 790, 165, 45)

        def fake_ocr(context, image, roi, want=None):
            if tuple(int(v) for v in roi) == count_roi:
                i = min(len(ctrl.clicks), len(plan) - 1)
                print(f"      [mock-ocr] count -> {plan[i]}/3 "
                      f"(clicks={len(ctrl.clicks)})", flush=True)
                return f"{plan[i]}/3", []
            return real_ocr(context, image, roi, want=want)

        loop_ctl._ocr_texts = fake_ocr

    try:
        task = tasker.post_task(entry).wait()
        print("    task done", flush=True)
        detail = tasker.get_task_detail(task.job_id)
        nodes = [getattr(n, "name", "?") for n in getattr(detail, "nodes", [])]
    finally:
        if real_ocr is not None:
            import loop_ctl
            loop_ctl._ocr_texts = real_ocr

    ok = bool(task.succeeded)
    print(f"[{name}] entry={entry} frames={frame_names}")
    print(f"    succeeded={ok}  clicks={ctrl.clicks}")
    print(f"    nodes={nodes}")
    passed = ok == expect_ok
    if expect_clicks is not None and len(ctrl.clicks) != expect_clicks:
        print(f"    [MISMATCH] expected {expect_clicks} clicks, got {len(ctrl.clicks)}")
        passed = False
    if expect_first_click_in is not None:
        # 两种写法，都不会把 720x1560 的绝对坐标写死：
        #   "plus_roi"        -> 取 layout 推导出的 ROI（跟着设备变）
        #   (x, y, w, h)      -> 基线坐标，按 canvas_h/1560 等比缩放
        want_box = expect_first_click_in
        if isinstance(want_box, str):
            import layout
            got_roi = layout.get(want_box)
            if got_roi is None:
                print(f"    [MISMATCH] layout has no {want_box!r}")
                passed = False
                want_box = None
            else:
                want_box = tuple(int(v) for v in got_roi)
        elif canvas_h is not None:
            s = canvas_h / 1560
            want_box = tuple(round(v * s) for v in want_box)
        got = ctrl.clicks[0] if ctrl.clicks else None
        if want_box is None:
            pass
        elif got is None:
            print("    [MISMATCH] expected a first click, got none")
            passed = False
        else:
            x, y, w, h = want_box
            # Maa 的 Click 在识别框内取随机点，所以只校验「落在框内」
            if not (x <= got[0] <= x + w and y <= got[1] <= y + h):
                print(f"    [MISMATCH] first click {got} outside "
                      f"{want_box} ({expect_first_click_in})")
                passed = False
    if expect_nodes_include:
        for want in expect_nodes_include:
            if want not in nodes:
                print(f"    [MISMATCH] node {want!r} never ran; nodes={nodes}")
                passed = False
    return passed


def main() -> int:
    argv = sys.argv[1:]
    canvas_h: int | None = None
    if "--canvas" in argv:
        i = argv.index("--canvas")
        try:
            canvas_h = int(argv[i + 1])
        except (IndexError, ValueError):
            print("用法: --canvas <长边像素，例如 1600>")
            return 2
        del argv[i:i + 2]
    only = argv[0] if argv else ""

    results: dict[str, bool] = {}
    skipped: dict[str, str] = {}

    cases = {
        "realm": ("确认已进入秘境@秘境面板", "确认已进入秘境", ["113_final2.png"],
                  True, {"expect_clicks": 0}),
        "nav": ("进入淫乱秘境@主界面", "进入淫乱秘境",
                ["21_main_720.png", "104_tap_home.png", "113_final2.png"],
                True, {"expect_clicks": 2}),
        # 进秘境后先弹「恭喜獲得」：点家園 → 点淫乱秘境 → 弹窗 ×2 → 秘境面板。
        # 4 次点击：家園、淫乱秘境、弹窗1、弹窗2；最后 `确认已进入秘境` 必须成功。
        "nav_popup": ("进入淫乱秘境@进场弹窗", "进入淫乱秘境",
                      ["21_main_720.png", "104_tap_home.png",
                       "100_now.png", "101_after_tap1.png",
                       "102_popup_gone.png"],
                      True, {"expect_clicks": 4}),
        "already": ("进入淫乱秘境@已在秘境", "进入淫乱秘境", ["113_final2.png"],
                    True, {"expect_clicks": 0}),
        # 菜单始终打不开：`打开家园菜单` 点 1 次没展开 → `开菜单闸门`
        # （每轮 3 次）放行 → 再点，直到闸门用光 → `重进失败` →
        # `重进失败重试门`（这一帧本来就是主页面，RecoverNet 第一轮就返回
        # True、不点蓝点）→ `重进-找菜单` 仍失败 → `异常收尾`。
        #
        # §3.50：`异常收尾` 判失败（需求方 11-15 指定）。
        # 断言：任务**有界结束**（4 次点击 = 首次 1 次 + 3 次重试，不空转），
        # 恢复门确实被执行过，且最终是红色失败而不是静默成功。
        "stuck": ("进入淫乱秘境@菜单打不开（有界收尾）", "进入淫乱秘境",
                  ["21_main_720.png"], False,
                  {"expect_clicks": 4,
                   "expect_nodes_include": ["重进失败重试门", "异常收尾"]}),
        "empty": ("关闭获得弹窗@无弹窗", "关闭获得弹窗", ["113_final2.png"],
                  True, {"expect_clicks": 0}),
        "popups": ("关闭获得弹窗@两条弹窗", "关闭获得弹窗",
                   ["100_now.png", "101_after_tap1.png", "102_popup_gone.png"],
                   True, {"expect_clicks": 2}),
        # 选格：159_before_drain 有 2 列被鸡妖占着，PickCell 会挑 honey @(614,703)，
        # 点完切到「資源詳情」浮窗（720_dispatch_menu），确认 btn_dispatch 命中
        # （`53_rob_popup.png` 与 `full/720_dispatch_menu.png` 逐像素相同，
        #  旧名字的文件已丢，直接用还在的那张）
        "pick": ("选格并点击@有目标资源", "选格并点击",
                 ["159_before_drain.png", "full/720_dispatch_menu.png"],
                 True, {"expect_clicks": 1}),
        # 负例：user_test_1 的 5 个格子全被鸡妖占着，PickCell 返回 None，
        # 应走 on_error 落到「没有目标资源」，任务成功且 0 次点击
        "pick_none": ("选格并点击@全部被抢（应成功且不点）", "选格并点击",
                      ["user_test_1.png"],
                      True, {"expect_clicks": 0}),
        # 选行：720_list_top 第 0 行不是采集中、含 honey，
        # PickRow 应返回该行「前往」按钮 (556,454,94,26)
        # 只校验**第一次点击**落在按钮框内：点完会继续走
        # 确认他人秘境 →（列表上没有管理按钮）→ 滑动 → 重选，循环次数由 max_hit 兜住，
        # 总点击数不是这个用例要断言的东西
        # 注意：这里的框是行内按钮的位置，跟 layout 推导无关（列表行的 y
        # 由 row_reco 自己扫），所以保留绝对值。
        "row": ("选行并点击@列表有目标", "选行并点击",
                ["full/720_list_top.png"],
                True, {"expect_first_click_in": (556, 454, 94, 26)}),
        # 采集中：20261108-152332 第 1、2 行是「采集中」（0.997/0.996），
        # 第 0 行不是，所以应该点第 0 行 (556,480,94,26) 而不是跳过整页
        "row_collect": ("选行并点击@跳过采集中行", "选行并点击",
                        ["snap/20261108-152332_720.png"],
                        True, {"expect_first_click_in": (556, 480, 94, 26)}),
        # 整页都是采集中：`选行并点击` 是**调试入口**（没有 next），
        # 识别不到行就 0 次点击、任务成功收尾。
        # 滑动循环在主链（`主循环选行` → `滑动闸门` → `列表需要滑动`）里，
        # 由下面的 `stuck_restart` 覆盖
        "row_all_collecting": ("选行并点击@整页采集中", "选行并点击",
                               ["full/720_fullscreen_activity.png"],
                               True, {"expect_clicks": 0}),
        # 点探尋：别人秘境右下角是「探尋」（OCR 读 0.99），自己的是「託管中」。
        # 这个节点是「别人秘境 → 回列表」的唯一出口，两个位置（x 483 / 591）都要能点中
        "tanxun": ("点探尋@别人秘境", "点探尋",
                   ["full/720_fullscreen_others_home.png",
                    "full/720_list_top.png"],
                   True, {"expect_clicks": 1}),
        # 第一次点击没进列表（同一帧再来一次），ClickTanxun 应重测后再点一次。
        "tanxun_retry": ("点探尋@第一次未进列表后重测", "点探尋",
                         ["full/720_fullscreen_others_home.png",
                          "full/720_fullscreen_others_home.png",
                          "full/720_list_top.png"],
                         True, {"expect_clicks": 2,
                                "expect_first_click_in": (591, 1210, 80, 100)}),
        # 别人秘境里点「探尋」并下翻：先点探尋（OCR 框 591,1264,67,43），
        # 再滑一页。断言第一次动作落在探尋按钮上。
        #
        # §3.51 修正：滑动判据从「灰度帧差」改成「行锚点 y 序列」后，
        # 原来喂 `720_list_top` ×2 的剧本会被正确判成「划不动」
        # （两帧行锚点完全一样 `[454,606,757,967]`）——那是剧本本身在
        # 模拟划不动，不是代码错。改成滑完换到**行锚点不同**的帧
        # （`720_fullscreen_activity` 是 `[397,548,700,851,1003]`）。
        #
        # §3.56：4 帧用完后，`ReplayController` 一直返回最后一帧，
        # `SwipeGate` 正确判「划不动」→ 整条重进链跑满 3 次配额 →
        # `异常收尾`。这是**剧本帧数不够**，不是链路错。所以断言改成
        # 「必须走到这些节点」而不是「任务成功」：少了帧就跑不到
        # `重进失败重试门`，接线断了同样跑不到。
        "others_tanxun": ("点探尋并下翻@别人秘境", "点探尋并下翻",
                          ["full/720_fullscreen_others_home.png",
                           "full/720_list_top.png",
                           "full/720_fullscreen_activity.png",
                           "full/720_fullscreen_activity.png"],
                          False,
                          {"expect_first_click_in": (591, 1210, 80, 100),
                           "expect_nodes_include": [
                               "点探尋并下翻", "主循环选行", "滑不动了",
                               "退到主页面", "回主界面", "重进-开菜单",
                               "重进失败", "重进失败重试门", "异常收尾"]}),
        # 派遣：点 + → 点 + → 掠奪 → 回到秘境面板。
        # 3 次点击：plus, plus, 掠奪；落地校验用 realm_marker（56_robbed 上有）
        #
        # 帧序列用重复的 1/3 帧，而不是历史上的 54_plus1(2/3) / 55_plus2(3/3)：
        # 那两张图仓库里没有（`--missing` 会列出来）。计数变化由
        # `count_plan` 模拟（见下面的注释）；「点击被弹窗吃掉、补点差额」
        # 那条路径由 test_dispatch_policy.py 的 `_AddCtrl` 覆盖。
        #
        # 断言用 layout 的 ROI key 而不是绝对坐标：点击落点由 layout 推导，
        # 换成别的长边重跑（--canvas 1600）时断言跟着一起变。
        "dispatch": ("派遣并掠奪@1/3 加到 3/3 再掠奪", "派遣并掠奪",
                     ["full/720_dispatch_menu.png",
                      "full/720_dispatch_menu.png",
                      "full/720_dispatch_menu.png",
                      "56_robbed.png"],
                     True, {"expect_clicks": 3,
                            "expect_first_click_in": "plus_roi",
                            # §3.67：AddChicken 现在点完会回读复核。
                            # 仓库里没有 2/3、3/3 的真实帧（on_error 里
                            # 那几张都是「恭喜獲得」盖住计数），而
                            # ReplayController 只在点击时切帧，回读只会
                            # 拿到同一张 1/3 → 永远补不到 3/3。
                            # 这里只替换计数 ROI 的 OCR 读数，按已点击数映射：
                            #   0 次点击 → 1/3（点击前 / 超时重试）
                            #   1 次点击 → 1/3
                            #   2 次点击 → 3/3（点完复核 / 读派遣数量）
                            # 点击、引擎、pipeline 路由仍走真实路径。
                            "count_plan": [1, 1, 3]}),
        # 划不动 → 回主界面 → 重进秘境（§5 的 (@1) 尾部闭环）。
        # 入口是 `主循环选行`（主链节点，带 next / on_error）。
        #
        # §3.51 后剧本需要改：滑动判据从「灰度帧差」换成「行锚点 y 序列」，
        # 喂同一帧两次会被**立刻**判成划不动（不再滑一页）。
        # 所以这里用「列表帧 → 主界面帧」直接模拟「滑不动了」的下一帧，
        # 省掉那一次注定被拒的滑动。
        #
        # 剧本：整页采集中（无行可点）→ 闸门判划不动 → 滑不动了
        #      → 循环闸门 → 点家園 → 主界面菜单收着 → 再点家園展开
        #      → 点淫乱秘境 → 落地校验 → 主循环选格 → 无格子 → 探尋 → 收尾
        "stuck_restart": ("主循环选行@划不动后回主界面重进", "主循环选行",
                          ["full/720_fullscreen_activity.png",
                           "21_main_720.png",
                           "104_tap_home.png",
                           "113_final2.png"],
                          False, {}),
        # 主任务前段：进秘境 → 选格（113_final2 有 honey）→ 点开 → 浮窗
        # → 派遣并掠奪（53 1/3 → 54 2/3 → 55 3/3）→ 掠奪 → 回到秘境面板
        # → 主循环选格（56_robbed 有 berry）→ 再点开 → …
        #
        # §3.56：只给 3 帧，跑到「掠奪后第二次选格」就没帧了，
        # 后面必然滑不动 → 重进链 → `异常收尾`。所以断言改成
        # 「根节点 → 进秘境 → 派遣 → 掠奪」这条前段接线跑通。
        # 这是**全仓库唯一**以 `主任务` 为 entry 的用例之一（另一个是
        # `main_head_popup`），别删。
        "main_head": ("主任务@进秘境→选格→派遣", "主任务",
                      ["113_final2.png", "full/720_dispatch_menu.png",
                       "56_robbed.png"],
                      False,
                      {"expect_nodes_include": [
                          "主任务", "进入淫乱秘境", "确认已进入秘境",
                          "读空闲鸡妖", "主循环选格", "确认资源详情",
                          "派遣并掠奪", "派遣策略", "读派遣数量", "掠奪"]}),
        # 主任务从主界面开始，点进秘境先弹两次「恭喜獲得」，再进入选格/派遣。
        # 这条同时验证 RewireMain 与 `清弹窗后继续` 的衔接。
        #
        # §3.56：同 `main_head`，7 帧跑到第二次选格就没了。
        # 断言集中在「弹窗链 → 进秘境 → 派遣」，这一段是它独有的覆盖。
        "main_head_popup": ("主任务@进场弹窗→选格→派遣", "主任务",
                            ["21_main_720.png", "104_tap_home.png",
                             "100_now.png", "101_after_tap1.png",
                             "113_final2.png", "full/720_dispatch_menu.png",
                             "56_robbed.png"],
                            False,
                            {"expect_nodes_include": [
                                "主任务", "进入淫乱秘境", "打开家园菜单",
                                "进入家园", "清弹窗后继续", "确认已进入秘境",
                                "主循环选格", "派遣并掠奪", "掠奪"]}),
    }

    if only == "--missing":
        return report_missing(cases)

    for key, case in cases.items():
        if only and key != only:
            continue
        label, entry, frames = case[0], case[1], case[2]
        expect_ok = case[3] if len(case) > 3 else True
        extra = case[4] if len(case) > 4 else {}
        missing = [n for n in frames if not frame_exists(n)]
        if missing:
            # 缺帧的用例跳过，不要让整批回归在 FileNotFoundError 上中断 ——
            # 后面的用例本来能跑，一炸就全都看不到结果。
            skipped[label] = ", ".join(missing)
            print("[skip] %s 缺帧: %s" % (label, skipped[label]), flush=True)
            continue
        if canvas_h is not None:
            extra = dict(extra, canvas_h=canvas_h)
        results[label] = run_case(label, entry, frames, expect_ok, **extra)

    print("\n=== summary ===")
    ok = True
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
        ok = ok and v
    if skipped:
        print("\n=== skipped (缺帧) ===")
        for k, v in skipped.items():
            print(f"  SKIP  {k}  <- {v}")
    return 0 if ok else 1


def report_missing(cases: dict) -> int:
    """列出所有用例里缺的帧，按文件名去重。"""
    need: dict[str, list[str]] = {}
    for key, case in cases.items():
        for name in case[2]:
            if not frame_exists(name):
                need.setdefault(name, []).append(key)
    if not need:
        print("所有用例的帧都在。")
        return 0
    print("缺 %d 张图：" % len(need))
    for name, users in sorted(need.items()):
        print("  %-32s 被这些用例用到: %s" % (name, ", ".join(users)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
