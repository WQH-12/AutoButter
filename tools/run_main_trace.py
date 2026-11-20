"""实跑主任务，并逐节点打印 trace（上设备用）。

和 `run_task.py` 的区别：

- `run_task.py` 只在最后打印 task detail，中间过程看不到
- 本脚本挂一个 ContextEventSink，**每执行完一个节点就打一行**：

      14:03:12  +0.8s  主循环选格           reco=hit  box=[556,454,94,26]  act=Click   ok
      14:03:13  +1.9s  确认他人秘境         reco=miss                     act=-       FAIL

- 同时在 `screenshots/trace/` 下存一份 `.log`（纯文本）和 `.jsonl`（结构化）
- 内置看门狗：超过 `--max-seconds` 还没结束就 `post_stop()`，不会无限跑
- Ctrl+C 也会优雅停止

`.log` 里除了逐节点事件，还会收录 agent 侧的 `print()` —— `[Layout]` /
`[RecoverNet]` / `[ClickTanxun]` 这些诊断行以前只打到控制台，落不到文件，
排查时只能靠 `reco=hit/miss` 反推。

用法：

    python tools\\run_main_trace.py                  # 默认跑「主任务」
    python tools\\run_main_trace.py --entry 进入淫乱秘境
    python tools\\run_main_trace.py --max-seconds 300
    python tools\\run_main_trace.py --check-only     # 只连设备+加载+注册，不跑任务
    python tools\\run_main_trace.py --verbose        # 连 Maa 的 Info 日志一起打
    python tools\\run_main_trace.py --no-popup-guard # 调试时关闭全局弹窗护栏

跑完把控制台最后 30 行，或者 `screenshots/trace/*.log`，发回来即可。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class _Tee:
    """把 stdout 同时写进 trace 的行缓冲。

    为什么需要：`agent/*.py` 里的 `print(...)` 是**直接写 stdout** 的，
    而 trace 文件原来只收 sink 里 `_emit()` 的内容（节点事件）。
    结果 `[Layout] measured screen=...`、`[RecoverNet] round=1/8 ...`
    这些最关键的诊断行全部丢失 —— 11-14 那次「恢复门从未执行」
    就是因为日志里看不到它的输出，只能靠节点序列反推。

    `print()` 会分多次 write（正文和换行是两次），所以这里按行缓冲，
    不完整的片段留到下次补全。
    """

    def __init__(self, stream, lines: list[str]):
        self._stream = stream
        self._lines = lines
        self._buf = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        # 先透传到真实 stdout，控制台才能实时看到（否则跑的时候一片空白）
        self._stream.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._lines.append(line)
        return len(text)

    def flush(self) -> None:
        # 末尾没有换行的残片也要落盘，否则最后一行会丢
        if self._buf:
            self._lines.append(self._buf)
            self._buf = ""
        self._stream.flush()

    def __getattr__(self, name):
        # 其余属性（encoding / isatty / fileno …）透传给真实 stdout
        return getattr(self._stream, name)


def _fmt_box(box) -> str:
    if not box:
        return "-"
    # 原始事件里是 [x,y,w,h] 列表；节点详情里是 Rect 对象
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return "[%d,%d,%d,%d]" % tuple(int(v) for v in box)
    if isinstance(box, dict):
        try:
            return "[%d,%d,%d,%d]" % (box["x"], box["y"], box["w"], box["h"])
        except Exception:  # noqa: BLE001
            return str(box)
    try:
        return "[%d,%d,%d,%d]" % (box.x, box.y, box.w, box.h)
    except Exception:  # noqa: BLE001
        return str(box)


def build_sink(started: float, log_lines: list[str], jsonl, verbose: bool):
    """构造事件回调。

    注意：**节点事件走 ContextEventSink，不走 TaskerEventSink**。
    Tasker 级别的 sink 只收 `Tasker.Task.*`，节点级别的 `Node.*` 全部投给
    `Tasker.add_context_sink()` 注册的 sink（实测 tasker sink 收不到任何 Node 事件）。
    """
    from maa.context import ContextEventSink

    class TraceSink(ContextEventSink):
        """只关心「一个节点执行完」这一类事件。"""

        def __init__(self):
            super().__init__()
            self.t0 = started
            self.lines = log_lines
            self.jsonl = jsonl
            self.verbose = verbose
            self.node_counts: dict[str, int] = {}
            self.fail_streak = 0
            self.max_fail_streak = 0

        # --- 内部 ---
        def _emit(self, text: str) -> None:
            # 不直接 append：`main()` 里把 stdout 换成了 _Tee，它会统一收集。
            # 两边都写会重复一行。单一来源还能保证节点事件和 agent 的
            # `[Layout]` / `[RecoverNet]` 按真实时间顺序交错。
            print(text, flush=True)

        def _jsonl(self, obj: dict) -> None:
            if self.jsonl is None:
                return
            self.jsonl.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.jsonl.flush()

        # --- Maa 回调 ---
        def on_node_pipeline_node(self, context, noti_type, detail):  # noqa: ANN001
            elapsed = time.perf_counter() - self.t0
            ok = str(getattr(noti_type, "name", "")).startswith("Succeeded")
            box = "-"
            action = "-"
            name = detail.name
            # 只在成功时查详情：失败的节点 Maa 侧没有 node_detail，
            # 查了会多打一条无关的 ERR 日志
            if ok:
                try:
                    nd = context.tasker.get_node_detail(detail.node_id)
                except Exception:  # noqa: BLE001
                    nd = None
                if nd is not None:
                    # 事件里的 detail.name 是父/焦点节点名，成功节点的
                    # node_details.name 才是刚执行完的节点名（11-11 实机确认）。
                    name = nd.name or name
                    reco = getattr(nd, "recognition", None)
                    act = getattr(nd, "action", None)
                    if reco is not None:
                        box = _fmt_box(getattr(reco, "box", None))
                    if act is not None:
                        action = getattr(act, "action", None) or "-"

            if ok:
                self._emit("%6.1fs  %-20s reco=hit  box=%-18s act=%-9s ok"
                           % (elapsed, name, box, action))
                self.node_counts[name] = \
                    self.node_counts.get(name, 0) + 1
                self.fail_streak = 0
            else:
                self._emit("%6.1fs  %-20s reco=miss" % (elapsed, name))
                self.fail_streak += 1
                self.max_fail_streak = max(self.max_fail_streak, self.fail_streak)
                if self.fail_streak in (5, 20, 50):
                    self._emit("  [WARN] 连续失败 %d 次，可能卡住了"
                               % self.fail_streak)

            self._jsonl({"t": round(elapsed, 2), "event": "node",
                         "name": name, "ok": ok,
                         "box": box, "action": action})

        def on_unknown_notification(self, context, msg: str, details: dict) -> None:
            if msg.startswith("Tasker.Task"):
                self._jsonl({"t": round(time.perf_counter() - self.t0, 2),
                             "event": "task", "msg": msg})
                if self.verbose:
                    self._emit("%6.1fs  %s"
                               % (time.perf_counter() - self.t0, msg))

    return TraceSink()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="主任务")
    ap.add_argument(
        "--serial",
        default=None,
        help="ADB serial；不填则用 ANDROID_SERIAL，或自动选唯一在线设备",
    )
    ap.add_argument("--short-side", type=int, default=720)
    ap.add_argument("--max-seconds", type=float, default=600.0,
                    help="看门狗：超过这个时间就停（默认 600s）")
    ap.add_argument("--verbose", action="store_true",
                    help="把 Maa 自己的 Info 日志也打出来（很吵）")
    ap.add_argument("--check-only", action="store_true",
                    help="只连设备 + 加载资源 + 注册识别，不跑任务（安全自查）")
    ap.add_argument("--no-popup-guard", action="store_true",
                    help="关闭全局弹窗护栏（默认每次截图前自动清「恭喜獲得」）")
    args = ap.parse_args()

    from maa.resource import Resource
    from maa.tasker import Tasker
    from maa.define import LoggingLevelEnum

    ADB = os.environ.get("ADB_PATH") or "adb"
    if not args.serial:
        try:
            from adb_util import pick_serial
            args.serial = pick_serial(ADB)
        except Exception as exc:  # noqa: BLE001
            print("[FATAL] 选择设备失败：%s" % exc)
            print("       先跑 adb devices；多台设备时用 --serial 指定")
            return 2
    print("[ok] adb serial: %s" % args.serial)
    if args.no_popup_guard:
        from maa.controller import AdbController
        ctrl = AdbController(ADB, args.serial)
        ctrl.set_screenshot_target_short_side(args.short_side)
    else:
        # 每次 screencap 前先检查并清掉「恭喜獲得」；清完才把画面交给 Maa。
        # 不能写 `from popup_guard import ...`：agent/ 下也有一个
        # popup_guard.py（清弹窗函数），同名模块会互相遮蔽（§3.66）。
        from cli_popup_guard import PopupGuardController
        ctrl = PopupGuardController(ADB, args.serial, short_side=args.short_side)
    job = ctrl.post_connection().wait()
    if not job.succeeded:
        print("[FATAL] 连不上设备 %s：%s" % (args.serial, job))
        print("       先跑 adb devices，确认手机已授权")
        return 2
    img = ctrl.post_screencap().wait().get()
    print("[ok] 已连接 %s，截图 %s" % (args.serial, img.shape))
    if not args.no_popup_guard:
        print("[ok] 全局弹窗护栏已开启（每次截图前自动清「恭喜獲得」）")

    res = Resource()
    if not res.post_bundle(str(ROOT / "resource")).wait().succeeded:
        print("[FATAL] 资源加载失败")
        return 2

    sys.path.insert(0, str(ROOT / "agent"))
    import registry
    reg = registry.register_all(res)
    print("[ok] 自定义识别/动作: %s" % reg)
    if not all(reg.values()):
        print("[FATAL] 有自定义识别没注册成功，主任务跑不起来")
        return 2

    tasker = Tasker()
    if not tasker.bind(res, ctrl):
        print("[FATAL] bind 失败")
        return 2
    # 枚举名是 Warn 不是 Warning（11-11 实测）。
    # 默认给 Error：Warn 档会刷一堆 adb 子进程退出 / MuMu 探测的噪声，
    # 而真正要看的逐节点 trace 是本脚本自己打的，与 Maa 日志级别无关。
    tasker.set_stdout_level(
        LoggingLevelEnum.Info if args.verbose else LoggingLevelEnum.Error)

    if args.check_only:
        print("\n[check-only] 连接、资源、注册都正常，不执行任务。")
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = ROOT / "screenshots" / "trace"
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / ("%s_%s.log" % (args.entry, stamp))
    jsonl_path = outdir / ("%s_%s.jsonl" % (args.entry, stamp))

    print("\n=== 开始跑 %s ===" % args.entry)
    print("    看门狗 %ds；Ctrl+C 可中止" % int(args.max_seconds))
    print("    trace 会存到 %s\n" % log_path)

    started = time.perf_counter()
    lines: list[str] = []
    jsonl = open(jsonl_path, "w", encoding="utf-8")
    sink = build_sink(started, lines, jsonl, args.verbose)
    # 节点事件必须挂 context sink；tasker sink 只收 Tasker.Task.*（实测）
    tasker.add_context_sink(sink)

    # 从这里开始接管 stdout：agent 侧的 `[Layout]` / `[RecoverNet]` 等
    # print 会同时进控制台和 trace 文件。
    real_stdout = sys.stdout
    tee = _Tee(real_stdout, lines)
    sys.stdout = tee
    ok = False
    stopped_by_watchdog = False
    try:
        task = tasker.post_task(args.entry)
        # 不用 task.wait()：它阻塞，看门狗和 Ctrl+C 都没机会插进来。
        # 轮询 task.done（不能用 task.running：刚 post 出去时是 Pending，
        # running 还是 False，那样循环会立刻退出，误报「已结束」）。
        deadline = started + args.max_seconds
        try:
            while not task.done:
                if time.perf_counter() > deadline:
                    print("\n[看门狗] 超过 %.0fs 还没结束，停止任务"
                          % args.max_seconds)
                    tasker.post_stop().wait()
                    stopped_by_watchdog = True
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n[中断] 收到 Ctrl+C，正在停止任务…")
            tasker.post_stop().wait()

        elapsed = time.perf_counter() - started
        try:
            ok = bool(task.succeeded)
        except Exception:  # noqa: BLE001
            ok = False
        print("\n=== 结束 ===")
        print("  succeeded=%s  elapsed=%.1fs%s"
              % (ok, elapsed, "（被看门狗停）" if stopped_by_watchdog else ""))
        print("  节点执行次数：")
        for name, cnt in sorted(sink.node_counts.items(), key=lambda kv: -kv[1]):
            print("    %3d  %s" % (cnt, name))
        if sink.max_fail_streak >= 5:
            print("  [注意] 最长连续失败 %d 次" % sink.max_fail_streak)
        if not args.no_popup_guard and hasattr(ctrl, "popup_clicks"):
            print("  弹窗护栏清理：%d 个" % ctrl.popup_clicks)
        tee.flush()
    finally:
        sys.stdout = real_stdout
        try:
            jsonl.close()
        except Exception:  # noqa: BLE001
            pass
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("  trace: %s" % log_path)
    print("  jsonl: %s" % jsonl_path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
