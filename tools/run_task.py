"""直接调用 MaaFramework Python 绑定跑 pipeline，绕过 MFA GUI 便于快速迭代。

用法:
    python tools/run_task.py 进入家园
    python tools/run_task.py 进入家园 --dump screenshots/30_after_task.png
"""
import argparse
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
ADB = os.environ.get("ADB_PATH") or "adb"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("entry")
    ap.add_argument(
        "--serial",
        default=None,
        help="ADB serial；不填则用 ANDROID_SERIAL，或自动选唯一在线设备",
    )
    ap.add_argument("--short-side", type=int, default=720)
    ap.add_argument("--dump", default="")
    ap.add_argument("--resource", default=str(ROOT / "resource"))
    ap.add_argument("--no-popup-guard", action="store_true",
                    help="关闭全局弹窗护栏（默认每次截图前自动清「恭喜獲得」）")
    args = ap.parse_args()

    if not args.serial:
        try:
            from adb_util import pick_serial
            args.serial = pick_serial(ADB)
        except Exception as exc:  # noqa: BLE001
            print("[FATAL] 选择设备失败：%s" % exc)
            print("       先跑 adb devices；多台设备时用 --serial 指定")
            return 2
    print("[ok] adb serial: %s" % args.serial)

    from maa.resource import Resource
    from maa.tasker import Tasker

    if args.no_popup_guard:
        from maa.controller import AdbController
        ctrl = AdbController(ADB, args.serial)
        ctrl.set_screenshot_target_short_side(args.short_side)
    else:
        # 不能写 `from popup_guard import ...`：agent/ 下也有一个
        # popup_guard.py（清弹窗函数），同名模块会互相遮蔽（§3.66）。
        from cli_popup_guard import PopupGuardController
        ctrl = PopupGuardController(ADB, args.serial, short_side=args.short_side)
    job = ctrl.post_connection().wait()
    if not job.succeeded:
        print(f"[FATAL] controller connect failed: {job}")
        return 2
    img0 = ctrl.post_screencap().wait().get()
    print(f"[ok] connected, screencap shape={img0.shape} resolution={ctrl.resolution}")
    if not args.no_popup_guard:
        print("[ok] popup guard: on")

    res = Resource()
    job = res.post_bundle(args.resource).wait()
    if not job.succeeded:
        print(f"[FATAL] resource load failed: {job}")
        return 2
    print(f"[ok] resource loaded, nodes={sorted(res.node_list)}")

    # 自定义识别要在 bind 之前挂到 Resource 上（进程内模式）。
    # 不注册的话 pipeline 里的 Custom 节点会直接判任务失败，
    # 日志是 "recognition is null [param_.name=...]"。
    sys.path.insert(0, str(ROOT / "agent"))
    import registry
    reg = registry.register_all(res)
    print(f"[ok] custom recognition: {reg}")
    if not all(reg.values()):
        print("[WARN] 有自定义识别没注册成功，带 Custom 节点的任务会失败")

    tasker = Tasker()
    if not tasker.bind(res, ctrl):
        print("[FATAL] bind failed")
        return 2
    from maa.define import LoggingLevelEnum
    tasker.set_stdout_level(LoggingLevelEnum.Info)

    job = tasker.post_task(args.entry).wait()
    print(f"[task] status={job.status} succeeded={job.succeeded}")
    detail = tasker.get_task_detail(job.job_id)
    print(f"[task] detail={detail}")

    if args.dump:
        img = ctrl.post_screencap().wait().get()
        out = Path(args.dump)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
        print(f"[dump] {out} shape={img.shape}")

    return 0 if job.succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
