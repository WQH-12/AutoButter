"""对比不同 ADB 截图方式的耗时，找出最快的。"""
from __future__ import annotations

import statistics
import sys
import time

from maa.controller import AdbController
from maa.define import MaaAdbScreencapMethodEnum
from adb_util import adb_path, pick_serial

sys.stdout.reconfigure(encoding="utf-8")

METHODS = [
    ("Default", MaaAdbScreencapMethodEnum.Default),
    ("Encode", MaaAdbScreencapMethodEnum.Encode),
    ("EncodeToFileAndPull", MaaAdbScreencapMethodEnum.EncodeToFileAndPull),
    ("RawWithGzip", MaaAdbScreencapMethodEnum.RawWithGzip),
    ("RawByNetcat", MaaAdbScreencapMethodEnum.RawByNetcat),
]


def main() -> int:
    serial = sys.argv[1] if len(sys.argv) > 1 else pick_serial()
    results = []

    for name, method in METHODS:
        try:
            ctrl = AdbController(adb_path(), serial, screencap_methods=int(method))
            ctrl.set_screenshot_target_short_side(720)
            job = ctrl.post_connection().wait()
            if not job.succeeded:
                print(f"{name:22s} connect failed")
                continue

            caps = []
            for _ in range(6):
                t = time.perf_counter()
                img = ctrl.post_screencap().wait().get()
                caps.append((time.perf_counter() - t) * 1000)
            shape = img.shape
            med = statistics.median(caps)
            results.append((med, name, shape))
            print(f"{name:22s} median={med:7.1f}ms min={min(caps):7.1f}ms "
                  f"max={max(caps):7.1f}ms shape={shape}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name:22s} ERROR {type(exc).__name__}: {exc}")

    if results:
        results.sort()
        print(f"\nfastest: {results[0][1]} at {results[0][0]:.0f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
