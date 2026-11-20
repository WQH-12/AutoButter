"""ADB 设备选择工具：优先显式 serial / ANDROID_SERIAL，否则自动选唯一在线设备。"""
from __future__ import annotations

import os
import subprocess


def adb_path() -> str:
    return os.environ.get("ADB_PATH") or "adb"


def device_statuses(adb: str | None = None) -> list[tuple[str, str]]:
    """返回 [(serial, state), ...]，state 可能是 device / unauthorized / offline。"""
    adb = adb or adb_path()
    try:
        cp = subprocess.run(
            [adb, "devices"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("无法执行 adb devices: %s" % exc) from exc
    if cp.returncode != 0:
        raise RuntimeError("adb devices 失败: %s" % (cp.stderr or cp.stdout).strip())

    out = []
    for line in (cp.stdout or "").splitlines():
        line = line.strip()
        if (not line or line.startswith("*")
                or line.startswith("List of devices")):
            continue
        parts = line.split()
        if len(parts) >= 2:
            out.append((parts[0], parts[1]))
    return out


def pick_serial(adb: str | None = None) -> str:
    """按优先级返回 serial；没有或有多台在线设备时抛 RuntimeError。"""
    env_serial = os.environ.get("ANDROID_SERIAL")
    if env_serial:
        return env_serial

    statuses = device_statuses(adb)
    online = [serial for serial, state in statuses if state == "device"]
    if len(online) == 1:
        return online[0]
    if not online:
        detail = ", ".join("%s=%s" % item for item in statuses) or "无设备"
        raise RuntimeError("adb 没有在线设备（%s）" % detail)
    raise RuntimeError(
        "检测到多个在线设备: %s；请用 --serial 指定"
        % ", ".join(online))
