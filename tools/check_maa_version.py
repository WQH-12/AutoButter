"""核对 pip 侧的 MaaFw 与 MFAAvalonia 内置的 MaaFramework 是否同一版本。

两边各带一份 MaaFramework，Agent 子进程握手时协议号必须相同
（HANDOVER §3.59）。协议号对不上时 GUI 只会报「Agent启动失败」，
CLI 却完全正常，所以单独做一个自检。

用法：
    python tools/check_maa_version.py
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTALL_NATIVE = os.path.join(ROOT, "install", "runtimes", "win-x64", "native")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main():
    try:
        pip_ver = importlib.metadata.version("MaaFw")
    except importlib.metadata.PackageNotFoundError:
        print("[FAIL] 没装 MaaFw（python -m pip install \"MaaFw==5.8.0\"）")
        return 1

    import maa
    pip_bin = os.path.join(os.path.dirname(os.path.abspath(maa.__file__)), "bin")

    print("pip   MaaFw          = %s" % pip_ver)
    print("pip   dll 目录       = %s" % pip_bin)
    print("install dll 目录     = %s" % INSTALL_NATIVE)

    if not os.path.isdir(INSTALL_NATIVE):
        print("\n[FAIL] 找不到 install/runtimes/win-x64/native")
        return 1

    names = sorted(n for n in os.listdir(INSTALL_NATIVE)
                   if n.lower().endswith(".dll"))
    if not names:
        print("\n[FAIL] install native 目录里没有 dll")
        return 1

    print()
    bad = []
    for n in names:
        a = os.path.join(INSTALL_NATIVE, n)
        b = os.path.join(pip_bin, n)
        if not os.path.exists(b):
            print("  [MISS] %-28s pip 侧没有这个文件" % n)
            bad.append(n)
            continue
        if _sha256(a) == _sha256(b):
            print("  [OK]   %-28s SAME" % n)
        else:
            print("  [DIFF] %-28s 两边不是同一版本" % n)
            bad.append(n)

    print()
    if bad:
        print("=== %d 个 dll 不一致 ===" % len(bad))
        print("    GUI 会报「Agent启动失败 / Protocol version mismatch」。")
        print("    先看 install/logs/log-YYYYMMDD.log 里的 "
              "`MaaFramework version: vX.Y.Z`，")
        print("    再 python -m pip install \"MaaFw==<同一个版本>\"")
        return 1

    print("=== 全部一致（%d 个 dll）===" % len(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
