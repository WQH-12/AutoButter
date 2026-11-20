#!/usr/bin/env python3
"""构建 AutoButter 发布包。

把「发布所需」从工作区（interface.json + resource/ + agent/ + install/）整理到
一个独立文件夹，每类元素各占一个子目录：

    release/AutoButter/
      MFAAvalonia.exe / *.dll / libs/ / runtimes/ / plugins/
                                运行时（MFA 要求与 exe 同级，无法再拆）
      interface.json            Maa 项目入口（必须在 exe 同级）
      resource/pipeline/*.json  流水线
      resource/image/*.png      模板
      resource/model/ocr/*      OCR 模型
      agent/*.py                自定义识别/动作
      docs/README.md            项目说明
      docs/使用说明.md          终端用户说明
      docs/LICENSE
      manifest-sha256.txt       发布包文件清单

用法：
    python tools/build_release.py --clean
    python tools/build_release.py --version v0.2.0 --zip
    python tools/build_release.py --no-runtime          # 只出 Maa 项目部分
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT
AGENT = ROOT / "agent"
INSTALL = ROOT / "install"
RELEASE_ROOT = ROOT / "release"
DEFAULT_OUT = RELEASE_ROOT / "AutoButter"

# 运行时：MFAAvalonia 自带的 .NET 程序，必须与 exe 同级
RUNTIME_FILES = [
    "MFAAvalonia.exe",
    "MFAAvalonia.dll",
    "MFAAvalonia.deps.json",
    "MFAAvalonia.runtimeconfig.json",
    "libloader.dll",
    "appsettings.json",
]
RUNTIME_DIRS = ["libs", "runtimes", "plugins"]
MODEL_FILES = ["det.onnx", "rec.onnx", "keys.txt", "README.md"]

VERSION_RE = re.compile(r'("version"\s*:\s*")([^"]*)(")')

USAGE_DOC = """# AutoButter 使用说明

## 运行环境

- Windows 10/11 x64
- Python 3.10+，命令名必须是 `python`（MFAAvalonia 用它在后台启动 agent）
- MaaFw Python 包：`python -m pip install "MaaFw==5.8.0"`
- adb 在 PATH 中（本包不自带 adb，可用 platform-tools 或 QtScrcpy 自带的那份）
- 手机开启 USB 调试并授权本机

## 首次启动

1. 双击 `MFAAvalonia.exe`。
2. 在「设置 → 设备」里选择手机，点截图确认画面正常（不是黑屏）。
3. 任务队列勾选「主任务」，其它调试任务不要勾。
4. 手机停在冒险主页面（或秘境列表 / 自家秘境面板）再点开始。

`config/` 和 `logs/` 由 MFAAvalonia 首次运行时自动生成，不在发布包内。

## 目录说明

| 路径 | 作用 |
|---|---|
| `interface.json` | Maa 项目入口：任务表、选项、agent 启动参数 |
| `resource/pipeline/` | 流水线 JSON |
| `resource/image/` | 模板素材（720 短边画布） |
| `resource/model/ocr/` | OCR 模型 |
| `agent/` | 自定义识别/动作（Python 子进程） |
| `docs/` | 说明与许可证 |
| `manifest-sha256.txt` | 发布包文件校验清单 |

## 校验

```powershell
Get-Content manifest-sha256.txt | ForEach-Object {
    $h, $p = $_ -split '  ', 2
    $a = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower()
    if ($a -ne $h) { "MISMATCH $p" }
}
```
"""


README_DOC = """# AutoButter

基于 [MaaFramework](https://github.com/MaaXYZ/MaaFramework) 的安卓小游戏资源自动获取工具。

本目录是可直接运行的发布包：`interface.json` + `resource/` + `agent/` + MFAAvalonia 运行时。

## 运行环境

- Windows 10/11 x64
- Python 3.10+（命令名 `python`）与 `MaaFw==5.8.0`
- adb 在 PATH 中；手机开启 USB 调试

详细步骤见 `docs/使用说明.md`。

## 目录

| 路径 | 作用 |
|---|---|
| `MFAAvalonia.exe` | 图形界面入口 |
| `interface.json` | Maa 项目入口：任务表、选项、agent 启动参数 |
| `resource/pipeline/` | 流水线 JSON |
| `resource/image/` | 模板素材（720 短边画布） |
| `resource/model/ocr/` | OCR 模型 |
| `agent/` | 自定义识别/动作（Python 子进程） |
| `docs/` | 使用说明、许可证 |
| `manifest-sha256.txt` | 文件校验清单 |

## 许可

MIT，见 `docs/LICENSE`。
"""

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True)


def dir_size(path: Path) -> tuple[int, int]:
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--version", default="",
                    help="写入 interface.json 的版本号；不填则沿用源文件")
    ap.add_argument("--clean", action="store_true",
                    help="构建前清空目标目录（仅允许 release\\ 下的路径）")
    ap.add_argument("--no-runtime", action="store_true",
                    help="不复制 MFAAvalonia 运行时，只出 Maa 项目部分")
    ap.add_argument("--zip", action="store_true", help="构建后打包成 zip")
    args = ap.parse_args()

    out = Path(args.out).resolve()

    if args.clean and out.exists():
        release_root = RELEASE_ROOT.resolve()
        try:
            out.relative_to(release_root)
            inside = True
        except ValueError:
            inside = False
        if not inside or out == release_root:
            print("[FATAL] 拒绝清理：--out 必须位于 %s 的子目录下（当前 %s）"
                  % (release_root, out))
            return 2
        shutil.rmtree(out)

    required = [
        PROJECT / "interface.json",
        PROJECT / "resource" / "pipeline" / "main.json",
        PROJECT / "resource" / "image",
        PROJECT / "resource" / "model" / "ocr",
        AGENT / "main.py",
    ]
    if not args.no_runtime:
        required.append(INSTALL / "MFAAvalonia.exe")
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("[FATAL] 缺少源文件：")
        for p in missing:
            print("        " + p)
        return 2

    out.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[Path, Path]] = []

    if not args.no_runtime:
        print("[1/6] 运行时 -> %s" % out)
        for name in RUNTIME_FILES:
            copy_file(INSTALL / name, out / name)
        for bat in sorted(INSTALL.glob("DependencySetup*.bat")):
            copy_file(bat, out / bat.name)
        for name in RUNTIME_DIRS:
            src = INSTALL / name
            if src.is_dir():
                copy_tree(src, out / name)
            else:
                print("      [WARN] 缺少运行时目录 %s" % src)
    else:
        print("[1/6] 跳过运行时（--no-runtime）")

    print("[2/6] interface.json")
    text = (PROJECT / "interface.json").read_bytes().decode("utf-8-sig")
    m = re.search(r'"version"\s*:\s*"([^"]*)"', text)
    version = args.version or (m.group(1) if m else "0.0.0")
    if args.version:
        text, n = VERSION_RE.subn(
            lambda mm: mm.group(1) + args.version + mm.group(3), text, count=1)
        if n != 1:
            print("[FATAL] interface.json 里找不到唯一的 version 字段")
            return 2
    (out / "interface.json").write_text(text, encoding="utf-8")

    print("[3/6] resource/pipeline, resource/image, resource/model/ocr")
    for p in sorted((PROJECT / "resource" / "pipeline").glob("*.json")):
        copy_file(p, out / "resource" / "pipeline" / p.name)
        pairs.append((p, out / "resource" / "pipeline" / p.name))
    for p in sorted((PROJECT / "resource" / "image").glob("*.png")):
        copy_file(p, out / "resource" / "image" / p.name)
        pairs.append((p, out / "resource" / "image" / p.name))
    extra = [p.name for p in (PROJECT / "resource" / "image").iterdir()
             if p.is_file() and p.suffix.lower() != ".png"]
    if extra:
        print("      [WARN] image/ 下非 png 文件未打包：%s" % ", ".join(extra))
    for name in MODEL_FILES:
        src = PROJECT / "resource" / "model" / "ocr" / name
        if not src.exists():
            print("[FATAL] 缺少 OCR 模型文件 %s" % src)
            return 2
        copy_file(src, out / "resource" / "model" / "ocr" / name)
        pairs.append((src, out / "resource" / "model" / "ocr" / name))

    print("[4/6] agent/*.py")
    for p in sorted(AGENT.glob("*.py")):
        copy_file(p, out / "agent" / p.name)
        pairs.append((p, out / "agent" / p.name))

    print("[5/6] docs/")
    (out / "docs").mkdir(parents=True, exist_ok=True)
    (out / "docs" / "README.md").write_text(README_DOC, encoding="utf-8")
    copy_file(ROOT / "LICENSE", out / "docs" / "LICENSE")
    (out / "docs" / "使用说明.md").write_text(USAGE_DOC, encoding="utf-8")

    print("[6/6] 校验 + 清单")
    errors = []
    for src, dst in pairs:
        if sha256(src) != sha256(dst):
            errors.append("%s 与源文件不一致" % dst.relative_to(out))

    for p in sorted((out / "agent").glob("*.py")):
        try:
            compile(p.read_text(encoding="utf-8"), str(p), "exec")
        except SyntaxError as exc:
            errors.append("%s 语法错误：%s" % (p.name, exc))

    for rel in ["interface.json", "resource/pipeline/main.json",
                "resource/model/ocr/det.onnx", "resource/model/ocr/rec.onnx",
                "resource/model/ocr/keys.txt", "agent/main.py",
                "agent/registry.py", "docs/使用说明.md"]:
        if not (out / rel).exists():
            errors.append("缺少 %s" % rel)
    if not args.no_runtime and not (out / "MFAAvalonia.exe").exists():
        errors.append("缺少 MFAAvalonia.exe")

    files = sorted(p for p in out.rglob("*") if p.is_file()
                   and p.name != "manifest-sha256.txt")
    lines = ["%s  %s" % (sha256(p), p.relative_to(out).as_posix()) for p in files]
    (out / "manifest-sha256.txt").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")

    if errors:
        print("[FAIL] 校验未通过：")
        for e in errors:
            print("       " + e)
        return 1

    total_n, total_b = dir_size(out)
    print()
    print("[ok] 发布包: %s" % out)
    print("[ok] 版本: %s" % version)
    print("[ok] 文件数: %d  总大小: %.1f MB" % (total_n, total_b / 1048576))
    print("[ok] 一致性: pipeline/image/model/agent 共 %d 个文件 SHA256 全部匹配"
          % len(pairs))
    for sub in ["resource/pipeline", "resource/image", "resource/model/ocr",
                "agent", "docs", "libs", "runtimes", "plugins"]:
        d = out / sub
        if d.is_dir():
            n, b = dir_size(d)
            print("     %-22s %4d files  %7.1f MB" % (sub, n, b / 1048576))
    if not args.no_runtime:
        print("     %-22s %4d files  %7.1f MB" % ("(运行时根目录)",
              len([p for p in out.iterdir() if p.is_file()]), 0.0))

    if args.zip:
        base = RELEASE_ROOT / ("AutoButter-%s" % version)
        print("[zip] %s.zip" % base)
        shutil.make_archive(str(base), "zip", root_dir=str(out.parent),
                            base_dir=out.name)
        print("[zip] 完成 %.1f MB"
              % (Path(str(base) + ".zip").stat().st_size / 1048576))

    return 0


if __name__ == "__main__":
    sys.exit(main())
