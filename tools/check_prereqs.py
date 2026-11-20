"""前置条件自检：链路开工前必须全绿。

比 _probe_prereqs.py 更进一步——这里真的去加载模板、真的注册一遍，
而不是只看文件在不在。

用法：
    python tools/check_prereqs.py            # 查工作区布局
    python tools/check_prereqs.py --install  # 查 install 布局
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BTN_TEMPLATES = ["btn_manage.png", "btn_dispatch.png", "btn_plus.png", "btn_minus.png"]


class Report:
    def __init__(self):
        self.items = []

    def add(self, ok, label, detail=""):
        self.items.append((ok, label, detail))
        print("  [%s] %-46s %s" % ("OK" if ok else "FAIL", label, detail))

    def summary(self):
        bad = [i for i in self.items if not i[0]]
        print()
        if bad:
            print("=== %d 项未通过 ===" % len(bad))
            for _, label, detail in bad:
                print("  - %s  %s" % (label, detail))
            return 1
        print("=== 全部通过（%d 项）===" % len(self.items))
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true",
                    help="检查 install 打包目录的布局")
    args = ap.parse_args()

    base = os.path.join(ROOT, "install") if args.install else ROOT
    image_dir = os.path.join(base, "resource", "image")
    agent_dir = os.path.join(base, "agent")

    r = Report()
    print("布局: %s" % ("install" if args.install else "工作区"))
    print("模板目录: %s" % image_dir)
    print()

    print("--- 1. 模板文件 ---")
    for n in BTN_TEMPLATES:
        r.add(os.path.exists(os.path.join(image_dir, n)), n)

    print("\n--- 2. 模板能真的读进来 ---")
    sys.path.insert(0, agent_dir)
    import importlib
    for mod_name in ["chicken_reco", "cell_reco", "row_reco"]:
        sys.modules.pop(mod_name, None)
    import chicken_reco
    import cell_reco
    import row_reco
    print("  chicken TPL_DIR = %s" % chicken_reco.TPL_DIR)
    print("  cell    IMG_DIR = %s" % cell_reco.IMG_DIR)
    print("  row     IMG_DIR = %s" % row_reco.IMG_DIR)
    f = chicken_reco.ChickenFinder()
    r.add(f.ready, "鸡妖模板加载", "%d 个" % len(f.templates))
    p = cell_reco.CellPicker()
    r.add(len(p.tpls) == 7, "物品模板加载", "%d 个" % len(p.tpls))
    rp = row_reco.RowPicker()
    r.add(rp.ready, "列表模板加载",
          "items=%d collect=%d" % (len(rp.items), len(rp.collect)))

    print("\n--- 3. registry 注册 ---")
    import registry
    names = [n for n, _, _ in registry.RECOGNITIONS]
    print("  识别: %s" % names)
    print("  动作: %s" % [n for n, _ in registry.ACTIONS])
    for name, mod_name, cls_name in registry.RECOGNITIONS:
        # 支持 `mod:Class` 写法（loop_ctl.py 一个文件里放了多个类）
        _mod, cls = registry._resolve(mod_name, cls_name)
        r.add(cls is not None, "%s 类可导入" % name, cls_name)
    for name, mod_name in registry.ACTIONS:
        _mod, cls = registry._resolve(mod_name, None)
        r.add(cls is not None, "%s 动作可导入" % name, mod_name)

    print("\n--- 4. interface.json ---")
    iface_path = os.path.join(base, "interface.json")
    txt = open(iface_path, encoding="utf-8").read()
    body = re.sub(r"^\s*//.*$", "", txt, flags=re.M)
    try:
        iface = json.loads(body)
    except json.JSONDecodeError as e:
        r.add(False, "interface.json 可解析", str(e))
        return r.summary()
    r.add(True, "interface.json 可解析")
    ag = iface.get("agent")
    r.add(isinstance(ag, dict) and "child_exec" in ag, "agent 段已启用",
          json.dumps(ag, ensure_ascii=False) if ag else "缺失")
    if isinstance(ag, dict):
        args_list = ag.get("child_args") or []
        target = None
        for a in args_list:
            if a.endswith(".py"):
                target = a
        if target:
            # MFAAvalonia 按 interface.json 所在目录解析相对路径。
            # 工作区与 install/ 都是 interface.json 与 agent/ 同级。
            beside = os.path.normpath(os.path.join(os.path.dirname(iface_path), target))
            r.add(os.path.exists(beside), "child_args 指向的脚本存在", beside)

    print("\n--- 5. pipeline 里引用的模板都存在 ---")
    pipe_path = os.path.join(base, "resource", "pipeline", "main.json")
    ptxt = open(pipe_path, encoding="utf-8").read()
    pbody = re.sub(r"^\s*//.*$", "", ptxt, flags=re.M)
    nodes = json.loads(pbody)
    refs = set()
    for v in nodes.values():
        t = v.get("template")
        if isinstance(t, str):
            refs.add(t)
        elif isinstance(t, list):
            refs.update(t)
    # 自定义动作/识别里硬编码的模板也要查，避免 nav_* 这类漏网。
    refs.update(("nav_home.png", "nav_adventure.png", "btn_manage.png",
                 "btn_dispatch.png", "obtain_resource.png",
                 "realm_marker.png", "home_menu_first.png",
                 "exit_anchor.png"))
    missing = sorted(t for t in refs if not os.path.exists(os.path.join(image_dir, t)))
    r.add(not missing, "pipeline 模板引用", "缺 %s" % missing if missing else "%d 个全部存在" % len(refs))

    print("\n--- 6. MaaFw 与 MFAAvalonia 内置 MaaFramework 是否同版本 ---")
    # 两边协议号对不上时，GUI 只会报「Agent启动失败 / Protocol version mismatch」，
    # 而 CLI（进程内注册，不走 Agent 子进程）完全正常。§3.59
    import hashlib
    import importlib.metadata
    native = os.path.join(ROOT, "install", "runtimes", "win-x64", "native")
    try:
        pip_ver = importlib.metadata.version("MaaFw")
    except importlib.metadata.PackageNotFoundError:
        pip_ver = None
    if pip_ver is None:
        r.add(False, "MaaFw 已安装", "python -m pip install \"MaaFw==5.8.0\"")
    elif not os.path.isdir(native):
        r.add(False, "install native 目录存在", native)
    else:
        import maa
        pip_bin = os.path.join(os.path.dirname(os.path.abspath(maa.__file__)), "bin")

        def _sha(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()

        diff = []
        for n in sorted(x for x in os.listdir(native) if x.lower().endswith(".dll")):
            a, b = os.path.join(native, n), os.path.join(pip_bin, n)
            if not os.path.exists(b) or _sha(a) != _sha(b):
                diff.append(n)
        r.add(not diff, "MaaFw 与内置 MaaFramework 一致",
              "MaaFw=%s，%d 个 dll 不同：%s"
              % (pip_ver, len(diff), diff) if diff
              else "MaaFw=%s，%d 个 dll 全同"
                   % (pip_ver, len(os.listdir(native))))

    return r.summary()


if __name__ == "__main__":
    sys.exit(main())
