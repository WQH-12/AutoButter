"""静态检查 pipeline 的 next / on_error 引用 + 自定义名字是否注册。

三类问题：

1. **`on_error` 里写节点自己 = 死循环**。Maa 会无条件重跑同一候选，
   不重新识别、不换候选。实测刷了 240+ 次 `Node.Action.Failed`，一点不前进。
2. **引用了不存在的节点名**。Maa 运行时才报错，静态查能提前发现。
3. **`custom_recognition` / `custom_action` 用了没注册的名字，或注册在
   错误的那一侧**。Maa 只在运行时打一行
   `CustomRecognition::analyze recognition is null`，节点每次直接失败，
   看起来像「节点被跳过」。11-14 的 `网络恢复门`（11-15 改名 `重进失败重试门`）
   就是这么死的：RecoverNet
   是 CustomAction（只有 run），却被写进 `custom_recognition`。

4. **带 `max_hit` 的节点是别人 next / on_error 的最后一个候选**。
   配额用光后 Maa 会**静默跳过**该节点（不报错、不触发它自己的 on_error），
   如果调用方没有别的候选，Maa 直接
   `invalid node id, handle error` 收尾 —— 任务 succeeded=False，
   后续收尾节点一个都不执行。11-15 11:53 的 `回主界面.on_error =
   ["重进-开菜单"]` 就是这么让任务无声退出的（§3.48）。

`next` 里的自引用是**允许**的（每次都会重新识别，失败才换候选），
`关闭获得弹窗` 就靠这个循环点弹窗，所以只提示、不算错误。

用法：python tools\\check_pipeline_refs.py
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))


def main() -> int:
    path = os.path.join(ROOT, "resource", "pipeline", "main.json")
    txt = open(path, encoding="utf-8").read()
    data = json.loads(re.sub(r"//[^\n]*", "", txt))
    names = set(data)

    dead_loops = []
    missing = []
    allowed = []
    for name, node in data.items():
        for field in ("next", "on_error"):
            for ref in node.get(field) or []:
                # 锚点写法 [Anchor] 跳过
                if ref.startswith("[") and ref.endswith("]"):
                    continue
                if ref not in names:
                    missing.append((name, field, ref))
                elif ref == name:
                    (dead_loops if field == "on_error" else allowed).append(
                        (name, field))

    print("nodes: %d" % len(data))
    print("on_error self-refs (dead loop): %s" % (dead_loops or "none"))
    print("missing node refs: %s" % (missing or "none"))
    print("next self-refs (allowed): %s" % (allowed or "none"))

    bad_custom = check_custom_names(data)
    print("custom name mismatches: %s" % (bad_custom or "none"))

    no_fallback = check_max_hit_last(data)
    print("max_hit 末位无兜底: %s"
          % (["%s.%s -> %s" % t for t in no_fallback] or "none"))
    return 1 if (dead_loops or missing or bad_custom or no_fallback) else 0


def check_custom_names(data: dict) -> list[tuple[str, str, str]]:
    """`custom_recognition` 必须注册在 RECOGNITIONS，`custom_action` 在 ACTIONS。

    返回 [(节点, 字段, 名字)]。
    """
    try:
        import registry
    except Exception as exc:  # noqa: BLE001
        print("  [WARN] 无法导入 registry，跳过自定义名字检查: %s" % exc)
        return []
    recs = {n for n, _m, _c in registry.RECOGNITIONS}
    acts = {n for n, _m in registry.ACTIONS}
    bad = []
    for name, node in data.items():
        r = node.get("custom_recognition")
        if r and r not in recs:
            bad.append((name, "custom_recognition", r))
        a = node.get("custom_action")
        if a and a not in acts:
            bad.append((name, "custom_action", a))
    return bad


def check_max_hit_last(data: dict) -> list[tuple[str, str, str]]:
    """找出「末位候选带 max_hit」的路由。

    Maa 的 `max_hit` 语义（schema）：超过次数后，该节点在**其他节点的
    next 列表里被跳过，既不会被识别也不会被执行**。实测 on_error 同理。
    被跳过时不会报错、也不会走它自己的 on_error，所以调用方若没有下一个
    候选，整条任务就在那里断掉（`invalid node id, handle error`）。

    返回 [(调用方, 字段, 末位节点名)]。
    """
    quota = {n for n, d in data.items()
             if isinstance(d, dict) and "max_hit" in d}
    bad = []
    for name, node in data.items():
        if not isinstance(node, dict):
            continue
        for field in ("next", "on_error"):
            refs = node.get(field) or []
            if isinstance(refs, str):
                refs = [refs]
            if not refs:
                continue
            last = refs[-1]
            if isinstance(last, dict):
                last = last.get("name")
            # 自引用（next 里写自己）是 §3.45 的重试写法：配额用光后
            # Maa 跳过该候选，本节点的 next 耗尽 → 走**它自己的** on_error。
            # 11-15 实测：`重进-开菜单` max_hit=1 时，点一次后直接落到
            # `重进失败`。所以自引用末位不算风险。
            if last in quota and last != name:
                bad.append((name, field, last))
    return bad


if __name__ == "__main__":
    sys.exit(main())
