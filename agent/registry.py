"""把 agent/ 下的自定义识别注册到一个 Resource 上（进程内模式）。

两种使用方式：

1. **Agent 子进程**（MFAAvalonia）：由 `agent/main.py` 在
   `AgentServer.start_up()` 之前调用 `register_agent_server()`，
   走 AgentServer 的注册表。

2. **进程内**（`tools/run_task.py`、`tools/verify_pipeline_offline.py`）：
   调 `register_all(resource)`，直接把实现挂到 Resource 上。
   这两条路互不影响：进程内注册不会启动子进程，也不需要 socket。

为什么要有这个文件：注册逻辑散在三个地方（main.py / run_task.py /
verify_pipeline_offline.py）时，加一个新的自定义识别就要改三处，
漏一处就会出现「GUI 能跑、离线剧本跑不了」这种半瘫状态。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# (注册名, 模块名, 类名)
RECOGNITIONS = [
    ("Chicken", "chicken_reco", "ChickenRecognition"),
    ("PickCell", "cell_reco", "CellRecognition"),
    ("PickRow", "row_reco", "RowRecognition"),
    # 列表循环的控制件都在 loop_ctl.py，它把多个类放在一个文件里，
    # 所以用 `模块:类` 的形式指明（下面的 _split 会解析）
    ("ListMoved", "loop_ctl:ListMoved", None),
    ("LoopGuard", "loop_ctl:LoopGuard", None),
    ("SwipeGate", "loop_ctl:SwipeGate", None),
    ("PlusGate", "loop_ctl:PlusGate", None),
    ("ChickenStock", "loop_ctl:ChickenStock", None),
    ("DispatchCountReached", "loop_ctl:DispatchCountReached", None),
    ("DispatchPolicy", "loop_ctl:DispatchPolicy", None),
    ("WaitReturnGate", "loop_ctl:WaitReturnGate", None),
    ("RealmOtherGate", "loop_ctl:RealmOtherGate", None),
    ("DispatchOkGate", "loop_ctl:DispatchOkGate", None),
    ("CooldownGate", "loop_ctl:CooldownGate", None),
    ("RefreshGate", "loop_ctl:RefreshGate", None),
    ("ReopenGate", "loop_ctl:ReopenGate", None),
]

# (注册名, 模块名:类名)  —— Custom Action 只能通过 AgentServer 的
# register_custom_action 或 Resource.register_custom_action 注册
ACTIONS = [
    ("SwipeRows", "loop_ctl:SwipeRows"),
    ("DispatchReset", "loop_ctl:DispatchReset"),
    ("AddChicken", "loop_ctl:AddChicken"),
    ("ClickReset", "loop_ctl:ClickReset"),
    ("ListReset", "loop_ctl:ListReset"),
    ("RewireMain", "loop_ctl:RewireMain"),
    ("SetRealm", "loop_ctl:SetRealm"),
    ("DeferCell", "loop_ctl:DeferCell"),
    ("MarkBadRow", "loop_ctl:MarkBadRow"),
    ("MarkRowDone", "loop_ctl:MarkRowDone"),
    ("ClickTanxun", "loop_ctl:ClickTanxun"),
    ("ExitOtherRealm", "loop_ctl:ExitOtherRealm"),
    ("BackToHome", "loop_ctl:BackToHome"),
    ("ClickAdventure", "loop_ctl:ClickAdventure"),
    ("RecoverNet", "loop_ctl:RecoverNet"),
    ("WaitCooldown", "loop_ctl:WaitCooldown"),
    ("MarkWait", "loop_ctl:MarkWait"),
    ("WaitPopup", "loop_ctl:WaitPopup"),
    ("ClickRefresh", "loop_ctl:ClickRefresh"),
    ("CloseList", "loop_ctl:CloseList"),
    ("AbortTask", "loop_ctl:AbortTask"),
    ("CountReopen", "loop_ctl:CountReopen"),
]


def _resolve(mod_name, cls_name):
    """返回 (module, class)。支持 `mod` + 显式类名，也支持 `mod:Class`。"""
    import importlib

    if ":" in mod_name:
        mod_name, cls_name = mod_name.split(":", 1)
    mod = importlib.import_module(mod_name)
    return mod, (getattr(mod, cls_name, None) if cls_name else None)


def register_all(resource) -> dict:
    """把全部自定义识别注册到 Resource。返回 {名字: 是否成功}。"""
    result = {}
    for name, mod_name, cls_name in RECOGNITIONS:
        try:
            _mod, cls = _resolve(mod_name, cls_name)
            if cls is None:
                # maa 不可用时模块会把这个类置 None
                result[name] = False
                continue
            result[name] = bool(resource.register_custom_recognition(name, cls()))
        except Exception as e:  # noqa: BLE001
            print("[registry] register %s failed: %s: %s"
                  % (name, type(e).__name__, e))
            result[name] = False
    for name, mod_name in ACTIONS:
        try:
            _mod, cls = _resolve(mod_name, None)
            if cls is None:
                result[name] = False
                continue
            result[name] = bool(resource.register_custom_action(name, cls()))
        except Exception as e:  # noqa: BLE001
            print("[registry] register action %s failed: %s: %s"
                  % (name, type(e).__name__, e))
            result[name] = False
    return result


def register_agent_server() -> dict:
    """给 AgentServer 子进程模式用，必须在 start_up() 之前调用。"""
    result = {}
    for name, mod_name, cls_name in RECOGNITIONS:
        try:
            _mod, cls = _resolve(mod_name, cls_name)
            if cls is None:
                result[name] = False
                continue
            from maa.agent.agent_server import AgentServer
            result[name] = bool(AgentServer.register_custom_recognition(name, cls()))
        except Exception as e:  # noqa: BLE001
            print("[registry] agent-server register %s failed: %s: %s"
                  % (name, type(e).__name__, e))
            result[name] = False
    for name, mod_name in ACTIONS:
        try:
            _mod, cls = _resolve(mod_name, None)
            if cls is None:
                result[name] = False
                continue
            from maa.agent.agent_server import AgentServer
            result[name] = bool(AgentServer.register_custom_action(name, cls()))
        except Exception as e:  # noqa: BLE001
            print("[registry] agent-server register action %s failed: %s: %s"
                  % (name, type(e).__name__, e))
            result[name] = False
    return result
