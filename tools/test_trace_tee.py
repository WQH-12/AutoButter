"""run_main_trace.py 的 stdout 收录逻辑单测（不连设备）。

背景：agent 侧的 `print()`（`[Layout]` / `[RecoverNet]` / `[ClickTanxun]`）
原来只打到控制台，落不到 `screenshots/trace/*.log`。11-14 排查
「重进失败重试门（当时还叫网络恢复门）从未执行」时，就是因为日志里没有它的
输出，只能靠节点序列反推。

用法：python tools\\test_trace_tee.py
"""
from __future__ import annotations

import importlib.util
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    path = os.path.join(ROOT, "tools", "run_main_trace.py")
    spec = importlib.util.spec_from_file_location("run_main_trace", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def case_line_buffering(mod) -> None:
    """print() 分两次 write（正文 + 换行），不能因此多出一行。"""
    lines: list[str] = []
    out = io.StringIO()
    tee = mod._Tee(out, lines)
    tee.write("abc")
    assert lines == [], "没换行时不该落盘"
    tee.write("def\n")
    assert lines == ["abcdef"], lines
    print("  line buffering: ok", flush=True)


def case_console_passthrough(mod) -> None:
    """必须同时写到真实 stdout，否则跑任务时控制台一片空白。"""
    lines: list[str] = []
    out = io.StringIO()
    tee = mod._Tee(out, lines)
    tee.write("[Layout] measured screen=720x1600\n")
    assert out.getvalue() == "[Layout] measured screen=720x1600\n", \
        out.getvalue()
    assert lines == ["[Layout] measured screen=720x1600"], lines
    print("  console passthrough: ok", flush=True)


def case_flush_tail(mod) -> None:
    """末尾没换行的残片也要进 trace，否则最后一行会丢。"""
    lines: list[str] = []
    out = io.StringIO()
    tee = mod._Tee(out, lines)
    tee.write("no newline at end")
    assert lines == []
    tee.flush()
    assert lines == ["no newline at end"], lines
    print("  flush tail: ok", flush=True)


def case_attribute_passthrough(mod) -> None:
    """encoding / isatty 等属性要透传，否则 `sys.stdout.reconfigure` 会炸。"""
    lines: list[str] = []
    out = io.StringIO()
    tee = mod._Tee(out, lines)
    assert tee.encoding == out.encoding, (tee.encoding, out.encoding)
    assert tee.isatty() == out.isatty()
    print("  attribute passthrough: ok", flush=True)


def case_real_print(mod) -> None:
    """真的把 sys.stdout 换掉，验证 print 能同时到两处。"""
    lines: list[str] = []
    out = io.StringIO()
    real = sys.stdout
    sys.stdout = mod._Tee(out, lines)
    try:
        print("[RecoverNet] round=1/8 home=False(0.000)")
        print("two lines")
    finally:
        sys.stdout = real
    assert lines == ["[RecoverNet] round=1/8 home=False(0.000)",
                     "two lines"], lines
    assert "[RecoverNet]" in out.getvalue()
    print("  real print capture: ok", flush=True)


def case_emit_not_duplicated(mod) -> None:
    """sink 的 _emit 现在只 print，不再自己 append，避免重复行。

    这里只做静态检查：`_emit` 的实现里不应该再有 `self.lines.append`。
    """
    import inspect
    src = inspect.getsource(mod.build_sink)
    i = src.find("def _emit")
    body = src[i:i + 400]
    assert "self.lines.append" not in body, \
        "_emit 还在自己 append，会和 _Tee 重复记录：\n" + body
    print("  no duplicate emit: ok", flush=True)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("[start] trace tee cases")
    mod = _load()
    case_line_buffering(mod)
    case_console_passthrough(mod)
    case_flush_tail(mod)
    case_attribute_passthrough(mod)
    case_real_print(mod)
    case_emit_not_duplicated(mod)
    print("trace tee tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
