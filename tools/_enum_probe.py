import sys

sys.stdout.reconfigure(encoding="utf-8")

from maa.define import MaaAdbScreencapMethodEnum, MaaAdbInputMethodEnum

print("=== screencap methods ===")
for name in dir(MaaAdbScreencapMethodEnum):
    if not name.startswith("_"):
        print(f"  {name:34s} = {getattr(MaaAdbScreencapMethodEnum, name)}")

print("=== input methods ===")
for name in dir(MaaAdbInputMethodEnum):
    if not name.startswith("_"):
        print(f"  {name:34s} = {getattr(MaaAdbInputMethodEnum, name)}")
