"""查 CustomRecognition 节点的字段名。只读。"""
from __future__ import annotations

import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

SCHEMA = r"E:\work\AutoButter\deps\tools\pipeline.schema.json"


def main():
    s = json.load(open(SCHEMA, encoding="utf-8"))
    for key in ("CustomRecognition", "CustomRecognitionV2"):
        if key not in s["$defs"]:
            continue
        d = s["$defs"][key]
        print("==", key)
        print("   keys:", list(d.get("properties", {})))
        for k, v in d.get("properties", {}).items():
            desc = (v.get("description") or "")[:90].replace("\n", " ")
            print("     %-28s %s" % (k, desc))
    n = s["$defs"]["Node"]["properties"]
    print("\nNode 里与 custom 相关的字段:")
    for k in n:
        if "custom" in k.lower():
            print("   ", k, "|", (n[k].get("description") or "")[:90].replace("\n", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
