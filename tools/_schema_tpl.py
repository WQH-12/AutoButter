import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

p = r"E:\work\AutoButter\deps\tools\pipeline.schema.json"
s = json.load(open(p, encoding="utf-8"))
defs = s.get("$defs") or s.get("definitions")
print("keys:", list(defs)[:30])
for name in ("TemplateMatch", "TemplateMatchV1", "TemplateMatchV2"):
    if name in defs:
        print(f"\n=== {name} ===")
        print(json.dumps(defs[name], ensure_ascii=False, indent=1)[:3000])
