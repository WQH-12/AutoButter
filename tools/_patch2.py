from pathlib import Path
p = Path(r"E:\work\AutoButter\tools\check_templates.py")
s = p.read_text(encoding="utf-8")
old = '''        key = name.replace(args.prefix, "")'''
new = '''        key = name.replace(args.prefix, "")
        if key.endswith("_1"):
            key = key[:-2]'''
s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
print("patched")
