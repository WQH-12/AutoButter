from pathlib import Path
p = Path(r"E:\work\AutoButter\tools\check_templates.py")
s = p.read_text(encoding="utf-8")
s = s.replace('"alcohol": "snap/20261108-130821_720.png",',
              '"alcohol": "snap/20261108-131441_720.png",')
p.write_text(s, encoding="utf-8")
print("patched: alcohol source -> 131441")
