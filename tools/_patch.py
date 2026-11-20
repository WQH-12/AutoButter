from pathlib import Path
for f in [r"E:\work\AutoButter\tools\ui.py", r"E:\work\AutoButter\tools\run_task.py"]:
    p = Path(f)
    s = p.read_text(encoding="utf-8")
    before = s
    s = s.replace("cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))", "cv2.imwrite(str(out), img)")
    s = s.replace("cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))", "cv2.imwrite(str(out), img)")
    s = s.replace('cv2.imwrite(rf"E:\\work\\AutoButter\\screenshots', 'cv2.imwrite(rf"E:\\work\\AutoButter\\screenshots')
    p.write_text(s, encoding="utf-8")
    print(f, "changed" if s != before else "unchanged")
