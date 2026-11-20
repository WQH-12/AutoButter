import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

# 干净的验收：对每张“确实含物品格子”的图，全屏匹配，报告命中数与分数
shots = [
    ("snap/20261108-130821_720.png", {"alcohol":1,"jade":1,"honey":1,"berry":2}),
    ("42_after_close.png",  {"alcohol":1,"fabric":4}),
    ("44_realm.png",        {"fabric":1,"stone":1}),
    ("52_go_row2.png",      {"silver":1,"fabric":1,"stone":1,"berry":3}),
    ("48_go_target.png",    {"stone":2,"berry":1,"fabric":2}),
    ("31_realm_popup.png",  {"alcohol":1,"fabric":3}),
    ("45_tanxun_list.png",  {"alcohol":1,"berry":1,"fabric":1,"jade":1,"silver":1,"stone":1}),
]

print(f"{'screenshot':30s} {'item':10s} {'expect':>6s} {'found':>5s} {'best':>6s}  verdict")
print("-"*78)
for rel, expect in shots:
    p = os.path.join(r"E:\work\AutoButter\screenshots", rel.replace("/", "\\"))
    img = cv2.imread(p)
    if img is None:
        print(f"{rel:30s} MISSING"); continue
    for item, want in expect.items():
        t = tpls[item + "_1"]
        res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
        rr = res.copy(); found = 0; scores = []
        while True:
            _, mx, _, ml = cv2.minMaxLoc(rr)
            if mx < 0.85: break
            found += 1; scores.append(mx)
            rr[max(0,ml[1]-20):ml[1]+20, max(0,ml[0]-20):ml[0]+20] = -1
        best = max(scores) if scores else 0.0
        verdict = "OK" if found >= want else f"MISS({want-found})"
        print(f"{rel[:30]:30s} {item:10s} {want:6d} {found:5d} {best:6.3f}  {verdict}")
