import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(p)[:-4]: cv2.imread(p, cv2.IMREAD_COLOR) for p in glob.glob(USER + r"\*.png")}

def detect(img, top_ratio=0.13, bot_ratio=0.62):
    h, w = img.shape[:2]
    top, bot = int(h*top_ratio), int(h*bot_ratio)
    roi = img[top:bot]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    mask = ((S > 90) & (V > 90)).astype(np.uint8) * 255
    purple = ((H >= 110) & (H <= 170) & (S > 45) & (V > 80)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask, purple)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if not (70 <= cw <= 115 and 70 <= ch <= 115): continue
        if not (0.7 <= cw/ch <= 1.45): continue
        if cv2.contourArea(c) < cw*ch*0.40: continue
        boxes.append((x, y+top, cw, ch))
    boxes.sort(key=lambda b: -b[2]*b[3])
    kept = []
    for b in boxes:
        if all(abs(b[0]-k[0]) > 30 or abs(b[1]-k[1]) > 30 for k in kept):
            kept.append(b)
    kept.sort(key=lambda b: (b[1]//45, b[0]))
    return kept

def border_hsv(img, box):
    x, y, cw, ch = box
    # 取上边缘中间 20px 的条带（避开圆角）
    strip = img[y+2:y+7, x+cw//2-10:x+cw//2+10]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    h = int(np.median(hsv[:,:,0])); s = int(np.median(hsv[:,:,1])); v = int(np.median(hsv[:,:,2]))
    if s < 60:
        return "gray", (h,s,v)
    if h < 22 or h > 170: return "ORANGE", (h,s,v)
    if 22 <= h < 40: return "yellow", (h,s,v)
    if 40 <= h < 90: return "green", (h,s,v)
    if 90 <= h < 118: return "BLUE", (h,s,v)
    return "PURPLE", (h,s,v)

# 对每个检出格子，用 6 个物品模板分类，并记录边框色
rows = []
for p in sorted(glob.glob(r"E:\work\AutoButter\screenshots\*.png")):
    img = cv2.imread(p)
    if img is None or img.shape[0] < 1000: continue
    for b in detect(img):
        x, y, cw, ch = b
        patch = img[y:y+ch, x:x+cw]
        scores = {}
        for n, t in tpls.items():
            if patch.shape[0] < t.shape[0] or patch.shape[1] < t.shape[1]: continue
            r = cv2.matchTemplate(patch, t, cv2.TM_CCOEFF_NORMED)
            scores[n] = float(r.max())
        if not scores: continue
        best = max(scores, key=scores.get)
        col, hsv = border_hsv(img, b)
        rows.append((os.path.basename(p), best, round(scores[best],2), col, hsv))

print("=== 每个物品出现过的边框色 ===")
from collections import defaultdict
agg = defaultdict(lambda: defaultdict(int))
for _, best, sc, col, hsv in rows:
    if sc >= 0.85:
        agg[best][col] += 1
for item in sorted(agg):
    print(f"  {item:11s} -> {dict(agg[item])}")

print("\n=== honey_1 相关的所有检出（含低分）===")
for f, best, sc, col, hsv in rows:
    if best == "honey_1":
        print(f"  {f:26s} score={sc:.2f} border={col:6s} hsv={hsv}")
