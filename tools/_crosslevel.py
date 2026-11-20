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
    blue = ((H >= 92) & (H <= 112) & (S > 55) & (V > 150)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(cv2.bitwise_or(mask, purple), blue)
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

def border_kind(img, box):
    x, y, cw, ch = box
    strip = img[y+2:y+6, x+cw//2-12:x+cw//2+12]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    h = int(np.median(hsv[:,:,0])); s = int(np.median(hsv[:,:,1]))
    if s < 55: return "gray"
    if h < 22 or h > 170: return "ORANGE"
    if 92 <= h <= 112: return "BLUE"
    if 112 < h <= 170: return "PURPLE"
    return f"h{h}"

# 关键测试：同一物品在不同底色下，用户模板能匹配吗
print("=== 每格: 最佳模板/分数 + 边框色 ===")
buckets = {}
for p in sorted(glob.glob(r"E:\work\AutoButter\screenshots\*.png")):
    base = os.path.basename(p)
    if base.startswith(("60_", "61_", "62_", "64_", "65_", "66_", "70_", "71_", "72_")):
        continue
    img = cv2.imread(p)
    if img is None or img.shape[0] < 1000: continue
    boxes = detect(img)
    if not boxes: continue
    line = []
    for b in boxes:
        x, y, cw, ch = b
        patch = img[y:y+ch, x:x+cw]
        best, bs = None, 0.0
        for n, t in tpls.items():
            if patch.shape[0] < t.shape[0] or patch.shape[1] < t.shape[1]: continue
            s = float(cv2.matchTemplate(patch, t, cv2.TM_CCOEFF_NORMED).max())
            if s > bs: best, bs = n, s
        col = border_kind(img, b)
        line.append(f"{best.replace('_1','') if best else '?'}@{bs:.2f}[{col}]")
        if best and bs >= 0.85:
            buckets.setdefault(best, {}).setdefault(col, 0)
            buckets[best][col] += 1
    print(f"{base:24s} " + " ".join(line))

print("\n=== 高分命中(>=0.85)按底色统计 ===")
for item in sorted(buckets):
    print(f"  {item:11s} {dict(buckets[item])}")
