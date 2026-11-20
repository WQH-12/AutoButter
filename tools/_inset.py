import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
raw = {os.path.basename(p)[:-4]: cv2.imread(p, cv2.IMREAD_COLOR) for p in glob.glob(USER + r"\*.png")}

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

# 收集所有格子：截图名、box、边框色
cells = []
for p in sorted(glob.glob(r"E:\work\AutoButter\screenshots\*.png")):
    base = os.path.basename(p)
    if base[:3] in ("60_","61_","62_","64_","65_","66_","70_","71_","72_"): continue
    img = cv2.imread(p)
    if img is None or img.shape[0] < 1000: continue
    for b in detect(img):
        cells.append((base, img, b, border_kind(img, b)))

# 三种策略对比
def evaluate(inset, label):
    ok = bad = 0
    detail = []
    for base, img, (x,y,cw,ch), col in cells:
        patch = img[y+inset:y+ch-inset, x+inset:x+cw-inset]
        if patch.shape[0] < 20: continue
        scores = {}
        for n, t0 in raw.items():
            t = t0[inset:t0.shape[0]-inset, inset:t0.shape[1]-inset]
            if t.shape[0] < 10 or t.shape[1] < 10: continue
            if patch.shape[0] < t.shape[0] or patch.shape[1] < t.shape[1]: continue
            scores[n] = float(cv2.matchTemplate(patch, t, cv2.TM_CCOEFF_NORMED).max())
        if not scores: continue
        best = max(scores, key=scores.get); bs = scores[best]
        if bs >= 0.85: ok += 1
        else: bad += 1
    print(f"inset={inset}px ({label}): 命中 {ok} / 未命中 {bad}")
    return ok, bad

print("=== 不同内缩量对全库格子的命中率 ===")
for inset in [0, 2, 4, 6, 8, 10, 12]:
    evaluate(inset, f"去掉 {inset}px 边框")

# 详细：inset=8 时每个物品在各底色下的分数
print("\n=== inset=8: 每个物品 x 每种底色的最高分 ===")
from collections import defaultdict
table = defaultdict(lambda: defaultdict(list))
for base, img, (x,y,cw,ch), col in cells:
    patch = img[y+8:y+ch-8, x+8:x+cw-8]
    if patch.shape[0] < 20: continue
    for n, t0 in raw.items():
        t = t0[8:t0.shape[0]-8, 8:t0.shape[1]-8]
        if patch.shape[0] < t.shape[0] or patch.shape[1] < t.shape[1]: continue
        s = float(cv2.matchTemplate(patch, t, cv2.TM_CCOEFF_NORMED).max())
        table[n][col].append(s)

for n in sorted(table):
    parts = []
    for col in sorted(table[n]):
        v = table[n][col]
        parts.append(f"{col}:max={max(v):.2f},n={len(v)}")
    print(f"  {n:11s} " + "  ".join(parts))
