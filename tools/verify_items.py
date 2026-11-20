import cv2, numpy as np, os, glob, json

IMG = r"E:\work\AutoButter\resource\image"
names = ["item_rice.png","item_scroll.png","item_yunying.png","item_goji.png","item_silver.png","item_jade.png"]
labels = ["酒米","无字密卷","云英","枸杞","饷银","玉璧"]

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

def crop_core(img, box, pad=9, drop_bottom=32):
    x, y, cw, ch = box
    return img[y+pad:y+ch-drop_bottom, x+pad:x+cw-pad]

tpls = []
for n in names:
    t = cv2.imread(os.path.join(IMG, n), cv2.IMREAD_COLOR)
    tpls.append(cv2.resize(t, (64, 48), interpolation=cv2.INTER_AREA))

# 交叉相似度：模板之间必须低
print("=== 模板互相相似度（越低越好，应 < 0.7）===")
print("        " + " ".join(f"{l:>6s}" for l in labels))
for i, a in enumerate(tpls):
    row = []
    for j, b in enumerate(tpls):
        r = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0][0]
        row.append(f"{r:6.2f}")
    print(f"{labels[i]:>6s}  " + " ".join(row))

# 各截图实际识别
print("\n=== 实测识别（每格 -> 最佳模板/分数）===")
for p in sorted(glob.glob(r"E:\work\AutoButter\screenshots\*.png")):
    img = cv2.imread(p)
    if img is None or img.shape[0] < 1000: continue
    boxes = detect(img)
    if not boxes: continue
    out = []
    for b in boxes:
        core = crop_core(img, b)
        core = cv2.resize(core, (64, 48), interpolation=cv2.INTER_AREA)
        scores = [float(cv2.matchTemplate(core, t, cv2.TM_CCOEFF_NORMED)[0][0]) for t in tpls]
        best = int(np.argmax(scores))
        out.append(f"{labels[best]}{scores[best]:.2f}")
    print(f"{os.path.basename(p):26s} {len(boxes)}格: " + " | ".join(out))
