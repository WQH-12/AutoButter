import cv2, numpy as np, glob, os

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
        if not (45 <= cw <= 115 and 45 <= ch <= 115): continue
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

files = ["42_after_close","44_realm","48_go_target","52_go_row2","31_realm_popup","49_list2"]
tiles, labels = [], []
for tag in files:
    p = rf"E:\work\AutoButter\screenshots\{tag}.png"
    img = cv2.imread(p)
    for i,(x,y,cw,ch) in enumerate(detect(img)):
        # 取内部图标（去掉 8px 边框）
        inner = img[y+8:y+ch-8, x+8:x+cw-8]
        if inner.size == 0: continue
        t = cv2.resize(inner, (72,72), interpolation=cv2.INTER_AREA)
        tiles.append(t); labels.append(f"{tag[:2]}_{i}")
print("slots:", len(tiles))

cols = 8
rows = (len(tiles)+cols-1)//cols
canvas = np.full((rows*96, cols*96, 3), 255, np.uint8)
for idx, t in enumerate(tiles):
    r, c = divmod(idx, cols)
    canvas[r*96+6:r*96+78, c*96+6:c*96+78] = t
    cv2.putText(canvas, labels[idx], (c*96+6, r*96+92), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 1)
cv2.imwrite(r"E:\work\AutoButter\screenshots\62_icon_montage.png", canvas)
print(canvas.shape)
