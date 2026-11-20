import cv2, numpy as np, os

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

# 只取 42_after_close（一屏内 6 个格子，类型齐全）用于人工标注
p = r"E:\work\AutoButter\screenshots\42_after_close.png"
img = cv2.imread(p)
boxes = detect(img)
tiles = [crop_core(img, b) for b in boxes]
H = max(t.shape[0] for t in tiles); W = max(t.shape[1] for t in tiles)
canvas = np.full((H+22, len(tiles)*(W+8), 3), 255, np.uint8)
for i, t in enumerate(tiles):
    x = i*(W+8)
    canvas[0:t.shape[0], x:x+t.shape[1]] = t
    cv2.putText(canvas, str(i), (x+2, H+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
canvas = cv2.resize(canvas, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(r"E:\work\AutoButter\screenshots\65_cores_42.png", canvas)
print(canvas.shape, [t.shape for t in tiles])
