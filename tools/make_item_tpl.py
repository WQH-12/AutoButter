import cv2, numpy as np, os, json

IMG_OUT = r"E:\work\AutoButter\resource\image"

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

# (文件名, 截图, 格子序号, 中文名)
SPECS = [
    ("item_rice.png",    "42_after_close.png", 0, "酒米"),
    ("item_scroll.png",  "42_after_close.png", 1, "无字密卷"),
    ("item_yunying.png", "42_after_close.png", 2, "云英"),
    ("item_goji.png",    "48_go_target.png",   1, "枸杞"),
    ("item_silver.png",  "52_go_row2.png",     0, "饷银"),
    ("item_jade.png",    "31_realm_popup.png", 0, "玉璧"),
]

made = []
for fname, shot, idx, cn in SPECS:
    p = os.path.join(r"E:\work\AutoButter\screenshots", shot)
    img = cv2.imread(p)
    boxes = detect(img)
    if idx >= len(boxes):
        print(f"[FAIL] {fname}: {shot} only {len(boxes)} slots"); continue
    core = crop_core(img, boxes[idx])
    cv2.imwrite(os.path.join(IMG_OUT, fname), core)
    made.append((fname, cn, core.shape, shot, idx))
    print(f"{fname:20s} {cn:6s} {core.shape}  <- {shot}#{idx} {boxes[idx]}")

json.dump([{"file": m[0], "name": m[1]} for m in made],
          open(os.path.join(IMG_OUT, "items_meta.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

# 拼图核对
tiles = [cv2.imread(os.path.join(IMG_OUT, m[0])) for m in made]
H = max(t.shape[0] for t in tiles); W = max(t.shape[1] for t in tiles)
canvas = np.full((H+24, len(tiles)*(W+10), 3), 255, np.uint8)
for i, (t, m) in enumerate(zip(tiles, made)):
    x = i*(W+10)
    canvas[0:t.shape[0], x:x+t.shape[1]] = t
    cv2.putText(canvas, m[1], (x+2, H+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
canvas = cv2.resize(canvas, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(r"E:\work\AutoButter\screenshots\66_tpl_v2.png", canvas)
print("montage", canvas.shape)
