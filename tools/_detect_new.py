import cv2, numpy as np, glob, os

p = r"E:\work\AutoButter\screenshots\snap\20261108-130821_720.png"
img = cv2.imread(p)

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

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

boxes = detect(img)
print(f"detected {len(boxes)} cells")

vis = img.copy()
for i, (x, y, cw, ch) in enumerate(boxes):
    patch = img[y:y+ch, x:x+cw]
    best, bs = None, 0.0
    for n, t in tpls.items():
        if patch.shape[0] < t.shape[0] or patch.shape[1] < t.shape[1]: continue
        s = float(cv2.matchTemplate(patch, t, cv2.TM_CCOEFF_NORMED).max())
        if s > bs: best, bs = n, s
    ok = bs >= 0.85
    color = (0,180,0) if ok else (0,0,255)
    cv2.rectangle(vis, (x,y), (x+cw,y+ch), color, 2)
    label = f"{best.replace('_1','')} {bs:.2f}" if best else "?"
    cv2.putText(vis, label, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    print(f"  #{i} box=({x},{y},{cw},{ch}) center=({x+cw//2},{y+ch//2}) -> {label}")

cv2.imwrite(r"E:\work\AutoButter\screenshots\75_detect_new.png", vis)
print("saved 75_detect_new.png")
