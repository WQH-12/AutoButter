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
        if not (70 <= cw <= 115 and 70 <= ch <= 115): continue   # 只要秘境大图标
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

# 收集所有秘境大格子（只取高 70+ 的），并计算颜色直方图
files = sorted(glob.glob(r"E:\work\AutoButter\screenshots\*.png"))
samples = []
for p in files:
    img = cv2.imread(p)
    if img is None or img.shape[0] < 1000: continue
    for i, b in enumerate(detect(img)):
        x, y, cw, ch = b
        inner = img[y+10:y+ch-10, x+10:x+cw-10]
        if inner.shape[0] < 40 or inner.shape[1] < 40: continue
        t = cv2.resize(inner, (56, 56), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1], None, [12,8], [0,180,0,256])
        cv2.normalize(hist, hist)
        samples.append((os.path.basename(p), i, b, t, hist.flatten()))

print("samples:", len(samples))
X = np.float32([s[4] for s in samples])
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
K = 6
_, labels, centers = cv2.kmeans(X, K, None, crit, 20, cv2.KMEANS_PP_CENTERS)
labels = labels.flatten()

# 每类挑一个样本展示
groups = {}
for idx, lab in enumerate(labels):
    groups.setdefault(int(lab), []).append(idx)

cols = 8
rows = (len(samples)+cols-1)//cols
canvas = np.full((rows*96, cols*96, 3), 255, np.uint8)
for idx, s in enumerate(samples):
    r, c = divmod(idx, cols)
    t = cv2.resize(s[3], (72,72), interpolation=cv2.INTER_NEAREST)
    canvas[r*96+4:r*96+76, c*96+4:c*96+76] = t
    cv2.putText(canvas, f"{labels[idx]}:{s[0][:5]}", (c*96+2, r*96+92), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0,0,0), 1)
cv2.imwrite(r"E:\work\AutoButter\screenshots\64_clusters.png", canvas)
print(canvas.shape)
for lab, idxs in sorted(groups.items()):
    print(lab, len(idxs), [samples[i][0][:12]+f"#{samples[i][1]}" for i in idxs[:6]])
