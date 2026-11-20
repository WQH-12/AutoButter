import cv2, numpy as np

def detect(path, debug_out=None, use_purple=True):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    top, bot = int(h*0.13), int(h*0.78)
    roi = img[top:bot]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    mask = ((S > 90) & (V > 90)).astype(np.uint8) * 255
    if use_purple:
        # 紫框：H 约 120-160（OpenCV 0-180），S 中等
        purple = ((H >= 110) & (H <= 170) & (S > 45) & (V > 80)).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, purple)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if not (45 <= cw <= 115 and 45 <= ch <= 115):
            continue
        if not (0.7 <= cw/ch <= 1.45):
            continue
        if cv2.contourArea(c) < cw*ch*0.40:
            continue
        boxes.append((x, y+top, cw, ch))
    # 去重（重叠框保留面积大的）
    boxes.sort(key=lambda b: -b[2]*b[3])
    kept = []
    for b in boxes:
        if all(abs(b[0]-k[0]) > 30 or abs(b[1]-k[1]) > 30 for k in kept):
            kept.append(b)
    kept.sort(key=lambda b: (b[1]//45, b[0]))
    if debug_out:
        vis = img.copy()
        for i,(x,y,cw,ch) in enumerate(kept):
            cv2.rectangle(vis, (x,y), (x+cw,y+ch), (0,0,255), 2)
            cv2.putText(vis, str(i), (x+2, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imwrite(debug_out, vis)
    return kept

tests = [
    (r"E:\work\AutoButter\screenshots\44_realm.png", "44"),
    (r"E:\work\AutoButter\screenshots\42_after_close.png", "42"),
    (r"E:\work\AutoButter\screenshots\52_go_row2.png", "52"),
    (r"E:\work\AutoButter\screenshots\31_realm_popup.png", "31"),
    (r"E:\work\AutoButter\screenshots\48_go_target.png", "48"),
]
for p, tag in tests:
    b = detect(p, rf"E:\work\AutoButter\screenshots\61_det_{tag}.png")
    print(tag, len(b), b)
