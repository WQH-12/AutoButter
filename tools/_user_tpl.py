import cv2, numpy as np, glob, os
files = sorted(glob.glob(r"E:\work\AutoButter\User_screenshot\*.png"))
tiles = []
for p in files:
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    print(f"{os.path.basename(p):16s} shape={im.shape} dtype={im.dtype} channels={'RGBA' if im.shape[2]==4 else 'RGB'}")
    tiles.append((os.path.basename(p), im))

H = max(t.shape[0] for _, t in tiles)
W = max(t.shape[1] for _, t in tiles)
print(f"max {W}x{H}")
scale = max(1, 96 // max(1, min(H, W)))
canvas = np.full((H*2 + 40, len(tiles)*(W*2+12), 3), 240, np.uint8)
for i, (n, t) in enumerate(tiles):
    if t.shape[2] == 4:
        bg = np.full((t.shape[0], t.shape[1], 3), 240, np.uint8)
        a = t[:,:,3:4].astype(float)/255.0
        t3 = (t[:,:,:3].astype(float)*a + bg.astype(float)*(1-a)).astype(np.uint8)
    else:
        t3 = t
    big = cv2.resize(t3, (W*2, H*2), interpolation=cv2.INTER_NEAREST)
    canvas[0:H*2, i*(W*2+12):i*(W*2+12)+W*2] = big
    cv2.putText(canvas, n.replace(".png",""), (i*(W*2+12)+2, H*2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
    cv2.putText(canvas, f"{t.shape[1]}x{t.shape[0]}", (i*(W*2+12)+2, H*2+34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1)
cv2.imwrite(r"E:\work\AutoButter\screenshots\71_user_tpl.png", canvas)
print("montage", canvas.shape)
