import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
files = sorted(glob.glob(USER + r"\*.png"))
print(f"{'file':22s} {'size':>10s} {'std':>6s} {'alpha255':>8s}")
for p in files:
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    a = (im.ndim == 2) or bool((im[:,:,3]==255).all())
    print(f"{os.path.basename(p):22s} {bgr.shape[1]:4d}x{bgr.shape[0]:<4d} {g.std():6.1f} {str(a):>8s}")

# 拼图：所有模板
tiles = []
for p in files:
    t = cv2.imread(p, cv2.IMREAD_COLOR)
    H = 110
    t = cv2.resize(t, (int(t.shape[1]*H/t.shape[0]), H), interpolation=cv2.INTER_NEAREST)
    t = cv2.copyMakeBorder(t, 20, 0, 0, 8, cv2.BORDER_CONSTANT, value=(255,255,255))
    cv2.putText(t, os.path.basename(p)[:-4], (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
    tiles.append(t)

cols = 6
rows = (len(tiles)+cols-1)//cols
Hm = max(t.shape[0] for t in tiles); Wm = max(t.shape[1] for t in tiles)
canvas = np.full((rows*(Hm+6), cols*(Wm+6), 3), 245, np.uint8)
for i, t in enumerate(tiles):
    r, c = divmod(i, cols)
    canvas[r*(Hm+6):r*(Hm+6)+t.shape[0], c*(Wm+6):c*(Wm+6)+t.shape[1]] = t
cv2.imwrite(r"E:\work\AutoButter\screenshots\92_all_user_tpl.png", canvas)
print("\nmontage:", canvas.shape)
