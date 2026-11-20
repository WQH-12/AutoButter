import cv2, numpy as np, glob, os

USER = r"E:\work\AutoButter\User_screenshot"
tpls = {os.path.basename(f)[:-4]: cv2.imread(f, cv2.IMREAD_COLOR) for f in glob.glob(USER + r"\*.png")}

# stone_1 = 云英，模板来自蓝框。找同物品在橙框里的实例
a = cv2.imread(r"E:\work\AutoButter\screenshots\42_after_close.png")   # 橙框云英 @ (163,680)
b = cv2.imread(r"E:\work\AutoButter\screenshots\snap\20261108-131441_720.png")  # 蓝框云英

t = tpls["stone_1"]
for tag, im in [("orange-frame 42", a), ("blue-frame 131441", b)]:
    res = cv2.matchTemplate(im, t, cv2.TM_CCOEFF_NORMED)
    _, mx, _, ml = cv2.minMaxLoc(res)
    print(f"{tag:22s} stone_1 score={mx:.3f} at {ml}")

# 拼图：模板 vs 两个实例
def crop(im, x, y, w=89, h=92): return im[y:y+h, x:x+w]
p1 = crop(a, 163, 680)
res = cv2.matchTemplate(b, t, cv2.TM_CCOEFF_NORMED)
_, mx, _, ml = cv2.minMaxLoc(res)
p2 = b[ml[1]-15:ml[1]+77, ml[0]-12:ml[0]+77]

H=180
def big(p): return cv2.resize(p,(int(p.shape[1]*H/p.shape[0]),H),interpolation=cv2.INTER_NEAREST)
tiles = [big(t), big(p1), big(p2)]
canvas = np.hstack([np.hstack([x, np.full((H,10,3),255,np.uint8)]) for x in tiles])
canvas = cv2.copyMakeBorder(canvas, 24, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255,255,255))
for i, s in enumerate(["TEMPLATE(blue frame)", "INSTANCE orange frame", "INSTANCE blue frame"]):
    cv2.putText(canvas, s, (4+i*(tiles[0].shape[1]+10), 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
cv2.imwrite(r"E:\work\AutoButter\screenshots\86_crossframe_stone.png", canvas)
print("saved 86_crossframe_stone.png")
