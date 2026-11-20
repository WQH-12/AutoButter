import cv2, numpy as np

# 从秘境截图裁图标（含边框），再放大拼图查看
src = cv2.imread(r"E:\work\AutoButter\screenshots\44_realm.png")
src2 = cv2.imread(r"E:\work\AutoButter\screenshots\42_after_close.png")

cands = [
    ("a_44_scroll1", src, 70, 565, 62, 62),
    ("b_44_ingot",   src, 180, 670, 62, 62),
    ("c_44_scroll2", src, 375, 670, 62, 62),
    ("d_42_rice",    src2, 268, 550, 62, 62),
    ("e_42_scroll",  src2, 78, 672, 62, 62),
    ("f_42_ingot",   src2, 182, 672, 62, 62),
]
tiles = []
for name, im, x, y, w, h in cands:
    c = im[y:y+h, x:x+w]
    tiles.append(cv2.resize(c, (186, 186), interpolation=cv2.INTER_CUBIC))
    cv2.imwrite(rf"E:\work\AutoButter\screenshots\_t_{name}.png", c)

canvas = np.hstack(tiles)
cv2.imwrite(r"E:\work\AutoButter\screenshots\57_icon_candidates.png", canvas)
print(canvas.shape)
