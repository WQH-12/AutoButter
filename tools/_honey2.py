import cv2, numpy as np

# honey_1 最高分只到 0.61，看看它匹配到了什么
tpl = cv2.imread(r"E:\work\AutoButter\User_screenshot\honey_1.png")
for shot, pt in [("43_manage_now.png", (396,792)), ("37_color_check.png", (396,905))]:
    img = cv2.imread(rf"E:\work\AutoButter\screenshots\{shot}")
    x, y = pt
    patch = img[y:y+tpl.shape[0], x:x+tpl.shape[1]]
    if patch.shape[:2] != tpl.shape[:2]:
        print(f"{shot}: patch shape {patch.shape} != tpl {tpl.shape}"); continue
    s = cv2.matchTemplate(patch, tpl, cv2.TM_CCOEFF_NORMED)[0][0]
    print(f"{shot} @{pt} score={s:.3f}")

# 拼图：honey 模板 vs 它匹配到的区域
img = cv2.imread(r"E:\work\AutoButter\screenshots\43_manage_now.png")
patch = img[792:792+tpl.shape[0], 396:396+tpl.shape[1]]
H = 200
a = cv2.resize(tpl, (int(tpl.shape[1]*H/tpl.shape[0]), H), interpolation=cv2.INTER_NEAREST)
b = cv2.resize(patch, (int(patch.shape[1]*H/patch.shape[0]), H), interpolation=cv2.INTER_NEAREST)
canvas = np.full((H+30, a.shape[1]+b.shape[1]+30, 3), 255, np.uint8)
canvas[0:H, 0:a.shape[1]] = a
canvas[0:H, a.shape[1]+30:a.shape[1]+30+b.shape[1]] = b
cv2.putText(canvas, "honey_1 TEMPLATE", (2, H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
cv2.putText(canvas, "BEST MATCH 0.61", (a.shape[1]+32, H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
cv2.imwrite(r"E:\work\AutoButter\screenshots\74_honey_problem.png", canvas)
print("saved")
