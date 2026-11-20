import cv2, numpy as np, os
# 列表页图标的原生尺寸：从 45_tanxun_list 里量
img = cv2.imread(r"E:\work\AutoButter\screenshots\45_tanxun_list.png")
# 第一行物品图标所在区域
crop = img[470:540, 250:680]
cv2.imwrite(r"E:\work\AutoButter\screenshots\87_list_icons.png",
            cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST))
print("list icon strip:", crop.shape)

# 用彩色边框找列表页图标框
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
mask = ((hsv[:,:,1] > 90) & (hsv[:,:,2] > 120)).astype(np.uint8)*255
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boxes = sorted([cv2.boundingRect(c) for c in cnts if 20 < cv2.boundingRect(c)[2] < 70 and 20 < cv2.boundingRect(c)[3] < 70],
               key=lambda b: b[0])
print("list page icon boxes:", boxes)
if boxes:
    ws = [b[2] for b in boxes]; hs = [b[3] for b in boxes]
    print(f"native size: {min(ws)}~{max(ws)} x {min(hs)}~{max(hs)}")
