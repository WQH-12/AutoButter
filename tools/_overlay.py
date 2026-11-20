import cv2
img = cv2.imread(r"E:\work\AutoButter\screenshots\42_after_close.png")
# 放大左上角的鸡妖 + 计时器区域
crop = img[540:680, 40:180]
cv2.imwrite(r"E:\work\AutoButter\screenshots\88_chicken_overlay.png",
            cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
print("88_chicken_overlay.png", crop.shape)

# 管理面板的一行（正在掠夺 04:46）
m = cv2.imread(r"E:\work\AutoButter\screenshots\43_manage_now.png")
crop2 = m[665:800, 75:650]
cv2.imwrite(r"E:\work\AutoButter\screenshots\89_manage_row.png",
            cv2.resize(crop2, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC))
print("89_manage_row.png", crop2.shape)
