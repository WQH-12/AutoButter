import cv2

ROOT = r"E:\work\AutoButter\screenshots"
img = cv2.imread(f"{ROOT}\\131_live_nav.png")
print("shape", img.shape)
# 弹窗整块
cv2.imwrite(f"{ROOT}\\_pop_full.png", img[200:1100, :])
# 标题区
cv2.imwrite(f"{ROOT}\\_pop_title.png", img[440:640, :])
# 物品区下方空白（点击落点附近）
cv2.imwrite(f"{ROOT}\\_pop_click_area.png", img[780:1000, 200:520])
# 底部按钮区
cv2.imwrite(f"{ROOT}\\_pop_bottom.png", img[1050:1400, :])
