import cv2, numpy as np, glob, os

# 判定截图是否被通道对调（我早期的 bug）
def is_swapped(img):
    # 正常界面：内容区以暖色为主 (H 10-25)；对调后变蓝 (H 95-115)
    roi = img[600:900, 60:660]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = ((hsv[:,:,0]>=95)&(hsv[:,:,0]<=115)&(hsv[:,:,1]>60)).mean()
    orange = (((hsv[:,:,0]<=25)|(hsv[:,:,0]>=170))&(hsv[:,:,1]>60)).mean()
    return blue > orange

bad = []
for p in sorted(glob.glob(r"E:\work\AutoButter\screenshots\*.png")):
    img = cv2.imread(p)
    if img is None or img.shape[0] < 1000: continue
    if is_swapped(img):
        bad.append(os.path.basename(p))
print("通道对调的截图（我的 bug，应删除）:")
for b in bad: print("  ", b)
print(f"共 {len(bad)} 张")
