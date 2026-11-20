import cv2, numpy as np
img = cv2.imread(r"E:\work\AutoButter\screenshots\snap\20261108-130821_720.png")

# 精确定位 alcohol 格子的橙色边框矩形
x0,y0,w,h = 163,568,87,91
sub = img[y0:y0+h, x0:x0+w]
hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
orange = ((hsv[:,:,0] <= 22) & (hsv[:,:,1] > 120) & (hsv[:,:,2] > 180)).astype(np.uint8)
ys, xs = np.where(orange)
print(f"orange pixels in box: {len(xs)}")
print(f"  x range {xs.min()}..{xs.max()}   y range {ys.min()}..{ys.max()}")
print(f"  => frame rect within box: x={xs.min()} y={ys.min()} w={xs.max()-xs.min()+1} h={ys.max()-ys.min()+1}")

# 逐行/逐列统计，看边框厚度
row_counts = orange.sum(axis=1)
col_counts = orange.sum(axis=0)
print("\n  top rows  :", row_counts[:12].tolist())
print("  bottom rows:", row_counts[-12:].tolist())
print("  left cols :", col_counts[:12].tolist())
print("  right cols:", col_counts[-12:].tolist())
