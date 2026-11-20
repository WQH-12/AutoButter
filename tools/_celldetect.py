import cv2, numpy as np
img = cv2.imread(r"E:\work\AutoButter\screenshots\burst\pad_test2\f00_720.png")

# cell0 检出框是 (62,624,86,92)，但其它格子都在 y~704
# 检查这个框到底框住了什么
for tag, (x,y,w,h) in [("cell0-detected", (62,624,86,92)),
                        ("cell1-detected", (163,708,87,91))]:
    crop = img[y:y+h, x:x+w]
    cv2.imwrite(rf"E:\work\AutoButter\screenshots\91_{tag}.png",
                cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    print(f"{tag}: box=({x},{y},{w},{h})")

# 如果 cell0 应该和 cell1 同一行，那么真实 y 应该在 708 附近
print("\n其它格子 y 值: 708, 704, 695, 692, 704  -> 中位数 704")
print("cell0 检出 y=624，比同排低 80px  -> 框到了鸡妖的木架")
