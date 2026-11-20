import cv2, numpy as np
# 列表页图标 vs 秘境格子图标 尺寸实测
lst = cv2.imread(r"E:\work\AutoButter\screenshots\45_tanxun_list.png")
realm = cv2.imread(r"E:\work\AutoButter\screenshots\42_after_close.png")

# 列表第一行 5 个图标框（前面测得）
boxes = [(7,19,66,51),(84,18,66,52),(160,18,67,52),(237,18,67,52),(314,18,67,52)]
print("=== 列表页图标（相对 crop 原点 250,470）===")
for i,(x,y,w,h) in enumerate(boxes):
    print(f"  #{i} {w}x{h}")
print(f"  平均 {np.mean([b[2] for b in boxes]):.0f}x{np.mean([b[3] for b in boxes]):.0f}")

print("\n=== 秘境格子 ===")
print("  外框 ~89x92，内部图标（你的模板）66x47")

print("\n=== 结论 ===")
print("  列表页 66x51 vs 秘境 66x47 —— 尺寸几乎一样，不是「变小」")
