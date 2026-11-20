import cv2, numpy as np

img31 = cv2.imread(r"E:\work\AutoButter\screenshots\31_realm_popup.png")
# 确认 31 里 slot#2 到底是什么物品
for i, (x,y,cw,ch) in enumerate([(61,616,89,92),(163,698,89,92),(367,696,89,92),(468,712,97,92),(570,688,89,91)]):
    p = img31[y:y+ch, x:x+cw]
    cv2.imwrite(rf"E:\work\AutoButter\screenshots\_s31_{i}.png", cv2.resize(p, (178,184), interpolation=cv2.INTER_NEAREST))

# 拼图
tiles = [cv2.imread(rf"E:\work\AutoButter\screenshots\_s31_{i}.png") for i in range(5)]
canvas = np.hstack([np.hstack([t, np.full((t.shape[0], 8, 3), 255, np.uint8)]) for t in tiles])
cv2.imwrite(r"E:\work\AutoButter\screenshots\76_slots_31.png", canvas)
print(canvas.shape)
