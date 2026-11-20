"""把 pipeline 算出来的点击点画到实机截图上，确认到底点在哪儿。"""
import cv2

ROOT = r"E:\work\AutoButter\screenshots"
img = cv2.imread(f"{ROOT}\\142_state.png")
h, w = img.shape[:2]
scale = w / 720.0

# pipeline: box=[206,496,306,110] (720 画布) + target_offset [0,300,0,0] -> 点中心
bx, by, bw, bh = 206, 496, 306, 110
cx720 = bx + bw // 2
cy720 = by + bh // 2 + 300
print(f"720 canvas click = ({cx720}, {cy720})")
print(f"real screen click = ({cx720 * scale:.0f}, {cy720 * scale:.0f})  scale={scale}")

pts = [(cx720 * scale, cy720 * scale)]
for px, py in pts:
    cv2.drawMarker(img, (int(px), int(py)), (0, 0, 255), cv2.MARKER_CROSS, 80, 6)
    cv2.circle(img, (int(px), int(py)), 60, (0, 255, 0), 4)

cv2.imwrite(f"{ROOT}\\_click_marked.png", img)

# 同时存一张 720 画布版本，方便和模板坐标对照
small = cv2.resize(img, (720, int(h / scale)))
cv2.imwrite(f"{ROOT}\\_click_marked_720.png", small)
print("saved _click_marked.png")
