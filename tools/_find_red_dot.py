"""量出用户标注的红色圆点位置，换算到 720 画布坐标。"""
import glob
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

import os

DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
cands = sorted(glob.glob(os.path.join(DESKTOP, "ChatGPT*.png")))
if not cands:
    raise SystemExit(f"no ChatGPT*.png on desktop: {DESKTOP}")
SRC = cands[-1]
print(f"using: {SRC}")

# cv2.imread 在 Windows 上不认非 ASCII 路径，改用 imdecode
raw = np.fromfile(SRC, dtype=np.uint8)
img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
if img is None:
    raise SystemExit(f"cannot decode {SRC}")
h, w = img.shape[:2]
print(f"image: {w}x{h}")

b, g, r = cv2.split(img)
mask = ((r > 180) & (g < 90) & (b < 90)).astype(np.uint8) * 255

# 我的十字是细线，用户的圆点是实心块：取面积最大的连通域
num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
best = None
for i in range(1, num):
    area = stats[i, cv2.CC_STAT_AREA]
    if best is None or area > stats[best, cv2.CC_STAT_AREA]:
        best = i

if best is None:
    raise SystemExit("no red component found")

x, y, bw, bh, area = stats[best]
cx, cy = cents[best]
print(f"largest red blob: bbox=({x},{y},{bw},{bh}) area={area} center=({cx:.1f},{cy:.1f})")
print(f"  normalized: ({cx / w:.4f}, {cy / h:.4f})")

# 截图是 1080x2340 -> 720 短边画布是 720x1560
scale = 720 / w
print(f"  in 720 canvas: ({cx * scale:.1f}, {cy * scale:.1f})")

# 也报告第二大的，方便确认十字和圆点被区分开
order = sorted(range(1, num), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
print("\ntop red components:")
for i in order[:4]:
    print(f"  area={stats[i, cv2.CC_STAT_AREA]:6d} "
          f"center=({cents[i][0]:.1f},{cents[i][1]:.1f}) "
          f"720=({cents[i][0] * scale:.1f},{cents[i][1] * scale:.1f})")

out = img.copy()
cv2.drawMarker(out, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 60, 3)
cv2.imwrite(r"E:\work\AutoButter\screenshots\_red_dot_found.png", out)
