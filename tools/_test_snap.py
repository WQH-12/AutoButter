import sys
sys.path.insert(0, r"E:\work\AutoButter\tools")
import cv2, numpy as np
from snap import to_canvas, draw_grid, parse_box

raw = cv2.imread(r"E:\work\AutoButter\screenshots\23_verify_now.png")
print("raw:", raw.shape)

for interp in ["area", "linear", "cubic", "lanczos"]:
    c = to_canvas(raw, 720, interp)
    print(f"  {interp:8s} -> {c.shape[1]}x{c.shape[0]}")

c = to_canvas(raw, 720, "area")
g = draw_grid(c)
print("grid:", g.shape)
cv2.imwrite(r"E:\work\AutoButter\screenshots\70_script_test_grid.png", g)

# crop test
patch = c[681:681+91, 62:62+86]
cv2.imwrite(r"E:\work\AutoButter\screenshots\70_script_test_crop.png", patch)
print("crop:", patch.shape)
print("parse_box:", parse_box("62,681,86,91"))
