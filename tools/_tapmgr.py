import time, cv2
from pathlib import Path
from maa.controller import AdbController

c = AdbController("adb", "3a1e6a75")
c.set_screenshot_target_short_side(720)
print(c.post_connection().wait().succeeded)
c.post_click(360, 1113).wait()
for i, d in enumerate([0.3, 0.8, 1.5]):
    time.sleep(d)
    img = c.post_screencap().wait().get()
    cv2.imwrite(rf"E:\work\AutoButter\screenshots\33_manage_t{i}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(i, img.shape)
