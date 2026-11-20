"""Snapshot the phone screen onto MaaFramework's 720 short-side canvas.

Why 720 short side: interface.json sets display_short_side = 720, so every
image the matcher ever sees is resized so that min(w,h) == 720. For a
1080x2340 phone that is 720x1560 -- NOT 720p (1280x720).

Usage (run from the repo root):
    python tools/snap.py --list
    python tools/snap.py                      # -> screenshots/snap/<ts>_720.png
    python tools/snap.py --name realm         # -> screenshots/snap/realm_720.png
    python tools/snap.py --name realm --raw   # also keep the 1080x2340 original
    python tools/snap.py --name realm --grid  # extra copy with a coordinate grid
    python tools/snap.py --via-maa            # use MaaFramework's own pipeline
    python tools/snap.py --crop 62,681,86,91 --out item_rice.png
    python tools/snap.py --crop 62,681,86,91 --out item_rice.png --src screenshots/snap/realm_720.png

--via-maa gives you the exact pixels the recogniser sees (no resize guessing).
--crop always cuts from an already-720 image, so the result is directly usable
as a template in resource/image/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SNAP_DIR = ROOT / "screenshots" / "snap"
IMG_DIR = ROOT / "resource" / "image"
LATEST = SNAP_DIR / "latest_720.png"

SHORT_SIDE = 720
INTERP = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
    "lanczos": cv2.INTER_LANCZOS4,
}


def die(msg: str) -> "None":
    print(f"[error] {msg}", file=sys.stderr)
    raise SystemExit(1)


def list_devices() -> list[tuple[str, str]]:
    p = subprocess.run(["adb", "devices", "-l"], capture_output=True, text=True)
    if p.returncode != 0:
        die(f"adb not usable: {p.stderr.strip() or p.stdout.strip()}")
    out = []
    for line in p.stdout.splitlines()[1:]:
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            out.append((parts[0], line))
    return out


def pick_serial(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    devs = list_devices()
    if not devs:
        die("no adb device found -- check the USB cable and USB debugging")
    if len(devs) > 1:
        names = ", ".join(d[0] for d in devs)
        die(f"more than one device ({names}); pass --serial")
    return devs[0][0]


def grab_adb(serial: str) -> np.ndarray:
    cmd = ["adb", "-s", serial, "exec-out", "screencap", "-p"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        die(f"screencap failed: {p.stderr.decode('utf-8', 'replace').strip()}")
    data = p.stdout
    if len(data) < 4096:
        die(f"screencap returned only {len(data)} bytes (device locked or asleep?)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        die("PNG decode failed")
    return img


def grab_maa(serial: str) -> np.ndarray:
    try:
        from maa.controller import AdbController
    except ImportError:
        die("maa python package missing -- run: pip install MaaFw==5.15.0")
    c = AdbController("adb", serial)
    c.set_screenshot_target_short_side(SHORT_SIDE)
    if not c.post_connection().wait().succeeded:
        die("MaaFramework controller connect failed")
    img = c.post_screencap().wait().get()
    return img


def to_canvas(img: np.ndarray, short_side: int, interp: str) -> np.ndarray:
    h, w = img.shape[:2]
    scale = short_side / float(min(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if size == (w, h):
        return img.copy()
    return cv2.resize(img, size, interpolation=INTERP[interp])


def draw_grid(img: np.ndarray, step: int = 50) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]
    for x in range(0, w, step):
        cv2.line(vis, (x, 0), (x, h), (0, 0, 255), 1)
    for y in range(0, h, step):
        cv2.line(vis, (0, y), (w, y), (0, 0, 255), 1)
    for x in range(0, w, step * 2):
        cv2.putText(vis, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    for y in range(step, h, step * 2):
        cv2.putText(vis, str(y), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    return vis


def parse_box(text: str) -> tuple[int, int, int, int]:
    try:
        parts = [int(v) for v in text.replace(" ", "").split(",")]
    except ValueError:
        die("--crop wants four integers: x,y,w,h")
    if len(parts) != 4:
        die("--crop wants four integers: x,y,w,h")
    return parts[0], parts[1], parts[2], parts[3]


def do_crop(box: tuple[int, int, int, int], src: Path, out: str) -> int:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        die(f"cannot read {src}")
    x, y, w, h = box
    ih, iw = img.shape[:2]
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > iw or y + h > ih:
        die(f"box {box} outside image {iw}x{ih}")
    patch = img[y:y + h, x:x + w]
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    dst = IMG_DIR / out
    cv2.imwrite(str(dst), patch)
    print(f"[crop] {dst}")
    print(f"       box={x},{y},{w},{h}  size={w}x{h}  from {src.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--list", action="store_true", help="list adb devices and exit")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--name", default=None, help="file stem, default = timestamp")
    ap.add_argument("--short-side", type=int, default=SHORT_SIDE)
    ap.add_argument("--interp", choices=sorted(INTERP), default="area")
    ap.add_argument("--raw", action="store_true", help="also save the original resolution")
    ap.add_argument("--grid", action="store_true", help="also save a copy with a 50px coordinate grid")
    ap.add_argument("--via-maa", action="store_true", help="capture through MaaFramework instead of adb")
    ap.add_argument("--crop", default=None, help="x,y,w,h -- cut a patch from an existing 720 image")
    ap.add_argument("--src", default=None, help="source image for --crop (default: latest_720.png)")
    ap.add_argument("--out", default=None, help="output file name for --crop")
    args = ap.parse_args()

    if args.list:
        devs = list_devices()
        if not devs:
            print("no device")
            return 1
        for serial, line in devs:
            print(line)
        return 0

    if args.crop:
        if not args.out:
            die("--crop needs --out <name.png>")
        src = Path(args.src) if args.src else LATEST
        if not src.exists():
            die(f"source image missing: {src} (run a snapshot first)")
        return do_crop(parse_box(args.crop), src, args.out)

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    serial = pick_serial(args.serial)

    t0 = time.time()
    if args.via_maa:
        raw = grab_maa(serial)
        canvas = raw  # MaaFramework already resized it
        mode = "maa"
    else:
        raw = grab_adb(serial)
        canvas = to_canvas(raw, args.short_side, args.interp)
        mode = f"adb/{args.interp}"
    dt = (time.time() - t0) * 1000

    stem = args.name or time.strftime("%Y%m%d-%H%M%S")
    out = SNAP_DIR / f"{stem}_720.png"
    cv2.imwrite(str(out), canvas)
    cv2.imwrite(str(LATEST), canvas)

    print(f"[snap] {mode}  {dt:.0f} ms")
    print(f"       raw    {raw.shape[1]}x{raw.shape[0]}")
    print(f"       canvas {canvas.shape[1]}x{canvas.shape[0]}  (short side {min(canvas.shape[:2])})")
    print(f"       saved  {out}")

    if args.raw:
        raw_path = SNAP_DIR / f"{stem}_raw.png"
        cv2.imwrite(str(raw_path), raw)
        print(f"       raw    {raw_path}")

    if args.grid:
        grid_path = SNAP_DIR / f"{stem}_720_grid.png"
        cv2.imwrite(str(grid_path), draw_grid(canvas))
        print(f"       grid   {grid_path}")

    if canvas.shape[0] * canvas.shape[1] != 720 * 1560:
        print(f"[warn] canvas is not 720x1560 -- interface.json display_short_side "
              f"must equal {args.short_side} for templates to line up")

    print()
    print("next: read x,y,w,h in any viewer, then")
    print(f"      python tools/snap.py --crop x,y,w,h --out item_xxx.png "
          f"--src {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
