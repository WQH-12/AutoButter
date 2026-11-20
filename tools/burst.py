"""Burst capture for animated UI (robbing chickens / countdown timers).

Single screenshots cannot tell you whether something is animating, and cannot
show you what a "being robbed" cell looks like in every phase. This captures a
burst and lays the frames out so the variation is obvious.

Usage
-----
    # 8 frames, 400 ms apart, plain full-screen shots
    python tools/burst.py --name rob --frames 8 --interval 400

    # also crop every detected item cell, with context above and below
    # (the chicken + countdown can sit above OR below the cell)
    python tools/burst.py --name rob --frames 8 --interval 400 --cells

    # capture the manage panel rows instead (shows 我 + 正在掠夺 mm:ss)
    python tools/burst.py --name manage --frames 1 --panel

    # longer burst to catch the whole animation loop
    python tools/burst.py --name rob --frames 20 --interval 250 --cells

Output (under screenshots/burst/<name>/)
    f00_720.png ...            full frames
    cell<i>_f<k>.png           per-cell crops, same index across frames
    montage_cells.png          rows = cells, columns = frames
    montage_full.png           contact sheet of full frames
    report.txt                 what was detected, with coordinates
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "screenshots", "burst")
SHORT_SIDE = 720


def grab(serial: str) -> np.ndarray:
    cmd = ["adb", "-s", serial, "exec-out", "screencap", "-p"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or len(p.stdout) < 4096:
        raise RuntimeError(f"screencap failed: {p.stderr.decode('utf-8', 'replace')[:200]}")
    img = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("PNG decode failed")
    return img


def to_canvas(img: np.ndarray, short_side: int = SHORT_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    s = short_side / float(min(h, w))
    if abs(s - 1.0) < 1e-6:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def detect_cells(img: np.ndarray, top_ratio=0.13, bot_ratio=0.62):
    """Locate item cells by their saturated coloured frame (orange/blue/purple)."""
    h, w = img.shape[:2]
    top, bot = int(h * top_ratio), int(h * bot_ratio)
    roi = img[top:bot]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    mask = ((S > 90) & (V > 90)).astype(np.uint8) * 255
    purple = ((H >= 110) & (H <= 170) & (S > 45) & (V > 80)).astype(np.uint8) * 255
    blue = ((H >= 92) & (H <= 112) & (S > 55) & (V > 150)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(cv2.bitwise_or(mask, purple), blue)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if not (70 <= cw <= 115 and 70 <= ch <= 115):
            continue
        if not (0.7 <= cw / ch <= 1.45):
            continue
        if cv2.contourArea(c) < cw * ch * 0.40:
            continue
        boxes.append((x, y + top, cw, ch))
    boxes.sort(key=lambda b: -b[2] * b[3])
    kept = []
    for b in boxes:
        if all(abs(b[0] - k[0]) > 30 or abs(b[1] - k[1]) > 30 for k in kept):
            kept.append(b)
    kept.sort(key=lambda b: (b[1] // 45, b[0]))
    return kept


def cell_context(img, box, pad_x=30, pad_up=240, pad_down=150):
    """Cell plus the space above/below where the chicken and timer appear."""
    x, y, w, h = box
    x0 = max(0, x - pad_x)
    x1 = min(img.shape[1], x + w + pad_x)
    y0 = max(0, y - pad_up)
    y1 = min(img.shape[0], y + h + pad_down)
    return img[y0:y1, x0:x1], (x0, y0)


def montage(tiles, cols, cell_w, cell_h, labels=None, pad=6, bg=245):
    if not tiles:
        return None
    rows = (len(tiles) + cols - 1) // cols
    lab_h = 18 if labels else 0
    canvas = np.full((rows * (cell_h + pad + lab_h) + pad,
                      cols * (cell_w + pad) + pad, 3), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y = pad + r * (cell_h + pad + lab_h)
        x = pad + c * (cell_w + pad)
        canvas[y:y + cell_h, x:x + cell_w] = t
        if labels:
            cv2.putText(canvas, labels[i], (x, y + cell_h + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None,
                    help="output folder under screenshots/burst/ (default: timestamp)")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--interval", type=int, default=400, help="milliseconds between frames")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--cells", action="store_true", help="also crop each detected cell with context")
    ap.add_argument("--panel", action="store_true", help="crop the manage panel rows instead of cells")
    ap.add_argument("--short-side", type=int, default=SHORT_SIDE)
    ap.add_argument("--pad-up", type=int, default=240,
                    help="pixels of context above each cell (chicken + timer live here)")
    ap.add_argument("--pad-down", type=int, default=150,
                    help="pixels of context below each cell")
    ap.add_argument("--pad-x", type=int, default=30,
                    help="pixels of context left and right of each cell")
    args = ap.parse_args()

    # Pressing Run in an editor passes no arguments at all. Default to a
    # useful burst instead of dying on "required: --name".
    bare_run = len(sys.argv) == 1
    if bare_run:
        args.frames = max(args.frames, 12)
        args.cells = True
        print("[info] no arguments given -> 12 frames, 400 ms, --cells")
    if not args.name:
        args.name = time.strftime("%Y%m%d-%H%M%S")

    if args.serial is None:
        p = subprocess.run(["adb", "devices"], capture_output=True, text=True)
        devs = [l.split()[0] for l in p.stdout.splitlines()[1:]
                if l.strip() and l.split()[1:2] == ["device"]]
        if not devs:
            print("[error] no adb device", file=sys.stderr)
            return 1
        if len(devs) > 1:
            print(f"[error] multiple devices {devs}, pass --serial", file=sys.stderr)
            return 1
        args.serial = devs[0]

    out = os.path.join(OUT_ROOT, args.name)
    os.makedirs(out, exist_ok=True)

    frames = []
    t_start = time.time()
    for i in range(args.frames):
        t0 = time.time()
        img = to_canvas(grab(args.serial), args.short_side)
        frames.append(img)
        cv2.imwrite(os.path.join(out, f"f{i:02d}_720.png"), img)
        print(f"  frame {i:02d}  {img.shape[1]}x{img.shape[0]}  {(time.time()-t0)*1000:.0f} ms")
        if i < args.frames - 1:
            time.sleep(max(0.0, args.interval / 1000.0 - (time.time() - t0)))

    print(f"[burst] {len(frames)} frames in {time.time()-t_start:.1f}s -> {out}")

    report = [f"name={args.name}", f"frames={len(frames)}",
              f"interval_ms={args.interval}",
              f"canvas={frames[0].shape[1]}x{frames[0].shape[0]}", ""]

    if args.cells:
        per_frame = [detect_cells(f) for f in frames]
        counts = [len(b) for b in per_frame]
        report.append(f"cells per frame: {counts}")
        print(f"[cells] per frame: {counts}")

        # cells must be stable across frames; use the frame with the most
        best = int(np.argmax(counts))
        boxes = per_frame[best]
        report.append(f"using frame {best} ({len(boxes)} cells)")
        for i, b in enumerate(boxes):
            report.append(f"  cell{i}: x={b[0]} y={b[1]} w={b[2]} h={b[3]} "
                          f"center=({b[0]+b[2]//2},{b[1]+b[3]//2})")

        # crop each cell across all frames
        for i, b in enumerate(boxes):
            crops = []
            for k, f in enumerate(frames):
                patch, origin = cell_context(f, b, args.pad_x, args.pad_up, args.pad_down)
                cv2.imwrite(os.path.join(out, f"cell{i}_f{k:02d}.png"), patch)
                crops.append((patch, k))
            # montage row: frames side by side for this cell
            H = 240
            tiles, labels = [], []
            for patch, k in crops:
                t = cv2.resize(patch, (int(patch.shape[1] * H / patch.shape[0]), H),
                               interpolation=cv2.INTER_AREA)
                tiles.append(t)
                labels.append(f"f{k:02d}")
            m = montage(tiles, len(tiles), tiles[0].shape[1], H, labels)
            cv2.imwrite(os.path.join(out, f"montage_cell{i}.png"), m)
        print(f"[cells] per-cell montages written (montage_cell0..{len(boxes)-1}.png)")

        # one big sheet: rows = cells, cols = frames, with a row label per cell
        if boxes:
            H = 170
            rows_img = []
            for i, b in enumerate(boxes):
                tiles = []
                for k, f in enumerate(frames):
                    patch, _ = cell_context(f, b, args.pad_x, args.pad_up, args.pad_down)
                    tiles.append(cv2.resize(patch, (int(patch.shape[1] * H / patch.shape[0]), H),
                                            interpolation=cv2.INTER_AREA))
                w = min(t.shape[1] for t in tiles)
                row = np.hstack([t[:, :w] for t in tiles])
                label = np.full((22, row.shape[1], 3), 245, np.uint8)
                cv2.putText(label, f"cell{i}  centre=({b[0]+b[2]//2},{b[1]+b[3]//2})",
                            (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
                rows_img.append(np.vstack([label, row]))
            wmax = max(r.shape[1] for r in rows_img)
            rows_img = [cv2.copyMakeBorder(r, 0, 3, 0, wmax - r.shape[1],
                                           cv2.BORDER_CONSTANT, value=(245, 245, 245))
                        for r in rows_img]
            sheet = np.vstack(rows_img)
            cv2.imwrite(os.path.join(out, "montage_cells.png"), sheet)
            print(f"[cells] montage_cells.png  ({len(boxes)} rows x {len(frames)} frames, "
                  f"{sheet.shape[1]}x{sheet.shape[0]})")

    if args.panel:
        # manage panel occupies roughly y 400..1200 in the 720x1560 canvas
        for k, f in enumerate(frames):
            crop = f[400:1210, 55:670]
            cv2.imwrite(os.path.join(out, f"panel_f{k:02d}.png"), crop)
        print("[panel] panel_f*.png written (y 400..1210)")

    # contact sheet of full frames
    if len(frames) > 1:
        H = 360
        tiles = [cv2.resize(f, (int(f.shape[1] * H / f.shape[0]), H), interpolation=cv2.INTER_AREA)
                 for f in frames]
        cols = min(5, len(tiles))
        cv2.imwrite(os.path.join(out, "montage_full.png"),
                    montage(tiles, cols, tiles[0].shape[1], H,
                            [f"f{i:02d}" for i in range(len(tiles))]))
        print("[full] montage_full.png")

    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report))
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
