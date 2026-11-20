from pathlib import Path
p = Path(r"E:\work\AutoButter\tools\burst.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
'''def cell_context(img, box, pad_x=26, pad_up=120, pad_down=130):''',
'''def cell_context(img, box, pad_x=30, pad_up=240, pad_down=150):''')

s = s.replace(
'''    ap.add_argument("--short-side", type=int, default=SHORT_SIDE)''',
'''    ap.add_argument("--short-side", type=int, default=SHORT_SIDE)
    ap.add_argument("--pad-up", type=int, default=240,
                    help="pixels of context above each cell (chicken + timer live here)")
    ap.add_argument("--pad-down", type=int, default=150,
                    help="pixels of context below each cell")
    ap.add_argument("--pad-x", type=int, default=30,
                    help="pixels of context left and right of each cell")''')

s = s.replace(
'''                patch, origin = cell_context(f, b)''',
'''                patch, origin = cell_context(f, b, args.pad_x, args.pad_up, args.pad_down)''')

s = s.replace(
'''                    patch, _ = cell_context(f, b)''',
'''                    patch, _ = cell_context(f, b, args.pad_x, args.pad_up, args.pad_down)''')

p.write_text(s, encoding="utf-8")
print("patched: pad_up 120 -> 240, now configurable")
