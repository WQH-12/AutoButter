from pathlib import Path
p = Path(r"E:\work\AutoButter\tools\crop_items.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
'''    ap.add_argument("--inset", type=int, default=6,
                    help="pixels trimmed from every side (default 6)")
    ap.add_argument("--drop-bottom", type=int, default=0,
                    help="extra pixels trimmed from the bottom (level text)")''',
'''    ap.add_argument("--inset", type=int, default=6,
                    help="pixels trimmed from every side (default 6)")
    ap.add_argument("--inset-x", type=int, default=None,
                    help="horizontal inset, overrides --inset")
    ap.add_argument("--inset-top", type=int, default=None,
                    help="top inset, overrides --inset")
    ap.add_argument("--drop-bottom", type=int, default=0,
                    help="extra pixels trimmed from the bottom (level text)")''')

s = s.replace(
'''        x0, y0 = x + args.inset, y + args.inset
        x1, y1 = x + w - args.inset, y + h - args.inset - args.drop_bottom''',
'''        ix = args.inset if args.inset_x is None else args.inset_x
        it = args.inset if args.inset_top is None else args.inset_top
        x0, y0 = x + ix, y + it
        x1, y1 = x + w - ix, y + h - it - args.drop_bottom''')

p.write_text(s, encoding="utf-8")
print("patched crop_items.py")
