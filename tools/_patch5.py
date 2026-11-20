from pathlib import Path
p = Path(r"E:\work\AutoButter\tools\burst.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
'''    ap.add_argument("--name", required=True, help="output folder under screenshots/burst/")''',
'''    ap.add_argument("--name", default=None,
                    help="output folder under screenshots/burst/ (default: timestamp)")''')

s = s.replace(
'''    args = ap.parse_args()
''',
'''    args = ap.parse_args()

    # Pressing Run in an editor passes no arguments at all. Default to a
    # useful burst instead of dying on "required: --name".
    bare_run = len(sys.argv) == 1
    if bare_run:
        args.frames = max(args.frames, 12)
        args.cells = True
        print("[info] no arguments given -> 12 frames, 400 ms, --cells")
    if not args.name:
        args.name = time.strftime("%Y%m%d-%H%M%S")
''')

p.write_text(s, encoding="utf-8")
print("patched")
