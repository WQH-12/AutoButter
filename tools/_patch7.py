from pathlib import Path
p = Path(r"E:\work\AutoButter\tools\burst.py")
s = p.read_text(encoding="utf-8")

old = '''        # one big sheet: rows = cells, cols = frames
        if boxes:
            H = 150
            rows_img = []
            for i, b in enumerate(boxes):
                tiles = []
                for k, f in enumerate(frames):
                    patch, _ = cell_context(f, b, args.pad_x, args.pad_up, args.pad_down)
                    tiles.append(cv2.resize(patch, (int(patch.shape[1] * H / patch.shape[0]), H),
                                            interpolation=cv2.INTER_AREA))
                w = min(t.shape[1] for t in tiles)
                rows_img.append(np.hstack([t[:, :w] for t in tiles]))
            wmax = max(r.shape[1] for r in rows_img)
            rows_img = [cv2.copyMakeBorder(r, 0, 2, 0, wmax - r.shape[1],
                                           cv2.BORDER_CONSTANT, value=(245, 245, 245))
                        for r in rows_img]
            cv2.imwrite(os.path.join(out, "montage_cells.png"), np.vstack(rows_img))
            print(f"[cells] montage_cells.png  ({len(boxes)} rows x {len(frames)} frames)")
'''

new = '''        # one big sheet: rows = cells, cols = frames, with a row label per cell
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
'''

if old not in s:
    raise SystemExit("anchor not found")
s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
print("patched montage_cells with row labels")
