#!/usr/bin/env python3
"""Convert official `HSI-<modality>-Falsecolor/<seq>/groundtruth_rect.txt` files into the
single GT CSV that `hsot/eval.py` scores against.

Evaluation helper only. **No part of the tracking pipeline reads ground truth or calls this
script** — `run_ranking_b.py`, `track_t1.py` and `finalize_submission.py` never open a
`groundtruth_rect.txt`, and the ingestion step copies only frames and `init_rect.txt` into
`--frames-root`. It is included so that our verification numbers on the released sample
sequences can be reproduced.

Output ID convention matches `sample_submission.csv`: `<modality>-<seq>_<1-based frame>`.

用法：python 3_src/gt_dir_to_csv.py <ranking 根（含 HSI-*-Falsecolor）> <out.csv>
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

MOD = re.compile(r"^HSI-(NIR|RedNIR|VIS)-False[-_]?colou?r$", re.IGNORECASE)


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows, n_seq = [], 0
    for mod_dir in sorted(root.iterdir()):
        m = MOD.match(mod_dir.name) if mod_dir.is_dir() else None
        if not m:
            continue
        prefix = m.group(1).lower() + "-"
        for seq_dir in sorted(p for p in mod_dir.iterdir() if p.is_dir()):
            gt = seq_dir / "groundtruth_rect.txt"
            if not gt.is_file():
                print(f"[skip] {seq_dir}: 無 groundtruth_rect.txt", file=sys.stderr)
                continue
            n_frames = len(list(seq_dir.glob("*.jpg")))
            lines = [ln for ln in gt.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
            if len(lines) != n_frames:
                raise SystemExit(f"FATAL {seq_dir.name}: GT {len(lines)} 列 ≠ {n_frames} 幀")
            for i, ln in enumerate(lines, start=1):
                x, y, w, h = [float(t) for t in re.split(r"[,\s]+", ln.strip())[:4]]
                rows.append((f"{prefix}{seq_dir.name}_{i}", x, y, w, h))
            n_seq += 1
    rows.sort(key=lambda r: (r[0].rsplit("_", 1)[0], int(r[0].rsplit("_", 1)[1])))
    with out.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["ID", "x", "y", "width", "height"])
        wr.writerows(rows)
    print(f"[gt] {n_seq} 支／{len(rows)} 列 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
