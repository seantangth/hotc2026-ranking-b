#!/usr/bin/env python3
"""用**既有** crop_meta.json 的窗裁切另一個 frames root。

⚠️ D094（08-29）：E-A' band×SAM3 主流程**不用**本工具——window 沿用被撤回，
改用 `hsot.crop_rerun prep` 對 band full-frame 軌跡重算窗（「同一套規則作用在
不同輸入」才是正確的單變因；窗是規則的產物、不是規則本身，且 9/7 部署形態
的窗本來就會重算）。本工具為「重用既有窗」情境保留（窗/解析度 fail-closed
邏輯日後可用），測試一併保留。

`hsot.crop_rerun prep` 是「從軌跡算窗＋裁切」一體；本工具只做後半——窗、
段結構、jpeg 編碼參數全部**沿用 meta**（v078 的窗），對新輸入（band-3ch
frames）裁出與原 crop run 同構的目錄（`<out>/<name>/*.jpg`＋`init_rect.txt`），
供 track_t1 與 `crop_rerun merge` 直接使用。窗不重算 ⇒ 唯一變因＝窗內像素。

init 語意對照 `crop_rerun.cmd_prep`：段首幀的 base 軌跡框平移 (−x1, −y1)
——base 軌跡從 `--base-csv`（沿用 v078 的 full 主線 submission）讀，
與原 prep 完全同源。幀→檔案以「序列內位置」對應（同 prep 的
convention-agnostic 修正，不假設幀號起點）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 3_src/


def load_tracks(csv_path) -> dict:
    """submission 型 CSV → {seq: [(frame:int, x, y, w, h)] 按 frame 升冪}。"""
    per = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        assert header[0].strip().lower() == "id", f"{csv_path} 標頭異常"
        for row in r:
            if not row or not row[0]:
                continue
            seq, fr = row[0].rsplit("_", 1)
            per.setdefault(seq, []).append(
                (int(fr), *[float(v) for v in row[1:5]]))
    for seq in per:
        per[seq].sort(key=lambda t: t[0])
    return per


def apply_one(name: str, win: dict, frames_root: Path, out_root: Path,
              tracks: dict) -> dict:
    from PIL import Image

    seq = win["seq"]
    x1, y1 = int(win["x1"]), int(win["y1"])
    x2, y2 = x1 + int(win["w"]), y1 + int(win["h"])
    f_lo, f_hi = win["frames"]
    rows = tracks.get(seq)
    assert rows, f"{name}: base-csv 無 {seq} 軌跡（init 需要它）"
    frames_all = [t[0] for t in rows]
    seq_dir = frames_root / seq
    jpgs = sorted(seq_dir.glob("*.jpg"), key=lambda q: int(q.stem))
    assert len(jpgs) == len(frames_all), (
        f"{name}: base {len(frames_all)} 列 vs {seq_dir} {len(jpgs)} 張——"
        "幀與影像不對齊，拒絕裁切（同 crop_rerun prep 的 fail-closed）")
    frame_to_jpg = dict(zip(frames_all, jpgs))
    orig_w, orig_h = win["orig"]
    with Image.open(jpgs[0]) as im0:
        assert im0.size == (orig_w, orig_h), (
            f"{name}: frames {im0.size} vs meta orig {(orig_w, orig_h)}——"
            "輸入解析度與開窗時不同，窗不可沿用")

    # jpeg 編碼沿用 meta 記載的 profile，經 crop_rerun._jpeg_encoding 反查
    # kwargs（同源函式，位元級同款輸出路徑）。
    from hsot.crop_rerun import _jpeg_encoding, JPEG_PROFILE_LEGACY

    profile = win.get("crop_image_encoding", {}).get("profile", JPEG_PROFILE_LEGACY)
    _, save_kwargs = _jpeg_encoding(profile)

    dest = out_root / name
    dest.mkdir(parents=True, exist_ok=True)
    seg_frames = [fr for fr in frames_all if f_lo <= fr <= f_hi]
    for fr in seg_frames:
        f = frame_to_jpg[fr]
        Image.open(f).crop((x1, y1, x2, y2)).save(dest / f.name, **save_kwargs)
    first = next(t for t in rows if t[0] == seg_frames[0])
    init = [first[1] - x1, first[2] - y1, first[3], first[4]]
    (dest / "init_rect.txt").write_text(" ".join(str(v) for v in init))
    return {"name": name, "seq": seq, "n_frames": len(seg_frames),
            "window": [x1, y1, win["w"], win["h"]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True, help="既有 crop_meta.json（v078 的窗）")
    ap.add_argument("--frames-root", required=True, help="要被裁的 frames root（band-3ch）")
    ap.add_argument("--base-csv", required=True,
                    help="開窗當時的 full 主線 submission（init 同源；沿用 v078 的）")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--only-seqs", help="序列清單檔（觸發支）；窗的 seq 不在清單者跳過")
    ap.add_argument("--names-out", help="輸出裁切段名清單（track_t1 --seq-list 用）")
    args = ap.parse_args(argv)

    meta = json.loads(Path(args.meta).read_text())
    tracks = load_tracks(args.base_csv)
    only = None
    if args.only_seqs:
        only = {s.strip() for s in Path(args.only_seqs).read_text().splitlines() if s.strip()}

    done = []
    for name, win in sorted(meta.items()):
        if only is not None and win["seq"] not in only:
            continue
        r = apply_one(name, win, Path(args.frames_root), Path(args.out_root), tracks)
        done.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    print(f"[apply_crop_meta] 裁切 {len(done)} 段（窗沿用 {args.meta}）")
    if args.names_out:
        Path(args.names_out).write_text("\n".join(r["name"] for r in done) + "\n")
    return 0 if done else 2


if __name__ == "__main__":
    sys.exit(main())
