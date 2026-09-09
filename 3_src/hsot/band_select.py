#!/usr/bin/env python3
"""E05 band selection(BABS/DeSAM2 式,training-free)——HOT2024 冠軍同源手法。

原理:官方假色是固定 CIE 線性投影;改為「首幀目標-背景可分性最大化」選出的
top-3 band 組 3ch,提升目標-背景對比(光譜相對可分性 1.525 已實測,D023)。
與 E04 的差異:這是「比較選最好」的相對操作(D023 支持),非絕對閾值判斷(D027 已敗)。

規則(確定性、僅用首幀 init box、不認序列名——Ranking B 合法):
  首幀 cube 上,每 band 算 Fisher 分數 |mean_t − mean_b| / sqrt(var_t + var_b)
  (目標區 = init box;背景 = box 外擴 1.5–2.5× 的環狀區,呼應 E04 局部背景教訓),
  取 top-3(按分數降序固定通道順序),全序列固定使用。

用法(生成 band-selected 3ch jpg 序列,供 track_t1 直接吃):
  python3 band_select.py --mosaic-root <HSI png 根> --seq <seq> --init "x y w h" \
      --out-dir <3ch 輸出> [--report <json>]
mosaic-root/<seq>/*.png 為 mosaic 幀(官方命名)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from hsot.io import MACRO, load_cube, modality_of, _read_u16  # package 模式
except ImportError:  # 平鋪模式(機器上與 io.py 同目錄)
    from io_hsot import MACRO, load_cube, modality_of, _read_u16  # type: ignore

RING = (1.5, 2.5)


def band_separability(cube: np.ndarray, box_xywh, ring=RING) -> np.ndarray:
    """每 band 的目標-背景 Fisher 分數。cube: (H, W, B) float。"""
    H, W, B = cube.shape
    x, y, w, h = [float(v) for v in box_xywh]
    cx, cy, s = x + w / 2, y + h / 2, float(np.sqrt(w * h))
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.maximum(np.abs(xx - cx) / max(w, 1e-6), np.abs(yy - cy) / max(h, 1e-6)) * 2  # 盒形距離(1.0=框邊)
    tgt = d <= 1.0
    bg = (d > ring[0]) & (d <= ring[1])  # 1.5–2.5× 環狀背景(E04 局部背景教訓)
    if tgt.sum() < 4 or bg.sum() < 16:
        return np.zeros(B)
    t = cube[tgt].astype(np.float64)   # (Nt, B)
    b = cube[bg].astype(np.float64)
    return np.abs(t.mean(0) - b.mean(0)) / np.sqrt(t.var(0) + b.var(0) + 1e-9)


def normalize_band(band: np.ndarray) -> np.ndarray:
    """per-frame percentile normalize(D024:VIS 熱像素離群、RedNIR 極暗)。"""
    lo, hi = np.percentile(band, [1, 99])
    if hi <= lo:
        hi = lo + 1
    return np.clip((band - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def main() -> None:
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--mosaic-root", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--init", required=True, help='"x y w h"')
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report")
    args = ap.parse_args()

    seq = args.seq
    mod = modality_of(seq)
    seq_dir = Path(args.mosaic_root) / seq
    pngs = sorted(seq_dir.glob("*.png"))
    assert pngs, f"{seq_dir} 無 mosaic png"
    init = [float(v) for v in args.init.split()]

    first = load_cube(_read_u16(pngs[0]), mod)
    scores = band_separability(first, init)
    top3 = np.argsort(scores)[::-1][:3].tolist()
    out = Path(args.out_dir) / seq
    out.mkdir(parents=True, exist_ok=True)
    for f in pngs:
        cube = load_cube(_read_u16(f), mod)
        rgb = np.stack([normalize_band(cube[:, :, b]) for b in top3], axis=-1)
        Image.fromarray(rgb).save(out / (f.stem + ".jpg"), quality=95)
    (out / "init_rect.txt").write_text(" ".join(str(v) for v in init))
    rep = {"seq": seq, "modality": mod, "top3_bands": top3,
           "scores": [round(float(s), 4) for s in scores]}
    print(f"{seq}: top3={top3} scores={[round(scores[b],3) for b in top3]} → {out}({len(pngs)} 幀)")
    if args.report:
        Path(args.report).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
