#!/usr/bin/env python3
"""E26：crop 窗內「感測器真實像素 vs 插值上採樣」的單變因對照 prep。

假說（D046 的正交延伸）：v008 的 crop-zoom 把約 100×100 的窗上採樣到 SAM3 的 1008
＝插值放大約 4–10 倍；而官方假色的每個像素本來就是 macro×macro（4×4 或 5×5）個
**真實感測器像素**壓縮而成 ⇒ 在同一個窗內改用 mosaic 原始像素，等於把「插值猜出來的」
換成「真的拍到的」，且**不觸犯 D046 的插值過量禁區**（v014 實測 −0.0208 的死因是插值，
真實像素與該機制正交）。

兩腿共用**同一組 crop 窗**（唯一變因＝窗內像素來源）：
  腿 A（fc）    ：官方假色裁窗 → 原尺寸存檔（SAM3 內部自行 resize，＝ v008 現行行為）
  腿 B（mosaic）：mosaic 裁 (窗 × macro) → p1–p99 灰階 ×3ch（D054 已驗證的轉檔）

座標映射（回假色原圖座標系，供與 GT 比對）：
  腿 A：box + (x1, y1)
  腿 B：box / macro + (x1, y1)
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def index_zip(zf: zipfile.ZipFile, exts) -> dict:
    out = {}
    for n in zf.namelist():
        p = Path(n)
        if p.suffix.lower() in exts and p.stem.isdigit():
            out[int(p.stem)] = n
    return out


def read_member(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as fh:
        return np.array(Image.open(io.BytesIO(fh.read())))


def prep_one(seq: str, hsi_zip: Path, fc_zip: Path, win: dict, gt: pd.DataFrame,
             out_root: Path, quality: int = 95) -> dict:
    zh, zf_ = zipfile.ZipFile(hsi_zip), zipfile.ZipFile(fc_zip)
    hidx, fidx = index_zip(zh, {".png", ".tif", ".tiff"}), index_zip(zf_, {".jpg", ".jpeg", ".png"})
    common = sorted(set(hidx) & set(fidx))
    if not common:
        raise RuntimeError(f"{seq}: HSI 與假色無共同幀")

    probe_h = read_member(zh, hidx[common[0]])
    probe_f = read_member(zf_, fidx[common[0]])
    if probe_h.ndim == 3:
        probe_h = probe_h[..., 0]
    macro = int(round(probe_h.shape[0] / probe_f.shape[0]))
    assert probe_h.shape[0] == probe_f.shape[0] * macro and probe_h.shape[1] == probe_f.shape[1] * macro, \
        f"{seq}: mosaic {probe_h.shape} 不是假色 {probe_f.shape[:2]} 的整數倍"
    assert [probe_f.shape[1], probe_f.shape[0]] == list(win["orig"]), \
        f"{seq}: 假色尺寸 {probe_f.shape[:2][::-1]} != 窗記錄的 orig {win['orig']}"

    x1, y1, w, h = int(win["x1"]), int(win["y1"]), int(win["w"]), int(win["h"])
    dir_fc = out_root / "fc" / seq
    dir_mo = out_root / "mosaic" / seq
    dir_fc.mkdir(parents=True, exist_ok=True)
    dir_mo.mkdir(parents=True, exist_ok=True)

    for i, fr in enumerate(common, start=1):
        fc = read_member(zf_, fidx[fr])
        if fc.ndim == 2:
            fc = np.stack([fc] * 3, -1)
        Image.fromarray(fc[y1:y1 + h, x1:x1 + w]).save(dir_fc / f"{i:04d}.jpg", quality=quality)

        mo = read_member(zh, hidx[fr])
        if mo.ndim == 3:
            mo = mo[..., 0]
        sub = mo[y1 * macro:(y1 + h) * macro, x1 * macro:(x1 + w) * macro]
        # p1–p99 正規化在**窗內**統計（D054 是全圖；窗內更貼合目標動態範圍）
        lo, hi = np.percentile(sub, 1), np.percentile(sub, 99)
        g = np.clip((sub.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
        Image.merge("RGB", [Image.fromarray(g)] * 3).save(dir_mo / f"{i:04d}.jpg", quality=quality)

    rows = gt[gt["seq"] == seq].sort_values("frame")
    r0 = rows.iloc[0]
    ix, iy = float(r0["x"]) - x1, float(r0["y"]) - y1
    iw, ih = float(r0["width"]), float(r0["height"])
    (dir_fc / "init_rect.txt").write_text(f"{ix} {iy} {iw} {ih}")
    (dir_mo / "init_rect.txt").write_text(
        f"{ix * macro} {iy * macro} {iw * macro} {ih * macro}")

    return {"seq": seq, "macro": macro, "n_frames": len(common),
            "win": [x1, y1, w, h], "orig": win["orig"],
            "fc_size": [w, h], "mosaic_size": [w * macro, h * macro],
            "frame_ids": [int(x) for x in common],
            "init_fc": [ix, iy, iw, ih]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hsi-dir", required=True, help="含 <seq>.zip 的 mosaic 目錄")
    ap.add_argument("--fc-dir", required=True, help="含 <seq>.zip 的官方假色目錄")
    ap.add_argument("--windows", required=True, help="crop_windows_val.json")
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    a = ap.parse_args()

    gt = pd.read_csv(a.gt_csv)
    p = gt["ID"].str.rsplit("_", n=1, expand=True)
    gt["seq"], gt["frame"] = p[0], p[1].astype(int)
    wins = json.loads(Path(a.windows).read_text())
    out_root = Path(a.out_root)

    meta = {}
    for s in a.seqs:
        if s not in wins:
            raise SystemExit(f"{s} 不在窗表內（crop 未選中它，不該進本實驗）")
        m = prep_one(s, Path(a.hsi_dir) / f"{s}.zip", Path(a.fc_dir) / f"{s}.zip",
                     wins[s], gt, out_root)
        meta[s] = m
        print(f"✅ {s}: macro={m['macro']} {m['n_frames']}f "
              f"fc={m['fc_size']} mosaic={m['mosaic_size']} init_fc={[round(v,1) for v in m['init_fc']]}")
    (out_root / "prep_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"\n→ {out_root}/prep_meta.json")


if __name__ == "__main__":
    main()
