#!/usr/bin/env python3
"""E29 全量 test：對 crop 選中序列產生 pan-sharpen 裁切窗（顏色 ＋ 真實解析度）。

與 val canary（prep_mosaic_crop.py ＋ pansharp_v1.py）的差異只在**資料型態**：
  假色  ：t1test_fc_75.tar 解開後的**目錄**（<seq>/NNNN.jpg）
  HSI   ：VIS 是 <name>.zip；**NIR/RedNIR 是逐檔目錄**（D036 禁止的樣態，官方就這樣存）
⇒ 本檔統一成一個 index 介面吃兩種型態，其餘合成邏輯完全沿用 pansharp_v1.pansharpen。

座標約定（與 E26 mosaic 腿相同，merge 時要除回 macro）：
  窗   ：由 crop_rerun prep 產生的 meta（假色座標系）
  裁切 ：假色窗 (x1,y1,w,h) → mosaic 窗 (x1*macro, y1*macro, w*macro, h*macro)
  init ：假色 init 座標 × macro

用法：
  python3 prep_pansharp_test_v1.py --fc-root ~/test_fc --hsi-zip-dir ~/test_hsi/zip \
      --hsi-dir-root ~/test_hsi/dir --meta ~/crop_meta.json --out-root ~/e29_test \
      --variant equalized_stretch
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pansharp_v1 import pansharpen  # noqa: E402

IMG_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


class HsiSource:
    """統一介面：不管 HSI 是 zip 還是逐檔目錄，都給「第 i 幀的 ndarray」。"""

    def __init__(self, seq: str, zip_dir: Path, dir_root: Path):
        self.zf = None
        z = zip_dir / f"{seq}.zip"
        d = dir_root / seq
        if z.exists():
            self.zf = zipfile.ZipFile(z)
            names = [n for n in self.zf.namelist()
                     if Path(n).suffix.lower() in IMG_EXT and re.search(r"\d", Path(n).stem)]
            self.items = sorted(names, key=lambda n: int(re.sub(r"\D", "", Path(n).stem)))
        elif d.is_dir():
            fs = [p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT]
            self.items = sorted(fs, key=lambda p: int(re.sub(r"\D", "", p.stem)))
        else:
            raise FileNotFoundError(f"{seq}: 既無 {z} 也無 {d}")

    def __len__(self) -> int:
        return len(self.items)

    def get(self, i: int) -> np.ndarray:
        it = self.items[i]
        if self.zf is not None:
            with self.zf.open(it) as fh:
                a = np.array(Image.open(io.BytesIO(fh.read())))
        else:
            a = np.array(Image.open(it))
        return a[..., 0] if a.ndim == 3 else a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc-root", required=True, help="解開的 t1test_fc_75（<seq>/NNNN.jpg）")
    ap.add_argument("--hsi-zip-dir", required=True)
    ap.add_argument("--hsi-dir-root", required=True)
    ap.add_argument("--meta", required=True, help="crop_rerun prep 產生的窗表 json")
    ap.add_argument("--base-csv", required=True,
                    help="base 軌跡（E15 test）——用來推 init 框，與 crop_rerun 同一條規則")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--variant", default="equalized_stretch")
    ap.add_argument("--quality", type=int, default=95)
    a = ap.parse_args()

    fc_root = Path(a.fc_root).resolve()
    zip_dir, dir_root = Path(a.hsi_zip_dir).resolve(), Path(a.hsi_dir_root).resolve()
    out_root = Path(a.out_root).resolve()
    wins = json.loads(Path(a.meta).read_text())
    pan_v, do_stretch = a.variant.replace("_stretch", ""), a.variant.endswith("_stretch")

    import pandas as pd
    base = pd.read_csv(a.base_csv)
    _p = base["ID"].str.rsplit("_", n=1, expand=True)
    base["seq"], base["frame"] = _p[0], _p[1].astype(int)

    meta_out, skipped = {}, []
    for name, w in sorted(wins.items()):
        seq = w.get("seq", name)
        fc_dir = fc_root / seq
        jpgs = sorted(fc_dir.glob("*.jpg"))
        if not jpgs:
            skipped.append(f"{name}: 假色缺"); continue
        try:
            src = HsiSource(seq, zip_dir, dir_root)
        except FileNotFoundError as e:
            skipped.append(f"{name}: {e}"); continue

        # fail-closed：幀數不等 ⇒ 位置對位不成立（AUDIT.md §2 的同型陷阱）
        if len(src) != len(jpgs):
            skipped.append(f"{name}: HSI {len(src)} 幀 != 假色 {len(jpgs)} 幀"); continue

        probe_f = np.array(Image.open(jpgs[0]))
        probe_h = src.get(0)
        macro = int(round(probe_h.shape[0] / probe_f.shape[0]))
        if (probe_h.shape[0] != probe_f.shape[0] * macro
                or probe_h.shape[1] != probe_f.shape[1] * macro):
            skipped.append(f"{name}: mosaic {probe_h.shape} 非假色 {probe_f.shape[:2]} 整數倍"); continue

        x1, y1, ww, hh = int(w["x1"]), int(w["y1"]), int(w["w"]), int(w["h"])
        dest = out_root / name
        dest.mkdir(parents=True, exist_ok=True)
        for i, jp in enumerate(jpgs):
            fc = np.array(Image.open(jp).convert("RGB"))[y1:y1 + hh, x1:x1 + ww]
            mo = src.get(i)[y1 * macro:(y1 + hh) * macro, x1 * macro:(x1 + ww) * macro]
            if mo.dtype != np.uint8:                     # 10-bit uint16 → 8-bit（窗內 p1–p99）
                lo, hi = np.percentile(mo, 1), np.percentile(mo, 99)
                mo = np.clip((mo.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
            img = pansharpen(fc, mo, macro, pan_v, "hpf", do_stretch)
            Image.fromarray(img).save(dest / jp.name, quality=a.quality)

        # init 與 crop_rerun 同一條規則：base 軌跡該段首幀的框，減去窗原點；再 ×macro
        f_lo = w.get("frames", [1, None])[0]
        r0 = base[(base["seq"] == seq) & (base["frame"] == f_lo)]
        if r0.empty:
            skipped.append(f"{name}: base 無首幀 {f_lo}"); continue
        r0 = r0.iloc[0]
        init = [(float(r0["x"]) - x1) * macro, (float(r0["y"]) - y1) * macro,
                float(r0["width"]) * macro, float(r0["height"]) * macro]
        (dest / "init_rect.txt").write_text(" ".join(str(v) for v in init))
        meta_out[name] = {"seq": seq, "macro": macro, "win": [x1, y1, ww, hh],
                          "orig": w.get("orig"), "n_frames": len(jpgs),
                          "pan_size": [ww * macro, hh * macro]}
        print(f"✅ {name}: macro={macro} {len(jpgs)}f {ww}x{hh} → {ww*macro}x{hh*macro}", flush=True)

    (out_root / "pansharp_test_meta.json").write_text(json.dumps(meta_out, indent=1))
    print(f"\n完成 {len(meta_out)} 支｜跳過 {len(skipped)} 支")
    for s in skipped:
        print(f"  ⚠️ {s}")
    if not meta_out:
        sys.exit(2)                                      # fail-closed


if __name__ == "__main__":
    main()
