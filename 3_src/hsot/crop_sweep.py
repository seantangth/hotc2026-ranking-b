#!/usr/bin/env python3
"""crop-zoom 覆蓋率與放大倍率的免費前測（純 CSV + 圖片尺寸，無 GPU、無 GT）。

回答 v3 的兩個 P1 問題，不必先燒一輪 GPU：
  Q1 門檻放寬——`SMALL_T` / `AREA_FRAC_MAX` 各檔位下，選中幾支、覆蓋多少幀、放大幾倍？
     （E19 用 32 / 0.40 選中 21 支，**22 支小目標因窗面積 >40% 被剔除**。）
  Q2 分段窗——把序列切成 K 段、每段各算窗，放大倍率能提高多少、能救回幾支？
     （現行為全序列固定窗；長序列目標移動大 → 窗必然大 → 被面積門檻剔除。）

放大倍率為何是關鍵量：SAM3 內部把整張圖 resize 到 image_size²（預設 1008，stride 14
→ 72×72 patch）。原圖約 409×216 時，5.5px 的目標只佔約 1 個 patch。裁切把 patch 預算
集中到目標周圍，**線性放大倍率 = sqrt(面積放大倍率)** 才是與 patch 數線性相關的量，
故本表同時列出兩者，避免把面積 4.2x 誤讀成「大 4.2 倍」（實際線性只有 2.05x）。

選序列規則與 crop_rerun.py 保持一致（確定性、無 GT、不認序列名 → Ranking B 合法）。

用法：
  python3 -m hsot.crop_sweep --frames-root <假色根> --base-csv <E15 軌跡> \
      --envelope-extra <E02 軌跡> [--segments 1 2 4 8] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MARGIN_SCALE = 2.0   # 與 crop_rerun.py 同值（E04 實測 R=2–3× 的域內證據）
MIN_MARGIN = 32.0


def parse(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    p = df["ID"].str.rsplit("_", n=1, expand=True)
    df["seq"], df["frame"] = p[0], p[1].astype(int)
    return df


def window_for(boxes: np.ndarray, scale: float, W: float, H: float) -> tuple[float, float, float, float]:
    """軌跡 envelope 外擴 margin，夾在原圖內。與 crop_rerun.cmd_prep 同一算式。"""
    m = max(MARGIN_SCALE * scale, MIN_MARGIN)
    x1 = max(0.0, float(np.min(boxes[:, 0])) - m)
    y1 = max(0.0, float(np.min(boxes[:, 1])) - m)
    x2 = min(W, float(np.max(boxes[:, 0] + boxes[:, 2])) + m)
    y2 = min(H, float(np.max(boxes[:, 1] + boxes[:, 3])) + m)
    return x1, y1, x2, y2


def analyse(frames_root: Path, base: pd.DataFrame, extra: pd.DataFrame | None,
            segments: list[int]) -> list[dict]:
    """逐序列算出：首幀尺度、幀數、各分段數下的窗面積佔比與放大倍率。"""
    from PIL import Image

    rows = []
    for seq, g in base.groupby("seq"):
        g = g.sort_values("frame")
        first = g.iloc[0]
        scale = float(np.sqrt(first["width"] * first["height"]))
        seq_dir = frames_root / seq
        jpgs = sorted(seq_dir.glob("*.jp*g"))
        if not jpgs:
            continue
        with Image.open(jpgs[0]) as im:
            W, H = im.size
        b = g[["x", "y", "width", "height"]].to_numpy(float)
        if extra is not None:
            e = extra[extra["seq"] == seq]
            if len(e):
                b = np.vstack([b, e.sort_values("frame")[["x", "y", "width", "height"]].to_numpy(float)])
        n_frames = len(g)
        row = {"seq": seq, "scale": round(scale, 1), "n_frames": n_frames,
               "orig": [W, H], "modality": seq.split("-")[0]}
        for k in segments:
            # 分 k 段各自算窗；每段的窗面積佔比取「幀數加權平均」，
            # 因為評分是全幀 pooling，長段的窗品質理當佔更大權重。
            bounds = np.linspace(0, n_frames, k + 1).astype(int)
            fracs, zooms, weights = [], [], []
            for i in range(k):
                lo, hi = bounds[i], bounds[i + 1]
                if hi <= lo:
                    continue
                seg = g.iloc[lo:hi][["x", "y", "width", "height"]].to_numpy(float)
                if extra is not None:
                    e = extra[extra["seq"] == seq].sort_values("frame")
                    if len(e):
                        seg = np.vstack([seg, e.iloc[lo:min(hi, len(e))][
                            ["x", "y", "width", "height"]].to_numpy(float)])
                x1, y1, x2, y2 = window_for(seg, scale, W, H)
                area = max((x2 - x1) * (y2 - y1), 1.0)
                fracs.append(area / (W * H))
                zooms.append(W * H / area)
                weights.append(hi - lo)
            w = np.array(weights, dtype=float)
            row[f"frac_k{k}"] = round(float(np.average(fracs, weights=w)), 4)
            row[f"zoom_k{k}"] = round(float(np.average(zooms, weights=w)), 2)
            row[f"lin_k{k}"] = round(float(np.sqrt(np.average(zooms, weights=w))), 2)
        rows.append(row)
    return rows


def report(rows: list[dict], small_ts: list[float], area_maxes: list[float],
           segments: list[int], total_frames: int) -> None:
    print(f"\n{'='*78}\n序列總數 {len(rows)}｜總幀數 {total_frames:,}\n{'='*78}")

    for k in segments:
        print(f"\n### 分 {k} 段（k=1 ＝ E19 現行的全序列固定窗）")
        print(f"{'SMALL_T':>8} {'AREA_MAX':>9} {'選中':>5} {'幀佔比':>7} "
              f"{'線性放大 中位':>13} {'範圍':>13}")
        for st in small_ts:
            for am in area_maxes:
                sel = [r for r in rows if r["scale"] < st and r[f"frac_k{k}"] < am]
                if not sel:
                    print(f"{st:>8.0f} {am:>9.2f} {0:>5}")
                    continue
                fr = sum(r["n_frames"] for r in sel) / total_frames
                lin = [r[f"lin_k{k}"] for r in sel]
                print(f"{st:>8.0f} {am:>9.2f} {len(sel):>5} {fr:>6.1%} "
                      f"{np.median(lin):>13.2f} {min(lin):>6.2f}–{max(lin):<6.2f}")

    # 被 E19 現行門檻剔除的小目標：差多少才過關？這 22 支是 P1 的直接標的。
    # 需要 k=1 當基準線；未傳 1 就跳過（否則撞 KeyError）。
    if 1 not in segments:
        return
    print(f"\n### E19 現行門檻(32/0.40)下「是小目標但被面積剔除」的序列")
    rej = sorted([r for r in rows if r["scale"] < 32 and r["frac_k1"] >= 0.40],
                 key=lambda r: r["frac_k1"])
    print(f"共 {len(rej)} 支，佔 {sum(r['n_frames'] for r in rej)/total_frames:.1%} 幀")
    hdr = f"{'序列':<22}{'尺度':>6}{'幀數':>7}" + "".join(f"{'k='+str(k):>9}" for k in segments)
    print(hdr)
    for r in rej:
        line = f"{r['seq']:<22}{r['scale']:>6.1f}{r['n_frames']:>7}"
        line += "".join(f"{r[f'frac_k{k}']:>9.2f}" for k in segments)
        print(line)
    print("（表中數字＝窗面積佔原圖比例；<0.40 即可通過現行門檻。"
          "看 k 增大時它掉到多少，就是分段窗能救回幾支。）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--base-csv", required=True)
    ap.add_argument("--envelope-extra")
    ap.add_argument("--segments", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--small-t", type=float, nargs="+", default=[32, 48, 64])
    ap.add_argument("--area-max", type=float, nargs="+", default=[0.40, 0.55, 0.70, 0.85])
    ap.add_argument("--json", help="逐序列明細輸出路徑")
    args = ap.parse_args()

    base = parse(args.base_csv)
    extra = parse(args.envelope_extra) if args.envelope_extra else None
    rows = analyse(Path(args.frames_root), base, extra, args.segments)
    report(rows, args.small_t, args.area_max, args.segments, len(base))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"\n逐序列明細 → {args.json}")


if __name__ == "__main__":
    main()
