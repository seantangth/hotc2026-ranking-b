#!/usr/bin/env python3
"""固定大小滑動窗的窗幾何前測（純 CSV + 尺寸表，無 GPU、無 GT、零成本）。

【為什麼是這個設計】D043 由 v009/E21 的失敗**反推**出來：分段窗 k=4 的窗幾何前測全部
算對（覆蓋 30%→51%、線性放大 2.05→2.54），LB 卻 −0.0139。敗因不在窗幾何，而在兩件
**破壞追蹤時間連續性**的事：
  (a) 段首用 base 軌跡的**預測框**（而非 GT）重新 init ＝ 拿 LB 0.674 的軌跡週期性
      覆寫 LB 0.691 的軌跡；
  (b) 每段 `init_state` 從零起，7 幀 memory bank 在段開頭是空的。
⇒ 正解 = **窗大小固定、位置逐段平移**：各段影像尺寸一致 → 可直接拼成**單一連續影片**
   → 只需首幀 GT init、memory 全程連續，(a)(b) 兩個敗因同時消失。

【窗尺寸的定義】先切 K 段、各段算 envelope+margin，取**所有段的最大寬高**為固定尺寸
(cw, ch)；每段窗位置 = 該段 envelope 中心（夾在原圖內）。取最大值保證每段的窗都涵蓋
該段全部軌跡框，同時所有段尺寸相同。K 越大 → 每段移動範圍越小 → cw 越小 → 放大越大，
但窗平移越頻繁（背景流動加劇，這一項幾何前測**測不出來**，只能上 LB）。

【範圍鐵律（D038 不對稱比）】本前測**只評估 E19 因面積門檻被剔除的那批小目標序列**。
E19 已驗證 +0.0175 的 21 支不得重做——v009/E21 正是死在「把已驗證的部分推翻重來」。

用法：
  python3 -m hsot.slide_sweep --sizes test_sizes.json \\
      --base-csv sub_v006_e15sam3.csv --envelope-extra exp003_samurai_large.csv \\
      [--segments 2 4 8 16] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SMALL_T = 32.0        # 與 crop_rerun.py 同值
AREA_FRAC_MAX = 0.40  # 同上；E19 用它剔除掉的那批就是本前測的標的
MARGIN_SCALE = 2.0    # E04 實測 R=2–3× 局部搜尋命中 86% 的域內證據
MIN_MARGIN = 32.0


def parse(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    p = df["ID"].str.rsplit("_", n=1, expand=True)
    df["seq"], df["frame"] = p[0], p[1].astype(int)
    return df


def _envelope(boxes: np.ndarray, margin: float) -> tuple[float, float, float, float]:
    """一組框的外包矩形 + margin（未夾邊界；夾邊界在窗位置決定時才做）。"""
    return (float(np.min(boxes[:, 0])) - margin,
            float(np.min(boxes[:, 1])) - margin,
            float(np.max(boxes[:, 0] + boxes[:, 2])) + margin,
            float(np.max(boxes[:, 1] + boxes[:, 3])) + margin)


def fixed_window(boxes_per_seg: list[np.ndarray], margin: float,
                 W: float, H: float) -> tuple[int, int, list[tuple[int, int]], int]:
    """回傳 (cw, ch, 各段左上角, 出窗幀數)。

    cw/ch = 所有段 envelope 的最大寬/高（向上取整），故每段窗必涵蓋該段全部框。
    段窗位置 = 段 envelope 中心，夾在 [0, W-cw] × [0, H-ch] 內。
    """
    envs = [_envelope(b, margin) for b in boxes_per_seg]
    cw = int(np.ceil(max(e[2] - e[0] for e in envs)))
    ch = int(np.ceil(max(e[3] - e[1] for e in envs)))
    cw, ch = min(cw, int(W)), min(ch, int(H))

    origins, outside = [], 0
    for b, (x1, y1, x2, y2) in zip(boxes_per_seg, envs):
        ox = int(round(np.clip((x1 + x2) / 2 - cw / 2, 0, W - cw)))
        oy = int(round(np.clip((y1 + y2) / 2 - ch / 2, 0, H - ch)))
        origins.append((ox, oy))
        # 出窗檢查：框必須完整落在窗內（cw 被原圖尺寸夾住時才可能失敗）
        outside += int(np.sum((b[:, 0] < ox) | (b[:, 1] < oy) |
                              (b[:, 0] + b[:, 2] > ox + cw) |
                              (b[:, 1] + b[:, 3] > oy + ch)))
    return cw, ch, origins, outside


def analyse(sizes: dict, base: pd.DataFrame, extra: pd.DataFrame | None,
            segments: list[int]) -> list[dict]:
    rows = []
    for seq, g in base.groupby("seq"):
        if seq not in sizes:
            continue
        g = g.sort_values("frame")
        first = g.iloc[0]
        scale = float(np.sqrt(first["width"] * first["height"]))
        if scale >= SMALL_T:
            continue  # 非小目標，不在 crop-zoom 的適用範圍
        W, H = float(sizes[seq]["W"]), float(sizes[seq]["H"])
        margin = max(MARGIN_SCALE * scale, MIN_MARGIN)
        n = len(g)
        gb = g[["x", "y", "width", "height"]].to_numpy(float)
        frames = g["frame"].to_numpy()

        # 聯集另一條軌跡（單一軌跡跟丟時會凍結在錯位置，其 envelope 特別小且
        # 會通過面積檢查，把真目標切在窗外——E19 讀碼時發現並修掉的結構性風險）
        eb = None
        if extra is not None:
            e = extra[extra["seq"] == seq].sort_values("frame")
            if len(e):
                eb = e.set_index("frame")[["x", "y", "width", "height"]]

        def seg_boxes(lo, hi):
            b = gb[lo:hi]
            if eb is not None:
                idx = [f for f in frames[lo:hi] if f in eb.index]
                if idx:
                    b = np.vstack([b, eb.loc[idx].to_numpy(float)])
            return b

        # k=1（E19 現行做法）當基準線：算它的窗面積佔比，用來判定該序列是否被剔除
        x1, y1, x2, y2 = _envelope(seg_boxes(0, n), margin)
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        frac1 = (x2 - x1) * (y2 - y1) / (W * H)

        row = {"seq": seq, "scale": round(scale, 1), "n_frames": n,
               "orig": [int(W), int(H)], "modality": seq.split("-")[0],
               "frac_k1": round(frac1, 4),
               "e19_selected": bool(frac1 < AREA_FRAC_MAX)}

        for k in segments:
            bounds = np.linspace(0, n, k + 1).astype(int)
            segs = [seg_boxes(int(bounds[i]), int(bounds[i + 1]))
                    for i in range(k) if bounds[i + 1] > bounds[i]]
            cw, ch, origins, outside = fixed_window(segs, margin, W, H)
            frac = cw * ch / (W * H)
            # 窗平移量：相鄰段左上角的位移（背景流動的代理指標）
            shifts = [float(np.hypot(origins[i + 1][0] - origins[i][0],
                                     origins[i + 1][1] - origins[i][1]))
                      for i in range(len(origins) - 1)]
            row[f"frac_k{k}"] = round(frac, 4)
            row[f"lin_k{k}"] = round(float(np.sqrt(1 / frac)), 2)
            row[f"win_k{k}"] = [cw, ch]
            row[f"out_k{k}"] = int(outside)
            row[f"shift_k{k}"] = round(float(np.median(shifts)) if shifts else 0.0, 1)
        rows.append(row)
    return rows


def report(rows: list[dict], segments: list[int], total_frames: int) -> None:
    sel = [r for r in rows if r["e19_selected"]]
    rej = [r for r in rows if not r["e19_selected"]]
    print(f"\n{'='*90}")
    print(f"小目標(<{SMALL_T:.0f}px) 共 {len(rows)} 支｜E19 已選中 {len(sel)} 支（**不得重做**）"
          f"｜被面積門檻剔除 {len(rej)} 支 ← 本前測的標的")
    print(f"標的幀數 {sum(r['n_frames'] for r in rej):,} / {total_frames:,} "
          f"= {sum(r['n_frames'] for r in rej)/total_frames:.1%}（＝替換比例，D038 下檔的乘數）")
    print("=" * 90)

    print(f"\n### 標的 {len(rej)} 支在各分段數下的固定窗（k=1 欄為 E19 剔除它們的原因）")
    hdr = f"{'序列':<24}{'尺度':>5}{'幀數':>6}{'k=1':>7}"
    for k in segments:
        hdr += f"{'k='+str(k):>17}"
    print(hdr)
    print(f"{'':<24}{'':>5}{'':>6}{'面積比':>7}" +
          "".join(f"{'面積比/線性/平移':>17}" for _ in segments))
    for r in sorted(rej, key=lambda r: r["frac_k1"]):
        line = f"{r['seq']:<24}{r['scale']:>5.1f}{r['n_frames']:>6}{r['frac_k1']:>7.2f}"
        for k in segments:
            mark = "*" if r[f"frac_k{k}"] < AREA_FRAC_MAX else " "
            line += f"{r[f'frac_k{k}']:>6.2f}/{r[f'lin_k{k}']:>4.1f}x/{r[f'shift_k{k}']:>4.0f}px{mark}"
        print(line)
    print("（* ＝該分段數下窗面積已降到 40% 門檻以下；平移 = 相鄰段窗左上角位移中位數，"
          "\n  是背景流動的代理——幾何前測唯一測不出的風險就是它對 memory attention 的影響）")

    print(f"\n### 彙總：各分段數能救回幾支（標的 {len(rej)} 支）")
    print(f"{'k':>4}{'通過門檻':>10}{'覆蓋幀':>10}{'線性放大 中位':>15}{'窗平移 中位':>13}{'出窗幀':>9}")
    for k in segments:
        ok = [r for r in rej if r[f"frac_k{k}"] < AREA_FRAC_MAX]
        if not ok:
            print(f"{k:>4}{0:>10}")
            continue
        fr = sum(r["n_frames"] for r in ok)
        print(f"{k:>4}{len(ok):>10}{fr/total_frames:>9.1%}"
              f"{np.median([r[f'lin_k{k}'] for r in ok]):>14.2f}x"
              f"{np.median([r[f'shift_k{k}'] for r in ok]):>12.0f}px"
              f"{sum(r[f'out_k{k}'] for r in ok):>9}")
    print("\n對照 E19 已驗證的 21 支：線性放大中位 2.05x → LB +0.0175。")
    print("出窗幀應為 0（固定窗尺寸取各段最大值即保證涵蓋）；非 0 代表窗被原圖尺寸夾住。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", required=True, help="{seq: {W,H,n}} 尺寸表 JSON")
    ap.add_argument("--base-csv", required=True)
    ap.add_argument("--envelope-extra")
    ap.add_argument("--segments", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--json")
    args = ap.parse_args()

    sizes = json.loads(Path(args.sizes).read_text())
    base = parse(args.base_csv)
    extra = parse(args.envelope_extra) if args.envelope_extra else None
    rows = analyse(sizes, base, extra, args.segments)
    report(rows, args.segments, len(base))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"\n逐序列明細 → {args.json}")


if __name__ == "__main__":
    main()
