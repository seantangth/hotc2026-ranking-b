#!/usr/bin/env python3
"""固定大小滑動窗 crop-zoom（D043 由 v009/E21 失敗反推出的正解）。

【與 crop_rerun.py --segments 的決定性差異】
  E21（crop_rerun --segments 4）把序列切成 K 段，**每段各自是一支獨立子序列**：
    · 各段窗尺寸不同 → 無法拼接 → 每段必須 `init_state` 從零起（memory bank 空）
    · 段首只能用 base 軌跡的**預測框**重新 init ＝ 拿 LB 0.674 的軌跡覆寫 LB 0.691 的軌跡
  實測 LB −0.0139，且窗幾何前測全部算對——**敗因不是窗，是時間連續性被破壞**。

  本模組：窗**尺寸固定**（= 各段 envelope 的最大寬高）、**位置逐段平移**。
    · 所有幀裁切後尺寸相同 → 直接輸出成**單一連續序列**
    · ⇒ track_t1.py 視之為一支普通影片：**只用首幀 GT init、memory 全程連續**
    · E21 的兩個敗因同時消失，且仍拿到「把 patch 預算集中到目標周圍」的收益

【殘餘風險（幾何前測測不出來，只能上 LB）】窗每段平移，裁切影像裡的**背景會流動**。
  這對 SAM 的 memory attention 是否有害無法先驗判定——但它正是自然影片裡的「相機跟拍」，
  底座在該分佈上訓練過大量資料。`slide_sweep.py` 輸出的「窗平移中位數」是其代理量。

【範圍鐵律（D038 不對稱比 + D043 教訓）】只處理 **E19 未選中的序列**，
  已驗證 +0.0175 的 21 支一律沿用 v008——v009/E21 正是死在把已驗證的部分推翻重做。
  由 `--exclude-meta`（E19 的 crop_meta.json）強制執行，不靠人記得。

用法：
  1) prep：產生單一連續裁切序列 + 窗表
     python3 -m hsot.slide_rerun prep --frames-root <原圖根> --base-csv <e15.csv> \
         --envelope-extra <e02.csv> --exclude-meta <e19_crop_meta.json> \
         --segments 16 --out-root <slide_fc> --meta <slide_meta.json>
  2)（外部）track_t1.py --frames-root <slide_fc> --seq-list <slide_seqs.txt>
  3) merge：逐幀加回該段窗偏移，未選中序列保留 base（＝v008）
     python3 -m hsot.slide_rerun merge --base-csv <v008.csv> --crop-csv <輸出> \
         --meta <slide_meta.json> --out <final.csv>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SMALL_T = 32.0        # 與 crop_rerun.py 同值（選序列規則保持一致）
AREA_FRAC_MAX = 0.40  # 同上
MARGIN_SCALE = 2.0
MIN_MARGIN = 32.0


def parse(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    p = df["ID"].str.rsplit("_", n=1, expand=True)
    df["seq"], df["frame"] = p[0], p[1].astype(int)
    return df


def _envelope(boxes: np.ndarray, margin: float) -> tuple[float, float, float, float]:
    return (float(np.min(boxes[:, 0])) - margin,
            float(np.min(boxes[:, 1])) - margin,
            float(np.max(boxes[:, 0] + boxes[:, 2])) + margin,
            float(np.max(boxes[:, 1] + boxes[:, 3])) + margin)


def plan_windows(boxes_per_seg: list[np.ndarray], margin: float, W: float, H: float):
    """固定尺寸 + 逐段平移的窗規劃。

    cw/ch 取各段 envelope 的最大寬高 → 每段窗必涵蓋該段全部軌跡框（除非被原圖夾住）。
    段窗位置 = 該段 envelope 中心，夾在 [0, W-cw] × [0, H-ch]。
    """
    envs = [_envelope(b, margin) for b in boxes_per_seg]
    cw = min(int(np.ceil(max(e[2] - e[0] for e in envs))), int(W))
    ch = min(int(np.ceil(max(e[3] - e[1] for e in envs))), int(H))
    origins, outside = [], 0
    for b, (x1, y1, x2, y2) in zip(boxes_per_seg, envs):
        ox = int(round(np.clip((x1 + x2) / 2 - cw / 2, 0, W - cw)))
        oy = int(round(np.clip((y1 + y2) / 2 - ch / 2, 0, H - ch)))
        origins.append((ox, oy))
        outside += int(np.sum((b[:, 0] < ox) | (b[:, 1] < oy) |
                              (b[:, 0] + b[:, 2] > ox + cw) |
                              (b[:, 1] + b[:, 3] > oy + ch)))
    return cw, ch, origins, outside


def cmd_prep(args) -> None:
    from PIL import Image

    base = parse(args.base_csv)
    extra = parse(args.envelope_extra) if args.envelope_extra else None
    # 範圍鐵律：E19 已選中的序列一律不碰（見檔頭）
    excluded = set()
    if args.exclude_meta:
        em = json.loads(Path(args.exclude_meta).read_text())
        excluded = {v.get("seq", k) for k, v in em.items()}
        print(f"排除 E19 已選中的 {len(excluded)} 支（已驗證 +0.0175，不得重做）")

    frames_root = Path(args.frames_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    K = max(2, int(args.segments))
    meta, n_small, n_excl, n_area = {}, 0, 0, 0

    for seq, g in base.groupby("seq"):
        g = g.sort_values("frame")
        first = g.iloc[0]
        scale = float(np.sqrt(first["width"] * first["height"]))
        if scale >= SMALL_T:
            continue
        n_small += 1
        if seq in excluded:
            n_excl += 1
            continue
        seq_dir = frames_root / seq
        jpgs = sorted(seq_dir.glob("*.jp*g"))
        if not jpgs:
            continue
        with Image.open(jpgs[0]) as im:
            W, H = im.size
        margin = max(MARGIN_SCALE * scale, MIN_MARGIN)
        n = len(g)
        gb = g[["x", "y", "width", "height"]].to_numpy(float)
        frames = g["frame"].to_numpy()

        eb = None
        if extra is not None:
            e = extra[extra["seq"] == seq].sort_values("frame")
            if len(e):
                eb = e.set_index("frame")[["x", "y", "width", "height"]]

        bounds = np.linspace(0, n, K + 1).astype(int)
        segs, spans = [], []
        for i in range(K):
            lo, hi = int(bounds[i]), int(bounds[i + 1])
            if hi <= lo:
                continue
            b = gb[lo:hi]
            if eb is not None:
                idx = [f for f in frames[lo:hi] if f in eb.index]
                if idx:
                    b = np.vstack([b, eb.loc[idx].to_numpy(float)])
            segs.append(b)
            spans.append((int(frames[lo]), int(frames[hi - 1])))

        cw, ch, origins, outside = plan_windows(segs, margin, W, H)
        if cw * ch >= AREA_FRAC_MAX * W * H:
            n_area += 1
            continue   # 放大倍率不足，保留 base
        assert outside == 0, f"{seq}: {outside} 幀軌跡框落在窗外（窗被原圖尺寸夾住）"

        # 輸出成**單一連續序列**：所有幀同尺寸 (cw, ch)，僅裁切位置逐段平移。
        # 這是本方法的全部要點——tracker 因此只需首幀 init、memory 不中斷。
        dest = out_root / seq
        dest.mkdir(exist_ok=True)
        seg_recs = []
        for (ox, oy), (f_lo, f_hi) in zip(origins, spans):
            for fr in range(f_lo, f_hi + 1):
                f = jpgs[fr - 1]   # base 的 frame 為 1-based 連續
                with Image.open(f) as im:
                    im.crop((ox, oy, ox + cw, oy + ch)).save(dest / f.name, quality=95)
            seg_recs.append({"lo": f_lo, "hi": f_hi, "ox": ox, "oy": oy})

        # 首幀 init：用**原圖的 GT init_rect**（若有）而非 base 預測框——E21 敗因(a)。
        gt_file = seq_dir / "init_rect.txt"
        if gt_file.exists():
            gt = [float(v) for v in gt_file.read_text().replace(",", " ").split()]
        else:
            f0 = g.iloc[0]
            gt = [float(f0["x"]), float(f0["y"]), float(f0["width"]), float(f0["height"])]
        ox0, oy0 = origins[0]
        (dest / "init_rect.txt").write_text(
            " ".join(str(v) for v in [gt[0] - ox0, gt[1] - oy0, gt[2], gt[3]]))

        meta[seq] = {"orig": [W, H], "win": [cw, ch], "segments": seg_recs,
                     "zoom": round(W * H / (cw * ch), 2),
                     "shift_median": round(float(np.median(
                         [np.hypot(origins[i + 1][0] - origins[i][0],
                                   origins[i + 1][1] - origins[i][1])
                          for i in range(len(origins) - 1)])) if len(origins) > 1 else 0.0, 1)}
        print(f"{seq}: 窗 {cw}×{ch}（原圖 {W}×{H}）放大 {meta[seq]['zoom']:.1f}x"
              f"｜{len(seg_recs)} 段、平移中位 {meta[seq]['shift_median']:.0f}px")

    Path(args.meta).write_text(json.dumps(meta, indent=1))
    print(f"\n小目標 {n_small} 支｜E19 已選中而排除 {n_excl} 支｜面積門檻剔除 {n_area} 支"
          f"｜**選中 {len(meta)} 支** → {args.out_root}")
    if meta:
        z = [m["zoom"] for m in meta.values()]
        print(f"放大倍率(面積) 中位 {np.median(z):.1f}x（線性 {np.sqrt(np.median(z)):.2f}x）"
              f"｜對照 E19 已驗證 4.2x（線性 2.05x）→ LB +0.0175")
    print(f"窗表 → {args.meta}")


def cmd_merge(args) -> None:
    """逐幀加回該幀所屬段的窗偏移；未選中序列一律沿用 base（＝v008）。"""
    base = parse(args.base_csv)
    crop = parse(args.crop_csv)
    meta = json.loads(Path(args.meta).read_text())
    done = set(crop["seq"])

    out = base.set_index("ID")[["x", "y", "width", "height"]].copy()
    n_seq = n_frame = 0
    for seq, w in meta.items():
        if seq not in done:
            continue
        c = crop[crop["seq"] == seq].sort_values("frame")
        # 裁切序列的輸出幀號是 1-based（track_t1 無 --gt-csv 時的行為），
        # 而本模組輸出的是**單一連續序列**，故位置 i ↔ 原序列第 (f_lo0 + i) 幀。
        f_lo0 = w["segments"][0]["lo"]
        cf = c["frame"].to_numpy()
        orig_frames = f_lo0 + cf - 1
        ox = np.zeros(len(c)); oy = np.zeros(len(c))
        covered = np.zeros(len(c), dtype=bool)
        for s in w["segments"]:
            m = (orig_frames >= s["lo"]) & (orig_frames <= s["hi"])
            ox[m], oy[m] = s["ox"], s["oy"]
            covered |= m
        assert covered.all(), f"{seq}: {int((~covered).sum())} 幀不屬於任何段"
        ids = [f"{seq}_{int(f)}" for f in orig_frames]
        miss = [i for i in ids if i not in out.index]
        assert not miss, f"{seq}: {len(miss)} 個幀號不存在於 base（首個 {miss[0]}）"
        out.loc[ids, "x"] = c["x"].to_numpy() + ox
        out.loc[ids, "y"] = c["y"].to_numpy() + oy
        out.loc[ids, "width"] = c["width"].to_numpy()
        out.loc[ids, "height"] = c["height"].to_numpy()
        n_seq += 1; n_frame += len(ids)

    out = out.reset_index()[["ID", "x", "y", "width", "height"]]
    assert len(out) == len(base), f"列數不符 {len(out)} vs {len(base)}"
    assert (out["ID"].values == base["ID"].values).all(), "ID 順序被改變"
    out.to_csv(args.out, index=False)
    print(f"合併完成：{n_seq} 支 / {n_frame} 幀 ({n_frame/len(base):.1%}) 採滑動窗重跑 "
          f"→ {args.out}")
    print(f"⚠️ D038 下檔＝{n_frame/len(base):.1%} × 選中序列最壞崩幅；上檔見前測外推")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prep")
    p1.add_argument("--frames-root", required=True)
    p1.add_argument("--base-csv", required=True, help="窗源主軌跡（如 E15）")
    p1.add_argument("--envelope-extra", help="窗源副軌跡，取聯集（如 E02）")
    p1.add_argument("--exclude-meta", help="E19 的 crop_meta.json —— 其序列一律不碰")
    p1.add_argument("--segments", type=int, default=16,
                    help="切 K 段決定窗尺寸與平移節奏（窗尺寸取各段最大，故仍是固定尺寸）")
    p1.add_argument("--out-root", required=True)
    p1.add_argument("--meta", required=True)
    p2 = sub.add_parser("merge")
    p2.add_argument("--base-csv", required=True, help="合併基底（應為 v008）")
    p2.add_argument("--crop-csv", required=True)
    p2.add_argument("--meta", required=True)
    p2.add_argument("--out", required=True)
    args = ap.parse_args()
    {"prep": cmd_prep, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
