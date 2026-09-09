#!/usr/bin/env python3
"""E13 條件式混合:T1 base + HiM2SAM,以無 GT frozen 指紋規則替換(Gate A pipeline)。

規則(Ranking B 合法:確定性、不認序列名、無 GT 可算):
  frozen_ratio(base 輸出中「與前一幀 box 完全相同」的幀比例)>= 閾值(預設 0.10)
  的序列,整段以 HiM2SAM 輸出替換;其餘保留 base。
閾值來源:canary 25 序列 oracle 掃描,0.08–0.15 皆正(平緩高原),取 0.10(D033:不細調)。

用法:
  # val 全量驗證(有 GT):
  python3.14 -m hsot.blend_e13 --base 5_outputs/t1_rerun_20260805/submission.csv \
      --him2sam <e13_full submission.csv> --out 5_outputs/blend_val.csv \
      --gt 1_data/raw/2026training.csv --seqs 1_data/val_split_v1.txt
  # test 提交生成(無 GT):
  python3.14 -m hsot.blend_e13 --base 5_outputs/submissions/exp003_samurai_large.csv \
      --him2sam <e13_test submission.csv> --out 5_outputs/submissions/sub_v003_e13blend.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FROZEN_T = 0.10


def parse(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    p = df["ID"].str.rsplit("_", n=1, expand=True)
    df["seq"], df["frame"] = p[0], p[1].astype(int)
    return df


def frozen_ratio(g: pd.DataFrame) -> float:
    b = g.sort_values("frame")[["x", "y", "width", "height"]].to_numpy(float)
    if len(b) < 2:
        return 0.0
    return float((np.abs(np.diff(b, axis=0)).sum(axis=1) == 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--him2sam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=FROZEN_T)
    ap.add_argument("--gt", help="有 GT 時輸出三方對照(base/him2sam/blend)")
    ap.add_argument("--seqs", help="序列清單過濾")
    args = ap.parse_args()

    base = parse(args.base)
    h2s = parse(args.him2sam)
    seqs = sorted(base["seq"].unique())
    if args.seqs:
        want = {s.strip() for s in Path(args.seqs).read_text().split() if s.strip()}
        seqs = [s for s in seqs if s in want]
        base = base[base["seq"].isin(seqs)]

    ratios = {s: frozen_ratio(g) for s, g in base.groupby("seq")}
    h2s_seqs = set(h2s["seq"].unique())
    picked, missing = [], []
    for s in seqs:
        if ratios[s] >= args.threshold:
            (picked if s in h2s_seqs else missing).append(s)
    if missing:
        print(f"⚠️ {len(missing)} 支選中但無 HiM2SAM 輸出(保留 base):{missing[:5]}")

    parts = [h2s[h2s["seq"] == s] if s in picked else base[base["seq"] == s] for s in seqs]
    blend = pd.concat(parts, ignore_index=True)
    # 對齊 base 的列順序與幀覆蓋(混合來源幀號應一致;保險 merge 檢查)
    assert len(blend) == len(base), f"列數不符:blend {len(blend)} vs base {len(base)}"
    blend[["ID", "x", "y", "width", "height"]].to_csv(args.out, index=False)
    print(f"混合完成:{len(picked)}/{len(seqs)} 支採 HiM2SAM(frozen≥{args.threshold})→ {args.out}")
    print(f"  選中:{sorted(picked)}")

    if args.gt:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from hsot import eval as ev
        r_base = ev.evaluate(args.base, args.gt, seqs)
        r_h2s = ev.evaluate(args.him2sam, args.gt, [s for s in seqs if s in h2s_seqs])
        r_blend = ev.evaluate(args.out, args.gt, seqs)
        print(f"\n=== 三方對照(pooled AUC)===")
        print(f"base(T1)   : {r_base['pooled']['auc']:.5f}")
        print(f"him2sam 全用: {r_h2s['pooled']['auc']:.5f}(僅覆蓋序列)")
        print(f"blend 混合  : {r_blend['pooled']['auc']:.5f}(Δ vs base {r_blend['pooled']['auc'] - r_base['pooled']['auc']:+.5f})")
        print("\n選中序列的 delta(blend 採用 HiM2SAM 者):")
        for s in sorted(picked):
            d = r_blend["per_seq"][s]["auc"] - r_base["per_seq"][s]["auc"]
            print(f"  {s:<26} {d:+.5f}(frozen {ratios[s]:.1%})")


if __name__ == "__main__":
    main()
