#!/usr/bin/env python3
"""make_pretend_new_data_v1 — 把本地 val 造成「假想新資料」封包，供冷啟動演練 #2（S-16）使用。

【為什麼需要這支】演練 #1 是**重現測試**不是泛化測試：四道閘門全部比對 `~/ref/` 的 Ranking A 參考
CSV、完整性檢查硬編 `len(d)==26860`、資料入口硬編 `t1test_fc_75.tar`、序列數 ≠75 即 `exit 1`
⇒ 直接套 9/7 的全新資料必掛（D052 桶1 第 f 項）。§6 第 6 條的原始規格是「拿本地 val 當**假想新資料**」。

【本支要模擬的關鍵事實（已實測核對，不是猜的）】
  官方 test `sample_submisson.csv` 的 ID ＝ **每序列 1-based 連續**（`nir-bee2_1`…`nir-bee2_N`）；
  而 `2026training.csv` 的 ID ＝ **跨序列流水號且各模態範圍重疊**（nir 1..22874／vis 30679..124909）。
  兩者是**不同的編號方案** ⇒ 演練若沿用 training 的 GT 當輸入，走的是 `track_t1.py:70-78` 的
  「有 GT → 全域幀號」分支，**根本沒測到 9/7 會走的「無 GT → 1-based」分支**。
  ⇒ 本支產生的封包一律用 **1-based**，且演練時**不得傳 `--gt-csv`**。

【輸出（out-dir）＝ 9/7 拿到的東西的等價物】
  sample_submission.csv        全部 <seq>_<1..N> 列，x/y/w/h 皆 0（模擬官方佔位）
  init_rect/<seq>.txt          首幀框（模擬官方 init_rect），供 read_init 使用
  frames/<seq>/init_rect.txt   同上，直接放進 frames-root（track_t1.py:49 的優先來源）
  _withheld_gt.csv             **僅供事後評分**，演練期間不得進入生成路徑
  MANIFEST.json                逐序列幀數與 init 來源，供閘門核對

【幀數以實際 jpg 為準、不以 GT 列數為準】——4 支序列的 GT 是雙區塊（n_gt != n_jpg，見
  5_outputs/peft_phase0_20260810/AUDIT.md §2），用 GT 列數會造出對不上的 sample。

用法（在已解開假色 tar 的機器上）：
  python3 make_pretend_new_data_v1.py --frames-root ~/val_fc --gt-csv 2026training.csv \
      --out-dir ~/drill2_pkg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True, help="每序列一個子目錄（jpg 幀）")
    ap.add_argument("--gt-csv", required=True, help="2026training.csv —— 只用來取首幀框與扣留評分用 GT")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seq-list", help="限定序列（預設 frames-root 下全部）")
    args = ap.parse_args()

    frames_root = Path(args.frames_root).resolve()
    out = Path(args.out_dir).resolve()
    (out / "init_rect").mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(Path(args.gt_csv).resolve())
    gt.columns = ["ID", "x", "y", "w", "h"]
    parts = gt["ID"].str.rsplit("_", n=1, expand=True)
    gt["seq"], gt["frame"] = parts[0], parts[1].astype(int)

    seqs = sorted(d.name for d in frames_root.iterdir() if d.is_dir() and any(d.glob("*.jp*g")))
    if args.seq_list:
        want = {s.strip() for s in Path(args.seq_list).read_text().split() if s.strip()}
        seqs = [s for s in seqs if s in want]
    if not seqs:
        sys.exit("frames-root 下沒有任何序列")

    sample_rows, gt_rows, manifest, warn = [], [], {}, []
    for seq in seqs:
        jpgs = sorted((frames_root / seq).glob("*.jp*g"))
        rows = gt[gt["seq"] == seq].sort_values("frame").reset_index(drop=True)
        if rows.empty:
            sys.exit(f"❌ {seq}: GT 無此序列，無法取 init 框")
        if len(rows) != len(jpgs):        # GT 雙區塊型（AUDIT.md §2）：位置對位不可信
            warn.append(f"{seq}: n_jpg={len(jpgs)} != n_gt={len(rows)}")
        n = len(jpgs)                     # ← 幀數以實際影格為準

        r0 = rows.iloc[0]
        init = [float(r0["x"]), float(r0["y"]), float(r0["w"]), float(r0["h"])]
        (out / "init_rect" / f"{seq}.txt").write_text(",".join(str(v) for v in init) + "\n")
        (frames_root / seq / "init_rect.txt").write_text(",".join(str(v) for v in init) + "\n")

        for pos in range(n):
            sample_rows.append((f"{seq}_{pos+1}", 0, 0, 0, 0))          # 1-based，模擬官方佔位
            if pos < len(rows):
                r = rows.iloc[pos]
                gt_rows.append((f"{seq}_{pos+1}", r["x"], r["y"], r["w"], r["h"]))
        manifest[seq] = {"n_frames": n, "n_gt_rows": len(rows), "init": init}

    cols = ["ID", "x", "y", "width", "height"]
    pd.DataFrame(sample_rows, columns=cols).to_csv(out / "sample_submission.csv", index=False)
    pd.DataFrame(gt_rows, columns=cols).to_csv(out / "_withheld_gt.csv", index=False)
    (out / "MANIFEST.json").write_text(json.dumps(
        {"n_seqs": len(seqs), "n_frames": len(sample_rows), "warnings": warn,
         "id_scheme": "per-sequence 1-based（與官方 test sample 一致）",
         "sequences": manifest}, indent=1))

    print(f"✅ 封包完成：{len(seqs)} 序列 / {len(sample_rows)} 幀 → {out}")
    print(f"   sample_submission.csv（佔位 0）、init_rect/、_withheld_gt.csv（僅事後評分）")
    if warn:
        print(f"⚠️ {len(warn)} 支序列 n_jpg != n_gt（GT 雙區塊型，評分時該序列位置對位不可信）：")
        for w in warn:
            print(f"     {w}")
    print("\n▶ 演練時務必：**不傳 --gt-csv**（否則會走全域幀號分支，等於沒測到 9/7 的情境）；")
    print("  傳 --sample-csv <out>/sample_submission.csv 讓 formal validator 做 exact-set 比對。")


if __name__ == "__main__":
    main()
