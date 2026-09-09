#!/usr/bin/env python3
"""g1_report_v1 — E-F Phase 1 的 G1 災難閘門 + G2 不對稱比 + diag19 機制讀數（DESIGN.md §4/§5）。

輸入皆為 track_t1 產出的 submission CSV（有 GT 全域 ID）。IoU 一律採官方遮罩語意
（GT 含 ≤0 座標的幀 IoU=-1 仍入分母，見 STRATEGY §2/D012）。

輸出 JSON：
  g1: base/peft pooled、逐序列 delta、worst、災難旗標（pooled 差 >0.05 或任一序列 <-0.15）
  g2: 下檔=0.3249×最壞崩幅、上檔=0.3249×中位改善、ratio、verdict
  diag19: 病灶機制讀數（逐序列 AUC delta 與跟丟率 delta；基線=E02 val65 既有輸出）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 3_src/（本機）
sys.path.insert(0, str(Path.home()))                             # ~（機上，hsot/ 在家目錄）
from hsot.eval import load_boxes, overlap_ratio, IOU_THRESHOLDS  # noqa: E402

NIR_TEST_FRAC = 0.3249  # DESIGN §5 G2：NIR 占 test 幀比例（8,727/26,860 實測）


def masked_iou(pred_csv: str, gt: pd.DataFrame, seqs: list[str]) -> pd.DataFrame:
    pred = load_boxes(pred_csv)
    pred = pred[pred["seq"].isin(set(seqs))].copy()
    sub = gt[gt["seq"].isin(set(seqs))].copy()
    m = sub.merge(pred, on="ID", suffixes=("_gt", "_pr"), how="left")
    missing = int(m["x_pr"].isna().sum())
    if missing:
        raise SystemExit(f"❌ {pred_csv} 缺 {missing} 幀預測（先跑完再評）")
    g = m[["x_gt", "y_gt", "w_gt", "h_gt"]].to_numpy(float)
    p = m[["x_pr", "y_pr", "w_pr", "h_pr"]].to_numpy(float)
    iou = np.full(len(g), -1.0)
    ok = np.sum(g > 0, axis=1) == 4
    iou[ok] = overlap_ratio(g[ok], p[ok])
    m["iou"] = iou
    return m


def pooled_auc(iou: np.ndarray) -> float:
    return float(np.mean([(iou >= t).mean() for t in IOU_THRESHOLDS]))


def per_seq(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seq, g in m.groupby("seq_gt"):
        rows.append({"seq": seq, "n": len(g), "auc": pooled_auc(g["iou"].to_numpy()),
                     "lost_rate": float((g["iou"] < 0.1).mean())})
    return pd.DataFrame(rows).set_index("seq")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--eval-seqs", required=True, help="G1 的 33 支清單")
    ap.add_argument("--base-csv", required=True, help="G1 baseline 腿（原始 ckpt）")
    ap.add_argument("--peft-csv", required=True, help="G1 PEFT 腿")
    ap.add_argument("--diag-seqs", help="diag19 清單（val_v1 NIR）")
    ap.add_argument("--diag-base-csv", help="diag 基線（E02 val65 既有輸出）")
    ap.add_argument("--diag-peft-csv", help="diag PEFT 腿")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()

    gt = load_boxes(a.gt_csv)
    seqs = [s.strip() for s in Path(a.eval_seqs).read_text().split() if s.strip()]

    mb = masked_iou(a.base_csv, gt, seqs)
    mp = masked_iou(a.peft_csv, gt, seqs)
    pb, pp = per_seq(mb), per_seq(mp)
    joined = pb.join(pp, lsuffix="_base", rsuffix="_peft")
    joined["delta"] = joined["auc_peft"] - joined["auc_base"]
    pool_b, pool_p = pooled_auc(mb["iou"].to_numpy()), pooled_auc(mp["iou"].to_numpy())
    worst = joined["delta"].min()
    disaster = (pool_p < pool_b - 0.05) or (worst < -0.15)

    med_gain = float(joined["delta"].clip(lower=0).median())
    up = NIR_TEST_FRAC * float(joined["delta"].median() if joined["delta"].median() > 0 else 0.0)
    down = NIR_TEST_FRAC * abs(min(worst, 0.0))
    ratio = (down / up) if up > 0 else float("inf")

    print("=" * 72)
    print(f"G1  base pooled={pool_b:.5f}  peft pooled={pool_p:.5f}  Δ={pool_p-pool_b:+.5f}")
    print(f"    改善 {(joined['delta']>0.005).sum()}／退步 {(joined['delta']<-0.005).sum()}"
          f"／持平 {(joined['delta'].abs()<=0.005).sum()}  worst={worst:+.4f}"
          f"（{joined['delta'].idxmin()}）")
    print(f"    災難旗標 = {'🚨 觸發（結案）' if disaster else '✅ 未觸發'}")
    print(joined.sort_values("delta")[["n_base", "auc_base", "auc_peft", "delta"]]
          .head(8).to_string())
    print(f"G2  下檔={down:+.5f}  上檔(中位)={up:+.5f}  比={ratio:.2f}:1 "
          f"{'⇒ >3:1 不發 LB' if ratio > 3 else '⇒ 可發'}")

    out = {"g1": {"pooled_base": pool_b, "pooled_peft": pool_p,
                  "delta_pooled": pool_p - pool_b, "worst_seq_delta": float(worst),
                  "disaster": bool(disaster),
                  "per_seq": {s: {"base": float(r["auc_base"]), "peft": float(r["auc_peft"]),
                                  "delta": float(r["delta"])} for s, r in joined.iterrows()}},
           "g2": {"down": down, "up_median": up, "ratio": ratio,
                  "median_gain_clip0": med_gain, "verdict": "NO-SEND" if ratio > 3 else "OK"}}

    if a.diag_seqs and a.diag_base_csv and a.diag_peft_csv:
        dseqs = [s.strip() for s in Path(a.diag_seqs).read_text().split() if s.strip()]
        db = per_seq(masked_iou(a.diag_base_csv, gt, dseqs))
        dp = per_seq(masked_iou(a.diag_peft_csv, gt, dseqs))
        dj = db.join(dp, lsuffix="_e02", rsuffix="_peft")
        dj["d_auc"] = dj["auc_peft"] - dj["auc_e02"]
        dj["d_lost"] = dj["lost_rate_peft"] - dj["lost_rate_e02"]
        print("diag19（病灶機制讀數；基線=E02 val65）——負 d_lost ＝ 跟丟改善")
        print(dj.sort_values("d_lost")[["auc_e02", "auc_peft", "d_auc",
                                        "lost_rate_e02", "lost_rate_peft", "d_lost"]].to_string())
        n_better = int((dj["d_lost"] < -0.02).sum())
        print(f"    跟丟率改善(>2pp) {n_better}/{len(dj)} 支；"
              f"AUC 改善 {(dj['d_auc']>0.005).sum()}/{len(dj)} 支")
        out["diag19"] = {"n_lost_improved": n_better,
                         "per_seq": {s: {"d_auc": float(r["d_auc"]), "d_lost": float(r["d_lost"])}
                                     for s, r in dj.iterrows()}}
    print("=" * 72)
    Path(a.out_json).write_text(json.dumps(out, indent=1))
    if disaster:
        sys.exit(3)  # 讓呼叫端能據此跳過 G3 test 腿


if __name__ == "__main__":
    main()
