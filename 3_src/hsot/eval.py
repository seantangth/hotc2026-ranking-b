"""HSOT 本地評測：忠實移植官方 compile_results（HyperTools.py）。

官方指標：Success AUC = 50 個 IoU 門檻（0.02..1.00）的 success rate 平均。
本模組同時輸出 pooled（全幀混算，推測與 Kaggle LB 一致）與 per-sequence 平均，
兩者與 LB 的對應關係由錨點提交校準（見 HSOT_EXPERIMENT_LOG.md 校準表）。

用法：
    python -m hsot.eval pred.csv gt.csv [--seqs val_split_v1.txt]
CSV 格式皆為 submission 格式：ID,x,y,width,height（ID = 序列名_幀號）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IOU_THRESHOLDS = np.arange(0.02, 1.02, 0.02)  # 官方：50 個
DIST_THRESHOLDS = np.linspace(1, 50, 50)      # 官方：dp_20 = precision[19]


def load_boxes(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = ["ID", "x", "y", "w", "h"]
    parts = df["ID"].str.rsplit("_", n=1, expand=True)
    df["seq"] = parts[0]
    df["frame"] = parts[1].astype(int)
    return df


def overlap_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """官方 overlap_ratio 逐列 IoU；a, b: (N,4) [x,y,w,h]。"""
    left = np.maximum(a[:, 0], b[:, 0])
    right = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    top = np.maximum(a[:, 1], b[:, 1])
    bottom = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - intersect
    iou = intersect / union
    return np.clip(iou, 0, 1)


def success_curve(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """官方 success_overlap：GT 含非正值的幀 IoU 記 -1（全門檻失敗）但留在分母。"""
    n = len(gt)
    iou = np.full(n, -1.0)
    mask = np.sum(gt > 0, axis=1) == 4
    if mask.any():
        iou[mask] = overlap_ratio(gt[mask], pred[mask])
    return np.array([(iou >= t).sum() / n for t in IOU_THRESHOLDS])


def center_distances(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    ga = gt[:, :2] + gt[:, 2:] / 2
    pa = pred[:, :2] + pred[:, 2:] / 2
    return np.linalg.norm(ga - pa, axis=1)


def evaluate(pred_csv: str | Path, gt_csv: str | Path, seqs: list[str] | None = None) -> dict:
    pred = load_boxes(pred_csv)
    gt = load_boxes(gt_csv)
    if seqs is not None:
        pred = pred[pred["seq"].isin(seqs)]
        gt = gt[gt["seq"].isin(seqs)]
    merged = gt.merge(pred, on="ID", suffixes=("_gt", "_pr"), how="left")
    n_missing = merged["x_pr"].isna().sum()
    if n_missing:
        print(f"警告：{n_missing} 幀缺預測（以 0 框計，IoU=0）", file=sys.stderr)
        merged[["x_pr", "y_pr", "w_pr", "h_pr"]] = merged[["x_pr", "y_pr", "w_pr", "h_pr"]].fillna(0)

    g = merged[["x_gt", "y_gt", "w_gt", "h_gt"]].to_numpy(float)
    p = merged[["x_pr", "y_pr", "w_pr", "h_pr"]].to_numpy(float)

    pooled_success = success_curve(g, p)
    dists = center_distances(g, p)
    pooled = {
        "auc": float(pooled_success.mean()),
        "dp20": float((dists < 20).mean()),
        "cle": float(dists.mean()),
        "n_frames": len(merged),
    }

    per_seq = {}
    for seq, grp in merged.groupby("seq_gt"):
        gg = grp[["x_gt", "y_gt", "w_gt", "h_gt"]].to_numpy(float)
        pp = grp[["x_pr", "y_pr", "w_pr", "h_pr"]].to_numpy(float)
        d = center_distances(gg, pp)
        per_seq[seq] = {
            "auc": float(success_curve(gg, pp).mean()),
            "dp20": float((d < 20).mean()),
            "n": len(grp),
        }
    seq_mean_auc = float(np.mean([v["auc"] for v in per_seq.values()])) if per_seq else float("nan")
    return {"pooled": pooled, "seq_mean_auc": seq_mean_auc, "per_seq": per_seq}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_csv")
    ap.add_argument("gt_csv")
    ap.add_argument("--seqs", help="限定序列清單檔（每行一個序列名）")
    ap.add_argument("--per-seq", action="store_true", help="列出每序列 AUC")
    args = ap.parse_args()
    seqs = Path(args.seqs).read_text().split() if args.seqs else None
    res = evaluate(args.pred_csv, args.gt_csv, seqs)
    p = res["pooled"]
    print(f"pooled   AUC={p['auc']:.5f}  DP@20={p['dp20']:.5f}  CLE={p['cle']:.2f}  ({p['n_frames']:,} 幀)")
    print(f"seq-mean AUC={res['seq_mean_auc']:.5f}  ({len(res['per_seq'])} 序列)")
    if args.per_seq:
        for seq, v in sorted(res["per_seq"].items(), key=lambda kv: kv[1]["auc"]):
            print(f"  {seq:35s} AUC={v['auc']:.4f} DP20={v['dp20']:.4f} n={v['n']}")


if __name__ == "__main__":
    main()
