#!/usr/bin/env python3
"""G2 同構性 smoke 的對照與災難檢查。

輸入＝兩份 track_t1 的 submission.csv（trained vs frozen，同一組 6 支 405 序列）
＋ 2026training.csv GT ＋ smoke plan json（{"failing": [...], "healthy": [...]}）。
join 鍵＝ID（track_t1 帶 --gt-csv 時輸出全域幀號，格式與 GT 完全相同，
rows_from_boxes：track_t1.py:443-458）。

IoU 語意：per-seq mean box-IoU、GT 無效幀（xywh 任一 ≤0）排除、**含首幀**
（兩側都是 init 照抄、IoU=1，對稱不偏，且與 frame-pooled Success AUC 慣例一致）。

災難判準（D062 canary 教義的健康對照；事前寫死 2026-08-29）：
任一 healthy 序列 trained − frozen < CATASTROPHE_DELTA ⇒ verdict=CATASTROPHE、
rc=5（**災難也是有效判決**——照樣寫 json、由 launch 收進 g2_summary，不炸腳本）。
failing 序列的改善與否照實記錄、不設限。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 3_src/
from train_tracker_g1.clip_dataset import frame_valid, load_gt_csv  # noqa: E402
from train_tracker_g1.losses import box_iou_xywh  # noqa: E402

CATASTROPHE_DELTA = -0.05  # 任一健康對照序列的容忍下限（事前判準，勿改）


def load_submission_csv(path) -> dict:
    """submission.csv → {ID: (x, y, w, h)}。欄位同 GT：ID,x,y,width,height。"""
    boxes = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header[0].strip().lower() == "id", f"非預期標頭 {header}"
        for row in reader:
            if row and row[0]:
                boxes[row[0]] = tuple(float(v) for v in row[1:5])
    return boxes


def seq_of(frame_id: str) -> str:
    """ID = f"{seq}_{global_frame}"；序列名可含底線（vis-L_person）⇒ rsplit。"""
    return frame_id.rsplit("_", 1)[0]


def per_seq_mean_iou(pred: dict, gt: dict, sequences) -> dict:
    """對每序列算 pred vs GT 的 mean IoU（GT 無效幀排除；pred 缺幀＝計 0，
    缺幀是 tracker 掉幀的實質失敗，不可靜默跳過）。"""
    per = {}
    for seq in sequences:
        ious = []
        for fid, gt_box in gt.items():
            if seq_of(fid) != seq or not frame_valid(gt_box):
                continue
            p = pred.get(fid)
            ious.append(box_iou_xywh(p, gt_box) if p is not None else 0.0)
        assert ious, f"{seq}: GT 內無有效幀（序列名寫錯？）"
        per[seq] = statistics.fmean(ious)
    return per


def compare(trained: dict, frozen: dict, gt: dict, plan: dict) -> dict:
    seqs = list(plan["failing"]) + list(plan["healthy"])
    t_iou = per_seq_mean_iou(trained, gt, seqs)
    f_iou = per_seq_mean_iou(frozen, gt, seqs)
    rows = [{"sequence": s,
             "group": "failing" if s in plan["failing"] else "healthy",
             "frozen": f_iou[s], "trained": t_iou[s],
             "delta": t_iou[s] - f_iou[s]} for s in seqs]
    catastrophes = [r for r in rows
                    if r["group"] == "healthy" and r["delta"] < CATASTROPHE_DELTA]
    verdict = "CATASTROPHE" if catastrophes else "OK"
    group_mean = {g: {
        "frozen": statistics.fmean(r["frozen"] for r in rows if r["group"] == g),
        "trained": statistics.fmean(r["trained"] for r in rows if r["group"] == g),
    } for g in ("failing", "healthy")}
    return {"verdict": verdict, "catastrophe_delta_limit": CATASTROPHE_DELTA,
            "catastrophe_sequences": [r["sequence"] for r in catastrophes],
            "per_sequence": rows, "group_mean": group_mean}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trained-csv", required=True)
    ap.add_argument("--frozen-csv", required=True)
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--plan-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    plan = json.loads(Path(args.plan_json).read_text())
    result = compare(load_submission_csv(args.trained_csv),
                     load_submission_csv(args.frozen_csv),
                     load_gt_csv(args.gt_csv), plan)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    for r in result["per_sequence"]:
        print(f"[smoke] {r['group']:7s} {r['sequence']:24s} "
              f"frozen={r['frozen']:.4f} trained={r['trained']:.4f} Δ={r['delta']:+.4f}")
    print(f"[smoke] verdict = {result['verdict']}")
    return 5 if result["verdict"] == "CATASTROPHE" else 0


if __name__ == "__main__":
    sys.exit(main())
