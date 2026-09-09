#!/usr/bin/env python3
"""P1 精度線（D105，2026-09-02）：**均勻取樣**的 clip 索引。

與 `build_hard_clips.py` 輸出同 schema（每 clip：sequence / modality / start_position /
length / frame_ids / mean_iou / min_iou / failing_frames / frozen_frames / fold），
`train_tracker_g1.clip_dataset.ClipDataset` 可直接吃；差別只有一個：**clip 起點是規則格點，
不是失敗事件**——這是 D105 要測的單一變因（failure-centric 取樣被懷疑是 G2 健康序列退步的來源）。

輸入
* `--frame-contract` ＝ `prep_oof405_frames.py` 產的 `contracts/frame_contract.csv`
  （欄：ID, sequence, position, modality, capture_group, fold, gt_valid, …）。position 是序列內
  0-based 幀位置＝frames_root/<seq>/ 下 sorted(*.jpg) 的索引（clip_dataset.py 檔頭的對齊契約）。
* `--gt-csv` ＝ 2026training.csv（只用來重驗 gt_valid，防 contract 與 GT 版本錯位）。
* `--pred-csv`（選配）＝ 405 zero-shot full-frame SAM3 預測（gDrive
  `oof405_exact_pair_20260827/run/full_sam3/submission.csv`），有給才算 mean_iou／min_iou／
  failing_frames／frozen_frames；沒給則四欄為 null、頂層 `has_pred_stats=false`
  ——下游 `group_breakdown_p1` 會據此跳過 identity 分組，不塞 NaN。

規則（寫死）：窗長 16、步長 16（非重疊）、窗內 ≥8 幀 gt_valid 才收、三模態全收。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CLIP_LEN = 16
STRIDE = 16
MIN_VALID = 8
FAIL_IOU = 0.50   # 同 build_hard_clips.py（identity clip ≡ min_iou >= 0.5）
FROZEN_RUN = 3    # 同 build_hard_clips.py（連續 ≥3 幀框完全相同視為凍結）
MODALITIES = ("vis", "nir", "rednir")


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "t")


def load_frame_contract(path) -> dict:
    """→ {sequence: [rows sorted by position]}；每 row: dict(ID, position:int, modality, fold:int, gt_valid:bool)。"""
    by_seq: dict[str, list] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        need = {"ID", "sequence", "position", "modality", "fold", "gt_valid"}
        missing = need - set(reader.fieldnames or [])
        assert not missing, f"frame_contract 缺欄位 {sorted(missing)}"
        for r in reader:
            by_seq[r["sequence"]].append({
                "ID": r["ID"], "position": int(r["position"]), "modality": r["modality"],
                "fold": int(r["fold"]), "gt_valid": _truthy(r["gt_valid"]),
            })
    for seq, rows in by_seq.items():
        rows.sort(key=lambda r: r["position"])
        assert [r["position"] for r in rows] == list(range(len(rows))), \
            f"{seq}: position 不連續（contract 損壞？）"
        assert len({r["fold"] for r in rows}) == 1, f"{seq}: fold 不一致"
        assert len({r["modality"] for r in rows}) == 1, f"{seq}: modality 不一致"
    return dict(by_seq)


def load_boxes_csv(path) -> dict:
    """submission 格式 CSV → {ID: (x, y, w, h)}。"""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        assert header[0].strip().lower() == "id", f"非預期標頭 {header}"
        for row in reader:
            if row and row[0]:
                out[row[0]] = tuple(float(v) for v in row[1:5])
    return out


def box_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def window_starts(n_frames: int, clip_len: int = CLIP_LEN, stride: int = STRIDE) -> list:
    """規則格點：0, stride, 2·stride, … 且窗完整落在序列內。"""
    if n_frames < clip_len:
        return []
    return list(range(0, n_frames - clip_len + 1, stride))


def clip_stats(frame_ids: list, valid: list, gt: dict, pred: dict | None) -> dict:
    """有 pred 才算；否則四欄 None。凍結＝連續 ≥FROZEN_RUN 幀框完全相同（含首幀起算）。"""
    if pred is None:
        return {"mean_iou": None, "min_iou": None, "failing_frames": None, "frozen_frames": None}
    ious = [box_iou(pred.get(fid), gt.get(fid)) for fid, v in zip(frame_ids, valid) if v]
    boxes = [pred.get(fid) for fid in frame_ids]
    run, frozen = 0, 0
    for i in range(len(boxes)):
        run = run + 1 if (i > 0 and boxes[i] is not None and boxes[i] == boxes[i - 1]) else 0
        if run >= FROZEN_RUN - 1 and valid[i]:
            frozen += 1
    return {
        "mean_iou": (sum(ious) / len(ious)) if ious else None,
        "min_iou": min(ious) if ious else None,
        "failing_frames": sum(1 for x in ious if x < FAIL_IOU),
        "frozen_frames": frozen,
    }


def build_uniform_clips(by_seq: dict, gt: dict, pred: dict | None,
                        clip_len: int = CLIP_LEN, stride: int = STRIDE,
                        min_valid: int = MIN_VALID, modalities=MODALITIES) -> list:
    clips = []
    for seq in sorted(by_seq):
        rows = by_seq[seq]
        modality = rows[0]["modality"]
        if modality not in modalities:
            continue
        for lo in window_starts(len(rows), clip_len, stride):
            win = rows[lo:lo + clip_len]
            frame_ids = [r["ID"] for r in win]
            # 用 GT 重驗有效性（contract 與 GT 版本若錯位，這裡會與 gt_valid 不一致 → fail closed）
            valid = []
            for r in win:
                g = gt.get(r["ID"])
                # 判準必須與契約端逐字相同（prep_oof405_frames.py:57 valid_box）：
                # 全為有限值 ＋ w>0 ＋ h>0。**x/y 不設限**——目標貼左緣/上緣時 x 或 y 合法為 0
                # （例：nir-ball_31 = 0,94,20,24），全訓練集有 1,352 幀屬此類。
                gv = (g is not None
                      and all(math.isfinite(v) for v in g)
                      and g[2] > 0 and g[3] > 0)
                assert gv == r["gt_valid"], f"{r['ID']}: gt_valid 與 GT 不一致（contract={r['gt_valid']}, gt={gv}）"
                valid.append(gv)
            if sum(valid) < min_valid:
                continue
            clips.append({
                "sequence": seq, "modality": modality,
                "start_position": lo, "length": clip_len,
                "frame_ids": frame_ids,
                **clip_stats(frame_ids, valid, gt, pred),
                "fold": rows[0]["fold"],
            })
    return clips


def summarize(clips: list, has_pred_stats: bool) -> dict:
    by_fold = Counter(str(c["fold"]) for c in clips)
    by_mod = Counter(c["modality"] for c in clips)
    doc = {
        "source": "uniform_grid",
        "clip_length": CLIP_LEN, "stride": STRIDE, "min_valid_frames": MIN_VALID,
        "modalities": sorted(by_mod),
        "n_clips": len(clips),
        "n_sequences_with_clips": len({c["sequence"] for c in clips}),
        "clips_per_modality": dict(sorted(by_mod.items())),
        "clips_per_fold": dict(sorted(by_fold.items())),
        "has_pred_stats": has_pred_stats,
    }
    if has_pred_stats:
        scored = [c for c in clips if c["min_iou"] is not None]
        identity = sum(1 for c in scored if c["min_iou"] >= FAIL_IOU)
        doc["identity_clips"] = identity
        doc["identity_rate"] = identity / len(scored) if scored else None
        doc["mean_clip_iou"] = (sum(c["mean_iou"] for c in scored) / len(scored)) if scored else None
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame-contract", required=True)
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--pred-csv", help="405 zero-shot full_sam3 submission.csv（選配，供統計）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    by_seq = load_frame_contract(args.frame_contract)
    gt = load_boxes_csv(args.gt_csv)
    pred = load_boxes_csv(args.pred_csv) if args.pred_csv else None
    clips = build_uniform_clips(by_seq, gt, pred)
    if not clips:
        raise SystemExit("no uniform clips built; refusing to emit an empty index")
    doc = {**summarize(clips, pred is not None), "clips": clips}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"uniform clips={doc['n_clips']} sequences={doc['n_sequences_with_clips']} "
          f"per_modality={doc['clips_per_modality']} per_fold={doc['clips_per_fold']} "
          f"pred_stats={doc['has_pred_stats']}"
          + (f" identity_rate={doc['identity_rate']:.1%} mean_clip_iou={doc['mean_clip_iou']:.4f}"
             if doc["has_pred_stats"] else ""))
    print(f"outputs={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
