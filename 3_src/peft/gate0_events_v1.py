#!/usr/bin/env python3
"""gate0_events_v1 — D068 Gate 0 分析器：恆等占比真值 ＋ 失敗事件清單 ＋ 事前判準裁決。

輸入＝track_t1（--backend samurai、--gt-csv、--mask-cache）在訓練池 76 支上的輸出。

【事前判準（執行前寫死；出處 report_audit_training.md §5 提案 D068）】
  G0-A 恆等占比：video-mode 傳播 mask 與 teacher（SAM2.1-L image-mode）mask 的
        逐標註幀 IoU ≥ 0.9 的占比 ≥ 90% ⇒ 「自蒸餾恆等監督」診斷確認。
        < 90% ⇒ 診斷錯誤，停止並重審（不進 Gate 1）。
  G0-B 失敗事件數：GT-box IoU 軌跡由 ≥0.5 跌破 <0.3 的轉換事件 ≥ 300 個。
        < 300 ⇒ 失敗段樣本不足，Gate 1 的 50/50 混採不成立，按缺項結案。
  G0-C 事件窗 loss 對比（不在本腳本；由 launch 的 lr=0 雙腿量測）：
        median(事件腿) / median(對照腿) ≥ 5 ⇒ 「事件窗有可優化信號」成立。
        （單位換算陷阱：舊 mask-loss 的 2.5/0.5 數字不適用於投影 loss ⇒ 對照腿當場實測地板。）

【已知量測偏差（記錄，不修）】Gate 0 推論走 SAMURAI backend（Kalman 記憶挑選），
比訓練期 vanilla 傳播更穩 ⇒ 恆等占比偏高（對 G0-A 的 90% 門檻略反保守）、
事件數偏低（對 G0-B 的 300 門檻保守）。判決書須引用本段。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ID_THR = 0.90          # G0-A 逐幀 mask IoU 門檻
ID_FRAC_PASS = 0.90    # G0-A 占比門檻
EVT_GOOD = 0.50        # 事件定義：由 ≥EVT_GOOD
EVT_BAD = 0.30         #           跌破 <EVT_BAD
EVT_MIN = 300          # G0-B 門檻
EVT_SKIP = 16          # 事件後跳過幀數（窗不重疊）


def box_iou_arr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a,b: [N,4] xywh → IoU [N]；GT<=0 的列由呼叫端遮罩。"""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / union, 0.0)


def load_masks_npz(path: Path):
    z = np.load(path)
    if int(z["n"]) == 0:
        return None
    m = np.unpackbits(z["bits"])[: int(z["n"]) * int(z["height"]) * int(z["width"])]
    return m.reshape(int(z["n"]), int(z["height"]), int(z["width"])).astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", required=True, help="track_t1 輸出（全域幀號 ID）")
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--seq-list", required=True)
    ap.add_argument("--masks-dir", default="", help="track_t1 out/masks（缺→跳過 G0-A mask 級）")
    ap.add_argument("--ann-root", default="", help="davis Annotations 根（teacher PNG）")
    ap.add_argument("--out-events", required=True)
    ap.add_argument("--out-verdict", required=True)
    a = ap.parse_args()

    gt = pd.read_csv(a.gt_csv); gt.columns = ["ID", "x", "y", "w", "h"]
    pr = pd.read_csv(a.pred_csv); pr.columns = ["ID", "x", "y", "w", "h"]
    parts = gt["ID"].str.rsplit("_", n=1, expand=True)
    gt["seq"], gt["frame"] = parts[0], parts[1].astype(int)
    pr = pr.set_index("ID")
    seqs = [s.strip() for s in Path(a.seq_list).read_text().split() if s.strip()]

    events, per_seq = [], {}
    id_num = id_den = 0
    id_ious_all = []

    for seq in seqs:
        rows = gt[gt["seq"] == seq].sort_values("frame").reset_index(drop=True)
        ids = rows["ID"].tolist()
        missing = [i for i in ids if i not in pr.index]
        if missing:
            per_seq[seq] = {"error": f"pred 缺 {len(missing)} 列"}
            continue
        g = rows[["x", "y", "w", "h"]].to_numpy(float)
        p = pr.loc[ids, ["x", "y", "w", "h"]].to_numpy(float)
        valid = (g > 0).all(axis=1)
        iou = box_iou_arr(p, g)
        iou[~valid] = np.nan

        # ── 事件抽取（GT-box IoU 軌跡）────────────────────────────────────
        n_evt_seq, t = 0, 1
        good = False
        while t < len(iou):
            v = iou[t]
            if np.isnan(v):
                t += 1; continue
            if v >= EVT_GOOD:
                good = True
            elif good and v < EVT_BAD:
                events.append({"seq": seq, "start": int(t), "iou_at": float(v)})
                n_evt_seq += 1
                good = False
                t += EVT_SKIP
                continue
            t += 1

        # ── 恆等占比（mask 級，G0-A）─────────────────────────────────────
        n_id = d_id = 0
        if a.masks_dir and a.ann_root:
            mz = Path(a.masks_dir) / f"{seq}.npz"
            ann_d = Path(a.ann_root) / seq
            if mz.exists() and ann_d.is_dir():
                masks = load_masks_npz(mz)
                if masks is not None:
                    for png in sorted(ann_d.glob("*.png")):
                        pos = int(png.stem)
                        if pos >= len(masks):
                            continue
                        tmask = np.array(Image.open(png)) > 0
                        pm = masks[pos]
                        if tmask.shape != pm.shape:
                            continue
                        inter = np.logical_and(pm, tmask).sum()
                        union = np.logical_or(pm, tmask).sum()
                        v = inter / union if union else 0.0
                        id_ious_all.append(v)
                        d_id += 1
                        if v >= ID_THR:
                            n_id += 1
        id_num += n_id; id_den += d_id
        per_seq[seq] = {"n_frames": int(len(iou)), "n_events": n_evt_seq,
                        "gt_iou_median": float(np.nanmedian(iou)),
                        "lost_frac": float(np.nanmean(iou < 0.1)),
                        "id_frames": d_id, "id_pass": n_id}

    id_frac = id_num / id_den if id_den else float("nan")
    g0a = (id_frac >= ID_FRAC_PASS) if id_den else None
    g0b = len(events) >= EVT_MIN
    verdict = {
        "G0A_identity_frac": id_frac, "G0A_pass": g0a, "G0A_n_frames": id_den,
        "G0A_identity_iou_median": float(np.median(id_ious_all)) if id_ious_all else None,
        "G0B_n_events": len(events), "G0B_pass": bool(g0b),
        "thresholds": {"id_thr": ID_THR, "id_frac_pass": ID_FRAC_PASS,
                       "evt_good": EVT_GOOD, "evt_bad": EVT_BAD, "evt_min": EVT_MIN},
        "bias_note": "SAMURAI KF backend：恆等占比偏高（反保守）、事件數偏低（保守）",
        "per_seq": per_seq,
    }
    Path(a.out_events).write_text(json.dumps({"events": events}, indent=1))
    Path(a.out_verdict).write_text(json.dumps(verdict, indent=1, ensure_ascii=False))
    print(f"[gate0] G0-A 恆等占比 {id_frac:.1%}（{id_num}/{id_den}，門檻 ≥{ID_FRAC_PASS:.0%}）"
          f" → {'✅' if g0a else ('—無 mask 資料' if g0a is None else '🚨 診斷不確認')}")
    print(f"[gate0] G0-B 事件數 {len(events)}（門檻 ≥{EVT_MIN}）→ {'✅' if g0b else '🚨 不足'}")
    print("GATE0-EVENTS-DONE")


if __name__ == "__main__":
    main()
