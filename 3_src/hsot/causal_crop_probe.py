#!/usr/bin/env python3
"""零 GPU：因果跟隨裁窗 vs 首幀釘死窗 vs two-pass envelope。

D066 只量過「窗由首幀決定」的 one-pass。本探針量「窗跟著上一幀預測走」
——仍是 one-pass / OPE 因果，但是 D066 沒測過的窗位置政策。

模擬限制（必須寫在讀數旁邊）：
  軌跡來自既有 full-frame 跑（e15），不是在因果窗內真的重跑 tracker。
  脫窗/天花板是「若窗跟著這條軌跡走，GT 還在不在窗裡」的上界診斷。
  真整合會有死亡螺旋（跟丟後窗跟著錯目標走）——本腳本另外報
  「pred IoU<0.1 之後」的條件脫窗，逼近那種失敗。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hsot.firstframe_crop_probe import (
    AFM_MAIN,
    MIN_MARGIN,
    MARGIN_SCALE,
    ROOT,
    SMALL_T,
    _parse_ids,
    clip_stats,
    envelope_window,
    infer_size,
    known_val_sizes,
    load_val_gt,
    size_candidates,
)


def _xywh_to_win(box, W, H, margin):
    x, y, w, h = [float(v) for v in box]
    x1 = max(0.0, x - margin)
    y1 = max(0.0, y - margin)
    x2 = min(float(W), x + w + margin)
    y2 = min(float(H), y + h + margin)
    return int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2))


def _area_frac(win, W, H):
    return max(0.0, (win[2] - win[0]) * (win[3] - win[1])) / float(W * H)


def causal_windows(guide: np.ndarray, W: int, H: int, cap: float = AFM_MAIN):
    """frame t 的窗由 guide[t-1] 決定（t=0 用 guide[0]＝init）。
    外擴 max(2×scale, 32)；若面積比 ≥ cap 則該幀改用全圖（不裁）。"""
    n = len(guide)
    wins = []
    cropped = np.zeros(n, dtype=bool)
    for t in range(n):
        src = guide[0] if t == 0 else guide[t - 1]
        scale = float(np.sqrt(max(src[2] * src[3], 1e-6)))
        margin = max(MARGIN_SCALE * scale, MIN_MARGIN)
        win = _xywh_to_win(src, W, H, margin)
        if _area_frac(win, W, H) >= cap:
            win = (0, 0, int(W), int(H))
            cropped[t] = False
        else:
            cropped[t] = True
        wins.append(win)
    return wins, cropped


def per_frame_clip(gt: np.ndarray, wins: list[tuple[int, int, int, int]]) -> np.ndarray:
    ub = np.empty(len(gt), dtype=float)
    for i, win in enumerate(wins):
        ub[i] = clip_stats(gt[i : i + 1], win)["iou_ub_mean"]
    return ub


def align(gt_df: pd.DataFrame, pred_df: pd.DataFrame):
    g = gt_df.sort_values("frame")
    p = pred_df.sort_values("frame")
    merged = g.merge(p, on="frame", suffixes=("_g", "_p"))
    if len(merged) != len(g):
        return None
    gt = merged[["x_g", "y_g", "width_g", "height_g"]].to_numpy(float)
    pr = merged[["x_p", "y_p", "width_p", "height_p"]].to_numpy(float)
    frames = merged["frame"].to_numpy(int)
    return gt, pr, frames


def box_iou(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main():
    gt = load_val_gt(ROOT / "1_data/val_split_v1.txt")
    e15 = _parse_ids(pd.read_csv(ROOT / "5_outputs/e15_sam3_20260806/submission_val65.csv"))
    cands, known = size_candidates(), known_val_sizes()

    rows = []
    ub_env, ub_causal_pred, ub_causal_gt, ub_ff = [], [], [], []
    all_causal_pred, all_causal_gt, all_ff = [], [], []
    n_small = 0
    for seq, g in gt.items():
        est, _ = infer_size(g, cands)
        W, H = known.get(seq, est)
        box0 = g.iloc[0][["x", "y", "width", "height"]].to_numpy(float)
        scale0 = float(np.sqrt(max(box0[2] * box0[3], 1e-6)))
        if scale0 >= SMALL_T:
            continue
        n_small += 1
        gb = e15[e15["seq"] == seq]
        al = align(g, gb)
        if al is None:
            continue
        gt_b, pr_b, _ = al
        ious = np.array([box_iou(gt_b[i], pr_b[i]) for i in range(len(gt_b))])
        lost = ious < 0.1

        env = envelope_window(pr_b, scale0, W, H)
        env_st = clip_stats(gt_b, env)
        env_af = _area_frac(env, W, H)

        wins_p, crop_p = causal_windows(pr_b, W, H)
        ub_p = per_frame_clip(gt_b, wins_p)
        wins_g, crop_g = causal_windows(gt_b, W, H)
        ub_g = per_frame_clip(gt_b, wins_g)

        # 首幀釘死、R 取 D066 最佳水平條帶附近：方形 R=12（對照，不是新主張）
        from hsot.firstframe_crop_probe import firstframe_window
        ff = firstframe_window(box0, scale0, 12.0, W, H)
        ff_st = clip_stats(gt_b, ff)

        rec = {
            "seq": seq,
            "n": int(len(gt_b)),
            "lost_frac": float(lost.mean()),
            "env_af": env_af,
            "env_sel": env_af < AFM_MAIN,
            "env_escape": env_st["escape_rate"],
            "env_ub_loss": env_st["ub_loss"],
            "causal_pred_crop_frac": float(crop_p.mean()),
            "causal_pred_escape": float((ub_p < 1.0).mean()),
            "causal_pred_ub_loss": float(1.0 - ub_p.mean()),
            "causal_pred_escape_while_tracked": float((ub_p[~lost] < 1.0).mean()) if (~lost).any() else None,
            "causal_pred_escape_while_lost": float((ub_p[lost] < 1.0).mean()) if lost.any() else None,
            "causal_gt_crop_frac": float(crop_g.mean()),
            "causal_gt_escape": float((ub_g < 1.0).mean()),
            "causal_gt_ub_loss": float(1.0 - ub_g.mean()),
            "ff12_escape": ff_st["escape_rate"],
            "ff12_ub_loss": ff_st["ub_loss"],
        }
        rows.append(rec)
        env_ub = per_frame_clip(gt_b, [env] * len(gt_b))
        ff_ub = per_frame_clip(gt_b, [ff] * len(gt_b))
        all_causal_pred.extend(ub_p.tolist())
        all_causal_gt.extend(ub_g.tolist())
        all_ff.extend(ff_ub.tolist())
        if env_af < AFM_MAIN:
            ub_env.extend(env_ub.tolist())
            ub_causal_pred.extend(ub_p.tolist())
            ub_causal_gt.extend(ub_g.tolist())
            ub_ff.extend(ff_ub.tolist())

    def pooled(xs):
        a = np.asarray(xs, float)
        return {
            "n": int(a.size),
            "escape": float((a < 1.0).mean()) if a.size else None,
            "ub_loss": float(1.0 - a.mean()) if a.size else None,
            "severe": float((a < 0.5).mean()) if a.size else None,
        }

    summary = {
        "n_small": n_small,
        "n_aligned": len(rows),
        "pooled_on_env_selected": {
            "envelope_twopass": pooled(ub_env),
            "causal_follow_pred": pooled(ub_causal_pred),
            "causal_follow_gt_oracle": pooled(ub_causal_gt),
            "firstframe_R12": pooled(ub_ff),
        },
        "pooled_all_small": {
            "causal_follow_pred": pooled(all_causal_pred),
            "causal_follow_gt_oracle": pooled(all_causal_gt),
            "firstframe_R12": pooled(all_ff),
        },
        "seq_mean": {
            "env_escape": float(np.mean([r["env_escape"] for r in rows if r["env_sel"]])) if any(r["env_sel"] for r in rows) else None,
            "causal_pred_escape": float(np.mean([r["causal_pred_escape"] for r in rows])),
            "causal_gt_escape": float(np.mean([r["causal_gt_escape"] for r in rows])),
            "ff12_escape": float(np.mean([r["ff12_escape"] for r in rows])),
            "causal_pred_ub_loss": float(np.mean([r["causal_pred_ub_loss"] for r in rows])),
            "causal_gt_ub_loss": float(np.mean([r["causal_gt_ub_loss"] for r in rows])),
        },
    }
    out = ROOT / "5_outputs/causal_crop_probe_20260821.json"
    out.write_text(json.dumps({"summary": summary, "per_seq": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
