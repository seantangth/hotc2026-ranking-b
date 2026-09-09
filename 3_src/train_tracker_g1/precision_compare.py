#!/usr/bin/env python3
"""P1 精度線 Stage B（D105）：full-sequence smoke 的對照、判決與 arm 選擇。

輸入＝兩份 track_t1 的 submission.csv（trained vs frozen，同一組 Stage B 序列）＋ 2026training.csv GT
＋ smoke plan json（train_p1.py --smoke-plan-out 的輸出，鍵 "all"）＋ 該 arm 的 Stage A clip_gain。
join 鍵＝ID（track_t1 帶 --gt-csv 輸出全域幀號）。GT 無效幀（xywh 任一 ≤0）排除；pred 缺幀計 0。

主讀數＝**frame-pooled Δ mean IoU**（與 Success AUC 同語意；序列不等權）；per-seq Δ 只當災難閘。
另報：官方 50 門檻 AUC Δ、已追到幀（frozen IoU≥0.5）的 mean IoU Δ、跟丟率（<0.1）變化——診斷用，不進判準。

═══ Stage B 事前判準（寫死於 2026-09-02；全文見 STRATEGY D105）═══
NO_GO_A       ⇔ clip_gain < +0.01（Stage A 未過；本腳本正常不會被呼叫，防線保留）
CATASTROPHE   ⇔ 任一序列 Δ < −0.05
NO_GO         ⇔ pooled Δ < −0.005
GREY          ⇔ −0.005 ≤ pooled Δ < +0.005   （不上 LB，呈 Sean）
GO            ⇔ pooled Δ ≥ +0.005
rc：GO=0／NO_GO=2／CATASTROPHE=5／GREY=6／NO_GO_A=7。

arm 選擇（--select a.json b.json …）：只有一個 GO 取它；多個 GO 取 pooled Δ 最高者，與次高差 <0.002 時取 A
（字典序最小的 arm 標籤）；無 GO ⇒ chosen=null（不打 LB）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 3_src/
from train_tracker_g1.clip_dataset import frame_valid, load_gt_csv  # noqa: E402
from train_tracker_g1.losses import box_iou_xywh  # noqa: E402
from train_tracker_g1.smoke_compare import load_submission_csv, seq_of  # noqa: E402

MIN_CLIP_GAIN = 0.01
CATASTROPHE_DELTA = -0.05
NO_GO_POOLED = -0.005
GO_POOLED = 0.005
ARM_TIE = 0.002
TRACKED_IOU = 0.5
LOST_IOU = 0.1
RC = {"GO": 0, "NO_GO": 2, "CATASTROPHE": 5, "GREY": 6, "NO_GO_A": 7}


def frame_ious(pred: dict, gt: dict, sequences) -> dict:
    """→ {seq: [iou per valid GT frame（GT 順序）]}；pred 缺幀＝0。"""
    seqs = set(sequences)
    out = {s: [] for s in sequences}
    for fid, gt_box in gt.items():
        s = seq_of(fid)
        if s not in seqs or not frame_valid(gt_box):
            continue
        p = pred.get(fid)
        out[s].append(box_iou_xywh(p, gt_box) if p is not None else 0.0)
    for s, v in out.items():
        assert v, f"{s}: GT 內無有效幀（序列名寫錯？）"
    return out


def auc50(ious) -> float:
    """官方 Success AUC：50 個門檻 0.02..1.00 的成功率平均（僅有效 GT 幀）。"""
    n = len(ious)
    return sum(sum(1 for x in ious if x >= k / 50.0) / n for k in range(1, 51)) / 50.0


def compare(trained: dict, frozen: dict, gt: dict, plan: dict, clip_gain: float) -> dict:
    seqs = list(plan["all"])
    t = frame_ious(trained, gt, seqs)
    f = frame_ious(frozen, gt, seqs)
    rows = []
    for s in seqs:
        rows.append({"sequence": s, "n": len(f[s]),
                     "frozen": statistics.fmean(f[s]), "trained": statistics.fmean(t[s]),
                     "delta": statistics.fmean(t[s]) - statistics.fmean(f[s])})
    all_f = [x for s in seqs for x in f[s]]
    all_t = [x for s in seqs for x in t[s]]
    pooled_f, pooled_t = statistics.fmean(all_f), statistics.fmean(all_t)
    pooled_delta = pooled_t - pooled_f
    tracked = [(a, b) for a, b in zip(all_f, all_t) if a >= TRACKED_IOU]
    diag = {
        "auc50_frozen": auc50(all_f), "auc50_trained": auc50(all_t),
        "auc50_delta": auc50(all_t) - auc50(all_f),
        "tracked_share_frozen": len(tracked) / len(all_f),
        "tracked_mean_iou_frozen": statistics.fmean(a for a, _ in tracked) if tracked else None,
        "tracked_mean_iou_trained": statistics.fmean(b for _, b in tracked) if tracked else None,
        "lost_rate_frozen": sum(1 for x in all_f if x < LOST_IOU) / len(all_f),
        "lost_rate_trained": sum(1 for x in all_t if x < LOST_IOU) / len(all_t),
    }
    worst = min(rows, key=lambda r: r["delta"])
    if clip_gain < MIN_CLIP_GAIN:
        verdict = "NO_GO_A"
    elif worst["delta"] < CATASTROPHE_DELTA:
        verdict = "CATASTROPHE"
    elif pooled_delta < NO_GO_POOLED:
        verdict = "NO_GO"
    elif pooled_delta < GO_POOLED:
        verdict = "GREY"
    else:
        verdict = "GO"
    return {"verdict": verdict, "clip_gain": clip_gain,
            "pooled_frozen": pooled_f, "pooled_trained": pooled_t, "pooled_delta": pooled_delta,
            "n_frames": len(all_f), "n_sequences": len(seqs),
            "worst_sequence": worst["sequence"], "worst_delta": worst["delta"],
            "catastrophe_sequences": [r["sequence"] for r in rows if r["delta"] < CATASTROPHE_DELTA],
            "criteria": {"min_clip_gain": MIN_CLIP_GAIN, "catastrophe_delta": CATASTROPHE_DELTA,
                         "no_go_pooled": NO_GO_POOLED, "go_pooled": GO_POOLED},
            "diagnostics": diag, "per_sequence": rows}


def select_arm(results: dict) -> dict:
    """results = {arm_label: compare() 的 dict}。回傳 {"chosen": label|None, "reason": str, "candidates": {...}}。"""
    go = {k: v["pooled_delta"] for k, v in results.items() if v["verdict"] == "GO"}
    cands = {k: {"verdict": v["verdict"], "pooled_delta": v["pooled_delta"]} for k, v in results.items()}
    if not go:
        return {"chosen": None, "reason": "no arm reached GO", "candidates": cands}
    if len(go) == 1:
        (k, _), = go.items()
        return {"chosen": k, "reason": "only GO arm", "candidates": cands}
    ranked = sorted(go.items(), key=lambda kv: (-kv[1], kv[0]))
    top, second = ranked[0], ranked[1]
    if top[1] - second[1] < ARM_TIE:
        chosen = min(go)  # 字典序最小 ＝ arm A（decoder，風險最低）
        return {"chosen": chosen, "reason": f"tie within {ARM_TIE} → lexicographically first", "candidates": cands}
    return {"chosen": top[0], "reason": "highest pooled_delta among GO arms", "candidates": cands}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trained-csv")
    ap.add_argument("--frozen-csv")
    ap.add_argument("--gt-csv")
    ap.add_argument("--plan-json")
    ap.add_argument("--clip-gain", type=float, help="該 arm 的 Stage A clip_gain")
    ap.add_argument("--arm", default="A")
    ap.add_argument("--out")
    ap.add_argument("--select", nargs="+", metavar="RESULT_JSON",
                    help="選擇模式：讀多個 compare 輸出 json（需含 arm 欄），寫 --out")
    args = ap.parse_args(argv)

    if args.select:
        results = {}
        for p in args.select:
            d = json.loads(Path(p).read_text())
            results[d.get("arm", Path(p).stem)] = d
        sel = select_arm(results)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(sel, indent=2, ensure_ascii=False))
        print(json.dumps(sel, ensure_ascii=False))
        return 0 if sel["chosen"] else 2

    for k in ("trained_csv", "frozen_csv", "gt_csv", "plan_json", "out"):
        assert getattr(args, k), f"--{k.replace('_', '-')} 必填"
    assert args.clip_gain is not None, "--clip-gain 必填"
    plan = json.loads(Path(args.plan_json).read_text())
    result = compare(load_submission_csv(args.trained_csv), load_submission_csv(args.frozen_csv),
                     load_gt_csv(args.gt_csv), plan, args.clip_gain)
    result["arm"] = args.arm
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    for r in result["per_sequence"]:
        print(f"[p1-B {args.arm}] {r['sequence']:26s} n={r['n']:4d} frozen={r['frozen']:.4f} "
              f"trained={r['trained']:.4f} Δ={r['delta']:+.4f}")
    d = result["diagnostics"]
    print(f"[p1-B {args.arm}] pooled {result['pooled_frozen']:.4f} → {result['pooled_trained']:.4f} "
          f"(Δ {result['pooled_delta']:+.4f}) | AUC50 Δ {d['auc50_delta']:+.4f} | "
          f"tracked meanIoU {d['tracked_mean_iou_frozen']} → {d['tracked_mean_iou_trained']} | "
          f"lost {d['lost_rate_frozen']:.3f} → {d['lost_rate_trained']:.3f}")
    print(f"[p1-B {args.arm}] verdict = {result['verdict']}")
    return RC[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
