#!/usr/bin/env python3
"""E32 Stage 1 探針——判準見 5_outputs/strategy_research_20260812/E32_STAGE1_DESIGN_20260813.md
（該檔於本腳本執行前寫入）。Lambda CUDA、零 LB 額度。

對 13 支序列（病灶 8＋對照 5）的凍結幀跑 2 幀迷你 video PCS session，量三件事：
 (i)  凍結跟丟幀的偵測命中率（top-1 IoU≥0.5）
 (ii) 「凍結但其實追得好」的幀被誤替換的比例＝08-10 副產發現的雙尾陷阱
 (iii) 逐序列模擬淨 delta（照替換政策把凍結框換成偵測框後的 IoU 變化）
"""
import json, os, shutil, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

SP = Path(os.environ.get("E32_DIR", str(Path.home() / "e32")))
CFG = json.load(open(SP / "stage1_frames.json"))
CKPT = SP / "ckpt" / "sam3.pt"
SCORE_TAUS = [0.0, 0.3, 0.5, 0.7]      # 事後掃描替換門檻（不是調參：Stage 2 前需固定一個）

def iou(a, b):
    iw = max(0.0, min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
    i = iw*ih; u = a[2]*a[3] + b[2]*b[3] - i
    return i/u if u > 0 else 0.0

from PIL import Image
from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor(checkpoint_path=str(CKPT))
print("[setup] predictor 就緒", flush=True)

def probe(frames_dir, first_name, tgt_name, box_xywh_rel):
    tmp = SP / "mini"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    shutil.copy(frames_dir / first_name, tmp / "00000.jpg")
    shutil.copy(frames_dir / tgt_name,   tmp / "00001.jpg")
    sid = predictor.start_session(str(tmp))
    sid = sid if isinstance(sid, str) else (sid.get("session_id") if isinstance(sid, dict) else str(sid))
    try:
        predictor.add_prompt(session_id=sid, frame_idx=0, bounding_boxes=[box_xywh_rel],
                             bounding_box_labels=[1], rel_coordinates=True)
        dets = []
        for out in predictor.propagate_in_video(session_id=sid, propagation_direction="forward",
                                                start_frame_idx=0, max_frame_num_to_track=1):
            if out.get("frame_index") != 1:
                continue
            o = out.get("outputs")
            if not o:
                continue
            bx, pr = o.get("out_boxes_xywh"), o.get("out_probs")
            if bx is None or len(bx) == 0:
                continue
            for k in range(len(bx)):
                x, y, w, h = [float(v) for v in bx[k]]
                dets.append(([x*W, y*H, w*W, h*H],
                             float(pr[k]) if pr is not None and k < len(pr) else 1.0))
        return dets
    finally:
        try: predictor.close_session(sid)
        except Exception: pass

records, errs = [], 0
for si, (seq, C) in enumerate(CFG["seqs"].items()):
    fdir = SP / "frames" / seq
    if not fdir.is_dir():
        print(f"[SKIP] {seq}: 無影格目錄", flush=True); continue
    first = C["first"]
    W, H = Image.open(fdir / f"{1:04d}.jpg").size
    ib = C["init_box"]
    box_rel = [ib[0]/W, ib[1]/H, ib[2]/W, ib[3]/H]
    for kind in ("frozen_lost", "frozen_fine"):
        for g in C[kind]:
            name = f"{g - first + 1:04d}.jpg"
            try:
                dets = sorted(probe(fdir, f"{1:04d}.jpg", name, box_rel), key=lambda d: -d[1])
            except Exception as e:
                errs += 1
                if errs <= 3: print(f"  EXC {seq}/{g}: {type(e).__name__} {str(e)[:120]}", flush=True)
                if errs >= 20: print("🚨 例外 ≥20 次，中止"); sys.exit(3)
                continue
            gt, pred = C["gt"][str(g)], C["pred"][str(g)]
            records.append({"seq": seq, "group": C["group"], "kind": kind, "frame": g,
                            "n_det": len(dets),
                            "det_iou": round(iou(dets[0][0], gt), 4) if dets else 0.0,
                            "det_score": round(dets[0][1], 4) if dets else 0.0,
                            "base_iou": round(iou(pred, gt), 4)})
    done = sum(1 for r in records if r["seq"] == seq)
    print(f"[{si+1}/{len(CFG['seqs'])}] {seq}: {done} 幀完成", flush=True)

# ── 三項判準讀數 ──────────────────────────────────────────────────────────
def rate(rs, f): return (sum(1 for r in rs if f(r)) / len(rs)) if rs else float("nan")
lost = [r for r in records if r["kind"] == "frozen_lost"]
fine = [r for r in records if r["kind"] == "frozen_fine"]
res = {"n_records": len(records), "n_exc": errs,
       "i_hit_rate@0.5": rate(lost, lambda r: r["det_iou"] >= 0.5),
       "i_detect_rate": rate(lost, lambda r: r["n_det"] > 0),
       "by_tau": {}, "per_seq": {}, "records": records}
for tau in SCORE_TAUS:
    fire = lambda r: r["n_det"] > 0 and r["det_score"] >= tau
    # (ii) 誤替換＝在「凍結但追得好」的幀上觸發且換成更差的框
    mis = [r for r in fine if fire(r) and r["det_iou"] < r["base_iou"]]
    deltas = {}
    for r in records:
        d = (r["det_iou"] - r["base_iou"]) if fire(r) else 0.0
        deltas.setdefault(r["seq"], []).append(d)
    per_seq = {s: round(sum(v)/len(v), 4) for s, v in deltas.items()}   # 每幀平均 IoU 變化
    res["by_tau"][str(tau)] = {
        "ii_mis_replace_rate": round(len(mis)/len(fine), 4) if fine else None,
        "iii_worst_seq_delta": min(per_seq.values()) if per_seq else None,
        "iii_worst_seq": min(per_seq, key=per_seq.get) if per_seq else None,
        "mean_delta_lost_frames": round(sum((r["det_iou"]-r["base_iou"]) for r in lost if fire(r))/len(lost), 4) if lost else None,
        "per_seq_delta": per_seq}
json.dump(res, open(SP / "stage1_result.json", "w"), indent=1, ensure_ascii=False)

print(f"\n{'='*64}")
print(f"(i)   凍結跟丟幀 n={len(lost)}：偵測率 {res['i_detect_rate']:.1%}｜top-1 IoU≥0.5 命中 {res['i_hit_rate@0.5']:.1%}（門檻 40%）")
for tau in SCORE_TAUS:
    b = res["by_tau"][str(tau)]
    print(f"τ={tau}: (ii) 誤替換率 {b['ii_mis_replace_rate']}（門檻 <2%）｜"
          f"(iii) 最壞序列 {b['iii_worst_seq']} {b['iii_worst_seq_delta']}（門檻 >−0.02）｜"
          f"跟丟幀平均 ΔIoU {b['mean_delta_lost_frames']}")
print(f"例外 {errs} 次")
