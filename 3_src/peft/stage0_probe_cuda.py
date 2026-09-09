#!/usr/bin/env python3
"""E32 Stage 0 探針——判準見 5_outputs/strategy_research_20260812/E32_STAGE0_DESIGN_20260813.md
（該檔於本腳本執行前寫入）。Lambda CUDA（本機 MPS 不可行：SAM3 上游 30+ 處硬編 cuda）、零 LB 額度。"""
import json, shutil, sys, warnings, traceback
from pathlib import Path
warnings.filterwarnings("ignore")

import os
SP = Path(os.environ.get("E32_DIR", str(Path.home()/"e32")))
CFG = json.load(open(SP / "stage0_frames.json"))
SEQ, FIRST = CFG["seq"], CFG["first_global"]
FRAMES = SP / "frames" / SEQ
CKPT = SP / "ckpt" / "sam3.pt"
DEV = "cuda"

def local_name(g):            # 全域幀號 → 本機檔名（首幀＝0001.jpg）
    return f"{g - FIRST + 1:04d}.jpg"

def iou(a, b):
    ax2, ay2, bx2, by2 = a[0]+a[2], a[1]+a[3], b[0]+b[2], b[1]+b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0])); ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw*ih; u = a[2]*a[3] + b[2]*b[3] - inter
    return inter/u if u > 0 else 0.0

def to_xywh(box, W, H, rel):
    x1, y1, x2, y2 = [float(v) for v in box]
    if rel: x1, x2, y1, y2 = x1*W, x2*W, y1*H, y2*H
    return [x1, y1, x2-x1, y2-y1]

import torch
from PIL import Image
from sam3.model_builder import build_sam3_video_predictor

W, H = Image.open(FRAMES / local_name(FIRST)).size
print(f"[setup] {SEQ} {W}x{H}｜首幀 {local_name(FIRST)}｜init box {CFG['init_box']}", flush=True)

predictor = build_sam3_video_predictor(checkpoint_path=str(CKPT))
print("[setup] predictor 建構完成", flush=True)

ib = CFG["init_box"]
box_xywh_rel = [ib[0]/W, ib[1]/H, ib[2]/W, ib[3]/H]   # 上游要 xywh 正規化，非 xyxy

def probe(g):
    """對單一目標幀組 2 幀迷你 session，回傳該幀的偵測清單（xywh, score）。"""
    tmp = SP / "mini"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    shutil.copy(FRAMES / local_name(FIRST), tmp / "00000.jpg")   # 幀0＝exemplar 來源
    shutil.copy(FRAMES / local_name(g),     tmp / "00001.jpg")   # 幀1＝受測幀
    sid = predictor.start_session(str(tmp))
    sid = sid if isinstance(sid, str) else (sid.get("session_id") if isinstance(sid, dict) else str(sid))
    try:
        predictor.add_prompt(session_id=sid, frame_idx=0, bounding_boxes=[box_xywh_rel],
                             bounding_box_labels=[1], rel_coordinates=True)
        dets = []
        for out in predictor.propagate_in_video(session_id=sid, propagation_direction="forward",
                                                start_frame_idx=0, max_frame_num_to_track=1):
            fi = out.get("frame_index", out.get("frame_idx")) if isinstance(out, dict) else getattr(out, "frame_index", None)
            if fi != 1:
                continue
            o = out.get("outputs") if isinstance(out, dict) else None
            if RAW["struct"] is None:
                RAW["struct"] = "None" if o is None else str(sorted(o.keys()))
            if o is None:                      # 該幀無輸出＝零偵測（合法情形，非錯誤）
                continue
            bx = o.get("out_boxes_xywh")       # 正規化 xywh（上游已除以 W_video/H_video）
            pr = o.get("out_probs")
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

RAW = {"struct": None}
res = {"lost": [], "good": []}
for grp in ("lost", "good"):
    for g in CFG[grp]:
        gt = CFG["gt"][str(g)]
        try:
            dets = probe(g)
            dets.sort(key=lambda d: -d[1])
            top1 = iou(dets[0][0], gt) if dets else 0.0
            best = max((iou(d[0], gt) for d in dets), default=0.0)
            res[grp].append({"frame": g, "n_det": len(dets), "top1_iou": round(top1, 4),
                             "best_iou": round(best, 4)})
            print(f"  [{grp}] {g}: 偵測 {len(dets)}｜top1 IoU {top1:.3f}｜best {best:.3f}", flush=True)
        except Exception as e:
            print(f"  [{grp}] {g}: 🚨 EXC {type(e).__name__}: {str(e)[:160]}", flush=True)
            res[grp].append({"frame": g, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            if len([r for r in res[grp] if "error" in r]) >= 2 and len(res[grp]) <= 2:
                traceback.print_exc(); print("前兩幀皆例外 ⇒ 判為可跑性問題，中止"); sys.exit(3)

ok = [r for r in res["lost"] if "error" not in r]
det_rate = sum(r["n_det"] > 0 for r in ok) / len(ok) if ok else 0.0
hit40 = sum(r["top1_iou"] >= 0.5 for r in ok) / len(ok) if ok else 0.0
gk = [r for r in res["good"] if "error" not in r]
ctrl_rate = sum(r["n_det"] > 0 for r in gk) / len(gk) if gk else 0.0
ctrl_hit = sum(r["top1_iou"] >= 0.5 for r in gk) / len(gk) if gk else 0.0

if det_rate < 0.30:      verdict, why = "KILL", f"跟丟幀偵測率 {det_rate:.0%} < 30% ⇒ SAM3.1 3/98 警訊重現"
elif hit40 >= 0.40:      verdict, why = "GO",   f"top-1 IoU≥0.5 命中率 {hit40:.0%} ≥ 40%"
else:                    verdict, why = "GREY", f"偵測率 {det_rate:.0%}（≥30%）但命中率 {hit40:.0%} < 40%"

out = {"seq": SEQ, "device": DEV, "n_lost": len(ok), "n_ctrl": len(gk),
       "lost_detect_rate": det_rate, "lost_top1_hit@0.5": hit40,
       "ctrl_detect_rate": ctrl_rate, "ctrl_top1_hit@0.5": ctrl_hit,
       "verdict": verdict, "why": why, "detail": res, "output_struct": RAW["struct"]}
json.dump(out, open(SP / "stage0_result.json", "w"), indent=1, ensure_ascii=False)
print(f"\n{'='*60}\n跟丟幀 n={len(ok)}：偵測率 {det_rate:.0%}｜top-1 IoU≥0.5 {hit40:.0%}")
print(f"對照幀 n={len(gk)}：偵測率 {ctrl_rate:.0%}｜top-1 IoU≥0.5 {ctrl_hit:.0%}")
print(f"【判決】{verdict} — {why}")
