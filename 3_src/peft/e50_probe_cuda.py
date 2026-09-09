#!/usr/bin/env python3
"""E50 persist-K=8 獨立窗探針。判準見 E50_PERSIST_K8_DESIGN_20260821.md（執行前已寫死）。"""
import json, os, shutil, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

SP = Path(os.environ.get("E50_DIR", str(Path.home() / "e50")))
CFG = json.load(open(SP / "e50_frames.json"))
CKPT = SP / "ckpt" / "sam3.pt"
TAU = 0.30
AGREE = 0.50
K = 8
M = 1
DIAG_K = [5, 8, 12]


def iou(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0


from PIL import Image
from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor(checkpoint_path=str(CKPT))
print("[setup] predictor 就緒", flush=True)


def probe(frames_dir, first_name, tgt_name, box_xywh_rel, W, H):
    tmp = SP / "mini"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copy(frames_dir / first_name, tmp / "00000.jpg")
    shutil.copy(frames_dir / tgt_name, tmp / "00001.jpg")
    sid = predictor.start_session(str(tmp))
    sid = sid if isinstance(sid, str) else (sid.get("session_id") if isinstance(sid, dict) else str(sid))
    try:
        predictor.add_prompt(session_id=sid, frame_idx=0, bounding_boxes=[box_xywh_rel],
                             bounding_box_labels=[1], rel_coordinates=True)
        dets = []
        for out in predictor.propagate_in_video(session_id=sid, propagation_direction="forward",
                                                start_frame_idx=0, max_frame_num_to_track=1):
            fi = out.get("frame_index") if isinstance(out, dict) else getattr(out, "frame_index", None)
            if fi != 1:
                continue
            o = out.get("outputs") if isinstance(out, dict) else None
            if not o:
                continue
            bx, pr = o.get("out_boxes_xywh"), o.get("out_probs")
            if bx is None or len(bx) == 0:
                continue
            for k in range(len(bx)):
                x, y, w, h = [float(v) for v in bx[k]]
                dets.append(([x * W, y * H, w * W, h * H],
                             float(pr[k]) if pr is not None and k < len(pr) else 1.0))
        return dets
    finally:
        try:
            predictor.close_session(sid)
        except Exception:
            pass


def hysteresis_latch(raw, agree, k, m):
    """確認 K 幀後回溯拼接整段，直到 M 幀重新對齊。"""
    n = len(raw)
    latch = [False] * n
    state = False
    d_streak = 0
    a_streak = 0
    for t in range(n):
        if not state:
            d_streak = d_streak + 1 if raw[t] else 0
            if d_streak >= k:
                state = True
                for j in range(t - k + 1, t + 1):
                    latch[j] = True
                d_streak = 0
                a_streak = 0
        else:
            latch[t] = True
            a_streak = a_streak + 1 if agree[t] else 0
            if a_streak >= m:
                latch[t] = False
                state = False
                a_streak = 0
    return latch


def causal_persist(raw, k):
    n = len(raw)
    out = [False] * n
    for t in range(n):
        if t + 1 >= k and all(raw[t - k + 1: t + 1]):
            out[t] = True
    return out


records, errs = [], 0
for si, (seq, C) in enumerate(CFG["seqs"].items()):
    fdir = SP / "frames" / seq
    if not fdir.is_dir():
        print(f"[SKIP] {seq}: 無影格目錄", flush=True)
        continue
    first = C["first"]
    W, H = Image.open(fdir / "0001.jpg").size
    ib = C["init_box"]
    box_rel = [ib[0] / W, ib[1] / H, ib[2] / W, ib[3] / H]
    for win in C["windows"]:
        for gi, g in enumerate(win["frames"]):
            name = f"{g - first + 1:04d}.jpg"
            try:
                dets = sorted(probe(fdir, "0001.jpg", name, box_rel, W, H), key=lambda d: -d[1])
            except Exception as e:
                errs += 1
                if errs <= 5:
                    print(f"  EXC {seq}/{g}: {type(e).__name__} {str(e)[:140]}", flush=True)
                if errs >= 20:
                    print("🚨 例外 ≥20 次，中止")
                    sys.exit(3)
                continue
            gt, pred = C["gt"][str(g)], C["pred"][str(g)]
            det_box = dets[0][0] if dets else None
            records.append({
                "seq": seq, "group": C["group"], "kind": win["kind"],
                "window": win["id"], "frame": g, "idx_in_win": gi,
                "n_det": len(dets),
                "det_iou_gt": round(iou(det_box, gt), 4) if det_box else 0.0,
                "det_iou_pred": round(iou(det_box, pred), 4) if det_box else 0.0,
                "det_score": round(dets[0][1], 4) if dets else 0.0,
                "base_iou": round(iou(pred, gt), 4),
            })
        print(f"  {seq} {win['id']} {win['kind']} {len(win['frames'])} 幀完成", flush=True)
    done = sum(1 for r in records if r["seq"] == seq)
    print(f"[{si+1}/{len(CFG['seqs'])}] {seq}: {done} 幀完成", flush=True)


def apply_policy(rs, k=K, m=M, tau=TAU, agree=AGREE):
    # 必須按 (seq, window)。E49 只按 window 分組把五支混成一條時間線，機上 GO 作廢。
    by_win = {}
    for r in rs:
        by_win.setdefault((r["seq"], r["window"]), []).append(r)
    fire = {}
    for key, wr in by_win.items():
        wr = sorted(wr, key=lambda x: x["idx_in_win"])
        raw = [x["n_det"] > 0 and x["det_iou_pred"] < tau for x in wr]
        ag = [x["n_det"] > 0 and x["det_iou_pred"] >= agree for x in wr]
        latch = hysteresis_latch(raw, ag, k, m)
        for x, f in zip(wr, latch):
            fire[(x["seq"], x["frame"], x["window"])] = f
    return fire


def mean_delta(rs, fire):
    if not rs:
        return None
    acc = []
    for r in rs:
        if fire.get((r["seq"], r["frame"], r["window"])):
            acc.append(r["det_iou_gt"] - r["base_iou"])
        else:
            acc.append(0.0)
    return round(sum(acc) / len(acc), 4)


def rate(xs):
    return None if not xs else sum(xs) / len(xs)


good = [r for r in records if r["kind"] == "alive_good"]
wrong = [r for r in records if r["kind"] == "alive_wrong" and r["group"] == "lesion"]
ctrl_good = [r for r in records if r["seq"] == "vis-officefan2" and r["kind"] == "alive_good"]
fire = apply_policy(records)

raw_good = [r["n_det"] > 0 and r["det_iou_pred"] < TAU for r in good]
persist_good = [bool(fire.get((r["seq"], r["frame"], r["window"]))) for r in good]
raw_wrong = [r["n_det"] > 0 and r["det_iou_pred"] < TAU for r in wrong]
persist_wrong = [bool(fire.get((r["seq"], r["frame"], r["window"]))) for r in wrong]

spliced_wrong = [r for r, f in zip(wrong, persist_wrong) if f]
i_rate = rate(persist_good)
ib_d = mean_delta(good, fire)
ii_hit = (sum(1 for r in spliced_wrong if r["det_iou_gt"] >= 0.5) / len(spliced_wrong)
          if spliced_wrong else None)
iib_cov = rate(persist_wrong)
ctrl_d = mean_delta(ctrl_good, fire)

go_i = i_rate is not None and i_rate < 0.05
go_ib = ib_d is not None and ib_d > -0.02
go_ii = ii_hit is not None and ii_hit >= 0.40
go_iib = iib_cov is not None and iib_cov >= 0.30
go_iii = ctrl_d is not None and ctrl_d > -0.02
verdict = "GO" if (go_i and go_ib and go_ii and go_iib and go_iii) else (
    "KILL" if (not go_i or not go_ib) else "GREY")

per_seq = {}
for seq in CFG["seqs"]:
    rs = [r for r in records if r["seq"] == seq]
    gs = [r for r in rs if r["kind"] == "alive_good"]
    ws = [r for r in rs if r["kind"] == "alive_wrong"]
    pf_g = [bool(fire.get((r["seq"], r["frame"], r["window"]))) for r in gs]
    pf_w = [bool(fire.get((r["seq"], r["frame"], r["window"]))) for r in ws]
    sw = [r for r, f in zip(ws, pf_w) if f]
    per_seq[seq] = {
        "n_good": len(gs), "n_wrong": len(ws),
        "raw_fire_good": rate([r["n_det"] > 0 and r["det_iou_pred"] < TAU for r in gs]),
        "persist_fire_good": rate(pf_g),
        "good_delta": mean_delta(gs, fire),
        "raw_fire_wrong": rate([r["n_det"] > 0 and r["det_iou_pred"] < TAU for r in ws]),
        "persist_cov_wrong": rate(pf_w),
        "persist_hit_wrong": (sum(1 for r in sw if r["det_iou_gt"] >= 0.5) / len(sw) if sw else None),
        "wrong_delta": mean_delta(ws, fire),
    }

by_k = {}
for kk in DIAG_K:
    fk = apply_policy(records, k=kk, m=M)
    gs_f = [bool(fk.get((r["seq"], r["frame"], r["window"]))) for r in good]
    ws_f = [bool(fk.get((r["seq"], r["frame"], r["window"]))) for r in wrong]
    sw = [r for r, f in zip(wrong, ws_f) if f]
    by_k[str(kk)] = {
        "good_fire": rate(gs_f),
        "good_delta": mean_delta(good, fk),
        "wrong_cov": rate(ws_f),
        "wrong_hit": (sum(1 for r in sw if r["det_iou_gt"] >= 0.5) / len(sw) if sw else None),
        "ctrl_delta": mean_delta(ctrl_good, fk),
    }

causal = {}
by_win_c = {}
for r in records:
    by_win_c.setdefault((r["seq"], r["window"]), []).append(r)
cfire = {}
for key, wr in by_win_c.items():
    wr = sorted(wr, key=lambda x: x["idx_in_win"])
    raw = [x["n_det"] > 0 and x["det_iou_pred"] < TAU for x in wr]
    cp = causal_persist(raw, K)
    for x, f in zip(wr, cp):
        cfire[(x["seq"], x["frame"], x["window"])] = f
causal = {
    "good_fire": rate([bool(cfire.get((r["seq"], r["frame"], r["window"]))) for r in good]),
    "good_delta": mean_delta(good, cfire),
    "wrong_cov": rate([bool(cfire.get((r["seq"], r["frame"], r["window"]))) for r in wrong]),
    "ctrl_delta": mean_delta(ctrl_good, cfire),
}

res = {
    "n_records": len(records), "n_exc": errs,
    "policy": {"tau": TAU, "agree": AGREE, "K": K, "M": M, "mode": "hysteresis_run_splice"},
    "i_persist_fire_good": i_rate,
    "i_b_good_mean_delta": ib_d,
    "ii_hit_spliced_wrong": ii_hit,
    "ii_b_cov_wrong": iib_cov,
    "iii_ctrl_mean_delta": ctrl_d,
    "raw_fire_good": rate(raw_good),
    "raw_fire_wrong": rate(raw_wrong),
    "gates": {"i": go_i, "i_b": go_ib, "ii": go_ii, "ii_b": go_iib, "iii": go_iii},
    "verdict": verdict,
    "per_seq": per_seq,
    "by_k": by_k,
    "causal_k3": causal,
    "n_good": len(good), "n_wrong": len(wrong), "n_ctrl_good": len(ctrl_good),
    "n_spliced_wrong": len(spliced_wrong),
    "records": records,
}

json.dump(res, open(SP / "e50_result.json", "w"), indent=1, ensure_ascii=False)

print(f"\n{'='*64}")
print(f"(i)    GOOD persist 開火 {i_rate}（門檻 <0.05） {'OK' if go_i else 'FAIL'}")
print(f"(i-b)  GOOD Δ {ib_d}（門檻 >−0.02） {'OK' if go_ib else 'FAIL'}")
print(f"(ii)   WRONG 拼接命中 {ii_hit}（門檻 ≥0.40） n_spliced={len(spliced_wrong)} {'OK' if go_ii else 'FAIL'}")
print(f"(ii-b) WRONG 覆蓋 {iib_cov}（門檻 ≥0.30） {'OK' if go_iib else 'FAIL'}")
print(f"(iii)  officefan2 Δ {ctrl_d}（門檻 >−0.02） {'OK' if go_iii else 'FAIL'}")
print(f"raw GOOD 開火 {rate(raw_good)}  raw WRONG 開火 {rate(raw_wrong)}")
print(f"VERDICT {verdict}  例外 {errs}")
if verdict != "GO":
    sys.exit(0)
