#!/usr/bin/env python3
"""E48 分歧閘探針。判準見 E48_DISAGREE_GATE_DESIGN_20260821.md（執行前已寫死）。"""
import json, os, shutil, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

SP = Path(os.environ.get("E48_DIR", str(Path.home() / "e48")))
CFG = json.load(open(SP / "e48_frames.json"))
CKPT = SP / "ckpt" / "sam3.pt"
TAU = 0.30
DIAG_TAUS = [0.20, 0.30, 0.50]


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
    for kind in ("alive_wrong", "alive_good"):
        for g in C.get(kind, []):
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
                "seq": seq, "group": C["group"], "kind": kind, "frame": g,
                "n_det": len(dets),
                "det_iou_gt": round(iou(det_box, gt), 4) if det_box else 0.0,
                "det_iou_pred": round(iou(det_box, pred), 4) if det_box else 0.0,
                "det_score": round(dets[0][1], 4) if dets else 0.0,
                "base_iou": round(iou(pred, gt), 4),
            })
    done = sum(1 for r in records if r["seq"] == seq)
    print(f"[{si+1}/{len(CFG['seqs'])}] {seq}: {done} 幀完成", flush=True)


def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


good = [r for r in records if r["kind"] == "alive_good"]
wrong = [r for r in records if r["kind"] == "alive_wrong" and r["group"] == "lesion"]
ctrl_good = [r for r in records if r["seq"] == "vis-officefan2" and r["kind"] == "alive_good"]
iab_good = [r["det_iou_pred"] for r in good if r["n_det"] > 0]

def fire(r, tau):
    return r["n_det"] > 0 and r["det_iou_pred"] < tau

def seq_mean_delta(rs, tau):
    d = {}
    for r in rs:
        delta = (r["det_iou_gt"] - r["base_iou"]) if fire(r, tau) else 0.0
        d.setdefault(r["seq"], []).append(delta)
    return {s: round(sum(v) / len(v), 4) for s, v in d.items()}

i_med = median(iab_good)
i_b = (sum(1 for x in iab_good if x < TAU) / len(iab_good)) if iab_good else None
ii_hit = (sum(1 for r in wrong if r["det_iou_gt"] >= 0.5) / len(wrong)) if wrong else None
ctrl_d = seq_mean_delta(ctrl_good, TAU).get("vis-officefan2")

go_i = i_med is not None and i_med >= 0.50
go_ib = i_b is not None and i_b < 0.15
go_ii = ii_hit is not None and ii_hit >= 0.40
go_iii = ctrl_d is not None and ctrl_d > -0.02
verdict = "GO" if (go_i and go_ib and go_ii and go_iii) else (
    "KILL" if (not go_i or not go_ib) else "GREY")

res = {
    "n_records": len(records), "n_exc": errs, "tau": TAU,
    "i_median_iou_det_pred_on_good": i_med,
    "i_b_disagree_rate_on_good": i_b,
    "ii_hit_rate_wrong": ii_hit,
    "ii_detect_rate_wrong": (sum(1 for r in wrong if r["n_det"] > 0) / len(wrong)) if wrong else None,
    "iii_ctrl_mean_delta": ctrl_d,
    "gates": {"i": go_i, "i_b": go_ib, "ii": go_ii, "iii": go_iii},
    "verdict": verdict,
    "by_tau": {},
    "per_seq": {},
    "n_good": len(good), "n_wrong": len(wrong), "n_ctrl_good": len(ctrl_good),
    "records": records,
}
for tau in DIAG_TAUS:
    sd = seq_mean_delta(records, tau)
    res["by_tau"][str(tau)] = {
        "n_fire": sum(1 for r in records if fire(r, tau)),
        "ctrl_delta": seq_mean_delta(ctrl_good, tau).get("vis-officefan2"),
        "lesion_wrong_mean_delta_if_fire": (
            round(sum(r["det_iou_gt"] - r["base_iou"] for r in wrong if fire(r, tau)) /
                  max(1, sum(1 for r in wrong if fire(r, tau))), 4)
            if any(fire(r, tau) for r in wrong) else None),
        "per_seq_delta": sd,
    }
for seq in CFG["seqs"]:
    rs = [r for r in records if r["seq"] == seq]
    res["per_seq"][seq] = {
        "n": len(rs),
        "good_med_iab": median([r["det_iou_pred"] for r in rs if r["kind"] == "alive_good" and r["n_det"] > 0]),
        "wrong_hit": (
            sum(1 for r in rs if r["kind"] == "alive_wrong" and r["det_iou_gt"] >= 0.5) /
            max(1, sum(1 for r in rs if r["kind"] == "alive_wrong"))),
    }

json.dump(res, open(SP / "e48_result.json", "w"), indent=1, ensure_ascii=False)

print(f"\n{'='*64}")
print(f"(i)   alive_good n={len(good)}  IoU(det,A) 中位 {i_med}（門檻 ≥0.50） {'OK' if go_i else 'FAIL'}")
print(f"(i-b) alive_good 分歧<0.30 比例 {i_b}（門檻 <0.15） {'OK' if go_ib else 'FAIL'}")
print(f"(ii)  alive_wrong n={len(wrong)}  det vs GT≥0.5 命中 {ii_hit}（門檻 ≥0.40） {'OK' if go_ii else 'FAIL'}")
print(f"(iii) officefan2 Δ {ctrl_d}（門檻 >−0.02） {'OK' if go_iii else 'FAIL'}")
print(f"VERDICT {verdict}  例外 {errs}")
if verdict != "GO":
    sys.exit(0)  # 非崩潰；判死也是合法完成
