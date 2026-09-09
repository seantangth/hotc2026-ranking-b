#!/usr/bin/env python3
"""E51 同意才更新 exemplar。判準見 E51_UPDATE_ON_AGREE_DESIGN_20260821.md。"""
import json, os, shutil, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

SP = Path(os.environ.get("E51_DIR", str(Path.home() / "e51")))
CFG = json.load(open(SP / "e51_frames.json"))
CKPT = SP / "ckpt" / "sam3.pt"
TAU = 0.30
AGREE = 0.50


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


def probe(frames_dir, src_name, tgt_name, box_xywh_rel, W, H):
    tmp = SP / "mini"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copy(frames_dir / src_name, tmp / "00000.jpg")
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
n_updates = {}
for si, (seq, C) in enumerate(CFG["seqs"].items()):
    fdir = SP / "frames" / seq
    if not fdir.is_dir():
        print(f"[SKIP] {seq}: 無影格目錄", flush=True)
        continue
    first = C["first"]
    W, H = Image.open(fdir / "0001.jpg").size
    ib = C["init_box"]
    ex_name = "0001.jpg"
    ex_rel = [ib[0] / W, ib[1] / H, ib[2] / W, ib[3] / H]
    n_upd = 0
    win = C["windows"][0]
    for gi, g in enumerate(win["frames"]):
        name = f"{g - first + 1:04d}.jpg"
        try:
            dets = sorted(probe(fdir, ex_name, name, ex_rel, W, H), key=lambda d: -d[1])
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
        iab = iou(det_box, pred) if det_box else 0.0
        updated = False
        if det_box is not None and iab >= AGREE:
            ex_name = name
            ex_rel = [pred[0] / W, pred[1] / H, pred[2] / W, pred[3] / H]
            updated = True
            n_upd += 1
        records.append({
            "seq": seq, "group": C["group"],
            "kind": C["kind"][str(g)],
            "window": win["id"], "frame": g, "idx_in_win": gi,
            "n_det": len(dets),
            "det_iou_gt": round(iou(det_box, gt), 4) if det_box else 0.0,
            "det_iou_pred": round(iab, 4) if det_box else 0.0,
            "det_score": round(dets[0][1], 4) if dets else 0.0,
            "base_iou": round(iou(pred, gt), 4),
            "updated": updated,
            "ex_age": gi if not updated else 0,
        })
        if (gi + 1) % 80 == 0:
            print(f"  {seq} {gi+1}/{len(win['frames'])}", flush=True)
    n_updates[seq] = n_upd
    print(f"[{si+1}/{len(CFG['seqs'])}] {seq}: {sum(1 for r in records if r['seq']==seq)} 幀  updates={n_upd}", flush=True)


def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def rate(xs):
    return None if not xs else sum(xs) / len(xs)


def fire(r):
    return r["n_det"] > 0 and r["det_iou_pred"] < TAU


good = [r for r in records if r["kind"] == "alive_good"]
wrong = [r for r in records if r["kind"] == "alive_wrong" and r["group"] == "lesion" and r["seq"] != "nir-pingpong"]
ctrl = [r for r in records if r["seq"] == "vis-officefan2"]
iab_good = [r["det_iou_pred"] for r in good if r["n_det"] > 0]

def mean_delta(rs):
    if not rs:
        return None
    acc = [(r["det_iou_gt"] - r["base_iou"]) if fire(r) else 0.0 for r in rs]
    return round(sum(acc) / len(acc), 4)

i_med = median(iab_good)
i_b = rate([r["det_iou_pred"] < TAU for r in good if r["n_det"] > 0])
ii_hit = rate([r["det_iou_gt"] >= 0.5 for r in wrong]) if wrong else None
ctrl_d = mean_delta(ctrl)

go_i = i_med is not None and i_med >= 0.50
go_ib = i_b is not None and i_b < 0.15
go_ii = ii_hit is not None and ii_hit >= 0.40
go_iii = ctrl_d is not None and ctrl_d > -0.02
verdict = "GO" if (go_i and go_ib and go_ii and go_iii) else (
    "KILL" if (not go_i or not go_ib) else "GREY")

per_seq = {}
for seq in CFG["seqs"]:
    rs = [r for r in records if r["seq"] == seq]
    gs = [r for r in rs if r["kind"] == "alive_good"]
    ws = [r for r in rs if r["kind"] == "alive_wrong"]
    per_seq[seq] = {
        "n": len(rs), "n_good": len(gs), "n_wrong": len(ws),
        "n_updates": n_updates.get(seq, 0),
        "good_med_iab": median([r["det_iou_pred"] for r in gs if r["n_det"] > 0]),
        "good_fire": rate([fire(r) for r in gs]),
        "good_delta": mean_delta(gs),
        "wrong_hit": rate([r["det_iou_gt"] >= 0.5 for r in ws]) if ws else None,
        "wrong_fire": rate([fire(r) for r in ws]) if ws else None,
        "wrong_delta": mean_delta(ws),
    }

# 診斷切片：E50 pingpong 殺手區、E49 yo_yo 誤火區
def slice_fire(seq, lo, hi, kind=None):
    rs = [r for r in records if r["seq"] == seq and lo <= r["frame"] <= hi]
    if kind:
        rs = [r for r in rs if r["kind"] == kind]
    return {
        "n": len(rs),
        "fire": rate([fire(r) for r in rs]) if rs else None,
        "delta": mean_delta(rs),
    }

diag = {
    "pingpong_e50_killzone": slice_fire("nir-pingpong", 12274, 12337, "alive_good"),
    "yoyo_e49_good": slice_fire("nir-yo_yo", 22131, 22194, "alive_good"),
}

res = {
    "n_records": len(records), "n_exc": errs,
    "policy": {"tau": TAU, "agree": AGREE, "mode": "update_on_agree_then_perframe_splice"},
    "i_median_iou_det_pred_on_good": i_med,
    "i_b_disagree_rate_on_good": i_b,
    "ii_hit_rate_wrong": ii_hit,
    "iii_ctrl_mean_delta": ctrl_d,
    "gates": {"i": go_i, "i_b": go_ib, "ii": go_ii, "iii": go_iii},
    "verdict": verdict,
    "n_updates": n_updates,
    "per_seq": per_seq,
    "diag_slices": diag,
    "n_good": len(good), "n_wrong": len(wrong), "n_ctrl": len(ctrl),
    "records": records,
}
json.dump(res, open(SP / "e51_result.json", "w"), indent=1, ensure_ascii=False)

print(f"\n{'='*64}")
print(f"(i)    GOOD IoU(det,A) 中位 {i_med}（≥0.50） {'OK' if go_i else 'FAIL'}")
print(f"(i-b)  GOOD 開火 {i_b}（<0.15） {'OK' if go_ib else 'FAIL'}")
print(f"(ii)   WRONG 命中 {ii_hit}（≥0.40） n_wrong={len(wrong)} {'OK' if go_ii else 'FAIL'}")
print(f"(iii)  officefan2 Δ {ctrl_d}（>−0.02） {'OK' if go_iii else 'FAIL'}")
print("diag", diag)
print(f"VERDICT {verdict}  例外 {errs}")
if verdict != "GO":
    sys.exit(0)
