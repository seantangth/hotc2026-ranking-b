#!/usr/bin/env python3
"""8/19 兩發預備：同一組 v056 候選（RedNIR ∧ iab<0.30 ∧ 雙方活）上換頭。

v061  DINOv2 外觀頭（模板＝各序列第 1 幀主線框）
v062  ViPT orig 第三者投票（test_rednir_orig + 同校正）

不認序列名；閘門與 v056 相同。今日不送 Kaggle。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "3_src"))
from hsot.quality_head_v1 import (  # noqa: E402
    apply_corr,
    feats,
    frozen_runs,
    iou,
    load_csv,
    modality,
    predict_p,
    validate,
    write_sub,
)

SPLICE_K = 6
SRC_MAX_RUN = 2
IAB_MAX = 0.30
SAMPLE = ROOT / "1_data/raw/sample_submisson.csv"
OUT = ROOT / "5_outputs" / "e43_heads_20260818"
SUBS = ROOT / "5_outputs" / "submissions"
TAR = ROOT / "1_data/packed/t1test_fc_75.tar"
WEIGHTS = ROOT / "3_src/hsot/qhead_weights_v056.npz"


def v056_pool(A, B, base):
    ra = {s: frozen_runs(A[s]) for s in A}
    rb = {s: frozen_runs(B[s]) for s in B}
    z = np.load(WEIGHTS)
    w, mu, sd = z["w"], z["mu"], z["sd"]
    cand = []
    for s in base:
        if s not in A or s not in B:
            continue
        ks = sorted(set(base[s]) & set(A[s]) & set(B[s]))
        nfr = max(len(ks), 1)
        prev_a = prev_b = None
        for i, f in enumerate(ks):
            a, b = A[s][f], B[s][f]
            if modality(s) != 1:
                prev_a, prev_b = a, b
                continue
            if ra[s][f] >= SPLICE_K or rb[s][f] >= SRC_MAX_RUN:
                prev_a, prev_b = a, b
                continue
            iab = float(iou(a, b))
            if iab >= IAB_MAX:
                prev_a, prev_b = a, b
                continue
            x = feats(a, b, prev_a, prev_b, ra[s][f], rb[s][f], i / nfr, 1)
            p = float(predict_p(x[None, :], w, mu, sd)[0])
            cand.append(
                {
                    "id": f"{s}_{f}",
                    "seq": s,
                    "frame": int(f),
                    "iab": iab,
                    "pA": p,
                    "logit_pick_b": bool(p < 0.5),
                    "a": [float(v) for v in a],
                    "b": [float(v) for v in b],
                }
            )
            prev_a, prev_b = a, b
    return cand


def apply_picks(base, order, cand, pick_b_ids, dest):
    outb = {s: {f: v.copy() for f, v in fm.items()} for s, fm in base.items()}
    by_id = {c["id"]: c for c in cand}
    n = 0
    for i in pick_b_ids:
        c = by_id[i]
        outb[c["seq"]][c["frame"]] = np.array(c["b"], dtype=np.float64)
        n += 1
    write_sub(order, outb, dest)
    validate(SAMPLE, dest)
    return n


def extract_needed(cand, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    want = set()
    for c in cand:
        want.add(f"{c['seq']}/{c['frame']:04d}.jpg")
        want.add(f"{c['seq']}/0001.jpg")
    copied = 0
    with tarfile.open(TAR, "r") as tf:
        for m in tf.getmembers():
            name = m.name.lstrip("./")
            if name not in want:
                continue
            src = tf.extractfile(m)
            if src is None:
                continue
            outp = dest / name
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_bytes(src.read())
            copied += 1
    missing = sorted(want - {str(p.relative_to(dest)) for p in dest.rglob("*.jpg")})
    return {"copied": copied, "want": len(want), "missing": missing[:20], "n_missing": len(missing)}


def crop_rgb(img, box, pad=0.1):
    """box = xywh, expand pad, clamp, return RGB uint8 crop (may be tiny)."""
    h, w = img.shape[:2]
    x, y, bw, bh = box
    x0 = int(np.floor(x - pad * bw))
    y0 = int(np.floor(y - pad * bh))
    x1 = int(np.ceil(x + bw + pad * bw))
    y1 = int(np.ceil(y + bh + pad * bh))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, h if False else y1)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]


def run_dinov2(cand, frames_root: Path, device: str = "cpu"):
    import torch
    from PIL import Image
    from torchvision import transforms

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    model.eval().to(device)
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    def embed(arr):
        if arr is None or arr.size == 0:
            return None
        im = Image.fromarray(arr).convert("RGB")
        x = tfm(im).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model(x)
        f = torch.nn.functional.normalize(f, dim=-1)
        return f.squeeze(0).cpu()

    # templates from frame 1, box A
    tmpl = {}
    from PIL import Image as PILImage

    for s in sorted({c["seq"] for c in cand}):
        p = frames_root / s / "0001.jpg"
        if not p.is_file():
            continue
        img = np.array(PILImage.open(p).convert("RGB"))
        a1 = next(c["a"] for c in cand if c["seq"] == s)  # dummy
        # use provided first-frame A from sidecar if present
        tpath = frames_root / f"{s}_tmpl_box.json"
        if tpath.is_file():
            box = json.loads(tpath.read_text())
        else:
            # fall back: any cand of this seq does not have frame1 A; load from sidecar A json
            box = json.loads((frames_root / "frame1_boxes.json").read_text())[s]
        tmpl[s] = embed(crop_rgb(img, box))

    picks = []
    detail = []
    for c in cand:
        p = frames_root / c["seq"] / f"{c['frame']:04d}.jpg"
        img = np.array(PILImage.open(p).convert("RGB")) if p.is_file() else None
        ea = embed(crop_rgb(img, c["a"])) if img is not None else None
        eb = embed(crop_rgb(img, c["b"])) if img is not None else None
        t = tmpl.get(c["seq"])
        if t is None or ea is None or eb is None:
            pick_b = c["logit_pick_b"]  # fail closed: keep logistic
            sa = sb = None
        else:
            sa = float((t * ea).sum())
            sb = float((t * eb).sum())
            pick_b = sb > sa
        if pick_b:
            picks.append(c["id"])
        detail.append({"id": c["id"], "cosA": sa, "cosB": sb, "pick_b": pick_b})
    return picks, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("pool", "vipt", "extract", "dinov2", "all"), default="all")
    ap.add_argument("--frames", default=str(OUT / "fc_crops"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    A, _ = load_csv(ROOT / "5_outputs/submissions/sub_v029_v027_left1px.csv")
    B = apply_corr(load_csv(ROOT / "5_outputs/submissions/sub_v012_e23b_sam21_ablation.csv")[0])
    base, order = load_csv(ROOT / "5_outputs/submissions/sub_v049_src_v012_K6.csv")
    cand = v056_pool(A, B, base)
    (OUT / "v056_pool.json").write_text(json.dumps(cand, indent=2))
    print(f"pool {len(cand)}  logitB {sum(c['logit_pick_b'] for c in cand)}")

    # frame-1 A boxes for DINOv2 template
    f1 = {s: [float(v) for v in A[s][1]] for s in {c["seq"] for c in cand} if s in A and 1 in A[s]}
    (OUT / "frame1_boxes.json").write_text(json.dumps(f1, indent=2))

    if args.stage in ("vipt", "all"):
        Craw, _ = load_csv(ROOT / "5_outputs/t2_bench/test_rednir_orig.csv")
        C = apply_corr(Craw)
        picks, n_miss = [], 0
        for c in cand:
            s, f = c["seq"], c["frame"]
            if s not in C or f not in C[s]:
                n_miss += 1
                continue
            iac, ibc = iou(np.array(c["a"]), C[s][f]), iou(np.array(c["b"]), C[s][f])
            c["iac"], c["ibc"] = float(iac), float(ibc)
            if ibc > iac:
                picks.append(c["id"])
        dest = OUT / "sub_v062_vipt_vote.csv"
        n = apply_picks(base, order, cand, picks, dest)
        dest2 = SUBS / "sub_v062_vipt_vote.csv"
        dest2.write_bytes(dest.read_bytes())
        agree = sum(1 for c in cand if c["logit_pick_b"] == (c["id"] in set(picks)))
        both_b = sum(1 for c in cand if c["logit_pick_b"] and c["id"] in set(picks))
        stats = {
            "replaced": n,
            "cand": len(cand),
            "c_missing": n_miss,
            "agree_logit": agree,
            "both_pick_b": both_b,
            "overlap_vs_v056": both_b,
            "seqs": dict(Counter(i.rsplit("_", 1)[0] for i in picks)),
        }
        (OUT / "v062_vipt_stats.json").write_text(json.dumps(stats, indent=2))
        print("v062", json.dumps(stats))

    if args.stage in ("extract", "all", "dinov2"):
        frames = Path(args.frames)
        info = extract_needed(cand, frames)
        (frames / "frame1_boxes.json").write_text(json.dumps(f1, indent=2))
        (frames / "v056_pool.json").write_text(json.dumps(cand))
        print("extract", info)
        if info["n_missing"]:
            print("MISSING", info["missing"], file=sys.stderr)

    if args.stage in ("dinov2",):
        frames = Path(args.frames)
        picks, detail = run_dinov2(cand, frames, device=args.device)
        dest = OUT / "sub_v061_dinov2.csv"
        n = apply_picks(base, order, cand, picks, dest)
        (SUBS / "sub_v061_dinov2.csv").write_bytes(dest.read_bytes())
        (OUT / "v061_dinov2_detail.json").write_text(json.dumps(detail, indent=2))
        agree = sum(1 for c in cand if c["logit_pick_b"] == (c["id"] in set(picks)))
        stats = {
            "replaced": n,
            "cand": len(cand),
            "agree_logit": agree,
            "seqs": dict(Counter(i.rsplit("_", 1)[0] for i in picks)),
        }
        (OUT / "v061_dinov2_stats.json").write_text(json.dumps(stats, indent=2))
        print("v061", json.dumps(stats))


if __name__ == "__main__":
    main()
