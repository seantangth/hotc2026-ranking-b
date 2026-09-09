#!/usr/bin/env python3
"""D079 Gate 0: mine failure-centred training clips from the 405 prediction cache.

Why this exists
---------------
D068's post-mortem named two causes for why three training attempts produced a
loss that never moved:

  1. 94-98% of clips were self-distillation identities (input == label), so
     there was nothing to learn;
  2. 30px targets made focal 20:1:1 degrade to roughly 0:1:1.

Cause (1) is a *sampling* problem, and it is fixable for free: the 405 cache
already tells us exactly where the tracker fails.  Rather than sampling clips
uniformly, this centres them on measured failure events -- frames where the
primary froze, diverged from GT, or lost the target outright.

No GPU, no new inference: every input here was produced by the 08-27 run.

Outputs a clip index plus the identity-rate statistic that Gate 0 is judged on:
**identity clips must be < 50%** (the old recipe was 94-98%; that rate *is* the
disease).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
import oof405_crossfit as oof  # noqa: E402

# A frame is "failing" when the primary is far enough off GT that a corrective
# gradient exists.  0.5 is the standard success-plot midpoint, not a tuned knob.
FAIL_IOU = 0.50
# Clip length in frames; centred on the failure event.
CLIP_LEN = 16
# Two clips from the same sequence must not overlap by more than this.
MAX_OVERLAP = 4


def frozen_runs(boxes: np.ndarray, sequence_indices) -> np.ndarray:
    return oof._frozen_runs(boxes, sequence_indices, tolerant=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", required=True)
    ap.add_argument("--main", required=True, help="primary predictions (the leg we would train)")
    ap.add_argument("--source", required=True)
    ap.add_argument("--main-diagnostics")
    ap.add_argument("--source-diagnostics")
    ap.add_argument("--pair-profile", default="rankB_robust", choices=sorted(oof.PAIR_SCOPES))
    ap.add_argument("--main-crop-diagnostics")
    ap.add_argument("--source-crop-diagnostics")
    ap.add_argument("--crop-meta")
    ap.add_argument("--modalities", default="vis,rednir",
                    help="D079's untested range is VIS/RedNIR; NIR is the already-refuted pool")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    ds = oof.load_dataset(
        args.contracts, args.main, args.source,
        args.main_diagnostics, args.source_diagnostics,
        pair_profile=args.pair_profile,
        main_crop_diagnostics=args.main_crop_diagnostics,
        source_crop_diagnostics=args.source_crop_diagnostics,
        crop_meta=args.crop_meta,
    )
    keep_mod = {m.strip() for m in args.modalities.split(",") if m.strip()}

    iou = oof._ious(ds.raw_main, ds.gt)
    valid = np.all(ds.gt > 0, axis=1)
    runs = frozen_runs(ds.raw_main, ds.sequence_indices)

    clips, per_seq, stats = [], {}, {
        "frames_total": 0, "frames_valid_gt": 0, "frames_failing": 0, "frames_frozen": 0}

    for seq, idx in sorted(ds.sequence_indices.items()):
        modality = seq.split("-", 1)[0]
        stats["frames_total"] += len(idx)
        if modality not in keep_mod:
            continue
        v = valid[idx]
        stats["frames_valid_gt"] += int(v.sum())
        seq_iou, seq_run = iou[idx], runs[idx]
        failing = v & (seq_iou < FAIL_IOU)
        frozen = seq_run >= 3
        stats["frames_failing"] += int(failing.sum())
        stats["frames_frozen"] += int((frozen & v).sum())

        # Event = a failing frame, or a frozen run: both are places the model is
        # demonstrably wrong and a box-regression gradient is informative.
        event = np.flatnonzero(failing | (frozen & v))
        chosen: list[int] = []
        for pos in event:
            lo = max(0, int(pos) - CLIP_LEN // 2)
            hi = min(len(idx), lo + CLIP_LEN)
            lo = max(0, hi - CLIP_LEN)
            if hi - lo < CLIP_LEN:
                continue
            if chosen and lo - chosen[-1] < CLIP_LEN - MAX_OVERLAP:
                continue
            chosen.append(lo)
            clip_slice = slice(lo, hi)
            clip_valid = v[clip_slice]
            if clip_valid.sum() < CLIP_LEN // 2:
                chosen.pop()
                continue
            clips.append({
                "sequence": seq, "modality": modality,
                "start_position": lo, "length": CLIP_LEN,
                "frame_ids": [ds.ids[i] for i in idx[clip_slice]],
                "mean_iou": float(np.mean(seq_iou[clip_slice][clip_valid])),
                "min_iou": float(np.min(seq_iou[clip_slice][clip_valid])),
                "failing_frames": int(failing[clip_slice].sum()),
                "frozen_frames": int((frozen & v)[clip_slice].sum()),
            })
        if chosen:
            per_seq[seq] = len(chosen)

    if not clips:
        raise SystemExit("no failure-centred clips found; refusing to emit an empty index")

    # The Gate 0 statistic.  An "identity clip" is one the tracker already gets
    # right everywhere -- exactly the self-distillation case that made the old
    # loss flat.  Sampling on failures is supposed to drive this far down.
    identity = sum(1 for c in clips if c["min_iou"] >= FAIL_IOU)
    identity_rate = identity / len(clips)

    fold_of = {}
    for i, seq in enumerate(ds.sequences):
        fold_of.setdefault(seq, int(ds.folds[i]))
    by_fold = {}
    for c in clips:
        by_fold[str(fold_of[c["sequence"]])] = by_fold.get(str(fold_of[c["sequence"]]), 0) + 1
        c["fold"] = fold_of[c["sequence"]]

    doc = {
        "source_pair": args.pair_profile,
        "modalities": sorted(keep_mod),
        "fail_iou_threshold": FAIL_IOU,
        "clip_length": CLIP_LEN,
        "n_clips": len(clips),
        "n_sequences_with_clips": len(per_seq),
        "identity_clips": identity,
        "identity_rate": identity_rate,
        "gate0_identity_rate_max": 0.50,
        "gate0_pass": identity_rate < 0.50,
        "frame_stats": stats,
        "clips_per_fold": by_fold,
        "clips": clips,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")

    print(f"modalities={sorted(keep_mod)}  clips={len(clips)}  "
          f"sequences={len(per_seq)}  clips/fold={by_fold}")
    print(f"frames: total={stats['frames_total']} valid_gt={stats['frames_valid_gt']} "
          f"failing(IoU<{FAIL_IOU})={stats['frames_failing']} frozen={stats['frames_frozen']}")
    print(f"identity rate = {identity_rate:.1%} (舊配方 94–98%；Gate 0 要求 < 50%) -> "
          f"{'✅ PASS' if doc['gate0_pass'] else '❌ FAIL'}")
    print(f"outputs={out}")
    return 0 if doc["gate0_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
