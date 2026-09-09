#!/usr/bin/env python3
"""E32b K=24 exemplar detector probe (CUDA; no tracker re-init).

Each record is an independent two-frame SAM3 detector session:
``[sequence first frame, current frame]`` with the first-frame init box as the
only exemplar.  This produces an offline replacement candidate.  It does not
modify SAM3 tracker memory and must never be described as an exact re-init run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent / "hsot"):
    if (candidate / "e32b_common.py").is_file():
        sys.path.insert(0, str(candidate))
        break

from e32b_common import (  # noqa: E402
    EXECUTION_MODE,
    EXPERIMENT,
    RESULT_SCHEMA_VERSION,
    ContractError,
    atomic_write_json,
    box_iou,
    manifest_records,
    payload_sha256,
    sha256_file,
    validate_box,
    validate_manifest_structure,
    verify_payload_sha256,
)


PINNED_SAM3_REVISION = "96914d2425f90a64f45ca977c2b5165418099543"


def session_id(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and value.get("session_id") is not None:
        return str(value["session_id"])
    raise ContractError(f"unexpected start_session return type: {type(value).__name__}")


def probe_one(predictor, first_path: Path, current_path: Path, init_box: list[float], mini: Path) -> dict:
    from PIL import Image

    with Image.open(first_path) as image:
        width, height = image.size
    with Image.open(current_path) as image:
        if image.size != (width, height):
            raise ContractError(
                f"frame size mismatch: first={(width, height)}, current={image.size}"
            )
    x, y, w, h = validate_box(init_box, "init box")
    if x + w > width + 1e-6 or y + h > height + 1e-6:
        raise ContractError(f"init box {init_box} outside {width}x{height}")
    prompt_xywh_rel = [x / width, y / height, w / width, h / height]

    shutil.rmtree(mini, ignore_errors=True)
    mini.mkdir(parents=True)
    shutil.copy2(first_path, mini / "00000.jpg")
    shutil.copy2(current_path, mini / "00001.jpg")

    sid = session_id(predictor.start_session(str(mini)))
    try:
        predictor.add_prompt(
            session_id=sid,
            frame_idx=0,
            bounding_boxes=[prompt_xywh_rel],
            bounding_box_labels=[1],
            rel_coordinates=True,
        )
        detections: list[tuple[list[float], float]] = []
        saw_frame_one = False
        for output in predictor.propagate_in_video(
            session_id=sid,
            propagation_direction="forward",
            start_frame_idx=0,
            max_frame_num_to_track=1,
        ):
            if not isinstance(output, dict):
                raise ContractError(f"unexpected propagate output {type(output).__name__}")
            if output.get("frame_index") != 1:
                continue
            saw_frame_one = True
            values = output.get("outputs")
            if values is None:
                continue  # documented no-detection abstention, not a runtime fallback
            boxes = values.get("out_boxes_xywh")
            probs = values.get("out_probs")
            if boxes is None or len(boxes) == 0:
                continue
            if probs is None or len(probs) != len(boxes):
                raise ContractError("SAM3 output boxes/probabilities length mismatch")
            for idx in range(len(boxes)):
                rx, ry, rw, rh = [float(v) for v in boxes[idx]]
                score = float(probs[idx])
                if not math.isfinite(score):
                    raise ContractError("SAM3 returned NaN/Inf probability")
                box = validate_box(
                    [rx * width, ry * height, rw * width, rh * height],
                    f"detector output {idx}",
                )
                detections.append((box, score))
        if not saw_frame_one:
            raise ContractError("SAM3 propagation never yielded frame_index=1")
    finally:
        predictor.close_session(sid)

    detections.sort(key=lambda item: item[1], reverse=True)
    top_box = detections[0][0] if detections else None
    top_score = detections[0][1] if detections else None
    return {
        "image_size": [width, height],
        "n_det": len(detections),
        "det_box": top_box,
        "det_score": top_score,
    }


def validate_sequence_result(result: dict, manifest: dict, seq: str) -> None:
    verify_payload_sha256(result)
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ContractError(f"{seq}: unsupported result schema")
    if result.get("experiment") != EXPERIMENT:
        raise ContractError(f"{seq}: wrong experiment")
    if result.get("execution_mode") != EXECUTION_MODE or result.get("true_tracker_reinit") is not False:
        raise ContractError(f"{seq}: result mode mislabelled")
    if result.get("manifest_payload_sha256") != manifest["payload_sha256"]:
        raise ContractError(f"{seq}: stale manifest identity")
    if result.get("sequence") != seq:
        raise ContractError(f"{seq}: sequence result says {result.get('sequence')!r}")
    expected = manifest["sequences"][seq]["target"] + manifest["sequences"][seq]["healthy_proxy"]
    records = result.get("records")
    if not isinstance(records, list) or [r.get("id") for r in records] != [r["id"] for r in expected]:
        raise ContractError(f"{seq}: record ID/order mismatch")
    for record, source in zip(records, expected, strict=True):
        if record.get("cohort") not in {"target", "healthy_proxy"}:
            raise ContractError(f"{record.get('id')}: invalid cohort")
        if record.get("a_box") != source["a_box"] or record.get("b_box") != source["b_box"]:
            raise ContractError(f"{record.get('id')}: stale A/B boxes")
        n_det = record.get("n_det")
        if not isinstance(n_det, int) or n_det < 0:
            raise ContractError(f"{record.get('id')}: invalid n_det")
        if n_det == 0:
            if record.get("det_box") is not None or record.get("det_score") is not None:
                raise ContractError(f"{record.get('id')}: abstention contains detector values")
        else:
            validate_box(record.get("det_box"), f"{record.get('id')} detector")
            score = record.get("det_score")
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise ContractError(f"{record.get('id')}: invalid detector score")
    if result.get("runtime_fallbacks") != 0 or result.get("n_errors") != 0:
        raise ContractError(f"{seq}: fallbacks/errors are forbidden")


def run_sequence(args: argparse.Namespace, manifest: dict) -> Path:
    seq = args.seq
    if seq not in manifest["sequences"]:
        raise ContractError(f"sequence {seq!r} is not in manifest")
    output_path = args.out_dir / "records" / f"{seq}.json"
    if output_path.is_file() and not args.force:
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        validate_sequence_result(cached, manifest, seq)
        if args.sam3_revision != PINNED_SAM3_REVISION:
            raise ContractError(
                f"SAM3 revision {args.sam3_revision} != pinned {PINNED_SAM3_REVISION}"
            )
        if not args.checkpoint.is_file():
            raise ContractError(f"missing checkpoint {args.checkpoint}")
        current_checkpoint_sha256 = sha256_file(args.checkpoint)
        if cached.get("sam3_revision") != args.sam3_revision:
            raise ContractError(f"{seq}: cached SAM3 revision differs from current run")
        if cached.get("checkpoint_sha256") != current_checkpoint_sha256:
            raise ContractError(f"{seq}: cached checkpoint differs from current run")
        print(f"RESUME {seq}: validated exact cached artifact {output_path}")
        return output_path

    if args.sam3_revision != PINNED_SAM3_REVISION:
        raise ContractError(
            f"SAM3 revision {args.sam3_revision} != pinned {PINNED_SAM3_REVISION}"
        )
    if not args.checkpoint.is_file():
        raise ContractError(f"missing checkpoint {args.checkpoint}")
    seq_dir = args.frames_root / seq
    if not seq_dir.is_dir():
        raise ContractError(f"missing frame directory {seq_dir}")

    import torch
    from sam3.model_builder import build_sam3_video_predictor

    if not torch.cuda.is_available():
        raise ContractError("CUDA is required; SAM3 MPS/CPU is not an accepted fallback")
    predictor = build_sam3_video_predictor(checkpoint_path=str(args.checkpoint))
    block = manifest["sequences"][seq]
    first_path = seq_dir / block["first_local_name"]
    source_records = [
        ("target", record) for record in block["target"]
    ] + [
        ("healthy_proxy", record) for record in block["healthy_proxy"]
    ]
    records = []
    mini = args.work_dir / seq / "mini"
    started = time.monotonic()
    for idx, (cohort, source) in enumerate(source_records, start=1):
        current_path = seq_dir / source["local_name"]
        probe = probe_one(predictor, first_path, current_path, block["init_box"], mini)
        det_box = probe["det_box"]
        record = {
            "id": source["id"],
            "frame": source["frame"],
            "cohort": cohort,
            "a_box": source["a_box"],
            "b_box": source["b_box"],
            "a_run": source["a_run"],
            "b_run": source["b_run"],
            **probe,
            "det_iou_a": round(box_iou(det_box, source["a_box"]), 6) if det_box else None,
            "det_iou_b": round(box_iou(det_box, source["b_box"]), 6) if det_box else None,
        }
        records.append(record)
        print(
            f"[{idx:3d}/{len(source_records)}] {source['id']} {cohort} "
            f"n_det={record['n_det']} iou_A={record['det_iou_a']} "
            f"iou_B={record['det_iou_b']}",
            flush=True,
        )

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "execution_mode": EXECUTION_MODE,
        "true_tracker_reinit": False,
        "sequence": seq,
        "manifest_payload_sha256": manifest["payload_sha256"],
        "sam3_revision": args.sam3_revision,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "device": torch.cuda.get_device_name(0),
        "runtime_fallbacks": 0,
        "n_errors": 0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "records": records,
    }
    result["payload_sha256"] = payload_sha256(result)
    validate_sequence_result(result, manifest, seq)
    atomic_write_json(output_path, result)
    shutil.rmtree(args.work_dir / seq, ignore_errors=True)
    print(f"COMPLETE {seq}: {len(records)} exact records -> {output_path}")
    return output_path


def assemble(args: argparse.Namespace, manifest: dict) -> Path:
    sequence_results = []
    records = []
    checkpoint_hashes = set()
    revisions = set()
    for seq in manifest["sequence_order"]:
        path = args.out_dir / "records" / f"{seq}.json"
        if not path.is_file():
            raise ContractError(f"missing sequence artifact {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        validate_sequence_result(result, manifest, seq)
        sequence_results.append({
            "sequence": seq,
            "payload_sha256": result["payload_sha256"],
            "elapsed_seconds": result["elapsed_seconds"],
        })
        records.extend(result["records"])
        checkpoint_hashes.add(result["checkpoint_sha256"])
        revisions.add(result["sam3_revision"])
    if len(checkpoint_hashes) != 1 or revisions != {PINNED_SAM3_REVISION}:
        raise ContractError("sequence artifacts used mixed checkpoint/revision identities")
    expected_ids = [r["id"] for r in manifest_records(manifest)]
    if [r["id"] for r in records] != expected_ids:
        raise ContractError("assembled record order differs from manifest")
    output = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "execution_mode": EXECUTION_MODE,
        "true_tracker_reinit": False,
        "manifest_payload_sha256": manifest["payload_sha256"],
        "sam3_revision": next(iter(revisions)),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "runtime_fallbacks": 0,
        "n_errors": 0,
        "sequence_results": sequence_results,
        "records": records,
    }
    output["payload_sha256"] = payload_sha256(output)
    output_path = args.out_dir / "probe_result.json"
    atomic_write_json(output_path, output)
    print(f"ASSEMBLED {len(records)} records -> {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path.home() / "e32b_work")
    parser.add_argument("--seq")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sam3-revision", default=PINNED_SAM3_REVISION)
    args = parser.parse_args()
    if args.assemble_only:
        if args.seq:
            parser.error("--assemble-only cannot be combined with --seq")
    else:
        if not args.seq or args.frames_root is None or args.checkpoint is None:
            parser.error("probe mode requires --seq, --frames-root, and --checkpoint")
    return args


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest_structure(manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.assemble_only:
        assemble(args, manifest)
    else:
        run_sequence(args, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
