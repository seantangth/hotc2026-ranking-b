#!/usr/bin/env python3
"""Validate E32b detector output and build an explicitly unscored Rank-A candidate.

Ranking-A test has no local GT.  Therefore this script can certify instrument
integrity and a no-GT control sanity check, but cannot claim an AUC gain.  A
``READY_FOR_ONE_LB_OFFLINE_AB`` verdict means exactly that: the offline A/B is
well-formed enough for one leaderboard measurement.  It is not an E32b GO and
it is not a tracker re-initialization result.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from e32b_common import (
    EXECUTION_MODE,
    EXPERIMENT,
    RESULT_SCHEMA_VERSION,
    ContractError,
    atomic_write_csv,
    atomic_write_json,
    box_iou,
    corrected_detector_box,
    load_submission,
    manifest_records,
    payload_sha256,
    sha256_file,
    validate_box,
    validate_manifest_structure,
    verify_payload_sha256,
)


TARGET_DETECT_RATE_MIN = 0.30
CONTROL_DETECT_RATE_MIN = 0.80
CONTROL_ALIGNMENT_MEDIAN_MIN = 0.50
CONTROL_SEVERE_DISAGREE_MAX = 0.15


def validate_probe(result: dict, manifest: dict) -> list[dict]:
    verify_payload_sha256(result)
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ContractError("unsupported probe result schema")
    if result.get("experiment") != EXPERIMENT:
        raise ContractError("probe result belongs to another experiment")
    if result.get("execution_mode") != EXECUTION_MODE:
        raise ContractError("probe result is not the locked offline execution mode")
    if result.get("true_tracker_reinit") is not False:
        raise ContractError("probe result incorrectly claims tracker re-init")
    if result.get("manifest_payload_sha256") != manifest["payload_sha256"]:
        raise ContractError("probe result was produced from another manifest")
    if result.get("runtime_fallbacks") != 0 or result.get("n_errors") != 0:
        raise ContractError("fallback/error-bearing results are forbidden")
    records = result.get("records")
    expected = manifest_records(manifest)
    if not isinstance(records, list) or [r.get("id") for r in records] != [r["id"] for r in expected]:
        raise ContractError("probe record ID/order differs from manifest")
    for record, source in zip(records, expected, strict=True):
        if record.get("a_box") != source["a_box"] or record.get("b_box") != source["b_box"]:
            raise ContractError(f"{record.get('id')}: stale A/B box")
        n_det = record.get("n_det")
        if not isinstance(n_det, int) or n_det < 0:
            raise ContractError(f"{record.get('id')}: invalid n_det")
        if n_det == 0:
            if record.get("det_box") is not None or record.get("det_score") is not None:
                raise ContractError(f"{record.get('id')}: abstention contains a detection")
        else:
            validate_box(record.get("det_box"), f"{record.get('id')} detector")
            score = record.get("det_score")
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise ContractError(f"{record.get('id')}: invalid score")
    return records


def rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def metrics(records: list[dict]) -> dict:
    target = [r for r in records if r["cohort"] == "target"]
    control = [r for r in records if r["cohort"] == "healthy_proxy"]
    if not target or not control:
        raise ContractError("target and healthy proxy cohorts must both be non-empty")
    target_detected = [r for r in target if r["n_det"] > 0]
    control_detected = [r for r in control if r["n_det"] > 0]
    control_alignment = [
        max(box_iou(r["det_box"], r["a_box"]), box_iou(r["det_box"], r["b_box"]))
        for r in control_detected
    ]
    severe = [
        r["n_det"] == 0
        or max(box_iou(r["det_box"], r["a_box"]), box_iou(r["det_box"], r["b_box"])) < 0.30
        for r in control
    ]
    target_novel = [
        max(box_iou(r["det_box"], r["a_box"]), box_iou(r["det_box"], r["b_box"])) < 0.30
        for r in target_detected
    ]
    return {
        "n_target": len(target),
        "n_control": len(control),
        "target_detect_rate": rate([r["n_det"] > 0 for r in target]),
        "target_novel_vs_both_rate_among_detected": rate(target_novel),
        "control_detect_rate": rate([r["n_det"] > 0 for r in control]),
        "control_alignment_median": (
            statistics.median(control_alignment) if control_alignment else None
        ),
        "control_severe_disagree_rate": rate(severe),
    }


def gate_values(observed: dict) -> dict[str, bool]:
    return {
        "target_detect_rate": observed["target_detect_rate"] >= TARGET_DETECT_RATE_MIN,
        "control_detect_rate": observed["control_detect_rate"] >= CONTROL_DETECT_RATE_MIN,
        "control_alignment_median": (
            observed["control_alignment_median"] is not None
            and observed["control_alignment_median"] >= CONTROL_ALIGNMENT_MEDIAN_MIN
        ),
        "control_severe_disagree_rate": (
            observed["control_severe_disagree_rate"] < CONTROL_SEVERE_DISAGREE_MAX
        ),
    }


def build_candidate(
    manifest: dict,
    records: list[dict],
    base_path: Path,
    sample_path: Path,
    output_path: Path,
) -> dict:
    base, base_order = load_submission(base_path)
    _, sample_order = load_submission(sample_path, boxes_required=False)
    if base_order != sample_order:
        raise ContractError("candidate base does not match sample canonical order")
    by_id = {r["id"]: r for r in records if r["cohort"] == "target"}
    manifest_target_ids = {r["id"] for r in manifest_records(manifest, "target")}
    if set(by_id) != manifest_target_ids:
        raise ContractError("candidate target record set differs from manifest")

    # The candidate base must still contain corrected canonical A on the deadzone
    # pool.  This prevents silently overlaying the detector on a different A/B pair.
    for source in manifest_records(manifest, "target"):
        seq, frame_text = source["id"].rsplit("_", 1)
        frame = int(frame_text)
        expected = corrected_detector_box(source["a_box"], "both")
        if base[seq][frame] != expected:
            raise ContractError(
                f"{source['id']}: base is not corrected canonical A; "
                f"base={base[seq][frame]}, expected={expected}"
            )

    out = {seq: {frame: list(box) for frame, box in frames.items()} for seq, frames in base.items()}
    n_detected = 0
    changed_ids: list[str] = []
    for rid, record in by_id.items():
        if record["n_det"] == 0:
            continue  # locked abstention, not an exception fallback
        seq, frame_text = rid.rsplit("_", 1)
        frame = int(frame_text)
        replacement = corrected_detector_box(
            record["det_box"],
            manifest["policy"]["detector_coordinate_correction_for_ranka_candidate"],
        )
        n_detected += 1
        if replacement != out[seq][frame]:
            changed_ids.append(rid)
        out[seq][frame] = replacement

    rows = []
    for rid in sample_order:
        seq, frame_text = rid.rsplit("_", 1)
        rows.append((rid, *validate_box(out[seq][int(frame_text)], rid)))
    atomic_write_csv(output_path, rows)
    # Reload the atomically written artifact; the writer is not the validator.
    _, written_order = load_submission(output_path)
    if written_order != sample_order:
        raise ContractError("written candidate failed canonical order validation")
    for seq, frames in base.items():
        first = min(frames)
        if out[seq][first] != frames[first]:
            raise ContractError(f"{seq}: first frame changed")
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "rows": len(rows),
        "detector_replacements": n_detected,
        "changed_rows": len(changed_ids),
        "changed_ids": changed_ids,
        "zero_detection_abstentions": len(by_id) - n_detected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe-result", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest_structure(manifest)
    result = json.loads(args.probe_result.read_text(encoding="utf-8"))
    records = validate_probe(result, manifest)
    observed = metrics(records)
    gates = gate_values(observed)
    ready = all(gates.values())
    if args.candidate_out.exists():
        args.candidate_out.unlink()  # explicit output only; prevents a stale READY artifact
    candidate = (
        build_candidate(manifest, records, args.base, args.sample, args.candidate_out)
        if ready else None
    )
    per_seq = {}
    for seq in manifest["sequence_order"]:
        seq_records = [r for r in records if r["id"].rsplit("_", 1)[0] == seq]
        per_seq[seq] = metrics(seq_records)
    report = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "verdict": "READY_FOR_ONE_LB_OFFLINE_AB" if ready else "NO_GO_PROXY_SANITY",
        "evaluation_scope": "no_gt_instrument_and_control_proxy_only",
        "auc_gain_measured": False,
        "requires_gt_or_one_lb_ab": True,
        "execution_mode": EXECUTION_MODE,
        "true_tracker_reinit": False,
        "warning": (
            "This verdict does not estimate score gain. The candidate changes boxes offline; "
            "it neither injects a prompt into tracker memory nor reruns downstream frames."
        ),
        "manifest_payload_sha256": manifest["payload_sha256"],
        "probe_payload_sha256": result["payload_sha256"],
        "base": {"path": str(args.base), "sha256": sha256_file(args.base)},
        "thresholds_locked_before_gpu": {
            "target_detect_rate_min": TARGET_DETECT_RATE_MIN,
            "control_detect_rate_min": CONTROL_DETECT_RATE_MIN,
            "control_alignment_median_min": CONTROL_ALIGNMENT_MEDIAN_MIN,
            "control_severe_disagree_rate_strict_max": CONTROL_SEVERE_DISAGREE_MAX,
        },
        "observed": observed,
        "gates": gates,
        "per_sequence": per_seq,
        "candidate": candidate,
    }
    report["payload_sha256"] = payload_sha256(report)
    atomic_write_json(args.report, report)
    print(f"VERDICT {report['verdict']}")
    print(json.dumps({"observed": observed, "gates": gates}, indent=1))
    if candidate:
        print(
            f"UNSCORED candidate: {candidate['detector_replacements']} replacements, "
            f"{candidate['zero_detection_abstentions']} abstentions -> {args.candidate_out}"
        )
    else:
        print("No candidate CSV written; proxy sanity gate did not pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

