#!/usr/bin/env python3
"""CPU-only physical-capture grouped OOF evaluator for the train405 raw cache.

This program deliberately does *not* run a tracker, start cloud infrastructure, or
build a Kaggle submission.  It consumes the immutable cache produced by
``run_oof405_zero_shot_lambda_v1.sh`` and evaluates a frozen policy grid:

* coordinate correction: ``none``, ``top-only``, ``both``;
* frozen-source splice at explicitly supplied K values;
* an optional v056-shaped quality head, refit separately on the four training
  folds and applied only to the held-out physical-capture fold.

The feature extractor never receives GT, a sequence basename, a capture-group
name, or a fold number.  GT is used only (a) to fit a fold's model from the other
four folds and (b) to score predictions after the held-out decisions are frozen.
All modalities sharing the same basename must be assigned to the same fold.

The loader is intentionally fail-closed.  It requires the prep hash manifest,
all fold/group contracts, exact prediction ID sets, and clean tracker diagnostics.
It will not infer a fold or silently drop a row when any of those contracts is
missing or inconsistent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from hsot.quality_head_v1 import fit_logit, predict_p


SUBMISSION_COLUMNS = ("ID", "x", "y", "width", "height")
FRAME_COLUMNS = (
    "ID", "sequence", "position", "modality", "capture_group", "fold", "gt_valid",
    "selected_block_start", "selected_block_end",
)
CAPTURE_COLUMNS = ("sequence", "modality", "capture_group", "fold", "n_frames", "evidence")
MIRROR_COLUMNS = ("source_ID", "target_ID", "sequence", "position")
N_FOLDS = 5
CORR_MODES = ("none", "top-only", "both")
IOU_THRESHOLDS = np.arange(0.02, 1.02, 0.02, dtype=np.float64)
SPLICE_SRC_MAX_RUN = 2
QHEAD_IAB_MAX = 0.30
QHEAD_MARGIN = 0.0
FEATURE_NAMES = (
    "iou_a_b", "log_area_a", "log_area_b", "log_area_ratio", "aspect_a", "aspect_b",
    "center_distance_over_sqrt_area_a", "a_dx", "a_dy", "a_dw", "a_dh", "b_dx",
    "b_dy", "b_dw", "b_dh", "frozen_run_a", "frozen_run_b", "frame_fraction",
    "is_nir", "is_rednir", "is_vis", "a_width", "a_height", "b_width", "b_height",
)
REQUIRED_HASHED_CONTRACTS = (
    "sample_contract.csv", "observed_gt.csv", "frame_contract.csv", "capture_groups.csv",
    "folds.json", "mirror_map.csv", "scoring_gt.csv", "prep_report.json",
)


class ContractError(ValueError):
    """A cache/contract invariant failed; evaluation must not continue."""


@dataclass(frozen=True)
class Dataset:
    ids: tuple[str, ...]
    sequences: tuple[str, ...]
    frame_numbers: np.ndarray
    positions: np.ndarray
    modalities: tuple[str, ...]
    capture_groups: tuple[str, ...]
    folds: np.ndarray
    first_mask: np.ndarray
    gt: np.ndarray
    raw_main: np.ndarray
    raw_source: np.ndarray
    sequence_indices: dict[str, np.ndarray]
    mirror_source_indices: np.ndarray
    mirror_gt: np.ndarray
    input_audit: dict[str, Any]


@dataclass(frozen=True)
class QHeadModel:
    heldout_fold: int
    train_folds: tuple[int, ...]
    n_train: int
    class_a_better: int
    class_b_better: int
    w: np.ndarray
    mu: np.ndarray
    sd: np.ndarray


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"{label} missing: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} unreadable/invalid: {path}: {exc}") from exc


def _read_csv(path: Path, columns: Sequence[str], label: str) -> list[dict[str, str]]:
    try:
        fh = path.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise ContractError(f"{label} unreadable: {path}: {exc}") from exc
    with fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ContractError(
                f"{label} columns {reader.fieldnames!r} != {list(columns)!r}")
        rows = [dict(row) for row in reader]
    if not rows and label not in {"mirror_map"}:
        raise ContractError(f"{label} has no rows: {path}")
    return rows


def _unique_ids(rows: Sequence[dict[str, str]], label: str, key: str = "ID") -> list[str]:
    ids = [row[key].strip() for row in rows]
    blank = [i for i, ident in enumerate(ids, 2) if not ident]
    if blank:
        raise ContractError(f"{label} has blank {key} at rows {blank[:5]}")
    if len(ids) != len(set(ids)):
        seen: set[str] = set()
        dup = next(ident for ident in ids if ident in seen or seen.add(ident))
        raise ContractError(f"{label} duplicate {key}: {dup}")
    return ids


def _parse_id(ident: str, label: str) -> tuple[str, int]:
    try:
        seq, frame_raw = ident.rsplit("_", 1)
        frame = int(frame_raw)
    except (ValueError, AttributeError) as exc:
        raise ContractError(f"{label} invalid ID (expected <sequence>_<integer>): {ident!r}") from exc
    if not seq or frame < 0:
        raise ContractError(f"{label} invalid ID: {ident!r}")
    return seq, frame


def _float_box(row: dict[str, str], label: str, *, prediction: bool) -> list[float]:
    try:
        box = [float(row[c]) for c in SUBMISSION_COLUMNS[1:]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"{label} non-numeric box for {row.get('ID')!r}") from exc
    if not all(math.isfinite(value) for value in box):
        raise ContractError(f"{label} NaN/Inf box for {row.get('ID')!r}: {box}")
    if prediction and (box[0] < 0 or box[1] < 0 or box[2] <= 0 or box[3] <= 0):
        raise ContractError(f"{label} invalid prediction box for {row.get('ID')!r}: {box}")
    return box


def _verify_contract_hashes(contracts: Path) -> dict[str, str]:
    manifest_path = contracts / "contract_sha256.json"
    manifest = _json(manifest_path, "contract_sha256")
    if not isinstance(manifest, dict) or not manifest:
        raise ContractError("contract_sha256.json must be a non-empty object")
    for name in REQUIRED_HASHED_CONTRACTS:
        expected = manifest.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ContractError(f"contract_sha256.json missing valid digest for {name}")
        path = contracts / name
        if not path.is_file():
            raise ContractError(f"required contract missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ContractError(
                f"contract hash mismatch for {name}: expected {expected}, got {actual}")
    return {name: str(manifest[name]) for name in sorted(manifest)}


PAIR_SCOPES = {
    "rankB_robust": {
        "pair_scope": "fresh rankB_robust full-frame SAM3/SAMURAI-reset raw pair",
        "does_not_certify":
            "Rank-A two-pass crop pair or legacy cross-sequence-KF pair",
        "offline_two_pass": False,
    },
    "rankB_robust_crop": {
        "pair_scope":
            "fresh rankB_robust_crop pair: full-frame SAM3/SAMURAI-reset legs with "
            "offline two-pass crop merged over the objectively selected small-target "
            "sequences",
        "does_not_certify":
            "rankA_best (v056 quality head / legacy cross-sequence-KF) or any "
            "full-frame-only pair",
        "offline_two_pass": True,
    },
}


def _validate_crop_meta(path: Path, seq_counts: dict[str, int]) -> dict[str, Any]:
    """Check crop_meta.json really describes a non-empty subset of this contract.

    ``hsot.crop_rerun prep`` writes a flat window-key -> window mapping; the
    sequence name lives in each window's ``seq`` field and one sequence may own
    several segment windows, so the selected set is derived, never assumed.
    """
    doc = _json(path, "crop_meta")
    if not isinstance(doc, dict) or not doc:
        raise ContractError("crop_meta must be a non-empty object")
    selected: set[str] = set()
    for key, window in doc.items():
        if not isinstance(window, dict) or "seq" not in window:
            raise ContractError(f"crop_meta window {key!r} has no 'seq' field")
        selected.add(str(window["seq"]))
    if not selected:
        raise ContractError("crop_meta lists no cropped sequence; the crop pair is empty")
    unknown = sorted(selected - set(seq_counts))
    if unknown:
        raise ContractError(f"crop_meta names sequences outside the contract: {unknown[:5]}")
    return {
        "path": str(path), "sha256": _sha256(path),
        "windows": len(doc),
        # run_ranking_b feeds `sorted(meta)` to the crop legs as their --seq-list,
        # so these keys are exactly the crop diagnostics' sequence keys.
        "windows_by_key": sorted(doc),
        "cropped_sequences": len(selected), "contract_sequences": len(seq_counts),
        "sequences": sorted(selected),
    }


def _validate_status_sidecar(
    diag: dict[str, Any], seq: str, backend: str, expected_rows: int | None, diag_path: Path,
) -> None:
    """Verify a resumed sequence against its own status sidecar.

    The 08-27 formal 405 run exposed this: after the runner resumed, every
    sequence's diagnostics entry became
    ``{"skipped": "already_done", "rows": N, "status_sidecar": ..., "run_signature": ...}``
    and the fresh-run check (`status == "complete"`, `n_avail == expected`) rejected
    all 405 -- the harness's fail-closed contract was incompatible with the
    runner's own resume path.  Reading the sidecar is the fix that keeps the
    contract strict: it carries the authoritative status, row count, artifact
    hash and the run config the sequence was actually produced with.
    """
    if diag.get("skipped") != "already_done":
        raise ContractError(
            f"{backend} diagnostics {seq}: unknown skip marker {diag.get('skipped')!r}")
    summary_signature = diag.get("run_signature")
    if not isinstance(summary_signature, str) or len(summary_signature) != 64:
        raise ContractError(f"{backend} diagnostics {seq} missing SHA256 run_signature")

    # The recorded path is absolute and from the GPU box; resolve beside the
    # diagnostics file we were actually handed so this works off a synced copy.
    recorded = diag.get("status_sidecar")
    if not isinstance(recorded, str) or not recorded:
        raise ContractError(f"{backend} diagnostics {seq} missing status_sidecar path")
    sidecar = diag_path.parent / "seq_csv" / Path(recorded).name
    if not sidecar.is_file():
        raise ContractError(
            f"{backend} {seq}: resumed sequence but status sidecar is absent: {sidecar}")

    doc = _json(sidecar, f"{backend} {seq} status sidecar")
    if not isinstance(doc, dict):
        raise ContractError(f"{backend} {seq}: status sidecar is not an object")
    if doc.get("sequence") != seq:
        raise ContractError(
            f"{backend} {seq}: sidecar names a different sequence {doc.get('sequence')!r}")
    if doc.get("status") != "complete":
        raise ContractError(
            f"{backend} {seq}: sidecar status={doc.get('status')!r}, required 'complete'")

    rows = doc.get("rows")
    if not isinstance(rows, int) or rows <= 0:
        raise ContractError(f"{backend} {seq}: sidecar rows={rows!r}")
    if rows != diag.get("rows"):
        raise ContractError(
            f"{backend} {seq}: sidecar rows={rows} disagrees with diagnostics {diag.get('rows')!r}")
    if expected_rows is not None and rows != expected_rows:
        raise ContractError(
            f"{backend} {seq}: sidecar rows={rows} != frame contract {expected_rows}")

    signature = doc.get("run_signature")
    if not isinstance(signature, dict) or signature.get("digest") != summary_signature:
        raise ContractError(
            f"{backend} {seq}: sidecar run_signature digest does not match diagnostics")

    # The sidecar records what the sequence was actually produced with, so the
    # semantics the profile promises are checked per sequence, not just once in
    # _meta.  A resumed leg mixing configs would otherwise pass silently.
    config = signature.get("run_config")
    if not isinstance(config, dict):
        raise ContractError(f"{backend} {seq}: sidecar has no run_config")
    required = {"backend": backend, "samurai_reset_kf": True}
    if backend == "sam3":
        required["sam3_eval"] = True
    for key, want in required.items():
        if config.get(key) != want:
            raise ContractError(
                f"{backend} {seq}: sidecar run_config.{key}={config.get(key)!r}, required {want!r}")

    details = doc.get("details")
    if isinstance(details, dict):
        hit = {"error", "fallback", "fallback_error"} & set(details)
        if hit:
            raise ContractError(
                f"{backend} {seq}: sidecar details contain forbidden status {sorted(hit)}")


def _validate_diagnostics(
    path: Path, backend: str, seq_counts: dict[str, int | None],
) -> dict[str, Any]:
    doc = _json(path, f"{backend} diagnostics")
    if not isinstance(doc, dict):
        raise ContractError(f"{backend} diagnostics must be an object")
    meta = doc.get("_meta")
    if not isinstance(meta, dict):
        raise ContractError(f"{backend} diagnostics missing _meta")
    required_meta = {
        "backend": backend,
        "validation": "pass",
        "allow_fallback": False,
        "samurai_reset_kf": True,
        "samurai_legacy_cross_seq_kf": False,
    }
    if backend == "sam3":
        required_meta["sam3_eval"] = True
    for key, want in required_meta.items():
        if meta.get(key) != want:
            raise ContractError(
                f"{backend} diagnostics _meta.{key}={meta.get(key)!r}, required {want!r}")
    if meta.get("n_seqs") != len(seq_counts):
        raise ContractError(
            f"{backend} diagnostics n_seqs={meta.get('n_seqs')!r} != {len(seq_counts)}")
    seq_keys = {key for key in doc if key != "_meta"}
    if seq_keys != set(seq_counts):
        raise ContractError(
            f"{backend} diagnostics sequence set mismatch: missing={sorted(set(seq_counts)-seq_keys)[:5]}, "
            f"extra={sorted(seq_keys-set(seq_counts))[:5]}")
    forbidden = {"error", "fallback", "fallback_error"}
    sidecars_verified = 0
    for seq, expected_rows in seq_counts.items():
        diag = doc[seq]
        if isinstance(diag, dict) and "skipped" in diag:
            # A resumed sequence: track_t1 recorded only a pointer, because the
            # work happened in an earlier run of the same contract.  The summary
            # alone would be weak evidence, so the per-sequence status sidecar --
            # which carries status, row count, artifact hash and the full run
            # config -- is what actually gets checked here.  This makes resumed
            # runs *more* strictly validated than fresh ones, not less.
            _validate_status_sidecar(diag, seq, backend, expected_rows, path)
            sidecars_verified += 1
            continue
        if expected_rows is None:
            # Crop legs are keyed by crop-window name and only cover that window's
            # frames, so the frame contract cannot predict n_avail.  Total coverage
            # is already enforced on the merged CSV's exact ID set; here we only
            # require the window to have completed with real frames.
            if not isinstance(diag, dict):
                raise ContractError(f"{backend} diagnostics {seq} is not an object")
            hit = forbidden & set(diag)
            if hit:
                raise ContractError(
                    f"{backend} diagnostics {seq} contains forbidden status {sorted(hit)}")
            n_avail = diag.get("n_avail")
            if diag.get("status") != "complete" or not isinstance(n_avail, int) or n_avail <= 0:
                raise ContractError(
                    f"{backend} diagnostics {seq} incomplete: "
                    f"status={diag.get('status')!r}, n_avail={n_avail!r}")
            signature = diag.get("run_signature")
            if not isinstance(signature, str) or len(signature) != 64:
                raise ContractError(f"{backend} diagnostics {seq} missing SHA256 run_signature")
            continue
        if not isinstance(diag, dict):
            raise ContractError(f"{backend} diagnostics {seq} is not an object")
        hit = forbidden & set(diag)
        if hit:
            raise ContractError(f"{backend} diagnostics {seq} contains forbidden status {sorted(hit)}")
        if diag.get("status") != "complete" or diag.get("n_avail") != expected_rows:
            raise ContractError(
                f"{backend} diagnostics {seq} incomplete/count mismatch: "
                f"status={diag.get('status')!r}, n_avail={diag.get('n_avail')!r}, "
                f"expected={expected_rows}")
        signature = diag.get("run_signature")
        if not isinstance(signature, str) or len(signature) != 64:
            raise ContractError(f"{backend} diagnostics {seq} missing SHA256 run_signature")
    return {
        "path": str(path), "sha256": _sha256(path), "backend": backend,
        "n_sequences": len(seq_counts), "validation": "pass",
        "resumed_sequences_verified_via_sidecar": sidecars_verified,
    }


def _prediction_boxes(path: Path, expected_ids: Sequence[str], label: str) -> np.ndarray:
    rows = _read_csv(path, SUBMISSION_COLUMNS, label)
    ids = _unique_ids(rows, label)
    if ids != list(expected_ids):
        want = set(expected_ids)
        got = set(ids)
        if got != want:
            raise ContractError(
                f"{label} exact ID set mismatch: missing={sorted(want-got)[:5]}, "
                f"extra={sorted(got-want)[:5]}")
        raise ContractError(f"{label} row order differs from sample_contract canonical order")
    return np.asarray([_float_box(row, label, prediction=True) for row in rows], dtype=np.float64)


def load_dataset(
    contracts: str | Path,
    main_csv: str | Path,
    source_csv: str | Path,
    main_diagnostics: str | Path | None = None,
    source_diagnostics: str | Path | None = None,
    pair_profile: str = "rankB_robust",
    main_crop_diagnostics: str | Path | None = None,
    source_crop_diagnostics: str | Path | None = None,
    crop_meta: str | Path | None = None,
) -> Dataset:
    if pair_profile not in PAIR_SCOPES:
        raise ContractError(
            f"unknown pair_profile {pair_profile!r}; expected one of {sorted(PAIR_SCOPES)}")
    scope = PAIR_SCOPES[pair_profile]
    crop_inputs = (main_crop_diagnostics, source_crop_diagnostics, crop_meta)
    if scope["offline_two_pass"]:
        if not all(crop_inputs):
            raise ContractError(
                f"{pair_profile} requires --main-crop-diagnostics, "
                "--source-crop-diagnostics and --crop-meta")
    elif any(crop_inputs):
        raise ContractError(
            f"{pair_profile} is a full-frame pair; crop diagnostics/meta must not be supplied")
    contracts = Path(contracts).expanduser().resolve()
    main_csv = Path(main_csv).expanduser().resolve()
    source_csv = Path(source_csv).expanduser().resolve()
    hashes = _verify_contract_hashes(contracts)

    sample_rows = _read_csv(contracts / "sample_contract.csv", SUBMISSION_COLUMNS, "sample_contract")
    gt_rows = _read_csv(contracts / "observed_gt.csv", SUBMISSION_COLUMNS, "observed_gt")
    frame_rows = _read_csv(contracts / "frame_contract.csv", FRAME_COLUMNS, "frame_contract")
    capture_rows = _read_csv(contracts / "capture_groups.csv", CAPTURE_COLUMNS, "capture_groups")
    mirror_rows = _read_csv(contracts / "mirror_map.csv", MIRROR_COLUMNS, "mirror_map")
    scoring_rows = _read_csv(contracts / "scoring_gt.csv", SUBMISSION_COLUMNS, "scoring_gt")
    sample_ids = _unique_ids(sample_rows, "sample_contract")
    if _unique_ids(gt_rows, "observed_gt") != sample_ids:
        raise ContractError("observed_gt IDs/order must exactly equal sample_contract")
    if _unique_ids(frame_rows, "frame_contract") != sample_ids:
        raise ContractError("frame_contract IDs/order must exactly equal sample_contract")
    for row in sample_rows:
        if any(float(row[col]) != 0.0 for col in SUBMISSION_COLUMNS[1:]):
            raise ContractError(f"sample_contract must contain zero placeholders: {row['ID']}")

    folds_doc = _json(contracts / "folds.json", "folds")
    prep_report = _json(contracts / "prep_report.json", "prep_report")
    if not isinstance(folds_doc, dict) or folds_doc.get("schema_version") != 1:
        raise ContractError("folds.json schema_version must be 1")
    if folds_doc.get("n_folds") != N_FOLDS:
        raise ContractError(f"folds.json n_folds must be exactly {N_FOLDS}")
    if folds_doc.get("group_rule") != "strip modality prefix only; preserve full basename":
        raise ContractError("folds.json group_rule is absent or not the physical-capture rule")
    if not isinstance(prep_report, dict) or prep_report.get("schema_version") != 1:
        raise ContractError("prep_report.json schema_version must be 1")
    if prep_report.get("physical_prediction_rows") != len(sample_ids):
        raise ContractError("prep_report physical_prediction_rows disagrees with contracts")

    n = len(sample_ids)
    sequences: list[str] = []
    frame_numbers = np.empty(n, dtype=np.int64)
    positions = np.empty(n, dtype=np.int64)
    modalities: list[str] = []
    capture_groups: list[str] = []
    fold_values = np.empty(n, dtype=np.int64)
    seq_indices_raw: dict[str, list[int]] = {}
    seq_meta: dict[str, tuple[str, str, int]] = {}
    for i, row in enumerate(frame_rows):
        seq_from_id, frame = _parse_id(row["ID"], "frame_contract")
        seq = row["sequence"]
        modality = row["modality"]
        group = row["capture_group"]
        try:
            position = int(row["position"])
            fold = int(row["fold"])
            gt_valid = int(row["gt_valid"])
            block_start = int(row["selected_block_start"])
            block_end = int(row["selected_block_end"])
        except ValueError as exc:
            raise ContractError(f"frame_contract non-integer metadata for {row['ID']}") from exc
        if seq != seq_from_id or modality not in {"nir", "rednir", "vis"}:
            raise ContractError(f"frame_contract ID/sequence/modality mismatch for {row['ID']}")
        if not seq.startswith(modality + "-"):
            raise ContractError(f"frame_contract modality prefix mismatch for {row['ID']}")
        objective_group = seq.split("-", 1)[1]
        if group != objective_group:
            raise ContractError(
                f"frame_contract capture_group for {seq} must be exact basename {objective_group!r}")
        if fold not in range(N_FOLDS) or position < 0 or gt_valid not in {0, 1}:
            raise ContractError(f"frame_contract invalid position/fold/gt_valid for {row['ID']}")
        if not block_start <= frame <= block_end:
            raise ContractError(f"frame_contract selected block excludes {row['ID']}")
        old = seq_meta.setdefault(seq, (modality, group, fold))
        if old != (modality, group, fold):
            raise ContractError(f"frame_contract has inconsistent metadata within sequence {seq}")
        sequences.append(seq)
        frame_numbers[i] = frame
        positions[i] = position
        modalities.append(modality)
        capture_groups.append(group)
        fold_values[i] = fold
        seq_indices_raw.setdefault(seq, []).append(i)

    if set(fold_values.tolist()) != set(range(N_FOLDS)):
        raise ContractError(f"frame_contract must populate all folds 0..{N_FOLDS-1}")
    sequence_indices: dict[str, np.ndarray] = {}
    seq_counts: dict[str, int] = {}
    for seq, raw_indices in seq_indices_raw.items():
        indices = np.asarray(sorted(raw_indices, key=lambda idx: positions[idx]), dtype=np.int64)
        got_positions = positions[indices].tolist()
        if got_positions != list(range(len(indices))):
            raise ContractError(f"frame_contract positions are not contiguous 0..N-1 for {seq}")
        if frame_numbers[indices].tolist() != sorted(frame_numbers[indices].tolist()):
            raise ContractError(f"frame_contract frame order is not increasing for {seq}")
        sequence_indices[seq] = indices
        seq_counts[seq] = len(indices)

    capture_ids = _unique_ids(capture_rows, "capture_groups", key="sequence")
    if set(capture_ids) != set(sequence_indices):
        raise ContractError("capture_groups sequence set disagrees with frame_contract")
    capture_by_seq = {row["sequence"]: row for row in capture_rows}
    group_folds: dict[str, set[int]] = {}
    for seq, (modality, group, fold) in seq_meta.items():
        row = capture_by_seq[seq]
        try:
            capture_fold = int(row["fold"])
            capture_n = int(row["n_frames"])
        except ValueError as exc:
            raise ContractError(f"capture_groups non-integer fold/count for {seq}") from exc
        if (
            row["modality"] != modality or row["capture_group"] != group or
            capture_fold != fold or capture_n != seq_counts[seq] or
            row["evidence"] != "exact basename after modality removal"
        ):
            raise ContractError(f"capture_groups disagrees with frame_contract for {seq}")
        group_folds.setdefault(group, set()).add(fold)
    leaked = {group: sorted(fs) for group, fs in group_folds.items() if len(fs) != 1}
    if leaked:
        raise ContractError(f"physical capture group crosses folds: {dict(list(leaked.items())[:5])}")

    seq_to_fold = folds_doc.get("sequence_to_fold")
    group_to_fold = folds_doc.get("group_to_fold")
    if not isinstance(seq_to_fold, dict) or not isinstance(group_to_fold, dict):
        raise ContractError("folds.json missing sequence_to_fold/group_to_fold objects")
    if seq_to_fold != {seq: meta[2] for seq, meta in sorted(seq_meta.items())}:
        raise ContractError("folds.json sequence_to_fold disagrees with frame/capture contracts")
    objective_group_fold = {group: next(iter(fs)) for group, fs in sorted(group_folds.items())}
    if group_to_fold != objective_group_fold:
        raise ContractError("folds.json group_to_fold disagrees with physical capture groups")
    if folds_doc.get("n_groups") != len(group_folds):
        raise ContractError("folds.json n_groups disagrees with capture contracts")

    gt = np.asarray([_float_box(row, "observed_gt", prediction=False) for row in gt_rows])
    gt_valid = np.all(gt > 0, axis=1)
    declared_valid = np.asarray([int(row["gt_valid"]) == 1 for row in frame_rows])
    expected_declared = np.isfinite(gt).all(axis=1) & (gt[:, 2] > 0) & (gt[:, 3] > 0)
    if not np.array_equal(declared_valid, expected_declared):
        raise ContractError("frame_contract gt_valid disagrees with observed_gt valid-box definition")

    main = _prediction_boxes(main_csv, sample_ids, "main prediction")
    source = _prediction_boxes(source_csv, sample_ids, "source prediction")
    main_diag_path = Path(main_diagnostics).expanduser().resolve() if main_diagnostics else main_csv.with_name("diagnostics.json")
    source_diag_path = Path(source_diagnostics).expanduser().resolve() if source_diagnostics else source_csv.with_name("diagnostics.json")
    diag_audit = {
        "main": _validate_diagnostics(main_diag_path, "sam3", seq_counts),
        "source": _validate_diagnostics(source_diag_path, "samurai", seq_counts),
    }
    crop_audit: dict[str, Any] | None = None
    if scope["offline_two_pass"]:
        meta_audit = _validate_crop_meta(
            Path(crop_meta).expanduser().resolve(), seq_counts)
        # The crop legs only ever ran on the selected sequences, so they are
        # validated against that subset -- with the same fallback-free,
        # complete-status, run-signature requirements as the full legs.
        crop_counts: dict[str, int | None] = {key: None for key in meta_audit["windows_by_key"]}
        crop_audit = {
            "crop_meta": meta_audit,
            "main": _validate_diagnostics(
                Path(main_crop_diagnostics).expanduser().resolve(), "sam3", crop_counts),
            "source": _validate_diagnostics(
                Path(source_crop_diagnostics).expanduser().resolve(), "samurai", crop_counts),
        }

    scoring_ids = _unique_ids(scoring_rows, "scoring_gt")
    scoring_by_id = {row["ID"]: row for row in scoring_rows}
    if not set(sample_ids).issubset(scoring_by_id):
        raise ContractError("scoring_gt is missing observed physical IDs")
    for ident, gt_row in zip(sample_ids, gt_rows, strict=True):
        if any(scoring_by_id[ident][col] != gt_row[col] for col in SUBMISSION_COLUMNS[1:]):
            raise ContractError(f"scoring_gt observed row differs from observed_gt: {ident}")
    id_to_index = {ident: i for i, ident in enumerate(sample_ids)}
    mirror_targets = _unique_ids(mirror_rows, "mirror_map", key="target_ID") if mirror_rows else []
    if set(scoring_ids) != set(sample_ids) | set(mirror_targets):
        raise ContractError("scoring_gt must equal observed IDs plus mirror_map target IDs")
    mirror_source_indices: list[int] = []
    mirror_gt: list[list[float]] = []
    for row in mirror_rows:
        source_id, target_id = row["source_ID"], row["target_ID"]
        if source_id not in id_to_index or target_id in id_to_index or target_id not in scoring_by_id:
            raise ContractError(f"mirror_map invalid source/target pair: {source_id} -> {target_id}")
        src_seq, _ = _parse_id(source_id, "mirror_map source")
        target_seq, _ = _parse_id(target_id, "mirror_map target")
        try:
            mirror_position = int(row["position"])
        except ValueError as exc:
            raise ContractError(f"mirror_map non-integer position for {target_id}") from exc
        src_idx = id_to_index[source_id]
        if (
            row["sequence"] != src_seq or target_seq != src_seq or
            positions[src_idx] != mirror_position
        ):
            raise ContractError(f"mirror_map sequence/position mismatch for {target_id}")
        mirror_source_indices.append(src_idx)
        mirror_gt.append(_float_box(scoring_by_id[target_id], "scoring_gt mirror", prediction=False))

    if prep_report.get("sequences") != len(sequence_indices):
        raise ContractError("prep_report sequence count disagrees with frame contract")
    if prep_report.get("mirrorable_duplicate_gt_rows") != len(mirror_rows):
        raise ContractError("prep_report mirror row count disagrees with mirror_map")
    if prep_report.get("scoring_rows_with_mirror_sensitivity") != len(scoring_rows):
        raise ContractError("prep_report scoring row count disagrees with scoring_gt")

    first_mask = positions == 0
    audit = {
        "formal_train405_contract": True,
        "pair_profile": pair_profile,
        "pair_scope": scope["pair_scope"],
        "does_not_certify": scope["does_not_certify"],
        "offline_two_pass_crop": crop_audit,
        "contracts_dir": str(contracts),
        "contract_sha256_manifest": hashes,
        "main_csv": {"path": str(main_csv), "sha256": _sha256(main_csv)},
        "source_csv": {"path": str(source_csv), "sha256": _sha256(source_csv)},
        "diagnostics": diag_audit,
        "rows": n,
        "sequences": len(sequence_indices),
        "capture_groups": len(group_folds),
        "fold_rows": {str(f): int(np.sum(fold_values == f)) for f in range(N_FOLDS)},
        "invalid_official_gt_rows": int(np.sum(~gt_valid)),
        "mirror_sensitivity_rows": len(mirror_rows),
    }
    return Dataset(
        tuple(sample_ids), tuple(sequences), frame_numbers, positions, tuple(modalities),
        tuple(capture_groups), fold_values, first_mask, gt, main, source, sequence_indices,
        np.asarray(mirror_source_indices, dtype=np.int64),
        np.asarray(mirror_gt, dtype=np.float64).reshape((-1, 4)), audit,
    )


def load_explicit_smoke_dataset(
    sample_contract: str | Path,
    gt_csv: str | Path,
    main_csv: str | Path,
    source_csv: str | Path,
) -> Dataset:
    """Build a deterministic grouped split for an explicitly requested legacy smoke pair.

    This is intentionally a separate, visibly non-formal path.  The exact sample IDs
    make GT alignment unambiguous and the already-approved modality-prefix rule makes
    grouping deterministic.  It cannot certify tracker diagnostics, zip selection, or
    mirror sensitivity, so its audit marks those formal train405 claims unavailable.
    """
    sample_contract = Path(sample_contract).expanduser().resolve()
    gt_csv = Path(gt_csv).expanduser().resolve()
    main_csv = Path(main_csv).expanduser().resolve()
    source_csv = Path(source_csv).expanduser().resolve()
    sample_rows = _read_csv(sample_contract, SUBMISSION_COLUMNS, "explicit smoke contract")
    sample_ids = _unique_ids(sample_rows, "explicit smoke contract")
    for row in sample_rows:
        if any(float(row[column]) != 0.0 for column in SUBMISSION_COLUMNS[1:]):
            raise ContractError(f"explicit smoke contract must contain zero placeholders: {row['ID']}")

    all_gt_rows = _read_csv(gt_csv, SUBMISSION_COLUMNS, "explicit smoke GT")
    all_gt_ids = _unique_ids(all_gt_rows, "explicit smoke GT")
    gt_by_id = dict(zip(all_gt_ids, all_gt_rows, strict=True))
    missing_gt = [ident for ident in sample_ids if ident not in gt_by_id]
    if missing_gt:
        raise ContractError(f"explicit smoke GT missing {len(missing_gt)} contract IDs: {missing_gt[:5]}")
    gt = np.asarray([
        _float_box(gt_by_id[ident], "explicit smoke GT", prediction=False)
        for ident in sample_ids
    ], dtype=np.float64)

    sequences: list[str] = []
    frame_numbers = np.empty(len(sample_ids), dtype=np.int64)
    positions = np.empty(len(sample_ids), dtype=np.int64)
    modalities: list[str] = []
    groups: list[str] = []
    seq_indices_raw: dict[str, list[int]] = {}
    for idx, ident in enumerate(sample_ids):
        seq, frame = _parse_id(ident, "explicit smoke contract")
        try:
            modality, group = seq.split("-", 1)
        except ValueError as exc:
            raise ContractError(f"explicit smoke sequence lacks modality prefix: {seq}") from exc
        if modality not in {"nir", "rednir", "vis"} or not group:
            raise ContractError(f"explicit smoke sequence has unsupported modality/group: {seq}")
        sequences.append(seq)
        frame_numbers[idx] = frame
        modalities.append(modality)
        groups.append(group)
        seq_indices_raw.setdefault(seq, []).append(idx)

    sequence_indices: dict[str, np.ndarray] = {}
    for seq, raw_indices in seq_indices_raw.items():
        indices = np.asarray(sorted(raw_indices, key=lambda idx: frame_numbers[idx]), dtype=np.int64)
        if len(set(frame_numbers[indices].tolist())) != len(indices):
            raise ContractError(f"explicit smoke contract repeats a frame in {seq}")
        positions[indices] = np.arange(len(indices), dtype=np.int64)
        sequence_indices[seq] = indices

    group_to_seqs: dict[str, list[str]] = {}
    for seq in sequence_indices:
        group_to_seqs.setdefault(seq.split("-", 1)[1], []).append(seq)
    if len(group_to_seqs) < N_FOLDS:
        raise ContractError(
            f"explicit smoke contract has only {len(group_to_seqs)} physical groups; need >= {N_FOLDS}")
    group_weight = {
        group: sum(len(sequence_indices[seq]) for seq in seqs)
        for group, seqs in group_to_seqs.items()
    }
    fold_frames = [0] * N_FOLDS
    fold_sequences = [0] * N_FOLDS
    group_to_fold: dict[str, int] = {}
    for group in sorted(group_to_seqs, key=lambda value: (-group_weight[value], value)):
        fold = min(range(N_FOLDS), key=lambda f: (fold_frames[f], fold_sequences[f], f))
        group_to_fold[group] = fold
        fold_frames[fold] += group_weight[group]
        fold_sequences[fold] += len(group_to_seqs[group])
    folds = np.asarray([group_to_fold[group] for group in groups], dtype=np.int64)
    if set(folds.tolist()) != set(range(N_FOLDS)):
        raise ContractError("explicit smoke deterministic split did not populate all five folds")

    main = _prediction_boxes(main_csv, sample_ids, "explicit smoke main prediction")
    source = _prediction_boxes(source_csv, sample_ids, "explicit smoke source prediction")
    audit = {
        "formal_train405_contract": False,
        "purpose": "legacy exact-pair CPU smoke only",
        "pair_scope": "caller-supplied exact prediction pair; this harness does not infer its tracker/crop profile",
        "does_not_certify": "formal train405 cache, Rank-A crop reproducibility, or tracker provenance",
        "limitations": [
            "tracker diagnostics/run signatures unavailable",
            "zip-to-GT observed-block mapping unavailable",
            "mirror-map official-row sensitivity unavailable",
            "must not be reported as the formal 405 OOF result",
        ],
        "sample_contract": {"path": str(sample_contract), "sha256": _sha256(sample_contract)},
        "gt_csv": {"path": str(gt_csv), "sha256": _sha256(gt_csv)},
        "main_csv": {"path": str(main_csv), "sha256": _sha256(main_csv)},
        "source_csv": {"path": str(source_csv), "sha256": _sha256(source_csv)},
        "rows": len(sample_ids),
        "sequences": len(sequence_indices),
        "capture_groups": len(group_to_seqs),
        "fold_rows": {str(f): int(np.sum(folds == f)) for f in range(N_FOLDS)},
        "derived_group_to_fold": dict(sorted(group_to_fold.items())),
        "invalid_official_gt_rows": int(np.sum(~np.all(gt > 0, axis=1))),
        "mirror_sensitivity_rows": 0,
    }
    return Dataset(
        tuple(sample_ids), tuple(sequences), frame_numbers, positions, tuple(modalities),
        tuple(groups), folds, positions == 0, gt, main, source, sequence_indices,
        np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.float64), audit,
    )


def apply_correction(raw: np.ndarray, first_mask: np.ndarray, mode: str) -> np.ndarray:
    if mode not in CORR_MODES:
        raise ValueError(f"unsupported correction mode {mode!r}")
    out = raw.copy()
    top = 1.0 if mode in {"top-only", "both"} else 0.0
    left = 1.0 if mode == "both" else 0.0
    active = ~first_mask
    if left:
        old_x = out[active, 0].copy()
        new_x = np.maximum(0.0, old_x - left)
        out[active, 0] = new_x
        out[active, 2] += old_x - new_x
    if top:
        old_y = out[active, 1].copy()
        new_y = np.maximum(0.0, old_y - top)
        out[active, 1] = new_y
        out[active, 3] += old_y - new_y
    out[first_mask] = raw[first_mask]
    return out


def _frozen_runs(boxes: np.ndarray, sequence_indices: dict[str, np.ndarray], *, tolerant: bool) -> np.ndarray:
    runs = np.zeros(len(boxes), dtype=np.int64)
    for indices in sequence_indices.values():
        run = 0
        for pos, idx in enumerate(indices):
            if pos:
                prev = indices[pos - 1]
                same = bool(np.allclose(boxes[idx], boxes[prev])) if tolerant else bool(np.array_equal(boxes[idx], boxes[prev]))
                run = run + 1 if same else 0
            runs[idx] = run
    return runs


def frozen_splice(
    main: np.ndarray,
    source: np.ndarray,
    sequence_indices: dict[str, np.ndarray],
    first_mask: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    if K < 1:
        raise ValueError("K must be >= 1")
    main_run = _frozen_runs(main, sequence_indices, tolerant=False)
    source_run = _frozen_runs(source, sequence_indices, tolerant=False)
    selected = (main_run >= K) & (source_run < SPLICE_SRC_MAX_RUN)
    selected &= ~first_mask
    out = main.copy()
    out[selected] = source[selected]
    out[first_mask] = main[first_mask]
    return out, selected


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _feature(
    a: np.ndarray,
    b: np.ndarray,
    prev_a: np.ndarray | None,
    prev_b: np.ndarray | None,
    run_a: int,
    run_b: int,
    fraction: float,
    modality: str,
) -> np.ndarray:
    """v056-shaped 25-D feature vector; no GT/fold/group/sequence-name input."""
    if modality not in {"nir", "rednir", "vis"}:
        raise ValueError(f"unknown modality {modality!r}")
    iab = _box_iou(a, b)
    area_a = max(float(a[2] * a[3]), 1e-6)
    area_b = max(float(b[2] * b[3]), 1e-6)
    aspect_a = a[2] / max(a[3], 1e-6)
    aspect_b = b[2] / max(b[3], 1e-6)
    ca = (a[0] + a[2] / 2, a[1] + a[3] / 2)
    cb = (b[0] + b[2] / 2, b[1] + b[3] / 2)
    center_dist = np.hypot(ca[0] - cb[0], ca[1] - cb[1]) / max(np.sqrt(area_a), 1.0)
    va = np.zeros(4, dtype=np.float64) if prev_a is None else a - prev_a
    vb = np.zeros(4, dtype=np.float64) if prev_b is None else b - prev_b
    return np.asarray([
        iab, np.log(area_a), np.log(area_b), np.log(area_a / area_b), aspect_a, aspect_b,
        center_dist, *va.tolist(), *vb.tolist(), float(run_a), float(run_b), fraction,
        float(modality == "nir"), float(modality == "rednir"), float(modality == "vis"),
        a[2], a[3], b[2], b[3],
    ], dtype=np.float64)


def _qhead_feature_cache(
    dataset: Dataset, main: np.ndarray, source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    main_runs = _frozen_runs(main, dataset.sequence_indices, tolerant=True)
    source_runs = _frozen_runs(source, dataset.sequence_indices, tolerant=True)
    features = np.empty((len(dataset.ids), len(FEATURE_NAMES)), dtype=np.float64)
    for seq, indices in dataset.sequence_indices.items():
        n_seq = max(len(indices), 1)
        prev_a: np.ndarray | None = None
        prev_b: np.ndarray | None = None
        for pos, idx in enumerate(indices):
            features[idx] = _feature(
                main[idx], source[idx], prev_a, prev_b, int(main_runs[idx]),
                int(source_runs[idx]), pos / n_seq, dataset.modalities[idx],
            )
            prev_a, prev_b = main[idx], source[idx]
    if not np.isfinite(features).all():
        raise ContractError("quality-head prediction-only feature matrix contains NaN/Inf")
    return features, main_runs, source_runs


def _qhead_gate(
    dataset: Dataset,
    main: np.ndarray,
    source: np.ndarray,
    main_runs: np.ndarray,
    source_runs: np.ndarray,
    K: int,
) -> np.ndarray:
    # This is the full deployment gate.  It intentionally has no GT term.
    rednir = np.asarray([modality == "rednir" for modality in dataset.modalities])
    iab = np.asarray([_box_iou(a, b) for a, b in zip(main, source, strict=True)])
    return (
        rednir & (~dataset.first_mask) & (main_runs < K) &
        (source_runs < SPLICE_SRC_MAX_RUN) & (iab < QHEAD_IAB_MAX)
    )


def _fit_qhead_fold_models(
    dataset: Dataset,
    main: np.ndarray,
    source: np.ndarray,
    features: np.ndarray,
    deploy_gate: np.ndarray,
) -> list[QHeadModel]:
    valid_gt = np.all(dataset.gt > 0, axis=1)
    iou_a = _ious(main, dataset.gt)
    iou_b = _ious(source, dataset.gt)
    non_tie = np.abs(iou_a - iou_b) >= 1e-9
    models: list[QHeadModel] = []
    for heldout in range(N_FOLDS):
        train_mask = (dataset.folds != heldout) & deploy_gate & valid_gt & non_tie
        train_folds = tuple(sorted(set(dataset.folds[train_mask].tolist())))
        expected_train_folds = tuple(fold for fold in range(N_FOLDS) if fold != heldout)
        if train_folds != expected_train_folds:
            raise ContractError(
                f"qhead heldout fold {heldout}: eligible training rows do not cover all four train folds; "
                f"got {train_folds}, required {expected_train_folds}")
        X = features[train_mask]
        y = (iou_a[train_mask] > iou_b[train_mask]).astype(np.float64)
        n_a = int(y.sum())
        n_b = int(len(y) - n_a)
        if len(y) < 10 or n_a < 2 or n_b < 2:
            raise ContractError(
                f"qhead heldout fold {heldout}: insufficient/degenerate train data "
                f"n={len(y)}, A_better={n_a}, B_better={n_b}")
        w, mu, sd = fit_logit(X, y)
        if w.shape != (len(FEATURE_NAMES),) or not all(
            np.isfinite(arr).all() for arr in (w, mu, sd)
        ) or np.any(sd <= 0):
            raise ContractError(f"qhead heldout fold {heldout}: invalid fitted parameters")
        models.append(QHeadModel(heldout, train_folds, len(y), n_a, n_b, w, mu, sd))
    return models


def apply_crossfit_qhead(
    dataset: Dataset,
    spliced: np.ndarray,
    main: np.ndarray,
    source: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray, list[QHeadModel]]:
    """Fit four folds/apply one fold five times, then assemble exact OOF decisions."""
    features, main_runs, source_runs = _qhead_feature_cache(dataset, main, source)
    gate = _qhead_gate(dataset, main, source, main_runs, source_runs, K)
    models = _fit_qhead_fold_models(dataset, main, source, features, gate)
    out = spliced.copy()
    selected = np.zeros(len(dataset.ids), dtype=bool)
    for model in models:
        heldout_gate = gate & (dataset.folds == model.heldout_fold)
        indices = np.flatnonzero(heldout_gate)
        if len(indices):
            p_a_better = predict_p(features[indices], model.w, model.mu, model.sd)
            take_source = indices[p_a_better < 0.5 - QHEAD_MARGIN]
            out[take_source] = source[take_source]
            selected[take_source] = True
    out[dataset.first_mask] = dataset.raw_main[dataset.first_mask]
    selected[dataset.first_mask] = False
    return out, selected, models


def _ious(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError(f"box arrays must both have shape (N,4), got {pred.shape}, {gt.shape}")
    result = np.full(len(gt), -1.0, dtype=np.float64)
    valid = np.all(gt > 0, axis=1)
    if np.any(valid):
        g, p = gt[valid], pred[valid]
        left = np.maximum(g[:, 0], p[:, 0])
        right = np.minimum(g[:, 0] + g[:, 2], p[:, 0] + p[:, 2])
        top = np.maximum(g[:, 1], p[:, 1])
        bottom = np.minimum(g[:, 1] + g[:, 3], p[:, 1] + p[:, 3])
        inter = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
        union = g[:, 2] * g[:, 3] + p[:, 2] * p[:, 3] - inter
        result[valid] = np.clip(inter / union, 0.0, 1.0)
    return result


def _auc_from_ious(ious: np.ndarray) -> float:
    if not len(ious):
        raise ValueError("cannot evaluate zero rows")
    return float(np.mean([(ious >= threshold).mean() for threshold in IOU_THRESHOLDS]))


def _metrics(dataset: Dataset, pred: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    if pred.shape != dataset.gt.shape or not np.isfinite(pred).all():
        raise ContractError("candidate predictions have invalid shape or NaN/Inf")
    if np.any(pred[:, :2] < 0) or np.any(pred[:, 2:] <= 0):
        raise ContractError("candidate predictions contain invalid boxes")
    use = np.ones(len(pred), dtype=bool) if mask is None else mask
    observed_iou = _ious(pred[use], dataset.gt[use])
    mirror_mask = use[dataset.mirror_source_indices] if len(dataset.mirror_source_indices) else np.zeros(0, dtype=bool)
    mirror_iou = (
        _ious(pred[dataset.mirror_source_indices[mirror_mask]], dataset.mirror_gt[mirror_mask])
        if np.any(mirror_mask) else np.empty(0, dtype=np.float64)
    )
    sensitivity_iou = np.concatenate([observed_iou, mirror_iou])
    return {
        "auc_observed": _auc_from_ious(observed_iou),
        "mean_iou_observed": float(observed_iou.mean()),
        "n_observed": int(len(observed_iou)),
        "n_invalid_gt": int(np.sum(observed_iou < 0)),
        "auc_mirror_sensitivity": _auc_from_ious(sensitivity_iou),
        "n_mirror_sensitivity": int(len(mirror_iou)),
    }


def _model_doc(model: QHeadModel) -> dict[str, Any]:
    payload = {
        "heldout_fold": model.heldout_fold,
        "train_folds": list(model.train_folds),
        "n_train": model.n_train,
        "class_a_better": model.class_a_better,
        "class_b_better": model.class_b_better,
        "w": model.w.tolist(), "mu": model.mu.tolist(), "sd": model.sd.tolist(),
    }
    payload["parameter_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def evaluate_grid(
    dataset: Dataset,
    corr_modes: Sequence[str],
    ks: Sequence[int],
    qhead_modes: Sequence[str] = ("none", "crossfit-v1"),
    *,
    include_predictions: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    corr_modes = tuple(dict.fromkeys(corr_modes))
    ks = tuple(dict.fromkeys(int(k) for k in ks))
    qhead_modes = tuple(dict.fromkeys(qhead_modes))
    if not corr_modes or any(mode not in CORR_MODES for mode in corr_modes):
        raise ValueError(f"corr_modes must be a non-empty subset of {CORR_MODES}")
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    if not qhead_modes or any(mode not in {"none", "crossfit-v1"} for mode in qhead_modes):
        raise ValueError("qhead_modes must be a non-empty subset of none,crossfit-v1")

    baseline_id = "corr=none__policy=main"
    candidates: list[dict[str, Any]] = []
    prediction_outputs: dict[str, np.ndarray] = {}
    model_outputs: dict[str, Any] = {}

    def add_candidate(
        candidate_id: str,
        pred: np.ndarray,
        *,
        corr: str,
        K: int | None,
        qhead: str,
        splice_selected: np.ndarray | None = None,
        qhead_selected: np.ndarray | None = None,
        models: list[QHeadModel] | None = None,
    ) -> None:
        if not np.array_equal(pred[dataset.first_mask], dataset.raw_main[dataset.first_mask]):
            raise ContractError(f"{candidate_id}: first frames are not exact raw main init")
        overall = _metrics(dataset, pred)
        folds = []
        for fold in range(N_FOLDS):
            mask = dataset.folds == fold
            fold_metrics = _metrics(dataset, pred, mask)
            fold_metrics.update({
                "fold": fold,
                "n_splice_selected": int(np.sum(splice_selected & mask)) if splice_selected is not None else 0,
                "n_qhead_selected": int(np.sum(qhead_selected & mask)) if qhead_selected is not None else 0,
            })
            folds.append(fold_metrics)
        candidate = {
            "candidate_id": candidate_id, "corr": corr, "K": K, "qhead": qhead,
            **overall,
            "n_splice_selected": int(np.sum(splice_selected)) if splice_selected is not None else 0,
            "n_qhead_selected": int(np.sum(qhead_selected)) if qhead_selected is not None else 0,
            "first_frames_exact_raw_main": int(np.sum(dataset.first_mask)),
            "folds": folds,
        }
        if models is not None:
            docs = [_model_doc(model) for model in models]
            candidate["qhead_fold_models"] = [{
                key: value for key, value in doc.items() if key not in {"w", "mu", "sd"}
            } for doc in docs]
            model_outputs[candidate_id] = docs
        candidates.append(candidate)
        if include_predictions:
            prediction_outputs[candidate_id] = pred.copy()

    # Baselines use no GT and make the first-frame invariant explicit.
    add_candidate(baseline_id, dataset.raw_main.copy(), corr="none", K=None, qhead="none")
    raw_source_safe = dataset.raw_source.copy()
    raw_source_safe[dataset.first_mask] = dataset.raw_main[dataset.first_mask]
    add_candidate("corr=none__policy=source", raw_source_safe, corr="none", K=None, qhead="none")

    corr_main_predictions: dict[str, np.ndarray] = {}
    for corr in corr_modes:
        main = apply_correction(dataset.raw_main, dataset.first_mask, corr)
        source = apply_correction(dataset.raw_source, dataset.first_mask, corr)
        corr_main_predictions[corr] = main
        corr_id = f"corr={corr}__policy=main"
        if corr_id != baseline_id:
            add_candidate(corr_id, main, corr=corr, K=None, qhead="none")
        source_id = f"corr={corr}__policy=source"
        if source_id != "corr=none__policy=source":
            source_safe = source.copy()
            source_safe[dataset.first_mask] = dataset.raw_main[dataset.first_mask]
            add_candidate(source_id, source_safe, corr=corr, K=None, qhead="none")
        for K in ks:
            spliced, splice_selected = frozen_splice(
                main, source, dataset.sequence_indices, dataset.first_mask, K)
            if "none" in qhead_modes:
                add_candidate(
                    f"corr={corr}__K={K}__qhead=none", spliced, corr=corr, K=K,
                    qhead="none", splice_selected=splice_selected,
                )
            if "crossfit-v1" in qhead_modes:
                qpred, qselected, models = apply_crossfit_qhead(
                    dataset, spliced, main, source, K)
                add_candidate(
                    f"corr={corr}__K={K}__qhead=crossfit-v1", qpred, corr=corr, K=K,
                    qhead="crossfit-v1", splice_selected=splice_selected,
                    qhead_selected=qselected, models=models,
                )

    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    raw_auc = by_id[baseline_id]["auc_observed"]
    for candidate in candidates:
        candidate["delta_vs_raw_main"] = candidate["auc_observed"] - raw_auc
        corr_baseline = by_id.get(f"corr={candidate['corr']}__policy=main", by_id[baseline_id])
        candidate["delta_vs_same_corr_main"] = candidate["auc_observed"] - corr_baseline["auc_observed"]
        positive_folds = 0
        worst = float("inf")
        for fold_result, corr_fold in zip(candidate["folds"], corr_baseline["folds"], strict=True):
            delta = fold_result["auc_observed"] - corr_fold["auc_observed"]
            fold_result["delta_vs_same_corr_main"] = delta
            positive_folds += int(delta > 0)
            worst = min(worst, delta)
        candidate["positive_folds_vs_same_corr_main"] = positive_folds
        candidate["worst_fold_delta_vs_same_corr_main"] = worst

    ranking = [
        candidate["candidate_id"] for candidate in
        sorted(candidates, key=lambda item: (-item["auc_observed"], item["candidate_id"]))
    ]
    report = {
        "schema_version": 1,
        "protocol": {
            "name": "physical-capture-grouped-5-fold-crossfit-v1",
            "n_folds": N_FOLDS,
            "group_rule": "strip modality prefix only; preserve full basename",
            "metric": "official Success AUC: mean success at IoU thresholds 0.02..1.00",
            "primary_metric": "auc_observed (one prediction row per physical image)",
            "secondary_metric": "auc_mirror_sensitivity (only exact duplicate GT blocks)",
            "candidate_grid_is_frozen": True,
            "selection_protocol": (
                "pre-locked fixed grid supplied on the CLI; every policy is evaluated unchanged "
                "on all five held-out folds; no fold-specific threshold/K/correction selection"
            ),
            "nested_hyperparameter_selection": False,
            "post_selection_warning": (
                "Ranking fixed candidates by this OOF table is model selection; do not describe the "
                "top-ranked OOF value as an unbiased post-selection test estimate.  A newly tuned "
                "threshold/grid requires nested CV or a new untouched evaluation set."
            ),
            "qhead": {
                "mode": "crossfit-v1",
                "fit_rule": "for heldout fold f, fit only rows from folds != f",
                "train_gate": (
                    "rednir, non-first, A_run<K, B_run<2, IoU(A,B)<0.30, valid train GT, non-tie"
                ),
                "heldout_gate": "same prediction-only gate without any GT term",
                "feature_names": list(FEATURE_NAMES),
                "forbidden_model_inputs": [
                    "ground-truth box", "GT-derived score/label at inference", "sequence basename",
                    "capture-group name", "fold number", "per-sequence learned constant",
                ],
            },
            "first_frame_policy": "exact raw main init after every stage",
        },
        "input_audit": dataset.input_audit,
        "grid": {"corr": list(corr_modes), "K": list(ks), "qhead": list(qhead_modes)},
        "ranking_by_observed_auc": ranking,
        "candidates": candidates,
    }
    return report, prediction_outputs, model_outputs


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _csv_text(columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> str:
    import io
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buf.getvalue()


def write_outputs(
    out_dir: Path,
    dataset: Dataset,
    report: dict[str, Any],
    predictions: dict[str, np.ndarray],
    models: dict[str, Any],
) -> None:
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ContractError(f"out-dir must not already be non-empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_columns = (
        "candidate_id", "corr", "K", "qhead", "auc_observed", "delta_vs_raw_main",
        "delta_vs_same_corr_main", "auc_mirror_sensitivity", "n_observed",
        "n_splice_selected", "n_qhead_selected", "positive_folds_vs_same_corr_main",
        "worst_fold_delta_vs_same_corr_main", "first_frames_exact_raw_main",
    )
    fold_columns = (
        "candidate_id", "corr", "K", "qhead", "fold", "auc_observed",
        "delta_vs_same_corr_main", "auc_mirror_sensitivity", "n_observed",
        "n_splice_selected", "n_qhead_selected",
    )
    metric_rows = [{key: candidate.get(key) for key in metric_columns}
                   for candidate in report["candidates"]]
    fold_rows = []
    for candidate in report["candidates"]:
        for fold in candidate["folds"]:
            fold_rows.append({
                **{key: candidate.get(key) for key in ("candidate_id", "corr", "K", "qhead")},
                **fold,
            })
    _atomic_text(out_dir / "candidate_metrics.csv", _csv_text(metric_columns, metric_rows))
    _atomic_text(out_dir / "fold_metrics.csv", _csv_text(fold_columns, fold_rows))
    _atomic_text(
        out_dir / "qhead_fold_models.json",
        json.dumps(models, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    if predictions:
        pred_root = out_dir / "oof_predictions"
        pred_root.mkdir()
        for candidate_id, boxes in predictions.items():
            safe_name = candidate_id.replace("=", "-").replace("__", "_") + ".csv"
            rows = []
            for ident, box in zip(dataset.ids, boxes, strict=True):
                rows.append({"ID": ident, **dict(zip(SUBMISSION_COLUMNS[1:], box.tolist(), strict=True))})
            _atomic_text(pred_root / safe_name, _csv_text(SUBMISSION_COLUMNS, rows))
    output_hashes = {
        path.name: _sha256(path) for path in sorted(out_dir.iterdir()) if path.is_file()
    }
    report["output_audit"] = output_hashes
    _atomic_text(
        out_dir / "report.json",
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def _comma_values(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values


def _comma_ints(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("K list must contain integers") from exc
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("K list must contain positive integers")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    contract_mode = ap.add_mutually_exclusive_group(required=True)
    contract_mode.add_argument("--contracts", help="formal prep_oof405_frames.py contracts directory")
    contract_mode.add_argument(
        "--smoke-contract",
        help="explicit legacy val sample contract; non-formal smoke path only",
    )
    ap.add_argument("--smoke-gt", help="required with --smoke-contract; may be a GT superset")
    ap.add_argument("--main", required=True, help="raw full_sam3/submission.csv")
    ap.add_argument("--source", required=True, help="raw full_samurai/submission.csv")
    ap.add_argument("--main-diagnostics", help="default: diagnostics.json beside --main")
    ap.add_argument("--source-diagnostics", help="default: diagnostics.json beside --source")
    ap.add_argument("--pair-profile", default="rankB_robust", choices=sorted(PAIR_SCOPES),
                    help="which production pair this formal cache certifies; "
                         "rankB_robust_crop additionally requires the crop diagnostics and meta")
    ap.add_argument("--main-crop-diagnostics",
                    help="rankB_robust_crop only: crop_sam3/diagnostics.json")
    ap.add_argument("--source-crop-diagnostics",
                    help="rankB_robust_crop only: crop_samurai/diagnostics.json")
    ap.add_argument("--crop-meta",
                    help="rankB_robust_crop only: offline_two_pass/crop_meta.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--corr", type=_comma_values, default=CORR_MODES,
                    help="comma list from none,top-only,both")
    ap.add_argument("--ks", type=_comma_ints, default=(2, 3, 5, 6, 8, 24),
                    help="comma-separated frozen splice K candidates")
    ap.add_argument("--qhead", type=_comma_values, default=("none", "crossfit-v1"),
                    help="comma list from none,crossfit-v1")
    ap.add_argument("--write-oof", action="store_true",
                    help="also write every full OOF prediction CSV (large)")
    args = ap.parse_args(argv)
    try:
        if args.smoke_contract:
            if not args.smoke_gt:
                raise ContractError("--smoke-gt is required with --smoke-contract")
            if (args.main_diagnostics or args.source_diagnostics
                    or args.main_crop_diagnostics or args.source_crop_diagnostics
                    or args.crop_meta or args.pair_profile != "rankB_robust"):
                raise ContractError(
                    "diagnostic overrides are not accepted in non-formal smoke mode; "
                    "use --contracts for a certified cache")
            dataset = load_explicit_smoke_dataset(
                args.smoke_contract, args.smoke_gt, args.main, args.source)
        else:
            if args.smoke_gt:
                raise ContractError("--smoke-gt is only valid with --smoke-contract")
            dataset = load_dataset(
                args.contracts, args.main, args.source,
                args.main_diagnostics, args.source_diagnostics,
                pair_profile=args.pair_profile,
                main_crop_diagnostics=args.main_crop_diagnostics,
                source_crop_diagnostics=args.source_crop_diagnostics,
                crop_meta=args.crop_meta,
            )
        report, predictions, models = evaluate_grid(
            dataset, args.corr, args.ks, args.qhead,
            include_predictions=bool(args.write_oof),
        )
        write_outputs(Path(args.out_dir).expanduser().resolve(), dataset, report, predictions, models)
    except (ContractError, ValueError, OSError) as exc:
        print(f"FAIL-CLOSED: {exc}", file=os.sys.stderr)
        return 2
    best = report["ranking_by_observed_auc"][0]
    best_row = next(c for c in report["candidates"] if c["candidate_id"] == best)
    print(
        f"OK: {len(dataset.ids)} rows / {len(dataset.sequence_indices)} sequences / "
        f"{N_FOLDS} grouped folds; top-ranked fixed-grid diagnostic {best} "
        f"AUC={best_row['auc_observed']:.5f}; outputs={args.out_dir}"
    )
    print("Post-selection warning: this top-ranked OOF value is not an unbiased test estimate.")
    print("No cloud action and no Kaggle submission were performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
