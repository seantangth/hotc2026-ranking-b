#!/usr/bin/env python3
"""Build the zero-GPU mixed-resolution Ranking-A candidate, fail closed.

The mixed raw main uses the proven crop output on crop-selected sequences and
the 1344 full-frame output everywhere else.  Selection comes from E46's
``test_meta.json`` and, by default, is independently cross-checked against the
set of sequences changed by v023 relative to its v006 full-frame base.

Optional finalized variants are produced by invoking ``finalize_submission.py``
rather than duplicating its correction, K6 splice, qhead, first-frame restore,
or formal validation logic.  This tool only writes local candidate artifacts;
it never talks to Kaggle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

import finalize_submission as fs  # noqa: E402


DEFAULT_CROP = ROOT / "5_outputs/submissions/sub_v023_cropwiden55.csv"
DEFAULT_NONCROP = ROOT / "5_outputs/submissions/sub_v013_e20_1344.csv"
DEFAULT_SELECTION_META = ROOT / "5_outputs/e46_sam3_crop_scores_20260819/test_meta.json"
DEFAULT_SELECTION_REFERENCE = ROOT / "5_outputs/submissions/sub_v006_e15sam3.csv"
DEFAULT_SOURCE = ROOT / "5_outputs/submissions/sub_v012_e23b_sam21_ablation.csv"
DEFAULT_SAMPLE = ROOT / "1_data/raw/sample_submisson.csv"
DEFAULT_OUT_DIR = ROOT / "5_outputs/mixed_resolution_20260823"
FINALIZER = SRC_ROOT / "finalize_submission.py"

RAW_NAME = "sub_mixed_v023crop_v013noncrop_raw.csv"
FINAL_NAME = "sub_mixed_v023crop_v013noncrop_both_K6.csv"
QHEAD_NAME = "sub_mixed_v023crop_v013noncrop_both_K6_qhead_v056.csv"
MANIFEST_NAME = "build_manifest.json"


class MixedCandidateError(ValueError):
    """An input or output violates a mixed-candidate provenance invariant."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sequence_set(data) -> set[str]:
    return set(data)


def load_selected_sequences(path: Path, sample) -> set[str]:
    """Read E46 metadata and ensure it describes whole sample sequences."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MixedCandidateError(f"selection metadata unreadable: {path}: {exc}") from exc

    if not isinstance(payload, dict) or not payload:
        raise MixedCandidateError("selection metadata must be a non-empty JSON object keyed by sequence")

    selected = set()
    for key, meta in payload.items():
        if not isinstance(key, str) or not key:
            raise MixedCandidateError("selection metadata contains an empty/non-string sequence key")
        if not isinstance(meta, dict):
            raise MixedCandidateError(f"selection metadata for {key} must be an object")
        if meta.get("seq", key) != key:
            raise MixedCandidateError(
                f"selection metadata key/seq mismatch: key={key!r}, seq={meta.get('seq')!r}"
            )
        if key not in sample:
            raise MixedCandidateError(f"selected sequence absent from sample: {key}")
        if not sample[key]:
            raise MixedCandidateError(f"selected sequence has no sample frames: {key}")
        frames = meta.get("frames")
        if not (isinstance(frames, list) and len(frames) == 2):
            raise MixedCandidateError(f"selection metadata frames must be [first,last] for {key}")
        declared = tuple(frames)
        actual = (min(sample[key]), max(sample[key]))
        if declared != actual:
            raise MixedCandidateError(
                f"selection metadata frame range mismatch for {key}: {declared} != {actual}"
            )
        selected.add(key)
    return selected


def validate_exact_input_sets(crop_order, noncrop_order, sample_order) -> None:
    errors = fs.validate_input_sets(crop_order, noncrop_order, sample_order)
    if errors:
        raise MixedCandidateError("crop/noncrop/sample exact-set failure: " + "; ".join(errors))


def first_frame_consensus(*named_data) -> int:
    """Require all inputs to carry the same canonical init box per sequence."""
    if not named_data:
        raise MixedCandidateError("no inputs supplied for first-frame consensus")
    base_name, base = named_data[0]
    seqs = sequence_set(base)
    for name, data in named_data[1:]:
        if sequence_set(data) != seqs:
            raise MixedCandidateError(f"{name} sequence set differs during first-frame validation")
    for seq in sorted(seqs):
        base_first = min(base[seq])
        base_box = base[seq][base_first]
        for name, data in named_data[1:]:
            first = min(data[seq])
            if first != base_first:
                raise MixedCandidateError(
                    f"first-frame index mismatch for {seq}: {base_name}={base_first}, {name}={first}"
                )
            if data[seq][first] != base_box:
                raise MixedCandidateError(
                    f"first-frame box mismatch for {seq}: {base_name} != {name}"
                )
    return len(seqs)


def changed_sequences(candidate, reference) -> tuple[set[str], int]:
    if sequence_set(candidate) != sequence_set(reference):
        raise MixedCandidateError("selection cross-check inputs have different sequence sets")
    changed = set()
    changed_rows = 0
    for seq in candidate:
        if set(candidate[seq]) != set(reference[seq]):
            raise MixedCandidateError(f"selection cross-check frame set differs for {seq}")
        for frame, box in candidate[seq].items():
            if box != reference[seq][frame]:
                changed.add(seq)
                changed_rows += 1
    return changed, changed_rows


def build_hybrid(crop, noncrop, selected: set[str]):
    seqs = sequence_set(crop)
    if sequence_set(noncrop) != seqs:
        raise MixedCandidateError("crop and noncrop sequence sets differ")
    unknown = selected - seqs
    if unknown:
        raise MixedCandidateError(f"selected sequences absent from inputs: {sorted(unknown)}")
    out = {}
    for seq in seqs:
        chosen = crop if seq in selected else noncrop
        out[seq] = {frame: list(box) for frame, box in chosen[seq].items()}
    return out


def assert_hybrid_provenance(hybrid, crop, noncrop, selected: set[str]) -> tuple[int, int]:
    crop_rows = noncrop_rows = 0
    for seq in hybrid:
        expected = crop[seq] if seq in selected else noncrop[seq]
        if hybrid[seq] != expected:
            branch = "crop" if seq in selected else "noncrop"
            raise MixedCandidateError(f"hybrid does not exactly equal its {branch} source for {seq}")
        if seq in selected:
            crop_rows += len(hybrid[seq])
        else:
            noncrop_rows += len(hybrid[seq])
    return crop_rows, noncrop_rows


def validate_candidate_file(path: Path, sample_path: Path, raw_main) -> tuple[int, int]:
    data, order = fs.load(path, Path(path).name)
    _, sample_order = fs.load(sample_path, "sample", validate_boxes=False)
    if order != sample_order:
        raise MixedCandidateError(f"output row order is not canonical sample order for {path}")
    errors = fs.validate(data, order, sample_path, raw_main=raw_main)
    if errors:
        raise MixedCandidateError(f"output validation failed for {path}: " + "; ".join(errors))
    return len(order), len(data)


def run_checked(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode:
        raise MixedCandidateError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def finalize_variant(
    raw_path: Path,
    source_path: Path,
    sample_path: Path,
    out_path: Path,
    k: int,
    qhead: str,
    qhead_weights: Path,
) -> str:
    cmd = [
        sys.executable,
        str(FINALIZER),
        "--main",
        str(raw_path),
        "--source",
        str(source_path),
        "--sample",
        str(sample_path),
        "--out",
        str(out_path),
        "--corr",
        "both",
        "--K",
        str(k),
        "--qhead",
        qhead,
    ]
    if qhead == "v056":
        cmd.extend(["--qhead-weights", str(qhead_weights)])
    return run_checked(cmd)


def atomic_json(payload, path: Path) -> None:
    path = Path(path)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp = Path(fh.name)
            json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def file_record(path: Path, kind: str, rows: int, sequences: int) -> dict:
    return {
        "path": str(Path(path).resolve()),
        "kind": kind,
        "rows": rows,
        "sequences": sequences,
        "bytes": Path(path).stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crop", type=Path, default=DEFAULT_CROP)
    ap.add_argument("--noncrop", type=Path, default=DEFAULT_NONCROP)
    ap.add_argument("--selected-meta", type=Path, default=DEFAULT_SELECTION_META)
    ap.add_argument(
        "--selection-reference",
        type=Path,
        default=DEFAULT_SELECTION_REFERENCE,
        help="full-frame base used to independently verify the metadata-selected sequence set",
    )
    ap.add_argument("--skip-selection-crosscheck", action="store_true")
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--K", type=int, default=fs.SPLICE_K)
    ap.add_argument("--raw-only", action="store_true", help="build only the uncorrected/unspliced hybrid")
    ap.add_argument(
        "--include-qhead",
        action="store_true",
        help="also emit the legacy v056 diagnostic variant (never auto-promoted or submitted)",
    )
    ap.add_argument("--qhead-weights", type=Path, default=fs.QHEAD_WEIGHTS)
    args = ap.parse_args()

    if args.K < 1:
        ap.error("--K must be >= 1")
    if args.raw_only and args.include_qhead:
        ap.error("--raw-only and --include-qhead are mutually exclusive")

    try:
        crop, crop_order = fs.load(args.crop, "crop")
        noncrop, noncrop_order = fs.load(args.noncrop, "noncrop")
        sample, sample_order = fs.load(args.sample, "sample", validate_boxes=False)
        validate_exact_input_sets(crop_order, noncrop_order, sample_order)
        selected = load_selected_sequences(args.selected_meta, sample)

        crosscheck = None
        reference = None
        reference_order = None
        if not args.skip_selection_crosscheck:
            reference, reference_order = fs.load(args.selection_reference, "selection-reference")
            errors = fs.validate_input_sets(crop_order, reference_order, sample_order)
            if errors:
                raise MixedCandidateError(
                    "crop/selection-reference/sample exact-set failure: " + "; ".join(errors)
                )
            changed, changed_rows = changed_sequences(crop, reference)
            if changed != selected:
                raise MixedCandidateError(
                    "E46 metadata disagrees with crop-vs-base changed-sequence set: "
                    f"metadata-only={sorted(selected - changed)}, diff-only={sorted(changed - selected)}"
                )
            crosscheck = {
                "reference": str(args.selection_reference.resolve()),
                "reference_sha256": sha256(args.selection_reference),
                "changed_sequences": len(changed),
                "changed_rows": changed_rows,
                "exact_match": True,
            }

        source = source_order = None
        consensus_inputs = [("crop", crop), ("noncrop", noncrop)]
        if reference is not None:
            consensus_inputs.append(("selection-reference", reference))
        if not args.raw_only:
            source, source_order = fs.load(args.source, "splice-source")
            errors = fs.validate_input_sets(crop_order, source_order, sample_order)
            if errors:
                raise MixedCandidateError("crop/source/sample exact-set failure: " + "; ".join(errors))
            consensus_inputs.append(("splice-source", source))
        consensus_sequences = first_frame_consensus(*consensus_inputs)

        hybrid = build_hybrid(crop, noncrop, selected)
        crop_rows, noncrop_rows = assert_hybrid_provenance(hybrid, crop, noncrop, selected)
        raw_errors = fs.validate(hybrid, sample_order, args.sample, raw_main=hybrid)
        if raw_errors:
            raise MixedCandidateError("raw hybrid validation failed: " + "; ".join(raw_errors))

        args.out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.out_dir, prefix=".mixed-build-") as stage_text:
            stage = Path(stage_text)
            raw_stage = stage / RAW_NAME
            fs.write(hybrid, sample_order, raw_stage)
            rows, sequences = validate_candidate_file(raw_stage, args.sample, hybrid)
            staged = [(raw_stage, args.out_dir / RAW_NAME, "raw mixed-resolution")]
            finalizer_logs = {}

            if not args.raw_only:
                finalizer_logs["selftest"] = run_checked(
                    [sys.executable, str(FINALIZER), "--selftest"]
                )
                final_stage = stage / FINAL_NAME
                finalizer_logs["both_K6"] = finalize_variant(
                    raw_stage,
                    args.source,
                    args.sample,
                    final_stage,
                    args.K,
                    "none",
                    args.qhead_weights,
                )
                validate_candidate_file(final_stage, args.sample, hybrid)
                staged.append(
                    (final_stage, args.out_dir / FINAL_NAME, "both correction + frozen splice")
                )

                if args.include_qhead:
                    qhead_stage = stage / QHEAD_NAME
                    finalizer_logs["both_K6_qhead_v056"] = finalize_variant(
                        raw_stage,
                        args.source,
                        args.sample,
                        qhead_stage,
                        args.K,
                        "v056",
                        args.qhead_weights,
                    )
                    validate_candidate_file(qhead_stage, args.sample, hybrid)
                    staged.append(
                        (
                            qhead_stage,
                            args.out_dir / QHEAD_NAME,
                            "both correction + frozen splice + legacy qhead v056 diagnostic",
                        )
                    )

            output_records = []
            for stage_path, final_path, kind in staged:
                record = file_record(stage_path, kind, rows, sequences)
                record["path"] = str(final_path.resolve())
                os.replace(stage_path, final_path)
                if sha256(final_path) != record["sha256"]:
                    raise MixedCandidateError(f"hash changed while publishing {final_path}")
                output_records.append(record)

        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "zero-GPU mixed-resolution Ranking-A candidate; local artifacts only",
            "kaggle_submitted": False,
            "recipe": {
                "selected_sequences": "v023 cropwiden55",
                "nonselected_sequences": "v013 E20 1344 full-frame",
                "selected_count": len(selected),
                "nonselected_count": len(sample) - len(selected),
                "selected_rows": crop_rows,
                "nonselected_rows": noncrop_rows,
                "selected_names": sorted(selected),
                "selection_crosscheck": crosscheck,
                "first_frame_consensus_sequences": consensus_sequences,
                "finalizer": str(FINALIZER.resolve()),
                "correction": None if args.raw_only else "both",
                "splice_K": None if args.raw_only else args.K,
                "qhead_diagnostic_included": args.include_qhead,
            },
            "inputs": {
                "crop": {"path": str(args.crop.resolve()), "sha256": sha256(args.crop)},
                "noncrop": {"path": str(args.noncrop.resolve()), "sha256": sha256(args.noncrop)},
                "selected_meta": {
                    "path": str(args.selected_meta.resolve()),
                    "sha256": sha256(args.selected_meta),
                },
                "sample": {"path": str(args.sample.resolve()), "sha256": sha256(args.sample)},
                "source": None
                if args.raw_only
                else {"path": str(args.source.resolve()), "sha256": sha256(args.source)},
                "qhead_weights": None
                if not args.include_qhead
                else {
                    "path": str(args.qhead_weights.resolve()),
                    "sha256": sha256(args.qhead_weights),
                },
            },
            "outputs": output_records,
            "finalizer_logs": finalizer_logs,
            "validation": {
                "sample_exact_set": True,
                "finite_boxes": True,
                "nonnegative_xy": True,
                "positive_wh": True,
                "first_frames_restored_to_raw_hybrid": True,
                "whole_sequence_branch_provenance": True,
            },
        }
        manifest_path = args.out_dir / MANIFEST_NAME
        atomic_json(manifest, manifest_path)

        print(
            f"selected {len(selected)} sequences/{crop_rows} rows from crop; "
            f"{len(sample) - len(selected)} sequences/{noncrop_rows} rows from 1344 noncrop"
        )
        if crosscheck:
            print(
                "selection cross-check passed: "
                f"{crosscheck['changed_sequences']} sequences/{crosscheck['changed_rows']} changed rows"
            )
        print(f"first-frame consensus passed: {consensus_sequences}/{len(sample)} sequences")
        for record in output_records:
            print(f"{record['sha256']}  {record['path']}")
        print(f"manifest: {manifest_path.resolve()}")
        print("Kaggle submission: NO")
        return 0
    except (fs.SubmissionValidationError, MixedCandidateError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
