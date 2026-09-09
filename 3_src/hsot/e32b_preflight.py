#!/usr/bin/env python3
"""Fail-closed preflight for the E32b production-pair pilot."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

from e32b_common import (
    CONTROL_AGREE_IOU_MIN,
    TRIGGER_A_RUN_MIN,
    TRIGGER_B_RUN_MIN,
    ContractError,
    atomic_write_json,
    box_iou,
    exact_frozen_runs,
    frame_tree_identity,
    load_submission,
    manifest_records,
    sha256_file,
    uniform_take,
    validate_manifest_structure,
)


def expected_records(manifest: dict, a: dict, b: dict) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    targets: dict[str, list[int]] = {}
    controls: dict[str, list[int]] = {}
    per_seq = int(manifest["control_definition"]["per_sequence"])
    for seq in manifest["sequence_order"]:
        frames = sorted(a[seq])
        ra = exact_frozen_runs(a[seq])
        rb = exact_frozen_runs(b[seq])
        target = [
            f for f in frames
            if f != frames[0]
            and ra[f] >= TRIGGER_A_RUN_MIN
            and rb[f] >= TRIGGER_B_RUN_MIN
        ]
        healthy = [
            f for f in frames
            if f != frames[0]
            and f not in set(target)
            and ra[f] < TRIGGER_B_RUN_MIN
            and rb[f] < TRIGGER_B_RUN_MIN
            and box_iou(a[seq][f], b[seq][f]) >= CONTROL_AGREE_IOU_MIN
        ]
        targets[seq] = target
        controls[seq] = uniform_take(healthy, per_seq)
    return targets, controls


def verify_frames(manifest: dict, frames_root: Path) -> dict:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ContractError("Pillow is required for --require-frames") from exc

    detail: dict[str, dict] = {}
    for seq in manifest["sequence_order"]:
        block = manifest["sequences"][seq]
        seq_dir = frames_root / seq
        if not seq_dir.is_dir():
            raise ContractError(f"missing frame directory {seq_dir}")
        frames = sorted([*seq_dir.glob("*.jpg"), *seq_dir.glob("*.jpeg")])
        if len(frames) != int(block["n_frames"]):
            raise ContractError(
                f"{seq}: {len(frames)} images != manifest n_frames={block['n_frames']}"
            )
        identity = frame_tree_identity(seq_dir)
        expected_identity = block["frame_identity"]
        for key in ("n_frames", "manifest_sha256", "content_manifest_sha256"):
            if identity[key] != expected_identity[key]:
                raise ContractError(
                    f"{seq}: extracted frame identity {key}={identity[key]!r} "
                    f"!= manifest {expected_identity[key]!r}"
                )
        first = seq_dir / block["first_local_name"]
        if not first.is_file():
            raise ContractError(f"{seq}: missing first frame {first.name}")
        with Image.open(first) as image:
            width, height = image.size
        x, y, w, h = [float(v) for v in block["init_box"]]
        if x + w > width + 1e-6 or y + h > height + 1e-6:
            raise ContractError(
                f"{seq}: init box {block['init_box']} outside {width}x{height}"
            )
        referenced = []
        for record in block["target"] + block["healthy_proxy"]:
            path = seq_dir / record["local_name"]
            if not path.is_file():
                raise ContractError(f"{record['id']}: missing {path}")
            with Image.open(path) as image:
                if image.size != (width, height):
                    raise ContractError(
                        f"{record['id']}: size {image.size} != first {(width, height)}"
                    )
            referenced.append(record["local_name"])
        detail[seq] = {
            "n_frames": len(frames),
            "size": [width, height],
            "referenced": len(referenced),
            "manifest_sha256": identity["manifest_sha256"],
            "content_manifest_sha256": identity["content_manifest_sha256"],
        }
    return detail


def verify_python_runtime(expected_python: Path, expected_revision: str, require_cuda: bool) -> dict:
    try:
        same_python = Path(sys.executable).samefile(expected_python)
    except (FileNotFoundError, OSError):
        same_python = False
    if not same_python:
        raise ContractError(
            f"preflight interpreter {sys.executable} is not declared Python {expected_python}"
        )
    try:
        direct_text = importlib.metadata.distribution("sam3").read_text("direct_url.json")
        direct = json.loads(direct_text) if direct_text else None
        actual_revision = direct["vcs_info"]["commit_id"]
    except Exception as exc:
        raise ContractError("cannot prove installed sam3 VCS revision from direct_url.json") from exc
    if actual_revision != expected_revision:
        raise ContractError(
            f"installed sam3 revision {actual_revision} != expected {expected_revision}"
        )
    try:
        import sam3  # noqa: F401
        import torch
    except Exception as exc:
        raise ContractError(f"shared Python cannot import sam3/torch: {exc}") from exc
    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise ContractError("CUDA unavailable; CPU/MPS fallback is forbidden")
    return {
        "python": str(expected_python),
        "sam3_revision": actual_revision,
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "device": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def run(args: argparse.Namespace) -> dict:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest_structure(manifest)
    inputs = manifest["input_identity"]
    for label, path in (("a", args.a_csv), ("b", args.b_csv), ("sample", args.sample)):
        actual = sha256_file(path)
        expected = inputs[label]["sha256"]
        if actual != expected:
            raise ContractError(f"{label} sha256 {actual} != manifest {expected}")
    archive_identity = inputs.get("frames_archive") or {}
    if args.frames_archive is not None:
        archive_hash = sha256_file(args.frames_archive)
        if archive_hash != archive_identity.get("sha256"):
            raise ContractError(
                f"frames archive sha256 {archive_hash} != manifest {archive_identity.get('sha256')}"
            )
        if args.frames_archive.stat().st_size != int(archive_identity.get("bytes", -1)):
            raise ContractError("frames archive byte size differs from manifest")
        archive_check = "pass"
    elif args.reuse_extracted_frames and args.require_frames:
        archive_check = "shared_reuse_exact_extracted_content_manifest"
    else:
        raise ContractError(
            "provide --frames-archive, or explicitly combine --reuse-extracted-frames "
            "with --require-frames"
        )

    a, a_order = load_submission(args.a_csv)
    b, b_order = load_submission(args.b_csv)
    sample, sample_order = load_submission(args.sample, boxes_required=False)
    if a_order != b_order or a_order != sample_order:
        raise ContractError("A/B/sample canonical row order differs")
    if len(a_order) != inputs["a"]["rows"]:
        raise ContractError("A row count differs from manifest")

    targets, controls = expected_records(manifest, a, b)
    actual_target_sequences = [s for s in a if targets.get(s)]
    if actual_target_sequences != manifest["sequence_order"]:
        raise ContractError(
            f"target sequence order changed: {actual_target_sequences} != "
            f"{manifest['sequence_order']}"
        )
    for seq in manifest["sequence_order"]:
        block = manifest["sequences"][seq]
        manifest_target = [int(r["frame"]) for r in block["target"]]
        manifest_control = [int(r["frame"]) for r in block["healthy_proxy"]]
        if targets[seq] != manifest_target:
            raise ContractError(f"{seq}: target membership differs from manifest")
        if controls[seq] != manifest_control:
            raise ContractError(f"{seq}: healthy proxy membership differs from manifest")
        frame_pos = {f: i for i, f in enumerate(sorted(a[seq]))}
        for record in block["target"] + block["healthy_proxy"]:
            f = int(record["frame"])
            if record["a_box"] != a[seq][f] or record["b_box"] != b[seq][f]:
                raise ContractError(f"{record['id']}: A/B box differs from manifest")
            expected_name = f"{frame_pos[f] + 1:04d}.jpg"
            if record["local_name"] != expected_name:
                raise ContractError(
                    f"{record['id']}: local name {record['local_name']} != {expected_name}"
                )

    frame_detail = None
    if args.require_frames:
        if args.frames_root is None:
            raise ContractError("--require-frames requires --frames-root")
        frame_detail = verify_frames(manifest, args.frames_root)

    checkpoint = None
    if args.checkpoint is not None:
        if not args.checkpoint.is_file() or args.checkpoint.stat().st_size < 1_000_000_000:
            raise ContractError(f"checkpoint missing or implausibly small: {args.checkpoint}")
        checkpoint_sha256 = sha256_file(args.checkpoint)
        if not args.expected_checkpoint_sha256:
            raise ContractError("checkpoint supplied without --expected-checkpoint-sha256")
        if checkpoint_sha256 != args.expected_checkpoint_sha256:
            raise ContractError(
                f"checkpoint sha256 {checkpoint_sha256} != expected "
                f"{args.expected_checkpoint_sha256}"
            )
        checkpoint = {
            "path": str(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": checkpoint_sha256,
        }

    runtime = None
    if args.expected_python is not None or args.expected_sam3_revision is not None or args.require_cuda:
        if args.expected_python is None or args.expected_sam3_revision is None:
            raise ContractError(
                "runtime validation requires both --expected-python and --expected-sam3-revision"
            )
        runtime = verify_python_runtime(
            args.expected_python, args.expected_sam3_revision, args.require_cuda
        )

    return {
        "schema_version": 1,
        "experiment": manifest["experiment"],
        "status": "PASS",
        "manifest_payload_sha256": manifest["payload_sha256"],
        "checks": {
            "manifest_structure": "pass",
            "input_hashes": "pass",
            "frames_archive_identity": archive_check,
            "exact_row_order": "pass",
            "trigger_membership": "pass",
            "healthy_proxy_membership": "pass",
            "frames": "pass" if args.require_frames else "not_requested",
            "checkpoint": "pass" if checkpoint else "not_requested",
            "python_runtime": "pass" if runtime else "not_requested",
        },
        "counts": manifest["counts"],
        "n_probe_records": len(manifest_records(manifest)),
        "frame_detail": frame_detail,
        "checkpoint": checkpoint,
        "python_runtime": runtime,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a-csv", type=Path, required=True)
    parser.add_argument("--b-csv", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--frames-archive", type=Path)
    parser.add_argument("--reuse-extracted-frames", action="store_true")
    parser.add_argument("--frames-root", type=Path)
    parser.add_argument("--require-frames", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-python", type=Path)
    parser.add_argument("--expected-sam3-revision")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except Exception as exc:  # fail report must survive for stepwise remote sync
        report = {
            "schema_version": 1,
            "status": "BLOCK",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        atomic_write_json(args.report, report)
        print(f"BLOCK: {type(exc).__name__}: {exc}")
        return 2
    atomic_write_json(args.report, report)
    print(
        f"PASS: {report['counts']['target']} target + "
        f"{report['counts']['healthy_proxy']} healthy proxies; "
        f"frames={report['checks']['frames']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
