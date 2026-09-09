#!/usr/bin/env python3
"""Build the frozen, rerunnable E32b production-pair frame manifest.

Canonical Rank-A invocation (the 125-frame/5-sequence pool):

  python3 3_src/hsot/make_e32b_manifest.py

An E46 fresh-rerun manifest is supported, but it is a different trajectory and
must state its independently observed count explicitly (currently 120):

  python3 3_src/hsot/make_e32b_manifest.py \
    --a-csv 5_outputs/e46_sam3_crop_scores_20260819/test_merged.csv \
    --a-label e46_fresh_crop_sam3 --expected-target-count 120 \
    --out /tmp/e32b_e46_frames.json
"""
from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path

from e32b_common import (
    CONTROL_AGREE_IOU_MIN,
    EXECUTION_MODE,
    EXPERIMENT,
    MANIFEST_SCHEMA_VERSION,
    TRIGGER_A_RUN_MIN,
    TRIGGER_B_RUN_MIN,
    ContractError,
    atomic_write_json,
    box_iou,
    canonical_json,
    exact_frozen_runs,
    load_submission,
    payload_sha256,
    sha256_file,
    uniform_take,
    validate_manifest_structure,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_A = ROOT / "5_outputs/submissions/sub_v023_cropwiden55.csv"
DEFAULT_B = ROOT / "5_outputs/submissions/sub_v012_e23b_sam21_ablation.csv"
DEFAULT_SAMPLE = ROOT / "1_data/raw/sample_submisson.csv"
DEFAULT_FRAMES_ARCHIVE = ROOT / "1_data/packed/t1test_fc_75.tar"
DEFAULT_OUT = ROOT / "3_src/peft/e32b_frames.json"


def archive_sequence_identities(archive: Path, sequences: list[str]) -> dict[str, dict]:
    wanted = set(sequences)
    members: dict[str, list[tarfile.TarInfo]] = {seq: [] for seq in sequences}
    with tarfile.open(archive, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name.lstrip("./")).parts
            if len(parts) != 2 or parts[0] not in wanted:
                continue
            if Path(parts[1]).suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            members[parts[0]].append(member)
        out = {}
        for seq in sequences:
            seq_members = sorted(members[seq], key=lambda member: Path(member.name).name)
            if not seq_members:
                raise ContractError(f"{archive}: no frames for {seq}")
            basic = []
            content = []
            for member in seq_members:
                name = Path(member.name).name
                item = {"name": name, "size": int(member.size)}
                stream = tf.extractfile(member)
                if stream is None:
                    raise ContractError(f"{archive}: cannot read {member.name}")
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
                basic.append(item)
                content.append({**item, "sha256": digest.hexdigest()})
            out[seq] = {
                "n_frames": len(seq_members),
                "manifest_sha256": hashlib.sha256(
                    canonical_json(basic).encode("utf-8")
                ).hexdigest(),
                "content_manifest_sha256": hashlib.sha256(
                    canonical_json(content).encode("utf-8")
                ).hexdigest(),
            }
        return out


def record_for(
    seq: str,
    frame: int,
    frame_position: dict[int, int],
    a_box: list[float],
    b_box: list[float],
    a_run: int,
    b_run: int,
) -> dict:
    position = frame_position[frame]
    return {
        "id": f"{seq}_{frame}",
        "frame": frame,
        "position": position,
        "local_name": f"{position + 1:04d}.jpg",
        "a_box": a_box,
        "b_box": b_box,
        "a_run": a_run,
        "b_run": b_run,
        "a_b_iou": round(box_iou(a_box, b_box), 6),
    }


def build_manifest(args: argparse.Namespace) -> dict:
    a, a_order = load_submission(args.a_csv)
    b, b_order = load_submission(args.b_csv)
    sample, sample_order = load_submission(args.sample, boxes_required=False)
    if a_order != sample_order:
        raise ContractError("A row order/ID contract differs from sample submission")
    if b_order != sample_order:
        raise ContractError("B row order/ID contract differs from sample submission")
    if set(a) != set(b) or set(a) != set(sample):
        raise ContractError("A/B/sample sequence sets differ")

    sequences: dict[str, dict] = {}
    target_total = 0
    control_total = 0
    target_sequences: list[str] = []
    for seq in a:
        frames = sorted(a[seq])
        if frames != sorted(b[seq]) or frames != sorted(sample[seq]):
            raise ContractError(f"{seq}: A/B/sample frame sets differ")
        positions = {frame: pos for pos, frame in enumerate(frames)}
        ra = exact_frozen_runs(a[seq])
        rb = exact_frozen_runs(b[seq])
        target_frames = [
            frame for frame in frames
            if frame != frames[0]
            and ra[frame] >= TRIGGER_A_RUN_MIN
            and rb[frame] >= TRIGGER_B_RUN_MIN
        ]
        if not target_frames:
            continue
        target_sequences.append(seq)

        healthy_pool = [
            frame for frame in frames
            if frame != frames[0]
            and frame not in set(target_frames)
            and ra[frame] < TRIGGER_B_RUN_MIN
            and rb[frame] < TRIGGER_B_RUN_MIN
            and box_iou(a[seq][frame], b[seq][frame]) >= CONTROL_AGREE_IOU_MIN
        ]
        controls = uniform_take(healthy_pool, args.controls_per_sequence)
        if len(controls) != args.controls_per_sequence:
            raise ContractError(
                f"{seq}: only {len(controls)} healthy proxies, expected "
                f"{args.controls_per_sequence}"
            )
        if a[seq][frames[0]] != b[seq][frames[0]]:
            raise ContractError(f"{seq}: A/B first-frame init boxes differ")

        sequences[seq] = {
            "n_frames": len(frames),
            "first_frame": frames[0],
            "first_local_name": "0001.jpg",
            "init_box": a[seq][frames[0]],
            "target": [
                record_for(seq, f, positions, a[seq][f], b[seq][f], ra[f], rb[f])
                for f in target_frames
            ],
            "healthy_proxy": [
                record_for(seq, f, positions, a[seq][f], b[seq][f], ra[f], rb[f])
                for f in controls
            ],
            "healthy_proxy_pool_size": len(healthy_pool),
        }
        target_total += len(target_frames)
        control_total += len(controls)

    if target_total != args.expected_target_count:
        raise ContractError(
            f"target count {target_total} != locked expectation {args.expected_target_count}; "
            "this A trajectory is not the declared production artifact"
        )
    if len(target_sequences) != args.expected_target_sequences:
        raise ContractError(
            f"target sequence count {len(target_sequences)} != "
            f"locked expectation {args.expected_target_sequences}"
        )

    frame_identities = archive_sequence_identities(args.frames_archive, target_sequences)
    for seq in target_sequences:
        identity = frame_identities[seq]
        if identity["n_frames"] != sequences[seq]["n_frames"]:
            raise ContractError(
                f"{seq}: archive has {identity['n_frames']} frames, "
                f"CSV contract has {sequences[seq]['n_frames']}"
            )
        sequences[seq]["frame_identity"] = identity

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "dataset": "ranking_a_test75",
        "input_identity": {
            "a": {
                "label": args.a_label,
                "path_hint": str(Path(args.a_csv).name),
                "sha256": sha256_file(args.a_csv),
                "rows": len(a_order),
            },
            "b": {
                "label": args.b_label,
                "path_hint": str(Path(args.b_csv).name),
                "sha256": sha256_file(args.b_csv),
                "rows": len(b_order),
            },
            "sample": {
                "path_hint": str(Path(args.sample).name),
                "sha256": sha256_file(args.sample),
                "rows": len(sample_order),
            },
            "frames_archive": {
                "path_hint": str(Path(args.frames_archive).name),
                "sha256": sha256_file(args.frames_archive),
                "bytes": Path(args.frames_archive).stat().st_size,
            },
        },
        "policy": {
            "a_run_min": TRIGGER_A_RUN_MIN,
            "b_run_min": TRIGGER_B_RUN_MIN,
            "run_counter_semantics": (
                "exact tuple equality; counter is 0 on the first box, so run>=24 "
                "begins on the 25th identical A box"
            ),
            "detector_input": "[first_frame,current_frame]",
            "detector_prompt": "first-frame init xywh normalized to frame size",
            "detector_selection": "top1_out_prob_no_score_gate",
            "zero_detection": "recorded abstention; candidate keeps its base box",
            "execution_mode": EXECUTION_MODE,
            "true_tracker_reinit": False,
            "detector_coordinate_correction_for_ranka_candidate": "both",
        },
        "control_definition": {
            "name": "healthy_proxy",
            "gt_available": False,
            "rule": (
                f"A_run<{TRIGGER_B_RUN_MIN} and B_run<{TRIGGER_B_RUN_MIN} and "
                f"IoU(A,B)>={CONTROL_AGREE_IOU_MIN}; deterministic uniform sample "
                "within each target sequence"
            ),
            "warning": "A/B agreement is a no-GT health proxy, not proof that either box is correct.",
            "per_sequence": args.controls_per_sequence,
        },
        "counts": {
            "target": target_total,
            "healthy_proxy": control_total,
            "target_sequences": len(target_sequences),
        },
        "sequence_order": target_sequences,
        "sequences": sequences,
    }
    manifest["payload_sha256"] = payload_sha256(manifest)
    validate_manifest_structure(manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-csv", type=Path, default=DEFAULT_A)
    parser.add_argument("--b-csv", type=Path, default=DEFAULT_B)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--frames-archive", type=Path, default=DEFAULT_FRAMES_ARCHIVE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--a-label", default="v023_crop_sam3_canonical_ranka")
    parser.add_argument("--b-label", default="v012_crop_samurai")
    parser.add_argument("--expected-target-count", type=int, default=125)
    parser.add_argument("--expected-target-sequences", type=int, default=5)
    parser.add_argument("--controls-per-sequence", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args)
    atomic_write_json(args.out, manifest)
    print(
        f"wrote {args.out}: target={manifest['counts']['target']} / "
        f"{manifest['counts']['target_sequences']} seqs, "
        f"healthy_proxy={manifest['counts']['healthy_proxy']}, "
        f"sha256={manifest['payload_sha256']}"
    )
    for seq in manifest["sequence_order"]:
        block = manifest["sequences"][seq]
        print(
            f"  {seq:<20} target={len(block['target']):3d} "
            f"healthy_proxy={len(block['healthy_proxy']):2d}/"
            f"{block['healthy_proxy_pool_size']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
