#!/usr/bin/env python3
"""Shared, dependency-free contracts for the E32b K=24 pilot.

This module deliberately contains only CSV/JSON/box plumbing.  The CUDA probe
imports it too, so manifest verification is identical on the local machine and
on the remote runner.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


BOX_COLUMNS = ("x", "y", "width", "height")
SUBMISSION_COLUMNS = ("ID", *BOX_COLUMNS)
MANIFEST_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
EXPERIMENT = "E32b_k24_exemplar"
EXECUTION_MODE = "offline_detector_box_substitution_proxy"
TRIGGER_A_RUN_MIN = 24
TRIGGER_B_RUN_MIN = 2
CONTROL_AGREE_IOU_MIN = 0.70


class ContractError(ValueError):
    """An artifact violated the frozen E32b contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_tree_identity(seq_dir: str | Path) -> dict[str, Any]:
    """Hash an extracted sequence without relying on path or mtime.

    ``manifest_sha256`` matches the production tracker run-signature convention
    (ordered basename + byte size).  ``content_manifest_sha256`` additionally
    binds every frame's bytes, which is required before reusing a frame tree
    produced by another job on a shared VM.
    """
    seq_dir = Path(seq_dir)
    frames = sorted([*seq_dir.glob("*.jpg"), *seq_dir.glob("*.jpeg")])
    if not frames:
        raise ContractError(f"{seq_dir}: no jpg/jpeg frames")
    basic = [{"name": path.name, "size": path.stat().st_size} for path in frames]
    content = [
        {**item, "sha256": sha256_file(path)}
        for item, path in zip(basic, frames, strict=True)
    ]
    return {
        "n_frames": len(frames),
        "manifest_sha256": hashlib.sha256(canonical_json(basic).encode("utf-8")).hexdigest(),
        "content_manifest_sha256": hashlib.sha256(
            canonical_json(content).encode("utf-8")
        ).hexdigest(),
    }


def payload_sha256(document: dict[str, Any], field: str = "payload_sha256") -> str:
    payload = {k: v for k, v in document.items() if k != field}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def verify_payload_sha256(document: dict[str, Any], field: str = "payload_sha256") -> None:
    recorded = document.get(field)
    actual = payload_sha256(document, field)
    if not isinstance(recorded, str) or recorded != actual:
        raise ContractError(f"{field} mismatch: recorded={recorded!r}, actual={actual}")


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
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
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=1, ensure_ascii=False) + "\n")


def atomic_write_csv(path: str | Path, rows: Iterable[Iterable[Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(SUBMISSION_COLUMNS)
            writer.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def split_id(value: str) -> tuple[str, int]:
    try:
        seq, frame_text = value.rsplit("_", 1)
        frame = int(frame_text)
    except (AttributeError, ValueError) as exc:
        raise ContractError(f"invalid ID {value!r}; expected <sequence>_<integer>") from exc
    if not seq or frame < 0:
        raise ContractError(f"invalid ID {value!r}")
    return seq, frame


def validate_box(box: Iterable[Any], label: str = "box") -> list[float]:
    try:
        out = [float(v) for v in box]
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label}: non-numeric box") from exc
    if len(out) != 4:
        raise ContractError(f"{label}: expected four values, got {len(out)}")
    x, y, w, h = out
    if not all(math.isfinite(v) for v in out):
        raise ContractError(f"{label}: NaN/Inf")
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ContractError(f"{label}: invalid xywh {out}")
    return out


def load_submission(path: str | Path, *, boxes_required: bool = True) -> tuple[dict[str, dict[int, list[float]]], list[str]]:
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"missing CSV: {path}")
    data: dict[str, dict[int, list[float]]] = defaultdict(dict)
    order: list[str] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != list(SUBMISSION_COLUMNS):
            raise ContractError(
                f"{path}: columns {reader.fieldnames!r} != {list(SUBMISSION_COLUMNS)!r}"
            )
        for row_no, row in enumerate(reader, start=2):
            rid = (row.get("ID") or "").strip()
            if rid in seen:
                raise ContractError(f"{path}:{row_no}: duplicate ID {rid}")
            seq, frame = split_id(rid)
            if frame in data[seq]:
                raise ContractError(f"{path}:{row_no}: duplicate sequence/frame {rid}")
            if boxes_required:
                box = validate_box([row[c] for c in BOX_COLUMNS], f"{path}:{row_no} {rid}")
            else:
                box = [0.0, 0.0, 0.0, 0.0]
            data[seq][frame] = box
            order.append(rid)
            seen.add(rid)
    if not order:
        raise ContractError(f"{path}: empty CSV")
    return dict(data), order


def exact_frozen_runs(frame_boxes: dict[int, list[float]]) -> dict[int, int]:
    """Historical E35/finalizer semantics: first box has counter 0.

    Consequently ``run >= 24`` begins on the 25th identical box, not the 24th.
    Exact float tuple equality is intentional and matches finalize_submission.py.
    """
    frames = sorted(frame_boxes)
    out: dict[int, int] = {}
    run = 0
    for idx, frame in enumerate(frames):
        if idx and frame_boxes[frame] == frame_boxes[frames[idx - 1]]:
            run += 1
        else:
            run = 0
        out[frame] = run
    return out


def box_iou(a: Iterable[Any] | None, b: Iterable[Any] | None) -> float:
    if a is None or b is None:
        return 0.0
    aa = [float(v) for v in a]
    bb = [float(v) for v in b]
    iw = max(0.0, min(aa[0] + aa[2], bb[0] + bb[2]) - max(aa[0], bb[0]))
    ih = max(0.0, min(aa[1] + aa[3], bb[1] + bb[3]) - max(aa[1], bb[1]))
    inter = iw * ih
    union = aa[2] * aa[3] + bb[2] * bb[3] - inter
    return inter / union if union > 0 else 0.0


def uniform_take(values: list[int], count: int) -> list[int]:
    """Deterministic endpoint-inclusive sample without NumPy."""
    if count < 0:
        raise ContractError("sample count must be non-negative")
    if count == 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    if count == 1:
        return [values[len(values) // 2]]
    indices = [(i * (len(values) - 1)) // (count - 1) for i in range(count)]
    if len(set(indices)) != len(indices):
        raise ContractError("uniform sample produced duplicate indices")
    return [values[i] for i in indices]


def corrected_detector_box(box: Iterable[Any], correction: str) -> list[float]:
    """Apply the already-locked Rank-A coordinate convention to a new detector box."""
    x, y, w, h = validate_box(box, "detector box")
    if correction == "none":
        return [x, y, w, h]
    if correction not in {"top-only", "both"}:
        raise ContractError(f"unsupported detector correction {correction!r}")
    ny = max(0.0, y - 1.0)
    h += y - ny
    y = ny
    if correction == "both":
        nx = max(0.0, x - 1.0)
        w += x - nx
        x = nx
    return validate_box([x, y, w, h], "corrected detector box")


def manifest_records(manifest: dict[str, Any], cohort: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seq in manifest.get("sequence_order", []):
        block = manifest["sequences"][seq]
        for name in ("target", "healthy_proxy"):
            if cohort is not None and name != cohort:
                continue
            out.extend(block[name])
    return out


def validate_manifest_structure(manifest: dict[str, Any]) -> None:
    verify_payload_sha256(manifest)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ContractError(f"unsupported manifest schema {manifest.get('schema_version')!r}")
    if manifest.get("experiment") != EXPERIMENT:
        raise ContractError(f"wrong experiment {manifest.get('experiment')!r}")
    identities = manifest.get("input_identity")
    if not isinstance(identities, dict):
        raise ContractError("manifest missing input_identity")
    for label in ("a", "b", "sample", "frames_archive"):
        value = identities.get(label)
        if not isinstance(value, dict) or not isinstance(value.get("sha256"), str):
            raise ContractError(f"manifest missing {label} sha256 identity")
    policy = manifest.get("policy") or {}
    expected_policy = {
        "a_run_min": TRIGGER_A_RUN_MIN,
        "b_run_min": TRIGGER_B_RUN_MIN,
        "detector_input": "[first_frame,current_frame]",
        "detector_selection": "top1_out_prob_no_score_gate",
        "execution_mode": EXECUTION_MODE,
        "true_tracker_reinit": False,
    }
    for key, value in expected_policy.items():
        if policy.get(key) != value:
            raise ContractError(f"manifest policy {key}={policy.get(key)!r}, expected {value!r}")
    sequence_order = manifest.get("sequence_order")
    sequences = manifest.get("sequences")
    if not isinstance(sequence_order, list) or not sequence_order:
        raise ContractError("manifest sequence_order must be non-empty")
    if not isinstance(sequences, dict) or set(sequence_order) != set(sequences):
        raise ContractError("manifest sequence_order/sequences mismatch")
    ids: set[str] = set()
    counts = {"target": 0, "healthy_proxy": 0}
    for seq in sequence_order:
        block = sequences[seq]
        first = int(block["first_frame"])
        validate_box(block["init_box"], f"{seq} init")
        frame_identity = block.get("frame_identity")
        if not isinstance(frame_identity, dict):
            raise ContractError(f"{seq}: missing frame_identity")
        if int(frame_identity.get("n_frames", -1)) != int(block.get("n_frames", -2)):
            raise ContractError(f"{seq}: frame identity count mismatch")
        for key in ("manifest_sha256", "content_manifest_sha256"):
            value = frame_identity.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ContractError(f"{seq}: invalid frame identity {key}")
        for cohort in counts:
            records = block.get(cohort)
            if not isinstance(records, list):
                raise ContractError(f"{seq}: missing list {cohort}")
            for record in records:
                rid = record.get("id")
                rseq, frame = split_id(rid)
                if rseq != seq or int(record.get("frame")) != frame:
                    raise ContractError(f"{seq}: inconsistent record {rid}")
                if frame == first:
                    raise ContractError(f"{rid}: first frame cannot be probed")
                if rid in ids:
                    raise ContractError(f"duplicate manifest record {rid}")
                ids.add(rid)
                validate_box(record["a_box"], f"{rid} A")
                validate_box(record["b_box"], f"{rid} B")
                local_name = record.get("local_name")
                if not isinstance(local_name, str) or not local_name.endswith(".jpg"):
                    raise ContractError(f"{rid}: invalid local_name {local_name!r}")
                counts[cohort] += 1
    recorded = manifest.get("counts") or {}
    if counts != {k: int(recorded.get(k, -1)) for k in counts}:
        raise ContractError(f"manifest counts mismatch: actual={counts}, recorded={recorded}")
