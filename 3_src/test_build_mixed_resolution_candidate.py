#!/usr/bin/env python3
"""CPU-only fail-closed tests for build_mixed_resolution_candidate.py."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_mixed_resolution_candidate as mixed  # noqa: E402


SCRIPT = Path(__file__).with_name("build_mixed_resolution_candidate.py")
COLS = ("ID", "x", "y", "width", "height")


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLS)
        writer.writerows(rows)


def fixture_tree(root: Path, mismatch_first: bool = False):
    crop = root / "crop.csv"
    noncrop = root / "noncrop.csv"
    reference = root / "reference.csv"
    sample = root / "sample.csv"
    meta = root / "meta.json"
    out = root / "out"
    crop_rows = [
        ("cropseq_1", 1, 2, 3, 4),
        ("cropseq_2", 11, 12, 3, 4),
        ("fullseq_1", 5, 6, 7, 8),
        ("fullseq_2", 15, 16, 7, 8),
    ]
    noncrop_rows = [
        ("cropseq_1", 1 if not mismatch_first else 2, 2, 3, 4),
        ("cropseq_2", 21, 22, 3, 4),
        ("fullseq_1", 5, 6, 7, 8),
        ("fullseq_2", 25, 26, 7, 8),
    ]
    reference_rows = [
        ("cropseq_1", 1, 2, 3, 4),
        ("cropseq_2", 31, 32, 3, 4),
        ("fullseq_1", 5, 6, 7, 8),
        ("fullseq_2", 15, 16, 7, 8),
    ]
    sample_rows = [(row[0], 0, 0, 0, 0) for row in crop_rows]
    write_csv(crop, crop_rows)
    write_csv(noncrop, noncrop_rows)
    write_csv(reference, reference_rows)
    write_csv(sample, sample_rows)
    meta.write_text(json.dumps({"cropseq": {"seq": "cropseq", "frames": [1, 2]}}))
    return crop, noncrop, reference, sample, meta, out


def run_raw_only(*paths):
    crop, noncrop, reference, sample, meta, out = paths
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--crop",
            str(crop),
            "--noncrop",
            str(noncrop),
            "--selection-reference",
            str(reference),
            "--sample",
            str(sample),
            "--selected-meta",
            str(meta),
            "--out-dir",
            str(out),
            "--raw-only",
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_raw_hybrid_uses_crop_only_on_selected_whole_sequences():
    with tempfile.TemporaryDirectory() as td:
        paths = fixture_tree(Path(td))
        proc = run_raw_only(*paths)
        assert proc.returncode == 0, proc.stderr
        out = paths[-1]
        rows = {r["ID"]: r for r in csv.DictReader((out / mixed.RAW_NAME).open())}
        assert float(rows["cropseq_2"]["x"]) == 11.0
        assert float(rows["fullseq_2"]["x"]) == 25.0
        assert float(rows["cropseq_1"]["x"]) == 1.0
        manifest = json.loads((out / mixed.MANIFEST_NAME).read_text())
        assert manifest["recipe"]["selected_count"] == 1
        assert manifest["recipe"]["selection_crosscheck"]["exact_match"] is True
        assert manifest["validation"]["first_frames_restored_to_raw_hybrid"] is True
        assert manifest["kaggle_submitted"] is False


def test_first_frame_mismatch_fails_without_publishing():
    with tempfile.TemporaryDirectory() as td:
        paths = fixture_tree(Path(td), mismatch_first=True)
        proc = run_raw_only(*paths)
        assert proc.returncode == 2
        assert "first-frame box mismatch" in proc.stderr
        assert not (paths[-1] / mixed.RAW_NAME).exists()


def test_stale_metadata_vs_diff_fails_closed():
    with tempfile.TemporaryDirectory() as td:
        paths = fixture_tree(Path(td))
        paths[4].write_text(json.dumps({"fullseq": {"seq": "fullseq", "frames": [1, 2]}}))
        proc = run_raw_only(*paths)
        assert proc.returncode == 2
        assert "metadata disagrees" in proc.stderr
        assert not (paths[-1] / mixed.RAW_NAME).exists()
