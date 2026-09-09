#!/usr/bin/env python3
"""crop_rerun must not assume base["frame"] is 1-based and contiguous.

The train405 contract numbers frames with global GT row numbers (nir-rider15 =
16305..16459), so the old `jpgs[fr - 1]` indexing raised IndexError partway
through the 405 OOF run.  test75 could never catch it: there K=1, so merge's
`f_lo + f - 1` degenerates to the identity and both readings agree.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

pytest.importorskip("pandas")
pytest.importorskip("PIL")
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from hsot import crop_rerun  # noqa: E402


def _sequence(root: Path, seq: str, frames: list[int], size=(200, 120)):
    """One sequence: images named by per-sequence 1-based stem, GT frames arbitrary."""
    d = root / seq
    d.mkdir(parents=True)
    for i in range(len(frames)):
        Image.new("RGB", size, (i, i, i)).save(d / f"{i + 1:04d}.jpg")
    return d


def _base_rows(seq: str, frames: list[int], box=(90.0, 50.0, 8.0, 8.0)):
    return [
        {"ID": f"{seq}_{fr}", "x": box[0] + k * 0.1, "y": box[1],
         "width": box[2], "height": box[3]}
        for k, fr in enumerate(frames)
    ]


def _run_prep(tmp: Path, seq: str, frames: list[int]) -> dict:
    frames_root = tmp / "frames"
    _sequence(frames_root, seq, frames)
    base = tmp / "base.csv"
    pd.DataFrame(_base_rows(seq, frames)).to_csv(base, index=False)
    out_root, meta = tmp / "crops", tmp / "crop_meta.json"
    args = argparse.Namespace(
        frames_root=str(frames_root), base_csv=str(base), envelope_extra=None,
        area_frac_max=0.55, out_root=str(out_root), meta=str(meta),
        segments=1, only_seqs=None, jpeg_quality=95, jpeg_subsampling=None,
    )
    crop_rerun.cmd_prep(args)
    import json
    return json.loads(meta.read_text())


def test_prep_handles_globally_numbered_frames(tmp_path):
    """The exact shape that broke the 405 run: frames start at 16305, not 1."""
    frames = list(range(16305, 16305 + 60))
    meta = _run_prep(tmp_path, "nir-rider15", frames)
    assert meta, "small target must be selected for cropping"
    window = next(iter(meta.values()))
    assert window["seq"] == "nir-rider15"
    # meta records ORIGINAL frame numbers -- merge relies on this for f_lo
    assert window["frames"] == [16305, 16305 + 59]
    # every frame produced a cropped image
    crop_dir = tmp_path / "crops" / next(iter(meta))
    assert len(sorted(crop_dir.glob("*.jpg"))) == len(frames)


def test_prep_still_handles_one_based_frames_identically(tmp_path):
    """Ranking-A convention must keep working -- this is a widening, not a change."""
    meta = _run_prep(tmp_path, "nir-bee2", list(range(1, 61)))
    window = next(iter(meta.values()))
    assert window["frames"] == [1, 60]
    crop_dir = tmp_path / "crops" / next(iter(meta))
    assert len(sorted(crop_dir.glob("*.jpg"))) == 60


def test_prep_fails_closed_when_frames_and_images_disagree(tmp_path):
    """Silently cropping a misaligned sequence would corrupt the pair."""
    frames_root = tmp_path / "frames"
    _sequence(frames_root, "nir-short", list(range(1, 21)))  # 20 images
    base = tmp_path / "base.csv"
    # 30 GT rows against 20 images
    pd.DataFrame(_base_rows("nir-short", list(range(1, 31)))).to_csv(base, index=False)
    args = argparse.Namespace(
        frames_root=str(frames_root), base_csv=str(base), envelope_extra=None,
        area_frac_max=0.55, out_root=str(tmp_path / "crops"),
        meta=str(tmp_path / "m.json"), segments=1, only_seqs=None,
        jpeg_quality=95, jpeg_subsampling=None,
    )
    with pytest.raises(SystemExit, match="幀與影像不對齊"):
        crop_rerun.cmd_prep(args)
