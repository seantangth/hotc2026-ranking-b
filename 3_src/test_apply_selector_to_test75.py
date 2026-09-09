#!/usr/bin/env python3
"""Guards for the script that produced v072 (LB 0.71550, current best).

It fits selector-v2 on all 405 train sequences and applies it to test75.  Two
properties matter and neither is obvious from reading it:

  * the output must preserve first frames as the *raw, uncorrected* main box --
    that is what makes them equal the official init box.  v056 lost +0.00031 by
    letting the coordinate correction touch first frames; v072 avoids it only
    because te_main_raw is used here.
  * rows the selector does not pick must keep the *spliced* box, not raw main,
    or the K=6 splice's own gains get silently reverted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import apply_selector_to_test75 as A  # noqa: E402


def test_structure_derives_sequences_first_frames_and_modalities_from_ids():
    ids = ["vis-a_1", "vis-a_2", "vis-a_3", "nir-b_10", "nir-b_11", "rednir-c_7"]
    seq_idx, modalities, first_mask = A.structure(ids)
    assert set(seq_idx) == {"vis-a", "nir-b", "rednir-c"}
    assert seq_idx["vis-a"].tolist() == [0, 1, 2]
    # first frame is the first *occurrence* in the file, not frame number 1:
    # test75 sequences do not all start at 1 and the 405 set uses global numbering
    assert first_mask.tolist() == [True, False, False, True, False, True]
    assert modalities == ["vis", "vis", "vis", "nir", "nir", "rednir"]


def test_structure_rejects_unknown_modality_prefix():
    with pytest.raises(SystemExit, match="unknown modality"):
        A.structure(["swir-x_1", "swir-x_2"])


def test_read_boxes_round_trips(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("ID,x,y,width,height\nvis-a_1,1,2,3,4\nvis-a_2,5,6,7,8\n")
    ids, boxes = A.read_boxes(p)
    assert ids == ["vis-a_1", "vis-a_2"]
    assert boxes.tolist() == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]


def test_first_frames_use_the_uncorrected_main_box():
    """The invariant that makes v072's 75 first frames equal the official init.

    Mirrors the script's final assignment: out[first_mask] = te_main_raw[first_mask].
    A correction applied here would silently cost the +0.00031 that v067 bought.
    """
    raw = np.array([[10.0, 20.0, 5.0, 6.0], [11.0, 21.0, 5.0, 6.0]])
    first_mask = np.array([True, False])
    corrected = A.oof.apply_correction(raw, first_mask, "both")
    # correction must already leave first frames alone...
    assert np.array_equal(corrected[first_mask], raw[first_mask])
    # ...and non-first frames must actually move, or the test proves nothing
    assert not np.array_equal(corrected[~first_mask], raw[~first_mask])
