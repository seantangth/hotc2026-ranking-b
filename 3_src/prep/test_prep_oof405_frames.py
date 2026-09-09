from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from prep_oof405_frames import (  # noqa: E402
    GtRow, assign_primary_folds, choose_observed_block, contiguous_blocks,
)


def row(seq: str, frame: int, box=(1.0, 2.0, 3.0, 4.0)) -> GtRow:
    return GtRow(f"{seq}_{frame}", seq, frame, *box)


def test_contiguous_blocks_and_unique_length_match():
    seq = "nir-basketball1"
    rows = [row(seq, i, (10, 10, 5, 5)) for i in range(1, 4)]
    rows += [row(seq, i, (20, 20, 6, 6)) for i in range(100, 105)]
    assert [len(x) for x in contiguous_blocks(rows)] == [3, 5]
    chosen, reason, _ = choose_observed_block(seq, rows, 5, (20, 20, 6, 6))
    assert [x.frame for x in chosen] == list(range(100, 105))
    assert reason == "matching_contiguous_block"


def test_identical_duplicate_block_uses_first_canonical():
    seq = "rednir-pills5"
    a = [row(seq, i, (i, 2, 3, 4)) for i in range(1, 4)]
    b = [row(seq, i + 99, (i, 2, 3, 4)) for i in range(1, 4)]
    chosen, reason, _ = choose_observed_block(seq, a + b, 3, a[0].box)
    assert [x.frame for x in chosen] == [1, 2, 3]
    assert reason == "identical_duplicate_block_first_canonical"


def test_ambiguous_nonidentical_blocks_fail_closed():
    seq = "vis-x"
    a = [row(seq, i, (i, 2, 3, 4)) for i in range(1, 4)]
    b = [row(seq, i + 99, (i + 50, 2, 3, 4)) for i in range(1, 4)]
    with pytest.raises(ValueError, match="拒絕猜測"):
        choose_observed_block(seq, a + b, 3, None)


def test_primary_group_keeps_numbered_families_separate_and_pairs_modalities():
    observed = {
        "vis-car": [row("vis-car", i) for i in range(1, 6)],
        "vis-car1": [row("vis-car1", i) for i in range(10, 14)],
        "vis-ball": [row("vis-ball", i) for i in range(20, 23)],
        "nir-ball": [row("nir-ball", i) for i in range(30, 33)],
    }
    seq_fold, meta = assign_primary_folds(observed, n_folds=2)
    assert meta["n_groups"] == 3
    assert meta["n_cross_modality_groups"] == 1
    assert seq_fold["vis-ball"] == seq_fold["nir-ball"]
    assert "car" in meta["group_to_fold"] and "car1" in meta["group_to_fold"]


def test_unambiguous_block_tolerates_1px_archive_annotation_drift():
    """nir-rider15 / vis-cup3: zip init_rect disagrees with GT by <=1px.

    The whole GT was taken verbatim (one block, len(rows) == n_images), so
    position 0 is GT row 0 by construction and no mapping was ever chosen --
    the residual disagreement is annotation drift, not misalignment.
    """
    rows = [row("nir-rider15", i, (20.0, 172.0, 10.0, 15.0)) for i in range(1, 6)]
    chosen, reason, delta = choose_observed_block(
        "nir-rider15", rows, 5, (19.0, 172.0, 9.0, 15.0))
    assert reason == "all_gt_rows"
    assert chosen is rows
    assert delta == (-1.0, 0.0, -1.0, 0.0)
    # the GT box, not the archive box, remains the authority
    assert chosen[0].box == (20.0, 172.0, 10.0, 15.0)

    _, _, none_delta = choose_observed_block(
        "nir-rider15", rows, 5, (20.0, 172.0, 10.0, 15.0))
    assert none_delta is None


def test_drift_tolerance_does_not_apply_where_a_block_was_chosen():
    """Where disambiguation actually happened, the init check stays exact."""
    # block lengths differ, so n_images=5 selects block b unambiguously as the
    # single candidate -- but a block *was* chosen, so 1px drift must still fail.
    a = [row("nir-two-blocks", i, (10.0, 10.0, 5.0, 5.0)) for i in range(1, 4)]
    b = [row("nir-two-blocks", i, (99.0, 99.0, 5.0, 5.0)) for i in range(10, 15)]
    chosen, reason, delta = choose_observed_block(
        "nir-two-blocks", a + b, 5, (99.0, 99.0, 5.0, 5.0))
    assert reason == "matching_contiguous_block" and delta is None
    with pytest.raises(ValueError, match="archive init"):
        choose_observed_block("nir-two-blocks", a + b, 5, (98.0, 99.0, 5.0, 5.0))


def test_large_archive_init_disagreement_still_fails_closed():
    rows = [row("nir-rider15", i, (20.0, 172.0, 10.0, 15.0)) for i in range(1, 6)]
    with pytest.raises(ValueError, match="archive init"):
        choose_observed_block("nir-broken", rows, 5, (119.0, 133.0, 3.0, 8.0))
