#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import finalize_submission as finalizer  # noqa: E402
import oof405_crossfit as oof  # noqa: E402


def _csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rehash(contracts: Path) -> None:
    manifest = {
        name: _hash(contracts / name) for name in oof.REQUIRED_HASHED_CONTRACTS
    }
    (contracts / "contract_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _formal_fixture(root: Path) -> tuple[Path, Path, Path]:
    contracts = root / "contracts"
    contracts.mkdir(parents=True)
    sample_rows = []
    gt_rows = []
    frame_rows = []
    capture_rows = []
    main_rows = []
    source_rows = []
    seq_to_fold = {}
    group_to_fold = {}
    seq_counts = {}
    for fold in range(5):
        group = f"cap{fold}"
        group_to_fold[group] = fold
        for modality in ("rednir", "vis"):
            seq = f"{modality}-{group}"
            seq_to_fold[seq] = fold
            seq_counts[seq] = 6
            capture_rows.append({
                "sequence": seq, "modality": modality, "capture_group": group,
                "fold": fold, "n_frames": 6,
                "evidence": "exact basename after modality removal",
            })
            for position, frame in enumerate(range(1, 7)):
                ident = f"{seq}_{frame}"
                if modality == "rednir":
                    a = [10.0 + frame, 10.0, 4.0, 4.0]
                    b = [30.0 + frame, 10.0, 4.0, 4.0]
                    gt = a if frame % 2 == 0 else b
                else:
                    # f2..f4 are an exact A freeze.  At K=2 only f4 is spliced.
                    a = ([20.0, 20.0, 4.0, 4.0] if 2 <= frame <= 4
                         else [10.0 + frame, 20.0, 4.0, 4.0])
                    b = [40.0 + frame, 20.0, 4.0, 4.0]
                    gt = b if frame == 4 else a
                if position == 0:
                    # Both trackers must start from the same init.
                    a = b = gt = [11.0, 10.0 if modality == "rednir" else 20.0, 4.0, 4.0]
                sample_rows.append(dict(zip(oof.SUBMISSION_COLUMNS, [ident, 0, 0, 0, 0])))
                gt_rows.append(dict(zip(oof.SUBMISSION_COLUMNS, [ident, *gt])))
                main_rows.append(dict(zip(oof.SUBMISSION_COLUMNS, [ident, *a])))
                source_rows.append(dict(zip(oof.SUBMISSION_COLUMNS, [ident, *b])))
                frame_rows.append({
                    "ID": ident, "sequence": seq, "position": position,
                    "modality": modality, "capture_group": group, "fold": fold,
                    "gt_valid": 1, "selected_block_start": 1, "selected_block_end": 6,
                })

    _csv(contracts / "sample_contract.csv", oof.SUBMISSION_COLUMNS, sample_rows)
    _csv(contracts / "observed_gt.csv", oof.SUBMISSION_COLUMNS, gt_rows)
    _csv(contracts / "frame_contract.csv", oof.FRAME_COLUMNS, frame_rows)
    _csv(contracts / "capture_groups.csv", oof.CAPTURE_COLUMNS, capture_rows)
    _csv(contracts / "mirror_map.csv", oof.MIRROR_COLUMNS, [])
    _csv(contracts / "scoring_gt.csv", oof.SUBMISSION_COLUMNS, gt_rows)
    folds = {
        "schema_version": 1,
        "group_rule": "strip modality prefix only; preserve full basename",
        "n_folds": 5, "n_groups": 5, "n_cross_modality_groups": 5,
        "fold_frames": [12] * 5, "fold_sequences": [2] * 5,
        "group_to_fold": group_to_fold, "sequence_to_fold": seq_to_fold,
    }
    (contracts / "folds.json").write_text(json.dumps(folds, indent=2) + "\n")
    prep = {
        "schema_version": 1, "sequences": 10, "gt_rows_total": len(gt_rows),
        "physical_prediction_rows": len(gt_rows), "mirrorable_duplicate_gt_rows": 0,
        "scoring_rows_with_mirror_sensitivity": len(gt_rows), "excluded_rows_total": 0,
        "unobservable_gt_rows": 0, "invalid_observed_gt_boxes": 0,
    }
    (contracts / "prep_report.json").write_text(json.dumps(prep, indent=2) + "\n")
    _rehash(contracts)

    main = root / "full_sam3" / "submission.csv"
    source = root / "full_samurai" / "submission.csv"
    _csv(main, oof.SUBMISSION_COLUMNS, main_rows)
    _csv(source, oof.SUBMISSION_COLUMNS, source_rows)
    for path, backend, signature in ((main, "sam3", "a" * 64),
                                     (source, "samurai", "b" * 64)):
        meta = {
            "backend": backend, "validation": "pass", "allow_fallback": False,
            "samurai_reset_kf": True, "samurai_legacy_cross_seq_kf": False,
            "n_seqs": 10,
        }
        if backend == "sam3":
            meta["sam3_eval"] = True
        diagnostics = {"_meta": meta}
        diagnostics.update({
            seq: {"status": "complete", "n_avail": count, "run_signature": signature}
            for seq, count in seq_counts.items()
        })
        path.with_name("diagnostics.json").write_text(json.dumps(diagnostics) + "\n")
    return contracts, main, source


def test_formal_contract_crossfit_excludes_heldout_fold_and_preserves_init(tmp_path):
    contracts, main, source = _formal_fixture(tmp_path)
    dataset = oof.load_dataset(contracts, main, source)
    report, predictions, models = oof.evaluate_grid(
        dataset, ["none"], [6], ["none", "crossfit-v1"], include_predictions=True)
    candidate_id = "corr=none__K=6__qhead=crossfit-v1"
    assert candidate_id in predictions
    assert candidate_id in models
    assert len(models[candidate_id]) == 5
    for model in models[candidate_id]:
        heldout = model["heldout_fold"]
        assert model["train_folds"] == [fold for fold in range(5) if fold != heldout]
        assert model["n_train"] >= 16
        assert model["class_a_better"] >= 2
        assert model["class_b_better"] >= 2
    assert np.array_equal(
        predictions[candidate_id][dataset.first_mask], dataset.raw_main[dataset.first_mask])
    candidate = next(c for c in report["candidates"] if c["candidate_id"] == candidate_id)
    assert candidate["first_frames_exact_raw_main"] == 10
    assert report["protocol"]["qhead"]["forbidden_model_inputs"]


def test_heldout_gt_cannot_change_its_own_qhead_decisions(tmp_path):
    contracts, main_path, source_path = _formal_fixture(tmp_path)
    dataset = oof.load_dataset(contracts, main_path, source_path)
    main = oof.apply_correction(dataset.raw_main, dataset.first_mask, "none")
    source = oof.apply_correction(dataset.raw_source, dataset.first_mask, "none")
    spliced, _ = oof.frozen_splice(
        main, source, dataset.sequence_indices, dataset.first_mask, 6)
    before, _, _ = oof.apply_crossfit_qhead(dataset, spliced, main, source, 6)

    changed_gt = dataset.gt.copy()
    # Reverse every eligible fold-0 label while retaining valid, non-tie examples.
    # Other outer models may change; the model whose held-out fold is 0 must not.
    for idx in np.flatnonzero((dataset.folds == 0) & (~dataset.first_mask)):
        if dataset.modalities[idx] == "rednir":
            changed_gt[idx] = (
                source[idx] if np.array_equal(changed_gt[idx], main[idx]) else main[idx])
    changed = replace(dataset, gt=changed_gt)
    after, _, _ = oof.apply_crossfit_qhead(changed, spliced, main, source, 6)
    assert np.array_equal(before[dataset.folds == 0], after[dataset.folds == 0])


def test_static_corr_and_splice_match_production_finalizer(tmp_path):
    contracts, main_path, source_path = _formal_fixture(tmp_path)
    dataset = oof.load_dataset(contracts, main_path, source_path)
    raw_main, _ = finalizer.load(main_path)
    raw_source, _ = finalizer.load(source_path)
    id_to_index = {ident: idx for idx, ident in enumerate(dataset.ids)}
    for mode, top, left in (("none", 0.0, 0.0), ("top-only", 1.0, 0.0),
                            ("both", 1.0, 1.0)):
        got_main = oof.apply_correction(dataset.raw_main, dataset.first_mask, mode)
        got_source = oof.apply_correction(dataset.raw_source, dataset.first_mask, mode)
        want_main = finalizer.apply_correction(raw_main, top=top, left=left)
        want_source = finalizer.apply_correction(raw_source, top=top, left=left)
        for ident, idx in id_to_index.items():
            seq, frame = ident.rsplit("_", 1)
            assert got_main[idx].tolist() == want_main[seq][int(frame)]
            assert got_source[idx].tolist() == want_source[seq][int(frame)]
        got, selected = oof.frozen_splice(
            got_main, got_source, dataset.sequence_indices, dataset.first_mask, 2)
        want, n_selected = finalizer.splice(want_main, want_source, K=2)
        assert int(selected.sum()) == n_selected == 5
        for ident, idx in id_to_index.items():
            seq, frame = ident.rsplit("_", 1)
            assert got[idx].tolist() == want[seq][int(frame)]


def test_physical_capture_cross_fold_leak_fails_closed(tmp_path):
    contracts, main, source = _formal_fixture(tmp_path)
    frame_path = contracts / "frame_contract.csv"
    capture_path = contracts / "capture_groups.csv"
    folds_path = contracts / "folds.json"
    frame_rows = list(csv.DictReader(frame_path.open()))
    capture_rows = list(csv.DictReader(capture_path.open()))
    for row in frame_rows:
        if row["sequence"] == "vis-cap0":
            row["fold"] = "1"
    for row in capture_rows:
        if row["sequence"] == "vis-cap0":
            row["fold"] = "1"
    folds = json.loads(folds_path.read_text())
    folds["sequence_to_fold"]["vis-cap0"] = 1
    _csv(frame_path, oof.FRAME_COLUMNS, frame_rows)
    _csv(capture_path, oof.CAPTURE_COLUMNS, capture_rows)
    folds_path.write_text(json.dumps(folds, indent=2) + "\n")
    _rehash(contracts)
    with pytest.raises(oof.ContractError, match="physical capture group crosses folds"):
        oof.load_dataset(contracts, main, source)


def test_tampered_contract_and_dirty_diagnostics_fail_closed(tmp_path):
    contracts, main, source = _formal_fixture(tmp_path)
    with (contracts / "frame_contract.csv").open("a") as fh:
        fh.write("tamper\n")
    with pytest.raises(oof.ContractError, match="hash mismatch"):
        oof.load_dataset(contracts, main, source)

    contracts, main, source = _formal_fixture(tmp_path / "second")
    diagnostics_path = main.with_name("diagnostics.json")
    diagnostics = json.loads(diagnostics_path.read_text())
    diagnostics["rednir-cap0"]["fallback"] = True
    diagnostics_path.write_text(json.dumps(diagnostics))
    with pytest.raises(oof.ContractError, match="forbidden status"):
        oof.load_dataset(contracts, main, source)


def test_real_q95_exact_pair_smoke_if_present(tmp_path):
    root = Path(__file__).parents[1]
    run = root / "5_outputs/q100_crop_pilot_20260824/20260824T0124Z-d7edb7f5"
    contract = run / "val_contract.csv"
    main = run / "val/legacy-q95/main_merged_global.csv"
    source = run / "val/legacy-q95/source_merged_global.csv"
    gt = root / "1_data/raw/2026training.csv"
    if not all(path.is_file() for path in (contract, main, source, gt)):
        pytest.skip("fresh q95 exact-pair smoke files are not present")
    dataset = oof.load_explicit_smoke_dataset(contract, gt, main, source)
    report, _, _ = oof.evaluate_grid(
        dataset, ["both"], [6], ["none"], include_predictions=False)
    by_id = {candidate["candidate_id"]: candidate for candidate in report["candidates"]}
    assert by_id["corr=both__policy=main"]["auc_observed"] == pytest.approx(0.69470, abs=7e-6)
    assert by_id["corr=both__policy=source"]["auc_observed"] == pytest.approx(0.69076, abs=7e-6)
    assert by_id["corr=both__K=6__qhead=none"]["auc_observed"] == pytest.approx(0.70206, abs=7e-6)
    assert dataset.input_audit["formal_train405_contract"] is False


def test_oof_launcher_finishes_crossfit_before_success():
    launcher = Path(__file__).parent / "launch/run_oof405_zero_shot_lambda_v1.sh"
    text = launcher.read_text(encoding="utf-8")
    inference_done = text.index("INFERENCE_DONE")
    crossfit_call = text.index('"$SRC/3_src/oof405_crossfit.py"')
    crossfit_done = text.index("CROSSFIT_DONE")
    assert inference_done < crossfit_call < crossfit_done
    assert "--contracts \"$WORK/contracts\"" in text
    assert "--main \"$WORK/run/full_sam3/submission.csv\"" in text
    assert "--source \"$WORK/run/full_samurai/submission.csv\"" in text
    assert "--ks 2,3,5,6,8,24" in text
    assert 'test -s "$WORK/crossfit/report.json"' in text


def _crop_pair_fixture(root: Path):
    """Add rankB_robust_crop artifacts on top of the full-frame formal fixture.

    Mirrors production shape: `hsot.crop_rerun prep` writes a flat window-key ->
    window mapping (one sequence may own several segment windows) and
    run_ranking_b feeds `sorted(meta)` to the crop legs as their --seq-list, so
    the crop diagnostics are keyed by window name, not sequence name.
    """
    contracts, main, source = _formal_fixture(root)
    crop_meta_path = root / "offline_two_pass" / "crop_meta.json"
    crop_meta_path.parent.mkdir(parents=True, exist_ok=True)
    windows = {
        "rednir-cap0__seg0": {"seq": "rednir-cap0", "orig": [64, 64],
                              "frames": [1, 3], "zoom": 4.2},
        "rednir-cap0__seg1": {"seq": "rednir-cap0", "orig": [64, 64],
                              "frames": [4, 6], "zoom": 4.0},
        "vis-cap1": {"seq": "vis-cap1", "orig": [64, 64], "frames": [1, 6], "zoom": 3.1},
    }
    crop_meta_path.write_text(json.dumps(windows, indent=1) + "\n")

    crop_diags = {}
    for leg, backend, signature in (("crop_sam3", "sam3", "c" * 64),
                                    ("crop_samurai", "samurai", "d" * 64)):
        meta = {
            "backend": backend, "validation": "pass", "allow_fallback": False,
            "samurai_reset_kf": True, "samurai_legacy_cross_seq_kf": False,
            "n_seqs": len(windows),
        }
        if backend == "sam3":
            meta["sam3_eval"] = True
        doc = {"_meta": meta}
        # window frame counts differ from the full sequence length on purpose
        doc.update({
            key: {"status": "complete", "n_avail": 3, "run_signature": signature}
            for key in windows
        })
        path = root / leg / "diagnostics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc) + "\n")
        crop_diags[leg] = path
    return contracts, main, source, crop_meta_path, crop_diags


def test_crop_pair_is_certified_and_audited_separately(tmp_path):
    contracts, main, source, crop_meta, diags = _crop_pair_fixture(tmp_path)
    dataset = oof.load_dataset(
        contracts, main, source,
        pair_profile="rankB_robust_crop",
        main_crop_diagnostics=diags["crop_sam3"],
        source_crop_diagnostics=diags["crop_samurai"],
        crop_meta=crop_meta,
    )
    audit = dataset.input_audit
    assert audit["pair_profile"] == "rankB_robust_crop"
    assert "offline two-pass crop" in audit["pair_scope"]
    assert "rankA_best" in audit["does_not_certify"]
    crop = audit["offline_two_pass_crop"]
    assert crop["crop_meta"]["windows"] == 3
    # two segment windows collapse to one sequence
    assert crop["crop_meta"]["sequences"] == ["rednir-cap0", "vis-cap1"]
    assert crop["crop_meta"]["cropped_sequences"] == 2
    assert crop["crop_meta"]["contract_sequences"] == 10

    # the full-frame scope must stay truthful and crop-free
    plain = oof.load_dataset(contracts, main, source)
    assert plain.input_audit["pair_profile"] == "rankB_robust"
    assert plain.input_audit["offline_two_pass_crop"] is None
    assert "full-frame" in plain.input_audit["pair_scope"]


def test_crop_pair_declaration_and_artifacts_fail_closed(tmp_path):
    contracts, main, source, crop_meta, diags = _crop_pair_fixture(tmp_path)
    kwargs = dict(
        main_crop_diagnostics=diags["crop_sam3"],
        source_crop_diagnostics=diags["crop_samurai"],
        crop_meta=crop_meta,
    )
    # declaring the crop pair without its artifacts is refused
    with pytest.raises(oof.ContractError, match="requires --main-crop-diagnostics"):
        oof.load_dataset(contracts, main, source, pair_profile="rankB_robust_crop")
    # smuggling crop artifacts into a full-frame declaration is refused
    with pytest.raises(oof.ContractError, match="full-frame pair"):
        oof.load_dataset(contracts, main, source, pair_profile="rankB_robust", **kwargs)
    with pytest.raises(oof.ContractError, match="unknown pair_profile"):
        oof.load_dataset(contracts, main, source, pair_profile="rankA_best")

    # a crop leg that fell back is refused
    doc = json.loads(diags["crop_sam3"].read_text())
    doc["vis-cap1"]["fallback"] = True
    diags["crop_sam3"].write_text(json.dumps(doc))
    with pytest.raises(oof.ContractError, match="forbidden status"):
        oof.load_dataset(contracts, main, source,
                         pair_profile="rankB_robust_crop", **kwargs)

    # crop_meta naming a sequence outside the contract is refused
    contracts2, main2, source2, meta2, diags2 = _crop_pair_fixture(tmp_path / "second")
    windows = json.loads(meta2.read_text())
    windows["nir-not-in-contract"] = {"seq": "nir-not-in-contract", "frames": [1, 3]}
    meta2.write_text(json.dumps(windows))
    with pytest.raises(oof.ContractError, match="outside the contract"):
        oof.load_dataset(contracts2, main2, source2, pair_profile="rankB_robust_crop",
                         main_crop_diagnostics=diags2["crop_sam3"],
                         source_crop_diagnostics=diags2["crop_samurai"],
                         crop_meta=meta2)


def test_oof_launcher_certifies_the_crop_pair_it_will_ship():
    launcher = Path(__file__).parent / "launch/run_oof405_zero_shot_lambda_v1.sh"
    text = launcher.read_text(encoding="utf-8")

    # the GPU legs must be the 9/7 delivery candidate, explicitly two-pass authorised
    assert "--profile rankB_robust_crop" in text
    assert "--allow-offline-two-pass" in text

    # the crop cross-fit must consume the merged pair plus its crop contracts
    assert "--pair-profile rankB_robust_crop" in text
    assert '--main "$WORK/run/offline_two_pass/main_merged.csv"' in text
    assert '--source "$WORK/run/offline_two_pass/source_merged.csv"' in text
    assert '--crop-meta "$WORK/run/offline_two_pass/crop_meta.json"' in text
    assert '--main-crop-diagnostics "$WORK/run/crop_sam3/diagnostics.json"' in text
    assert '--source-crop-diagnostics "$WORK/run/crop_samurai/diagnostics.json"' in text

    # both delivery options get a reading, so 9/6 is a comparison and not a guess
    assert "--pair-profile rankB_robust\n" in text or "--pair-profile rankB_robust \\" in text
    crop_report = text.index('test -s "$WORK/crossfit_crop/report.json"')
    plain_report = text.index('test -s "$WORK/crossfit/report.json"')
    assert text.index("INFERENCE_DONE") < crop_report < plain_report < text.index("CROSSFIT_DONE")

    # a new destination: this cache certifies a different pair than the 08-22 path
    assert "oof405_exact_pair_20260827" in text
    assert "oof405_zero_shot_20260822" not in text

    # official gated weight, fail-closed
    assert "facebook/sam3/resolve/main/sam3.pt" in text
    assert "1038lab" not in text
    assert "sha256sum -c -" in text
    assert 'HF_TOKEN:?' in text


def test_oof_launcher_env_setup_is_idempotent_for_resume():
    """A restart after a mid-run failure must not die on an existing venv.

    The runner advertises --resume-existing-work and the 08-27 405 run proved it
    matters: after crop_rerun failed, the restart died instantly on
    `uv venv: A virtual environment already exists`, throwing away six hours of
    completed full-frame legs' worth of machine time before resume could apply.
    """
    launcher = Path(__file__).parent / "launch/run_oof405_zero_shot_lambda_v1.sh"
    text = launcher.read_text(encoding="utf-8")
    for env in ("T1ENV", "SAM3ENV"):
        assert f'[ -x "${env}/bin/python" ] || uv venv' in text, f"{env} venv not guarded"
    # no bare unconditional creation left behind
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("uv venv"):
            raise AssertionError(f"unguarded venv creation: {stripped}")


def _resumed_leg(root: Path, backend: str, seq_counts: dict[str, int], *, good=True):
    """Diagnostics in the resumed shape plus the sidecars they point at."""
    leg = root / f"full_{backend}"
    (leg / "seq_csv").mkdir(parents=True, exist_ok=True)
    meta = {
        "backend": backend, "validation": "pass", "allow_fallback": False,
        "samurai_reset_kf": True, "samurai_legacy_cross_seq_kf": False,
        "n_seqs": len(seq_counts),
    }
    if backend == "sam3":
        meta["sam3_eval"] = True
    doc = {"_meta": meta}
    for i, (seq, rows) in enumerate(seq_counts.items()):
        digest = f"{i:064d}"
        sidecar = leg / "seq_csv" / f"{seq}.status.json"
        config = {"backend": backend, "samurai_reset_kf": True}
        if backend == "sam3":
            config["sam3_eval"] = True if good else False
        sidecar.write_text(json.dumps({
            "schema_version": 2, "sequence": seq,
            "status": "complete" if good else "failed",
            "rows": rows,
            "run_signature": {"digest": digest, "run_config": config},
            "details": {"n_avail": rows, "status": "complete"},
        }))
        doc[seq] = {
            "skipped": "already_done", "rows": rows,
            # absolute GPU-box path on purpose: resolution must work off a synced copy
            "status_sidecar": f"/home/ubuntu/run/full_{backend}/seq_csv/{seq}.status.json",
            "run_signature": digest,
        }
    path = leg / "diagnostics.json"
    path.write_text(json.dumps(doc))
    return path


def test_resumed_diagnostics_are_validated_through_their_sidecars(tmp_path):
    """The 08-27 formal 405 run died here: after resume every sequence's entry is
    {"skipped": "already_done", ...} and the fresh-run shape check rejected all 405."""
    counts = {"nir-ball": 626, "vis-cup3": 493}
    path = _resumed_leg(tmp_path, "sam3", counts)
    audit = oof._validate_diagnostics(path, "sam3", counts)
    assert audit["validation"] == "pass"
    assert audit["resumed_sequences_verified_via_sidecar"] == 2


def test_resumed_sequence_fails_closed_when_the_sidecar_is_missing(tmp_path):
    counts = {"nir-ball": 626}
    path = _resumed_leg(tmp_path, "sam3", counts)
    (path.parent / "seq_csv" / "nir-ball.status.json").unlink()
    with pytest.raises(oof.ContractError, match="status sidecar is absent"):
        oof._validate_diagnostics(path, "sam3", counts)


def test_resumed_sequence_fails_closed_on_bad_status_rows_digest_or_config(tmp_path):
    counts = {"nir-ball": 626}

    # sidecar says the run did not complete
    bad = _resumed_leg(tmp_path / "a", "sam3", counts, good=False)
    with pytest.raises(oof.ContractError, match="sidecar status="):
        oof._validate_diagnostics(bad, "sam3", counts)

    # row count disagrees with the frame contract
    path = _resumed_leg(tmp_path / "b", "sam3", counts)
    with pytest.raises(oof.ContractError, match="!= frame contract"):
        oof._validate_diagnostics(path, "sam3", {"nir-ball": 999})

    # sidecar digest does not match the diagnostics summary
    path = _resumed_leg(tmp_path / "c", "sam3", counts)
    sidecar = path.parent / "seq_csv" / "nir-ball.status.json"
    doc = json.loads(sidecar.read_text())
    doc["run_signature"]["digest"] = "f" * 64
    sidecar.write_text(json.dumps(doc))
    with pytest.raises(oof.ContractError, match="run_signature digest does not match"):
        oof._validate_diagnostics(path, "sam3", counts)

    # a leg that silently mixed in a non-eval config
    path = _resumed_leg(tmp_path / "d", "sam3", counts)
    sidecar = path.parent / "seq_csv" / "nir-ball.status.json"
    doc = json.loads(sidecar.read_text())
    doc["run_signature"]["run_config"]["sam3_eval"] = False
    sidecar.write_text(json.dumps(doc))
    with pytest.raises(oof.ContractError, match="run_config.sam3_eval"):
        oof._validate_diagnostics(path, "sam3", counts)
