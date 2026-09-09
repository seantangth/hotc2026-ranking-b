#!/usr/bin/env python3
"""Phase 0 fail-closed 護欄的回歸測試（純 CPU、無模型依賴）。

可直接執行：`python3 3_src/test_failclosed.py`，亦相容 pytest。
涵蓋 D052b 指出的三個缺口：exception fallback、resume 身分驗證、formal validator。
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from track_t1 import (  # noqa: E402
    SUB_COLS, build_run_signature, count_frames, frame_ids_for, load_sample_frame_ids,
    mask_cache_ok, resume_ok, rows_from_boxes, score_cache_ok,
    resolve_samurai_reset_kf, status_path_for, validate_submission, write_csv_atomic,
    write_scores_atomic, write_sequence_status,
)


def test_write_csv_atomic_leaves_no_partial():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.csv"
        write_csv_atomic(p, [("seq_1", 1, 2, 3, 4), ("seq_2", 5, 6, 7, 8)])
        assert p.exists() and not p.with_suffix(".csv.part").exists()
        d = pd.read_csv(p)
        assert list(d.columns) == SUB_COLS and len(d) == 2


def test_samurai_reset_default_legacy_opt_in_and_conflict():
    assert resolve_samurai_reset_kf(False, False) is True
    assert resolve_samurai_reset_kf(True, False) is True  # 舊 flag 相容
    assert resolve_samurai_reset_kf(False, True) is False
    try:
        resolve_samurai_reset_kf(True, True)
    except ValueError:
        pass
    else:
        raise AssertionError("相互矛盾的 reset/legacy flags 應被拒絕")


def test_resume_rejects_truncated_and_foreign_csv():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.csv"
        sig = {"digest": "run-A"}
        write_csv_atomic(p, [(f"seqA_{i}", 1, 2, 3, 4) for i in range(1, 11)])
        write_sequence_status(p, "seqA", "complete", sig, 10)
        assert resume_ok(p, "seqA", 10, sig)[0] is True
        # 列數不符 = 上次跑到一半（舊版只看存在性 → fail-open）
        ok, why = resume_ok(p, "seqA", 20, sig)
        assert ok is False and "row_mismatch" in why
        # 別支序列的檔案混入（快取目錄複用時的真實風險）
        ok, why = resume_ok(p, "seqB", 10, sig)
        assert ok is False and "bad_id_prefix" in why
        # 半截/損壞檔
        bad = Path(td) / "bad.csv"
        bad.write_text("ID,x,y,wid")
        assert resume_ok(bad, "seqA", 10, sig)[0] is False


def test_resume_requires_complete_sidecar_and_exact_run_signature():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.csv"
        write_csv_atomic(p, [(f"seqA_{i}", 1, 2, 3, 4) for i in range(1, 4)])
        sig_a, sig_b = {"digest": "run-A"}, {"digest": "run-B"}

        assert resume_ok(p, "seqA", 3, sig_a) == (False, "missing_status")
        write_sequence_status(p, "seqA", "complete", sig_a, 3)
        assert resume_ok(p, "seqA", 3, sig_a) == (True, "ok")
        ok, why = resume_ok(p, "seqA", 3, sig_b)
        assert ok is False and "run_signature_mismatch" in why

        write_sequence_status(p, "seqA", "fallback", sig_a, 3,
                              {"error": "OOM", "fallback": "init_copy x3"})
        ok, why = resume_ok(p, "seqA", 3, sig_a)
        assert ok is False and why == "status:fallback"
        sidecar = json.loads(status_path_for(p).read_text())
        assert sidecar["status"] == "fallback"
        assert sidecar["details"]["fallback"] == "init_copy x3"


def test_resume_rejects_same_shape_csv_if_content_changed_after_status():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.csv"
        sig = {"digest": "run-A"}
        write_csv_atomic(p, [(f"seqA_{i}", 1, 2, 3, 4) for i in range(1, 4)])
        write_sequence_status(p, "seqA", "complete", sig, 3)
        assert resume_ok(p, "seqA", 3, sig) == (True, "ok")
        write_csv_atomic(p, [(f"seqA_{i}", 9, 8, 7, 6) for i in range(1, 4)])
        ok, why = resume_ok(p, "seqA", 3, sig)
        assert ok is False and why == "artifact_sha256_mismatch"


def test_sam3_score_cache_requires_hash_rows_and_exact_ids():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        boxes, scores = root / "s.csv", root / "s.scores.csv"
        rows = [(f"seqA_{i}", 1, 2, 3, 4) for i in range(1, 4)]
        write_csv_atomic(boxes, rows)
        write_scores_atomic(scores, [r[0] for r in rows], [0.1, 0.2, 0.3])
        import hashlib
        score_sha = hashlib.sha256(scores.read_bytes()).hexdigest()
        write_sequence_status(boxes, "seqA", "complete", {"digest": "run-A"}, 3,
                              {"obj_scores_sha256": score_sha})
        assert score_cache_ok(scores, boxes, 3) is True
        write_scores_atomic(scores, [r[0] for r in rows], [9.1, 9.2, 9.3])
        assert score_cache_ok(scores, boxes, 3) is False


def test_run_signature_binds_config_checkpoint_and_frame_manifest():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        seq = root / "seq"
        seq.mkdir()
        (seq / "00001.jpg").write_bytes(b"frame-one")
        (seq / "init_rect.txt").write_text("1 2 3 4")
        cfg_a = {"backend": "samurai", "checkpoint": {"path": "a.pt"}}
        cfg_b = {"backend": "sam3", "checkpoint": {"path": "b.pt"}}
        sig_a = build_run_signature(seq, cfg_a)
        assert build_run_signature(seq, cfg_a) == sig_a
        assert build_run_signature(seq, cfg_b)["digest"] != sig_a["digest"]
        (seq / "00001.jpg").write_bytes(b"changed-frame-with-new-size")
        assert build_run_signature(seq, cfg_a)["digest"] != sig_a["digest"]


def test_resume_rejects_missing():
    with tempfile.TemporaryDirectory() as td:
        assert resume_ok(Path(td) / "nope.csv", "s", 5) == (False, "missing")


def test_mask_cache_ok_unlinks_corrupt():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.npz"
        p.write_bytes(b"not an npz")
        assert mask_cache_ok(p) is False
        assert not p.exists(), "壞快取應被移除以便自我修復"


def test_validator_catches_missing_extra_dup_nan():
    with tempfile.TemporaryDirectory() as td:
        smp = Path(td) / "sample.csv"
        pd.DataFrame([(f"s_{i}", 0, 0, 1, 1) for i in range(1, 6)],
                     columns=SUB_COLS).to_csv(smp, index=False)
        good = pd.DataFrame([(f"s_{i}", 0, 0, 1, 1) for i in range(1, 6)], columns=SUB_COLS)
        assert validate_submission(good, smp) == []

        short = good.iloc[:3]
        assert any("缺 2 個" in e for e in validate_submission(short, smp))

        extra = pd.concat([good, pd.DataFrame([("s_99", 0, 0, 1, 1)], columns=SUB_COLS)])
        assert any("多出 1 個" in e for e in validate_submission(extra, smp))

        dup = pd.concat([good, good.iloc[:1]])
        assert any("重複 ID" in e for e in validate_submission(dup, smp))

        nonpos = good.copy()
        nonpos.loc[0, "width"] = 0
        assert any("非正 w/h" in e for e in validate_submission(nonpos, smp))

        nan = good.copy()
        nan.loc[0, "x"] = float("nan")
        assert any("NaN" in e for e in validate_submission(nan, smp))

        inf = good.copy()
        inf["x"] = inf["x"].astype(float)
        inf.loc[0, "x"] = float("inf")
        assert any("finite" in e for e in validate_submission(inf, smp))

        negative = good.copy()
        negative.loc[0, "y"] = -1
        assert any("負 x/y" in e for e in validate_submission(negative, smp))

        reversed_order = good.iloc[::-1].reset_index(drop=True)
        assert any("順序" in e for e in validate_submission(reversed_order, smp))

        assert any("fallback" in e for e in validate_submission(good, smp, ["s"], False))
        assert validate_submission(good, smp, ["s"], True) == []


def test_fallback_produces_full_length_csv_but_cannot_resume():
    """序列失敗時以 init 框補滿，但 sidecar 必須使 resume 拒絕它。"""
    with tempfile.TemporaryDirectory() as td:
        seq, init, fids = "vis-x", [10.0, 20.0, 30.0, 40.0], list(range(101, 151))
        p = Path(td) / f"{seq}.csv"
        sig = {"digest": "run-fallback"}
        write_csv_atomic(p, [(f"{seq}_{f}", *init) for f in fids])
        write_sequence_status(p, seq, "fallback", sig, len(fids),
                              {"fallback": f"init_copy x{len(fids)}"})
        d = pd.read_csv(p)
        assert len(d) == len(fids)
        assert (d[["x", "y", "width", "height"]].nunique() == 1).all(), "fallback 應為同一 init 框"
        ok, why = resume_ok(p, seq, len(fids), sig)
        assert ok is False and why == "status:fallback"


def test_missing_positions_use_latest_box_known_at_that_time():
    boxes = {
        0: [0.0, 0.0, 10.0, 10.0],
        1: [1.0, 0.0, 10.0, 10.0],
        3: [30.0, 0.0, 10.0, 10.0],
        5: [50.0, 0.0, 10.0, 10.0],
    }
    rows = rows_from_boxes("s", [1, 2, 3, 4, 5, 6], boxes)
    xs = [r[1] for r in rows]
    assert xs == [0.0, 1.0, 1.0, 30.0, 30.0, 50.0]


def test_count_frames_matches_glob_semantics():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for i in range(3):
            (d / f"{i:04d}.jpg").write_bytes(b"x")
        (d / "0003.jpeg").write_bytes(b"x")
        assert count_frames(d) == 4
        assert count_frames(Path(td) / "nonexistent") == 0


def test_sample_contract_controls_global_frame_ids_and_fails_closed():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "contract.csv"
        pd.DataFrame([
            ("nir-x_1617", 0, 0, 0, 0),
            ("nir-x_1618", 0, 0, 0, 0),
        ], columns=SUB_COLS).to_csv(p, index=False)
        groups = load_sample_frame_ids(p)
        assert frame_ids_for("nir-x", 2, None, groups) == [1617, 1618]
        try:
            frame_ids_for("nir-x", 3, None, groups)
        except ValueError as exc:
            assert "實際影像" in str(exc)
        else:
            raise AssertionError("contract 列數與影像數不符時必須拒絕")


def test_gt_mismatch_requires_explicit_contract():
    gt = pd.DataFrame({"seq": ["s", "s", "s"], "frame": [1, 2, 99]})
    try:
        frame_ids_for("s", 2, gt)
    except ValueError as exc:
        assert "--sample-csv contract" in str(exc)
    else:
        raise AssertionError("GT 多區塊/短幀不得再靜默取全部 frame ID")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"✅ {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"❌ {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-fails}/{len(fns)} 通過")
    sys.exit(1 if fails else 0)
