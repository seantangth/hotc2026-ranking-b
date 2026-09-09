#!/usr/bin/env python3
"""finalize_submission.py P0 交付護欄回歸測試（純 CPU）。"""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import finalize_submission as fs  # noqa: E402


SCRIPT = Path(__file__).with_name("finalize_submission.py")


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fs.SUBMISSION_COLUMNS)
        writer.writerows(rows)


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_correction_preserves_earliest_frame_and_legacy_opt_out():
    raw = {"s": {7: [10.0, 20.0, 3.0, 4.0], 9: [12.0, 22.0, 3.0, 4.0]}}
    safe = fs.apply_correction(raw)
    assert safe["s"][7] == raw["s"][7]
    assert safe["s"][9] == [11.0, 21.0, 4.0, 5.0]

    legacy = fs.apply_correction(raw, preserve_first=False)
    assert legacy["s"][7] == [9.0, 19.0, 4.0, 5.0]


def test_qhead_gate_uses_callers_k():
    # A 連續凍結，B 逐幀移動。K=2 時 A run>=2 必須跳過；K=10 則全是候選。
    seq = "rednir-unit"
    A = {seq: {f: [0.0, 0.0, 10.0, 10.0] for f in range(1, 5)}}
    B = {seq: {f: [100.0 + f, 0.0, 10.0, 10.0] for f in range(1, 5)}}
    spliced = {seq: {f: list(box) for f, box in A[seq].items()}}
    with tempfile.TemporaryDirectory() as td:
        weights = Path(td) / "w.npz"
        np.savez(weights, w=np.zeros(25), mu=np.zeros(25), sd=np.ones(25))
        with patch("hsot.quality_head_v1.predict_p", return_value=np.array([0.0])):
            _, _, cand_k2 = fs.apply_qhead_v056(spliced, A, B, weights, K=2)
            _, _, cand_k10 = fs.apply_qhead_v056(spliced, A, B, weights, K=10)
    assert cand_k2 == 2
    assert cand_k10 == 4


def test_production_uses_sample_order_default_no_qhead_and_keeps_init():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        main = td / "main.csv"
        source = td / "source.csv"
        sample = td / "sample.csv"
        out = td / "final.csv"
        main_rows = [
            ("seq_1", 10, 10, 2, 2),
            ("seq_2", 20, 20, 2, 2),
            ("other_5", 5, 6, 3, 4),
            ("other_7", 8, 9, 3, 4),
        ]
        source_rows = [
            ("other_7", 30, 30, 3, 4),
            ("seq_1", 40, 40, 2, 2),
            ("other_5", 31, 31, 3, 4),
            ("seq_2", 41, 41, 2, 2),
        ]
        sample_rows = [
            ("seq_2", 0, 0, 0, 0),
            ("other_7", 0, 0, 0, 0),
            ("seq_1", 0, 0, 0, 0),
            ("other_5", 0, 0, 0, 0),
        ]
        _write_csv(main, main_rows)
        _write_csv(source, source_rows)
        _write_csv(sample, sample_rows)

        proc = _run("--main", main, "--source", source, "--sample", sample, "--out", out)
        assert proc.returncode == 0, proc.stderr
        assert "--qhead none" in proc.stdout
        assert "放棄 RedNIR" not in proc.stdout
        rows = list(csv.DictReader(out.open()))
        assert [r["ID"] for r in rows] == [r[0] for r in sample_rows]
        boxes = {r["ID"]: [float(r[c]) for c in fs.SUBMISSION_COLUMNS[1:]] for r in rows}
        assert boxes["seq_1"] == [10.0, 10.0, 2.0, 2.0]
        assert boxes["other_5"] == [5.0, 6.0, 3.0, 4.0]
        assert boxes["seq_2"] == [19.0, 19.0, 3.0, 3.0]


def test_production_requires_sample_and_rejects_bad_inputs():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        main = td / "main.csv"
        source = td / "source.csv"
        sample = td / "sample.csv"
        out = td / "final.csv"
        _write_csv(main, [("s_1", 0, 0, 1, 1)])
        _write_csv(source, [("s_1", 0, 0, 1, 1)])

        missing_sample = _run("--main", main, "--source", source, "--out", out)
        assert missing_sample.returncode == 2
        assert "--sample" in missing_sample.stderr
        assert not out.exists()

        # 官方 sample 的 box 是 0 placeholder；它只定義 ID/order，不應擋下 production。
        _write_csv(sample, [("s_1", 0, 0, 0, 0)])
        placeholder_sample = _run(
            "--main", main, "--source", source, "--sample", sample, "--out", out,
        )
        assert placeholder_sample.returncode == 0, placeholder_sample.stderr
        out.unlink()

        # main/source prediction 則仍必須 finite 且正尺寸。
        _write_csv(main, [("s_1", 0, 0, float("inf"), 1)])
        bad_main = _run(
            "--main", main, "--source", source, "--sample", sample, "--out", out,
        )
        assert bad_main.returncode == 2
        assert "NaN/Inf" in bad_main.stderr
        assert not out.exists()
        _write_csv(main, [("s_1", 0, 0, 1, 1)])

        # source duplicate 不得被 dict 靜默覆寫。
        _write_csv(sample, [("s_1", 0, 0, 0, 0)])
        _write_csv(source, [("s_1", 0, 0, 1, 1), ("s_1", 2, 2, 1, 1)])
        duplicate = _run(
            "--main", main, "--source", source, "--sample", sample, "--out", out,
        )
        assert duplicate.returncode == 2
        assert "重複 ID" in duplicate.stderr
        assert not out.exists()


def test_real_official_sample_placeholder_is_accepted():
    sample = Path(__file__).parents[1] / "1_data" / "raw" / "sample_submisson.csv"
    if not sample.is_file():
        return
    _, order = fs.load(sample, "sample", validate_boxes=False)
    assert order and len(order) == len(set(order))


def test_exact_set_row_count_and_box_domain_are_fail_closed():
    assert fs.validate_input_sets(["s_1"], ["s_1", "s_2"], ["s_1"])
    raw = {"s": {1: [0.0, 0.0, 1.0, 1.0]}}
    bad = {"s": {1: [-1.0, 0.0, 1.0, 1.0]}}
    errs = fs.validate(bad, ["s_1"], raw_main=raw)
    assert any("負 x/y" in e for e in errs)
    assert any("首幀未等於" in e for e in errs)


def test_atomic_write_does_not_replace_existing_file_on_failure():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out_path = td / "final.csv"
        out_path.write_text("known-good\n")
        out = {"s": {1: [0.0, 0.0, 1.0, 1.0]}}
        try:
            fs.write(out, ["s_1", "missing_2"], out_path)
        except KeyError:
            pass
        else:
            raise AssertionError("write 應因缺 ID 失敗")
        assert out_path.read_text() == "known-good\n"
        assert list(td.glob(".final.csv.*.tmp")) == []


def test_selftest_d_reproduces_v090_bit_exactly():
    """(d) 三窗共識鏈的位元級守護（D101）。

    這是 `rankB_deliver_v090` 的唯一位元級證據：medoid(A,B,C) → 交付後處理 → sub_v090。
    輸入在 5_outputs（gDrive 為唯一真相源、不進 git），缺檔時 skip——但**交付機器上必須跑到**，
    9/7 若走 v090 profile，launch 腳本的 selftest 閘門會 grep 這一段。
    """
    import pytest

    root = fs.ROOT
    needed = [
        root / "5_outputs/submissions/sub_v090_cropwin_medoid.csv",
        *(root / f"5_outputs/cropwindow_ensemble_20260831/{t}_main_merged.csv"
          for t in ("A", "B", "C")),
        root / "5_outputs/cropwindow_ensemble_20260831/A_source_merged.csv",
        root / "5_outputs/cropwindow_ensemble_20260831/A_full_sam3.csv",
        root / "1_data/raw/sample_submisson.csv",
        fs.SELECTOR_WEIGHTS,
    ]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        pytest.skip(f"缺 (d) 的輸入（gDrive 檔案未拉下）：{missing[0]}")
    assert fs._selftest_d() == 0
    # 分層計數與 (c) 不同是預期的：main 腿換成 medoid ⇒ 每層的候選都變了
    assert fs.V090_LAYER_COUNTS == (513, 7652, 127, 37)
    assert fs.V090_LAYER_COUNTS != (496, 7623, 110, 103)


# ─── G11：selftest 拿到「本次要跑的 sample」而非歷史 test75 sample ─────────────
def test_selftest_sample_guard_passes_on_matching_sample():
    """ID 集合相符 ⇒ 回 None（不擋）。"""
    chain = {"nir-ball": {1: [0, 0, 2, 2], 2: [0, 0, 2, 2]}, "vis-toy1": {1: [0, 0, 2, 2]}}
    order = ["nir-ball_1", "nir-ball_2", "vis-toy1_1"]
    assert fs.selftest_sample_matches(order, chain, "c") is None


def test_selftest_sample_guard_fires_on_9_7_style_new_sample():
    """9/7 的新 sample 是完全不同的序列 ⇒ 必須 fail-fast 並指名 drill 的 SAMPLE 覆蓋。

    舊碼在這裡是 write() 丟未捕捉的 KeyError，腳本死、機器 5 分鐘後自毀，
    症狀完全看不出是「sample 拿錯」（09-04 稽核 G11）。
    """
    chain = {"nir-ball": {1: [0, 0, 2, 2]}, "vis-toy1": {1: [0, 0, 2, 2]}}
    new_sample = ["rankb-newseq1_1", "rankb-newseq2_1"]
    msg = fs.selftest_sample_matches(new_sample, chain, "c")
    assert msg is not None
    assert "SELFTEST_SAMPLE" in msg      # 訊息要直接給修法
    assert "rankb-newseq1_1" in msg      # 要指出是哪些 ID 多出來


def test_selftest_sample_guard_fires_on_partial_overlap():
    """只差幾幀（例如新資料序列較長）也要擋——部分重疊最容易被誤判成「差不多」。"""
    chain = {"nir-ball": {1: [0, 0, 2, 2], 2: [0, 0, 2, 2]}}
    almost = ["nir-ball_1", "nir-ball_2", "nir-ball_3"]
    msg = fs.selftest_sample_matches(almost, chain, "d")
    assert msg is not None
    assert "nir-ball_3" in msg


# ─── G14：9/7 現場退路 profile（lean＝關掉 selector 與 qhead）─────────────────
def test_lean_fallback_profile_is_single_variable_vs_v090():
    """lean 設定檔必須與 v090 逐字相同，只差 selector 與 quality_head 兩個欄位。

    9/7 是「只執行不修改」的窗口 ⇒ 退路必須是**事前存在且被測過**的組態，
    不能當場改設定。刻意放獨立檔（--config 指過去），不動已封板的 ranking_profiles.json。
    """
    import json
    main = json.loads((fs.ROOT / "3_src/configs/ranking_profiles.json").read_text())
    lean = json.loads((fs.ROOT / "3_src/configs/ranking_profiles_lean.json").read_text())
    a = main["profiles"]["rankB_deliver_v090"]
    b = lean["profiles"]["rankB_deliver_v090_lean"]
    assert [k for k in a if k != "description" and a[k] != b[k]] == ["postprocess"]
    changed = {k: (a["postprocess"][k], b["postprocess"][k])
               for k in a["postprocess"] if a["postprocess"][k] != b["postprocess"][k]}
    assert changed == {"selector": ("v2", "none"), "quality_head": ("v056", "none")}


def test_lean_fallback_reference_is_bit_reproducible():
    """lean 參照檔必須能由磁碟上既有的 08-31 中間產物位元級重現（CPU，<1 秒）。"""
    import csv
    import pytest
    import tempfile
    from pathlib import Path
    import run_ensemble_medoid as medoid_tool

    D = fs.ROOT / "5_outputs/cropwindow_ensemble_20260831"
    ref = fs.ROOT / "5_outputs/submissions/sub_v090_lean_no_selector_no_qhead.csv"
    variants = [D / f"{t}_main_merged.csv" for t in ("A", "B", "C")]
    needed = [*variants, D / "A_source_merged.csv", D / "A_full_sam3.csv", ref,
              fs.ROOT / "1_data/raw/sample_submisson.csv"]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        pytest.skip(f"缺 lean 參照鏈輸入：{missing[0]}")

    tmp = Path(tempfile.gettempdir()) / "_test_lean_medoid.csv"
    assert medoid_tool.main([*map(str, variants), "--out", str(tmp)]) == 0
    main, _ = fs.load(tmp)
    src, _ = fs.load(D / "A_source_merged.csv")
    raw_third, _ = fs.load(D / "A_full_sam3.csv", "third-leg")
    main_p, src_p = fs.apply_correction(main), fs.apply_correction(src)
    chain, n_splice = fs.splice(main_p, src_p)
    # lean 的定義就是這裡**沒有** apply_selector_v2、**沒有** apply_qhead_v056
    chain, n_third, _ = fs.apply_third_leg_rescue(chain, main, src, fs.apply_correction(raw_third))
    fs.restore_first_frames(chain, main)
    _, order = fs.load(fs.ROOT / "1_data/raw/sample_submisson.csv", "sample", validate_boxes=False)
    out = Path(tempfile.gettempdir()) / "_test_lean_out.csv"
    fs.write(chain, order, out)

    assert (n_splice, n_third) == (513, 37)
    assert out.read_bytes() == ref.read_bytes(), "lean 參照檔位元級不符"
    # 與 v090 的差異就是被關掉的那兩層；若變成 0 代表 lean 根本沒生效
    v090 = fs.ROOT / "5_outputs/submissions/sub_v090_cropwin_medoid.csv"
    if v090.is_file():
        a = list(csv.DictReader(open(out)))
        b = list(csv.DictReader(open(v090)))
        diff = sum(1 for x, y in zip(a, b)
                   if (x["x"], x["y"], x["width"], x["height"]) != (y["x"], y["y"], y["width"], y["height"]))
        assert diff == 5548, f"lean vs v090 應差 5548 列，實得 {diff}"
