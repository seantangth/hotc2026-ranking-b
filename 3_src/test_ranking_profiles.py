#!/usr/bin/env python3
"""ranking profiles / dry-run runner 的純 CPU 回歸測試（只用標準庫）。"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_ranking_b as runner  # noqa: E402


def _plan_kwargs(root: Path, allow: bool = False) -> dict:
    return {
        "frames_root": root / "frames",
        "sample": root / "sample.csv",
        "work_dir": root / "work",
        "out": root / "final.csv",
        "sam3_python": Path(sys.executable),
        "samurai_python": Path(sys.executable),
        "sam3_ckpt": root / "sam3.pt",
        "samurai_dir": root / "samurai",
        "samurai_ckpt": root / "sam21.pt",
        "qhead_weights": runner.SRC_ROOT / "hsot/qhead_weights_v056.npz",
        "selector_weights": runner.SRC_ROOT / "hsot/selector_weights_v2.npz",
        "allow_offline_two_pass": allow,
        "sam3_source_revision": "sam3-test-rev",
        "samurai_source_revision": "samurai-test-rev",
    }


def _command_text(steps) -> str:
    return "\n".join(runner.render_command(s) for s in steps if s.kind == "command")


def test_profiles_lock_required_semantics():
    doc = runner.load_profiles()
    # 08-29：default 改為 rankB_deliver_v078（v078 鏈＝LB 0.71666）。rankB_robust
    # 仍是合規 one-pass 對照檔，語意鎖定照舊逐項驗。
    # 09-01：v090 完整 drill 四閘門全過（selftest／pytest／dry-run BLOCK=0／端到端 rc=0）
    # 且端到端產出對 sub_v090 逐列 100% 相同 ⇒ 依事前授權定案切換（D104）。
    assert doc["default_profile"] == "rankB_deliver_v090"
    robust = doc["profiles"]["rankB_robust"]
    # both, not top-only: the formal 405 grouped OOF showed both wins on *both*
    # pairs (crop +0.00251, full-frame +0.00149).  top-only's only support was
    # the 08-24 fresh q95 val smoke, which that file itself marks non-formal.
    assert robust["postprocess"] == {
        "coordinate_correction": "both",
        "frozen_splice_k": 6,
        "quality_head": "none",
        "first_frame_policy": "preserve_init",
        # 08-29：one-pass 合規對照檔刻意不帶 selector／第三腿——它的用途是
        # 「若主辦方採嚴格單遍解讀時的可交付版本」，層數越少越好審。
        "selector": "none",
        "third_leg_rescue": False,
    }
    assert robust["tracking"]["samurai_reset_between_sequences"] is True
    assert robust["tracking"]["sam3_eval"] is True
    assert robust["validation"]["sample_submission_required"] is True
    assert robust["offline_two_pass_crop"]["profile_intent"] == "disabled"

    best = doc["profiles"]["rankA_best"]
    assert best["postprocess"]["coordinate_correction"] == "both"
    assert best["postprocess"]["quality_head"] == "v056"
    assert best["postprocess"]["first_frame_policy"] == "preserve_init"
    assert best["offline_two_pass_crop"]["profile_intent"] == "enabled"

    # 9/7 交付主檔＝v078 鏈（LB 0.71666）。每一層都有 LB 實測支持：
    # selector-v2 +0.00234（D085）／qhead 疊加 +0.00087／第三腿 +0.00029（D091）。
    # finalize_submission.py --selftest 的 (c) 段位元級守護整條鏈。
    deliver = doc["profiles"]["rankB_deliver_v078"]
    assert deliver["postprocess"] == {
        "coordinate_correction": "both",
        "frozen_splice_k": 6,
        "quality_head": "v056",
        "first_frame_policy": "preserve_init",
        "selector": "v2",
        "third_leg_rescue": True,
    }
    assert deliver["tracking"]["samurai_reset_between_sequences"] is True
    assert deliver["tracking"]["sam3_eval"] is True
    assert deliver["offline_two_pass_crop"]["profile_intent"] == "enabled"
    assert deliver["validation"]["sample_submission_required"] is True


def test_executable_path_preserves_virtualenv_symlink():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        venv_python = root / "venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.symlink_to(Path(sys.executable))
        got = runner._executable_arg(venv_python)
        assert got == venv_python.absolute()
        assert got != venv_python.resolve(), "venv symlink 不可被解參考成 shared base Python"
        assert runner._resolve_executable(got) == venv_python.absolute()


def test_rankb_plan_is_causal_reset_and_sample_required():
    doc = runner.load_profiles()
    profile = doc["profiles"]["rankB_robust"]
    with tempfile.TemporaryDirectory() as td:
        steps, paths = runner.build_plan("rankB_robust", profile, **_plan_kwargs(Path(td)))
    text = _command_text(steps)
    assert paths["use_crop"] is False
    assert "hsot.crop_rerun" not in text
    assert "--samurai-reset-kf" in text
    assert "--samurai-legacy-cross-seq-kf" not in text
    assert "--sam3-eval" in text
    assert "--source-revision sam3-test-rev" in text
    assert "--source-revision samurai-test-rev" in text
    assert text.count("--sample-csv") == 2, "兩套 full tracker 都必須拿 sample 做 exact-set 驗證"
    # 08-28: both, per the formal 405 grouped OOF (see the profile description)
    assert "--corr both" in text
    assert "--corr top-only" not in text
    assert "--qhead none" in text
    assert steps[-1].name == "preserve-first-frame-and-validate"


def test_two_pass_needs_explicit_allow_flag():
    doc = runner.load_profiles()
    profile = doc["profiles"]["rankA_best"]
    assert runner.effective_two_pass(profile, False) is False
    assert runner.effective_two_pass(profile, True) is True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        blocked_steps, blocked_paths = runner.build_plan(
            "rankA_best", profile, **_plan_kwargs(root, allow=False))
        allowed_steps, allowed_paths = runner.build_plan(
            "rankA_best", profile, **_plan_kwargs(root, allow=True))
    assert blocked_paths["use_crop"] is False
    assert "hsot.crop_rerun" not in _command_text(blocked_steps)
    allowed_text = _command_text(allowed_steps)
    assert allowed_paths["use_crop"] is True
    assert "hsot.crop_rerun prep" in allowed_text
    assert allowed_text.count("hsot.crop_rerun merge") == 2
    assert "--corr both" in allowed_text
    assert "--qhead v056" in allowed_text
    assert "--samurai-legacy-cross-seq-kf" in allowed_text
    assert "--samurai-reset-kf" not in allowed_text


def _write_csv(path: Path, rows: list[tuple]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(runner.SUBMISSION_COLUMNS)
        writer.writerows(rows)


def test_official_zero_box_sample_is_not_validated_as_prediction():
    with tempfile.TemporaryDirectory() as td:
        sample = Path(td) / "sample.csv"
        _write_csv(sample, [("vis-a_1", 0, 0, 0, 0), ("vis-a_2", 0, 0, 0, 0)])
        rows, sample_errors = runner._read_unique_submission(sample, validate_boxes=False)
        _, prediction_errors = runner._read_unique_submission(sample, validate_boxes=True)
    assert len(rows) == 2 and sample_errors == []
    assert any("非正" in error for error in prediction_errors)


def test_runner_restores_first_frames_and_sample_order():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        frames = root / "frames"
        for seq, init_text in (("vis-a", "1, 2, 3, 4\n"), ("nir-b", "10 20 30 40\n")):
            (frames / seq).mkdir(parents=True)
            (frames / seq / "init_rect.txt").write_text(init_text)
        sample = root / "sample.csv"
        # sample 順序刻意不是 pred 順序；首幀也不是永遠 suffix=1。
        # 官方 sample 的四個座標欄皆可為 0 placeholder；不可誤判成 invalid pred。
        _write_csv(sample, [
            ("vis-a_5", 0, 0, 0, 0),
            ("vis-a_6", 0, 0, 0, 0),
            ("nir-b_10", 0, 0, 0, 0),
            ("nir-b_11", 0, 0, 0, 0),
        ])
        pred = root / "pre.csv"
        _write_csv(pred, [
            ("nir-b_11", 90, 91, 8, 9),
            ("vis-a_6", 50, 51, 6, 7),
            ("nir-b_10", 11, 21, 30, 40),
            ("vis-a_5", 2, 3, 3, 4),
        ])
        out = root / "final.csv"
        stats = runner.preserve_first_frames_and_validate(pred, out, sample, frames)
        with out.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    assert stats == {"rows": 4, "sequences": 2, "restored": 2}
    assert [r["ID"] for r in rows] == ["vis-a_5", "vis-a_6", "nir-b_10", "nir-b_11"]
    by_id = {r["ID"]: r for r in rows}
    assert [float(by_id["vis-a_5"][k]) for k in runner.SUBMISSION_COLUMNS[1:]] == [1, 2, 3, 4]
    assert [float(by_id["nir-b_10"][k]) for k in runner.SUBMISSION_COLUMNS[1:]] == [10, 20, 30, 40]
    assert float(by_id["vis-a_6"]["x"]) == 50
    assert float(by_id["nir-b_11"]["x"]) == 90


def test_runner_rejects_duplicate_or_nonfinite_final():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        frames = root / "frames" / "s"
        frames.mkdir(parents=True)
        (frames / "init_rect.txt").write_text("1 2 3 4")
        sample = root / "sample.csv"
        _write_csv(sample, [("s_1", 0, 0, 0, 0), ("s_2", 0, 0, 0, 0)])
        bad = root / "bad.csv"
        _write_csv(bad, [("s_1", 0, 0, 1, 1), ("s_1", float("nan"), 0, 1, 1)])
        try:
            runner.preserve_first_frames_and_validate(bad, root / "out.csv", sample, root / "frames")
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("duplicate/NaN 必須 fail-closed")
    assert "重複 ID" in message
    assert "非 finite" in message


def test_runner_rejects_negative_prediction_coordinates():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        frames = root / "frames" / "s"
        frames.mkdir(parents=True)
        (frames / "init_rect.txt").write_text("1 2 3 4")
        sample = root / "sample.csv"
        _write_csv(sample, [("s_1", 0, 0, 0, 0), ("s_2", 0, 0, 0, 0)])
        bad = root / "bad.csv"
        _write_csv(bad, [("s_1", 1, 2, 3, 4), ("s_2", -1, 2, 3, 4)])
        try:
            runner.preserve_first_frames_and_validate(bad, root / "out.csv", sample, root / "frames")
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("負 x/y 必須 fail-closed")
    assert "負座標" in message


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"❌ {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests)-failed}/{len(tests)} 通過")
    raise SystemExit(1 if failed else 0)


def test_rankb_robust_crop_is_robust_plus_crop_only():
    """9/7 主檔候選：只在 crop 與 corr 上與 rankB_robust 不同，其餘 robust 修復全保留。"""
    doc = runner.load_profiles()
    crop = doc["profiles"]["rankB_robust_crop"]
    robust = doc["profiles"]["rankB_robust"]

    # D078 的四個 P0 修復一個都不能因為開 crop 而掉回去
    assert crop["tracking"] == robust["tracking"]
    assert crop["validation"] == robust["validation"]
    assert crop["postprocess"]["quality_head"] == "none"
    assert crop["postprocess"]["first_frame_policy"] == "preserve_init"
    assert crop["postprocess"]["frozen_splice_k"] == 6

    # 與 rankB_robust 的差異剛好只有這兩項
    assert crop["offline_two_pass_crop"]["profile_intent"] == "enabled"
    assert robust["offline_two_pass_crop"]["profile_intent"] == "disabled"
    assert crop["postprocess"]["coordinate_correction"] == "both"
    # both profiles now agree on corr; the crop/no-crop intent is the only split
    assert (crop["postprocess"]["coordinate_correction"]
            == robust["postprocess"]["coordinate_correction"])

    # 絕不可退化成 rankA_best：那兩項正是 D078 判定的洩漏與負增益來源
    best = doc["profiles"]["rankA_best"]
    assert crop["postprocess"]["quality_head"] != best["postprocess"]["quality_head"]
    assert (crop["tracking"]["samurai_reset_between_sequences"]
            is not best["tracking"]["samurai_reset_between_sequences"])


def test_rankb_robust_crop_still_gates_two_pass_behind_explicit_flag():
    doc = runner.load_profiles()
    profile = doc["profiles"]["rankB_robust_crop"]
    assert runner.effective_two_pass(profile, False) is False
    assert runner.effective_two_pass(profile, True) is True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        blocked_steps, blocked_paths = runner.build_plan(
            "rankB_robust_crop", profile, **_plan_kwargs(root, allow=False))
        allowed_steps, allowed_paths = runner.build_plan(
            "rankB_robust_crop", profile, **_plan_kwargs(root, allow=True))
    assert blocked_paths["use_crop"] is False
    assert "hsot.crop_rerun" not in _command_text(blocked_steps)

    allowed_text = _command_text(allowed_steps)
    assert allowed_paths["use_crop"] is True
    assert "hsot.crop_rerun prep" in allowed_text
    assert allowed_text.count("hsot.crop_rerun merge") == 2
    # crop 腿也必須帶著每序列 reset，且全程不得掛上 v056 qhead
    assert "--samurai-reset-kf" in allowed_text
    assert "--samurai-legacy-cross-seq-kf" not in allowed_text
    assert "qhead_weights_v056" not in allowed_text


# ── crop 窗三方共識（D101 / rankB_deliver_v090）─────────────────────────────


def test_rankb_deliver_v090_is_v078_plus_window_consensus():
    """v090 ＝ v078 ＋ 窗共識。後處理鏈一個字都不能動——D100 已把落差歸因到 crop 階段，
    改後處理等於同時動兩個變因。"""
    doc = runner.load_profiles()
    v078 = doc["profiles"]["rankB_deliver_v078"]
    v090 = doc["profiles"]["rankB_deliver_v090"]
    assert v090["postprocess"] == v078["postprocess"]
    assert v090["tracking"] == v078["tracking"]
    assert v090["validation"] == v078["validation"]
    crop078 = dict(v078["offline_two_pass_crop"])
    crop090 = dict(v090["offline_two_pass_crop"])
    wc = crop090.pop("window_consensus")
    assert "window_consensus" not in crop078, "v078 必須維持單一窗（drill 腳本與 PROVENANCE 都引用它）"
    assert crop090 == crop078, "除了 window_consensus 之外，crop 設定必須與 v078 相同"

    tags = [v["tag"] for v in wc["variants"]]
    assert tags == ["A", "B", "C"], "順序有語意：medoid 平手取第一個輸入"
    assert wc["variants"][0] == {**wc["variants"][0], "envelope_extra": True, "segments": 1}, \
        "A 必須是現行窗（envelope_extra＋segments=1），否則 medoid 的現任換人"
    assert wc["variants"][1]["segments"] == 2
    assert wc["variants"][2]["envelope_extra"] is False
    assert wc["source_leg_variant"] == "A"
    assert wc["medoid_tie_breaker"] == "first_variant"
    # 09-01 起 default ＝ v090（D104）。v078 仍須留著——它是 selftest (c) 守護的基準鏈，
    # 也是「把窗共識關掉」時的單變因對照，不得刪除。
    assert doc["default_profile"] == "rankB_deliver_v090"
    assert "rankB_deliver_v078" in doc["profiles"]


def test_window_consensus_plan_runs_three_windows_then_medoid():
    doc = runner.load_profiles()
    profile = doc["profiles"]["rankB_deliver_v090"]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        steps, paths = runner.build_plan(
            "rankB_deliver_v090", profile, **_plan_kwargs(root, allow=True))
    names = [s.name for s in steps]
    text = _command_text(steps)

    # 三組窗各一次 prep／track／merge；SAMURAI 腿只跑一次（窗 A）
    assert text.count("hsot.crop_rerun prep") == 3
    assert text.count("hsot.crop_rerun merge") == 4  # 三組主腿 ＋ 一組 source 腿
    assert names.count("track-crop-source-samurai") == 1

    prep = {s.name: s.argv for s in steps if s.name.startswith("offline-two-pass-prep-")}
    assert "--segments" not in prep["offline-two-pass-prep-A"], \
        "A ＝ crop_rerun 預設，不發旗標才能與單一窗指令逐字相同"
    assert "--envelope-extra" in prep["offline-two-pass-prep-A"]
    assert prep["offline-two-pass-prep-B"][-2:] == ["--segments", "2"]
    assert "--envelope-extra" not in prep["offline-two-pass-prep-C"]
    assert "--segments" not in prep["offline-two-pass-prep-C"]
    # 三組窗都只吃本次 run 自己的 full 輸出（9/7 可複製的硬約束）
    for argv in prep.values():
        assert str(paths["full_primary"]) in argv
        assert not any("5_outputs/submissions" in a for a in argv)

    medoid = next(s for s in steps if s.name == "medoid-crop-primary")
    ins = [a for a in medoid.argv if a.endswith(".csv")]
    assert [Path(a).name for a in ins[:3]] == [
        "main_merged_A.csv", "main_merged_B.csv", "main_merged_C.csv"], "順序即現任"
    assert str(paths["merged_primary"]) in medoid.argv

    # SAMURAI 腿吃窗 A 的 meta/frames；finalize 吃 medoid 輸出
    src_merge = next(s for s in steps if s.name == "merge-crop-source")
    assert paths["crop_variants"]["A"]["crop_meta"] in src_merge.argv
    finalize = next(s for s in steps if s.name == "finalize")
    assert finalize.argv[finalize.argv.index("--main") + 1] == str(paths["merged_primary"])
    assert finalize.argv[finalize.argv.index("--source") + 1] == str(paths["merged_source"])
    # 第三腿仍是 full-frame SAM3（crop 前那一腿），不是 medoid
    assert finalize.argv[finalize.argv.index("--third-leg") + 1] == str(paths["full_primary"])

    # 每組窗各有自己的選中條件；沒選中就走 passthrough，medoid 仍拿得到三份輸入
    for tag in ("A", "B", "C"):
        assert f"crop_selected:{tag}" in {s.condition for s in steps}
        assert f"no_crop_selected:{tag}" in {s.condition for s in steps}


def test_v078_plan_is_unchanged_by_the_consensus_feature():
    """回歸護欄：窗共識是加法，v078 鏈（selftest (c) 位元級守護的那條）一步都不能變。"""
    doc = runner.load_profiles()
    profile = doc["profiles"]["rankB_deliver_v078"]
    with tempfile.TemporaryDirectory() as td:
        steps, paths = runner.build_plan(
            "rankB_deliver_v078", profile, **_plan_kwargs(Path(td), allow=True))
    assert [s.name for s in steps] == [
        "track-full-source-samurai", "track-full-primary-sam3", "reject-base-fallbacks",
        "offline-two-pass-prep", "materialize-crop-sequence-list",
        "track-crop-primary-sam3", "merge-crop-primary",
        "track-crop-source-samurai", "merge-crop-source",
        "reject-crop-fallbacks", "no-crop-passthrough",
        "finalize", "preserve-first-frame-and-validate",
    ]
    assert paths["window_consensus"] is False
    assert paths["crop_variants"] == {}
    assert "run_ensemble_medoid" not in _command_text(steps)


def test_window_consensus_config_is_validated():
    import copy
    doc = runner.load_profiles()
    base = doc["profiles"]["rankB_deliver_v090"]

    def reject(mutate, needle):
        p = copy.deepcopy(base)
        mutate(p["offline_two_pass_crop"]["window_consensus"], p)
        try:
            runner._validate_profile("probe", p)
        except ValueError as exc:
            assert needle in str(exc), f"錯誤訊息未提到 {needle}：{exc}"
            return
        raise AssertionError(f"應被拒絕但通過了：{needle}")

    # N=2 時 medoid 恆等於現任 ⇒ 兩組窗測不到任何東西（v084 就是栽在這）
    reject(lambda wc, p: wc.__setitem__("variants", wc["variants"][:2]), "至少要三組")
    reject(lambda wc, p: wc["variants"].append(dict(wc["variants"][0])), "重複 tag")
    reject(lambda wc, p: wc.__setitem__("source_leg_variant", "Z"), "source_leg_variant")
    reject(lambda wc, p: wc.__setitem__("medoid_tie_breaker", "random"), "medoid_tie_breaker")
    reject(lambda wc, p: wc["variants"][1].__setitem__("segments", 0), "segments")
    reject(lambda wc, p: wc["variants"][1].__setitem__("envelope_extra", "yes"), "envelope_extra")
    # 窗共識只作用在 crop 階段：intent=disabled 時開它是設定錯誤，不是靜默 no-op
    reject(lambda wc, p: p["offline_two_pass_crop"].__setitem__("profile_intent", "disabled"),
           "profile_intent=enabled")

    # enabled=false 仍是合法設定（等同單一窗）
    off = copy.deepcopy(base)
    off["offline_two_pass_crop"]["window_consensus"] = {"enabled": False}
    runner._validate_profile("probe", off)
    assert runner.window_consensus(off) is None


def test_per_window_passthrough_routes_by_tag():
    """某組窗選不到序列時，只有那一組走 passthrough，其餘照跑——medoid 仍有三份輸入。"""
    import json as _json
    doc = runner.load_profiles()
    profile = doc["profiles"]["rankB_deliver_v090"]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        steps, paths = runner.build_plan(
            "rankB_deliver_v090", profile, **_plan_kwargs(root, allow=True))
        Path(paths["full_primary"]).parent.mkdir(parents=True, exist_ok=True)
        Path(paths["full_primary"]).write_text("ID,x,y,width,height\ns_1,1,2,3,4\n")
        # A 有序列、B 空、C 有序列
        for tag, meta in (("A", {"s": {}}), ("B", {}), ("C", {"s": {}})):
            mp = Path(paths["crop_variants"][tag]["crop_meta"])
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text(_json.dumps(meta))
        internal = [s for s in steps
                    if s.kind == "internal"
                    and (s.name.startswith("materialize-crop-sequence-list-")
                         or s.name.startswith("no-crop-passthrough-"))]
        runner.execute_plan(internal, paths, profile=profile,
                            sample=root / "sample.csv", frames_root=root / "frames")
        # 只有 B 被複製；A/C 留給真正的 merge 步驟寫
        assert not Path(paths["crop_variants"]["A"]["merged_primary"]).exists()
        assert Path(paths["crop_variants"]["C"]["merged_primary"]).exists() is False
        assert Path(paths["crop_variants"]["B"]["merged_primary"]).read_text() == \
            Path(paths["full_primary"]).read_text()
        assert Path(paths["crop_variants"]["A"]["crop_seqs"]).read_text() == "s\n"
        assert Path(paths["crop_variants"]["B"]["crop_seqs"]).read_text() == ""


# ─── Q02：sample 檔內列序是隱性契約（09-04 稽核）────────────────────────────
def _write_sample(tmp_path, ids):
    p = tmp_path / "sample.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        fh.write("ID,x,y,width,height\n")
        for i in ids:
            fh.write(f"{i},0,0,0,0\n")
    return p


def test_sample_contract_accepts_numeric_ascending(tmp_path):
    import run_ranking_b as rb
    p = _write_sample(tmp_path, [f"nir-ball_{i}" for i in range(1, 13)])
    groups, errors = rb._sample_contract(p)
    assert errors == []
    assert len(groups["nir-ball"]) == 12


def test_sample_contract_blocks_lexicographic_order(tmp_path):
    """Excel／naive sort() 產生的字典序：seq_1, seq_10, seq_11, …, seq_2 …

    ID 集合、唯一性、筆數、<seq>_<int> 可拆全部通過，首幀在字典序下仍排第一
    ⇒ 舊碼零警告放行，但每一列的框都會綁到錯的幀，且 Ranking B 無 GT 無法察覺。
    """
    import run_ranking_b as rb
    ids = sorted([f"nir-ball_{i}" for i in range(1, 13)])   # 字典序
    assert ids[1] == "nir-ball_10"                          # 確認確實不是數字序
    p = _write_sample(tmp_path, ids)
    groups, errors = rb._sample_contract(p)
    assert errors, "字典序必須 BLOCK"
    assert "未嚴格遞增" in errors[0] and "nir-ball" in errors[0]


def test_sample_contract_blocks_duplicate_frame_number_in_order(tmp_path):
    import run_ranking_b as rb
    p = _write_sample(tmp_path, ["vis-toy1_1", "vis-toy1_3", "vis-toy1_3b".replace("b", ""), "vis-toy1_5"])
    _, errors = rb._sample_contract(p)
    assert errors and ("重複" in errors[0] or "未嚴格遞增" in errors[0])


def test_sample_contract_allows_non_contiguous_but_ascending(tmp_path):
    """非 1..N 連續但遞增（短幀／多區塊序列）是合法的，不可誤擋。"""
    import run_ranking_b as rb
    p = _write_sample(tmp_path, [f"rednir-duck4_{i}" for i in (5, 9, 11, 40, 41)])
    _, errors = rb._sample_contract(p)
    assert errors == []


def test_track_t1_frame_ids_for_rejects_non_ascending():
    import track_t1
    import pytest
    with pytest.raises(ValueError, match="未嚴格遞增"):
        track_t1.frame_ids_for("nir-ball", 4, None, {"nir-ball": [1, 10, 2, 3]})
    # 遞增則放行
    assert track_t1.frame_ids_for("nir-ball", 4, None, {"nir-ball": [1, 2, 3, 10]}) == [1, 2, 3, 10]


# ─── G15：init_rect 座標慣例與幀檔名（09-05 稽核）──────────────────────────
def _minimal_jpeg(width, height):
    """能被 _image_size 的 header parser 讀出寬高的最小 JPEG（SOI + SOF0 + EOI）。"""
    sof = (b"\xff\xc0" + (17).to_bytes(2, "big") + b"\x08"
           + height.to_bytes(2, "big") + width.to_bytes(2, "big")
           + b"\x03" + b"\x01\x11\x00\x02\x11\x01\x03\x11\x01")
    return b"\xff\xd8" + sof + b"\xff\xd9"


def _frames_root(tmp_path, seq, n, init_text, w=640, h=480, stem=lambda i: f"{i:04d}"):
    d = tmp_path / "frames" / seq
    d.mkdir(parents=True)
    for i in range(1, n + 1):
        (d / f"{stem(i)}.jpg").write_bytes(_minimal_jpeg(w, h))
    (d / "init_rect.txt").write_text(init_text)
    return tmp_path / "frames"


def test_image_size_reads_jpeg_header():
    import run_ranking_b as rb
    import tempfile
    from pathlib import Path
    p = Path(tempfile.mkdtemp()) / "a.jpg"
    p.write_bytes(_minimal_jpeg(1280, 720))
    assert rb._image_size(p) == (1280, 720)


def _frames_finding(tmp_path, seq, n, init_text, **kw):
    """呼叫真正的 preflight，回傳 frames-root 那一條 finding（端到端，不重寫判準）。"""
    import run_ranking_b as rb
    fr = _frames_root(tmp_path, seq, n, init_text, **kw)
    sample = _write_sample(tmp_path, [f"{seq}_{i}" for i in range(1, n + 1)])
    prof = rb.load_profiles()["profiles"]["rankB_robust"]
    findings = rb.preflight(
        "rankB_robust", prof,
        frames_root=fr, sample=sample,
        work_dir=tmp_path / "w", out=tmp_path / "o.csv",
        sam3_python=Path("/nonexistent"), samurai_python=Path("/nonexistent"),
        sam3_ckpt=Path("/nonexistent"), samurai_dir=Path("/nonexistent"),
        samurai_ckpt=Path("/nonexistent"), qhead_weights=Path("/nonexistent"),
        selector_weights=Path("/nonexistent"),
        allow_offline_two_pass=False, execute=False, resume_existing_work=False,
    )
    return next(f for f in findings if f.check == "frames-root")


def test_preflight_accepts_valid_init_rect(tmp_path):
    f = _frames_finding(tmp_path, "nir-ball", 4, "100,50,40,30\n")
    assert f.level == "OK", f.detail


def test_preflight_blocks_x1y1x2y2_style_init_rect(tmp_path):
    """(x1,y1,x2,y2) 被當成 (x,y,w,h)：所有既有檢查都通過，但整輪 4.5 小時全錯。

    真值框 (600,400)-(630,440) 若寫成 x1y1x2y2 而被讀成 w=630/h=440，
    x+w = 1230 > 640 ⇒ 唯一抓得到它的判準就是「框必須落在影像內」。
    """
    f = _frames_finding(tmp_path, "vis-toy1", 3, "600,400,630,440\n", w=640, h=480)
    assert f.level == "BLOCK" and "超出影像" in f.detail, f.detail


def test_preflight_blocks_non_numeric_frame_stem(tmp_path):
    """檔名 stem 非純數字 ⇒ 排序順序與數字序不同，同 Q02 的錯位問題。"""
    f = _frames_finding(tmp_path, "vis-toy2", 3, "10,10,20,20\n",
                        stem=lambda i: f"frame_{i:03d}a")
    assert f.level == "BLOCK" and "stem 非純數字" in f.detail, f.detail


# ── 09-07：官方佈局無模態前綴 ⇒ 後處理靜默全當 VIS，preflight 必須擋 ──────────────
def _prefix_finding(tmp_path, seqs):
    import run_ranking_b as rb
    fr = tmp_path / "frames"
    ids = []
    for s in seqs:
        _frames_root(tmp_path, s, 2, "10,10,20,15\n")
        ids += [f"{s}_{i}" for i in range(1, 3)]
    sample = _write_sample(tmp_path, ids)
    prof = rb.load_profiles()["profiles"]["rankB_robust"]
    findings = rb.preflight(
        "rankB_robust", prof, frames_root=fr, sample=sample,
        work_dir=tmp_path / "w", out=tmp_path / "o.csv",
        sam3_python=Path("/nonexistent"), samurai_python=Path("/nonexistent"),
        sam3_ckpt=Path("/nonexistent"), samurai_dir=Path("/nonexistent"),
        samurai_ckpt=Path("/nonexistent"), qhead_weights=Path("/nonexistent"),
        selector_weights=Path("/nonexistent"),
        allow_offline_two_pass=False, execute=False, resume_existing_work=False,
    )
    return next(f for f in findings if f.check == "modality-prefix")


def test_preflight_blocks_when_no_sequence_has_modality_prefix(tmp_path):
    f = _prefix_finding(tmp_path, ["blackball2", "cardpigs1"])
    assert f.level == "BLOCK" and "prep_rankingb_frames_v1" in f.detail, f.detail


def test_preflight_warns_on_partial_prefix_and_ok_on_full(tmp_path):
    assert _prefix_finding(tmp_path / "partial", ["nir-a", "cardpigs1"]).level == "WARN"
    assert _prefix_finding(tmp_path / "full", ["nir-a", "rednir-b", "vis-c"]).level == "OK"
