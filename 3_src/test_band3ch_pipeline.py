#!/usr/bin/env python3
"""E-A' band×SAM3 管線 CPU 單元測試（make_band3ch_frames / merge_band_submission）。

不需 torch、不讀真資料——合成 mosaic 已知 band 訊號，驗 X2Cube 語意、
band 選擇與觸發規則的確定性、stem 對齊 fail-closed、merge 驗證語意。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from hsot.io import load_cube, x2cube  # noqa: E402
from prep import make_band3ch_frames as mb  # noqa: E402
from prep import merge_band_submission as ms  # noqa: E402


# ─── X2Cube 語意（修正版：cube[y,x,k] = mosaic[y*m + k//m, x*m + k%m]）───

def test_x2cube_band_layout_4x4():
    m = 4
    H, W = 8, 12
    mosaic = np.zeros((H, W), dtype=np.uint16)
    for yy in range(H):
        for xx in range(W):
            mosaic[yy, xx] = (yy % m) * m + (xx % m)  # 每 macro-pixel 內寫 band index
    cube = x2cube(mosaic, m)
    assert cube.shape == (2, 3, 16)
    for k in range(16):
        assert (cube[:, :, k] == k).all(), f"band {k} 佈局錯位"


def test_load_cube_rednir_drops_last_band():
    mosaic = np.ones((8, 8), dtype=np.uint16)
    assert load_cube(mosaic, "rednir").shape[2] == 15  # 4×4=16 丟末全零 band
    assert load_cube(mosaic, "vis").shape[2] == 16
    mosaic5 = np.ones((10, 10), dtype=np.uint16)
    assert load_cube(mosaic5, "nir").shape[2] == 25


# ─── 合成場景：band 7 有訊號、其餘噪聲 ───────────────────────────────

def _synth_scene(tmp_path, seq="nir-synth1", n_frames=3, signal_band=7,
                 fc_signal=False):
    """建 fc_root/<seq>（jpg＋init_rect）與 hsi_root/<seq>（mosaic png）。
    目標方塊在 signal_band 上亮、其他 band 平坦；fc_signal 控制假色有無對比。"""
    from PIL import Image

    m, hc, wc = 5, 40, 60  # nir：5×5 → cube 40×60×25
    box = (20.0, 10.0, 16.0, 12.0)
    rng = np.random.default_rng(0)
    fc_dir = tmp_path / "fc" / seq
    hsi_dir = tmp_path / "hsi" / seq
    fc_dir.mkdir(parents=True)
    hsi_dir.mkdir(parents=True)
    (fc_dir / "init_rect.txt").write_text("20 10 16 12")
    x, y, w, h = (int(v) for v in box)
    for i in range(n_frames):
        cube = rng.integers(90, 110, size=(hc, wc, 25)).astype(np.uint16)
        cube[y:y + h, x:x + w, signal_band] += 400  # 目標只在 signal band 亮
        mosaic = np.zeros((hc * m, wc * m), dtype=np.uint16)
        for k in range(25):
            mosaic[k // m::m, k % m::m] = cube[:, :, k]
        Image.fromarray(mosaic).save(hsi_dir / f"{i + 1:04d}.png")
        fc = np.full((hc, wc, 3), 128, dtype=np.uint8)
        if fc_signal:
            fc[y:y + h, x:x + w] = 250
        Image.fromarray(fc).save(fc_dir / f"{i + 1:04d}.jpg", quality=95)
    return tmp_path / "fc", tmp_path / "hsi", box


def test_band_selection_finds_signal_band_and_is_deterministic(tmp_path):
    fc_root, hsi_root, box = _synth_scene(tmp_path)
    top3_a, img_a, _ = mb.band3_first_frame(hsi_root / "nir-synth1", "nir-synth1",
                                            list(box))
    top3_b, img_b, _ = mb.band3_first_frame(hsi_root / "nir-synth1", "nir-synth1",
                                            list(box))
    assert top3_a[0] == 7  # 訊號 band 必居首
    assert top3_a == top3_b and (img_a == img_b).all()  # 確定性


def test_decide_triggers_when_band_beats_fc(tmp_path):
    fc_root, hsi_root, _ = _synth_scene(tmp_path, fc_signal=False)
    r = mb.decide_one("nir-synth1", fc_root, hsi_root)
    assert r["triggered"] and r["chosen"] in ("band3", "proj3")
    assert max(r["auc_band3"], r["auc_proj3"]) > r["auc_fc"] + mb.MARGIN
    assert r["top3_bands"][0] == 7


def test_decide_no_trigger_when_fc_equally_good(tmp_path):
    fc_root, hsi_root, _ = _synth_scene(tmp_path, seq="nir-synth2", fc_signal=True)
    r = mb.decide_one("nir-synth2", fc_root, hsi_root)
    # 假色同樣可分（AUC≈1）⇒ delta < MARGIN ⇒ 不觸發（規則全域、資料決定）
    assert not r["triggered"] and r["chosen"] == "fc"


def _synth_combo_scene(tmp_path, seq="nir-combo1"):
    """組合訊號場景：目標＝band 7 增/band 11 減的**差分**訊號，單 band Fisher
    低（各 band 邊際分佈與背景大量重疊）、線性組合（b7−b11）幾乎完美可分
    ——重現 5 支真序列補測的「判別力在組合不在個別 band」型態。"""
    from PIL import Image

    m, hc, wc = 5, 40, 60
    box = (20.0, 10.0, 16.0, 12.0)
    rng = np.random.default_rng(1)
    fc_dir = tmp_path / "fc" / seq
    hsi_dir = tmp_path / "hsi" / seq
    fc_dir.mkdir(parents=True)
    hsi_dir.mkdir(parents=True)
    (fc_dir / "init_rect.txt").write_text("20 10 16 12")
    x, y, w, h = (int(v) for v in box)
    for i in range(2):
        base = rng.integers(200, 800, size=(hc, wc)).astype(np.int64)  # 共模大噪聲
        cube = np.stack([base + rng.integers(-20, 20, size=(hc, wc))
                         for _ in range(25)], axis=-1)
        cube[y:y + h, x:x + w, 7] += 150   # 差分訊號：b7↑ b11↓
        cube[y:y + h, x:x + w, 11] -= 150
        cube = np.clip(cube, 0, 65535).astype(np.uint16)
        mosaic = np.zeros((hc * m, wc * m), dtype=np.uint16)
        for k in range(25):
            mosaic[k // m::m, k % m::m] = cube[:, :, k]
        Image.fromarray(mosaic).save(hsi_dir / f"{i + 1:04d}.png")
        Image.fromarray(np.full((hc, wc, 3), 128, dtype=np.uint8)).save(
            fc_dir / f"{i + 1:04d}.jpg", quality=95)
    return tmp_path / "fc", tmp_path / "hsi"


def test_proj_candidate_wins_on_combination_signal(tmp_path):
    fc_root, hsi_root = _synth_combo_scene(tmp_path)
    r = mb.decide_one("nir-combo1", fc_root, hsi_root)
    # 共模噪聲淹沒單 band ⇒ band3 吃不到；Fisher 投影（差分組合）吃得到
    assert r["auc_proj3"] > r["auc_band3"], r
    assert r["triggered"] and r["chosen"] == "proj3", r


def test_proj_matrix_deterministic_and_causal(tmp_path):
    fc_root, hsi_root = _synth_combo_scene(tmp_path, seq="nir-combo2")
    from hsot.io import load_cube, _read_u16

    cube = load_cube(_read_u16(sorted((hsi_root / "nir-combo2").glob("*.png"))[0]),
                     "nir")
    box = [20.0, 10.0, 16.0, 12.0]
    Wa = mb.fisher_proj_matrix(cube, box)
    Wb = mb.fisher_proj_matrix(cube, box)
    assert (Wa == Wb).all() and Wa.shape == (25, 3)
    img = mb.proj3_image(cube, Wa)
    assert img.dtype == np.uint8 and img.shape == (40, 60, 3)


def test_convert_proj_mode_writes_aligned_frames(tmp_path):
    fc_root, hsi_root = _synth_combo_scene(tmp_path, seq="nir-combo3")
    r = mb.convert_one("nir-combo3", fc_root, hsi_root, tmp_path / "band",
                       chosen="proj3")
    assert r["chosen"] == "proj3" and r["n_frames"] == 2
    stems = [p.stem for p in sorted((tmp_path / "band" / "nir-combo3").glob("*.jpg"))]
    assert stems == ["0001", "0002"]


def test_convert_writes_aligned_stems_and_init(tmp_path):
    fc_root, hsi_root, _ = _synth_scene(tmp_path)
    out_root = tmp_path / "band"
    r = mb.convert_one("nir-synth1", fc_root, hsi_root, out_root)
    assert r["n_frames"] == 3
    stems = [p.stem for p in sorted((out_root / "nir-synth1").glob("*.jpg"))]
    assert stems == ["0001", "0002", "0003"]
    assert (out_root / "nir-synth1" / "init_rect.txt").read_text() == "20 10 16 12"


def test_convert_fails_closed_on_stem_mismatch(tmp_path):
    fc_root, hsi_root, _ = _synth_scene(tmp_path)
    extra = fc_root / "nir-synth1" / "9999.jpg"  # 假色多一幀 ⇒ stem 不齊必炸
    from PIL import Image

    Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(extra)
    with pytest.raises(AssertionError, match="幀名與假色不齊"):
        mb.convert_one("nir-synth1", fc_root, hsi_root, tmp_path / "band2")


def test_decide_is_order_independent(tmp_path):
    """觸發判定不得依批次處理順序——合規主張「每序列自足」的落地驗證
    （per-seq RNG 由 [SEED, crc32(seq), crc32(stream)] 導出）。"""
    fc_root, hsi_root, _ = _synth_scene(tmp_path, seq="nir-synth1")
    _synth_scene(tmp_path, seq="nir-synth3", signal_band=11)
    solo = mb.decide_one("nir-synth3", fc_root, hsi_root)
    after_other = None
    for s in ("nir-synth1", "nir-synth3"):  # 批次順序：先處理另一支
        r = mb.decide_one(s, fc_root, hsi_root)
        if s == "nir-synth3":
            after_other = r
    assert solo == after_other  # 單獨算與批次中算完全相同


def test_decide_cli_end_to_end(tmp_path):
    fc_root, hsi_root, _ = _synth_scene(tmp_path)
    seqs = tmp_path / "seqs.txt"
    seqs.write_text("nir-synth1\n")
    out = tmp_path / "decide.json"
    rc = mb.main(["--fc-root", str(fc_root), "--hsi-root", str(hsi_root),
                  "--seqs", str(seqs), "--decide-out", str(out)])
    assert rc == 0
    plan = json.loads(out.read_text())
    assert plan["triggered"] == ["nir-synth1"] and plan["margin"] == mb.MARGIN


# ─── apply_crop_meta（既有窗裁新 frames——單變因設計核心）──────────────

def _mk_crop_world(tmp_path, seq="nir-a", n=4, W=60, H=40):
    """frames_root/<seq>/1..n.jpg（像素值編碼幀號與座標，可驗裁切正確性）＋
    base 軌跡 CSV ＋ v078 型 meta（窗 [10,5,30,20]、K=1 全段）。"""
    from PIL import Image

    frames = tmp_path / "frames" / seq
    frames.mkdir(parents=True)
    for i in range(1, n + 1):
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        arr[:, :, 0] = i * 10          # R 通道編碼幀號
        arr[5:25, 10:40, 1] = 200      # 窗內 G 亮（驗窗位置）
        Image.fromarray(arr).save(frames / f"{i:04d}.jpg", quality=95)
    base = tmp_path / "base.csv"
    rows = [["ID", "x", "y", "width", "height"]] + [
        [f"{seq}_{i}", "12", "7", "6", "5"] for i in range(1, n + 1)]
    import csv as _csv

    _csv.writer(open(base, "w", newline="")).writerows(rows)
    meta = {seq: {"x1": 10, "y1": 5, "w": 30, "h": 20, "orig": [W, H],
                  "seq": seq, "frames": [1, n], "zoom": 4.0,
                  "crop_image_encoding": {"profile": "legacy-q95"}}}
    mp = tmp_path / "crop_meta.json"
    mp.write_text(json.dumps(meta))
    return tmp_path / "frames", base, mp


def test_apply_crop_meta_window_and_init(tmp_path):
    from prep import apply_crop_meta as ac

    frames, base, meta = _mk_crop_world(tmp_path)
    out = tmp_path / "cropped"
    rc = ac.main(["--meta", str(meta), "--frames-root", str(frames),
                  "--base-csv", str(base), "--out-root", str(out),
                  "--names-out", str(tmp_path / "names.txt")])
    assert rc == 0
    from PIL import Image

    crops = sorted((out / "nir-a").glob("*.jpg"))
    assert [p.stem for p in crops] == ["0001", "0002", "0003", "0004"]
    im = np.array(Image.open(crops[0]))
    assert im.shape == (20, 30, 3)            # 窗大小
    assert im[:, :, 1].mean() > 150           # 裁到的是窗內亮區
    init = (out / "nir-a" / "init_rect.txt").read_text().split()
    assert [float(v) for v in init] == [2.0, 2.0, 6.0, 5.0]  # (12-10, 7-5, w, h)
    assert (tmp_path / "names.txt").read_text().strip() == "nir-a"


def test_apply_crop_meta_only_seqs_filter_and_resolution_guard(tmp_path):
    from prep import apply_crop_meta as ac

    frames, base, meta = _mk_crop_world(tmp_path, seq="nir-b")
    only = tmp_path / "only.txt"
    only.write_text("nir-zzz\n")  # 不含 nir-b ⇒ 全跳過 ⇒ rc 2
    rc = ac.main(["--meta", str(meta), "--frames-root", str(frames),
                  "--base-csv", str(base), "--out-root", str(tmp_path / "o2"),
                  "--only-seqs", str(only)])
    assert rc == 2
    # 解析度與 meta.orig 不符 ⇒ 窗不可沿用 ⇒ AssertionError（fail-closed）
    m = json.loads(meta.read_text())
    m["nir-b"]["orig"] = [999, 999]
    meta.write_text(json.dumps(m))
    with pytest.raises(AssertionError, match="窗不可沿用"):
        ac.main(["--meta", str(meta), "--frames-root", str(frames),
                 "--base-csv", str(base), "--out-root", str(tmp_path / "o3")])


def test_apply_crop_meta_segment_init_from_base_track(tmp_path):
    from prep import apply_crop_meta as ac

    frames, base, meta = _mk_crop_world(tmp_path, seq="nir-c", n=4)
    m = json.loads(meta.read_text())
    m["nir-c__seg01"] = {**m.pop("nir-c"), "frames": [3, 4]}  # 段：幀 3-4
    meta.write_text(json.dumps(m))
    out = tmp_path / "o4"
    rc = ac.main(["--meta", str(meta), "--frames-root", str(frames),
                  "--base-csv", str(base), "--out-root", str(out)])
    assert rc == 0
    crops = sorted((out / "nir-c__seg01").glob("*.jpg"))
    assert [p.stem for p in crops] == ["0003", "0004"]  # 只裁段內幀


# ─── merge_band_submission ───────────────────────────────────────────

def _rows(entries):
    return [["ID", "x", "y", "width", "height"]] + \
        [[i, str(x), str(y), str(w), str(h)] for i, x, y, w, h in entries]


def _write(path, rows):
    import csv

    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def test_merge_replaces_triggered_and_keeps_rest(tmp_path):
    base = _rows([("nir-a_1", 1, 1, 5, 5), ("nir-a_2", 2, 2, 5, 5),
                  ("vis-b_1", 9, 9, 4, 4)])
    band = _rows([("nir-a_1", 1, 1, 5, 5), ("nir-a_2", 3, 3, 6, 6)])
    merged, replaced, kept = ms.merge(base, band, {"nir-a"})
    assert replaced == 2 and kept == 1
    assert merged[2][1:] == ["3", "3", "6", "6"]  # band 版換入
    assert merged[3] == base[3]                     # 未觸發支位元級沿用


def test_merge_validate_catches_first_frame_and_id_order(tmp_path):
    fc = tmp_path / "fc" / "nir-a"
    fc.mkdir(parents=True)
    (fc / "init_rect.txt").write_text("1 1 5 5")
    sample = _rows([("nir-a_1", 0, 0, 0, 0), ("nir-a_2", 0, 0, 0, 0)])
    good = _rows([("nir-a_1", 1, 1, 5, 5), ("nir-a_2", 2, 2, 5, 5)])
    assert ms.validate(good, sample, tmp_path / "fc", {"nir-a"}) == []
    bad_first = _rows([("nir-a_1", 9, 9, 5, 5), ("nir-a_2", 2, 2, 5, 5)])
    assert any("preserve_init" in e
               for e in ms.validate(bad_first, sample, tmp_path / "fc", {"nir-a"}))
    bad_order = _rows([("nir-a_2", 2, 2, 5, 5), ("nir-a_1", 1, 1, 5, 5)])
    assert any("順序" in e or "ID" in e
               for e in ms.validate(bad_order, sample, tmp_path / "fc", {"nir-a"}))


def test_merge_cli_end_to_end_with_report(tmp_path):
    fc = tmp_path / "fc" / "nir-a"
    fc.mkdir(parents=True)
    (fc / "init_rect.txt").write_text("1 1 5 5")
    base_p, band_p = tmp_path / "base.csv", tmp_path / "band.csv"
    _write(base_p, _rows([("nir-a_1", 1, 1, 5, 5), ("nir-a_2", 2, 2, 5, 5),
                          ("vis-b_1", 9, 9, 4, 4)]))
    _write(band_p, _rows([("nir-a_1", 1, 1, 5, 5), ("nir-a_2", 2.5, 2, 5, 5)]))
    _write(tmp_path / "sample.csv",
           _rows([("nir-a_1", 0, 0, 0, 0), ("nir-a_2", 0, 0, 0, 0),
                  ("vis-b_1", 0, 0, 0, 0)]))
    trig = tmp_path / "trig.json"
    trig.write_text(json.dumps({"triggered": ["nir-a"]}))
    (tmp_path / "fc" / "vis-b").mkdir()
    (tmp_path / "fc" / "vis-b" / "init_rect.txt").write_text("9 9 4 4")
    rc = ms.main(["--base-csv", str(base_p), "--band-csv", str(band_p),
                  "--triggered-json", str(trig), "--sample", str(tmp_path / "sample.csv"),
                  "--fc-root", str(tmp_path / "fc"),
                  "--out", str(tmp_path / "final.csv"),
                  "--report", str(tmp_path / "rep.json")])
    assert rc == 0
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["rows_replaced"] == 2 and rep["rows_kept"] == 1
    assert rep["per_seq_diff_vs_base"]["nir-a"]["n"] == 2
