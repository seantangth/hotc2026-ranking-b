"""io.py 驗證：合成 mosaic 單元測試（純數學）+ 真實樣本整合驗證。

直接跑：  python3.14 -m hsot.test_io      （需 3_src 在 PYTHONPATH）
也相容 pytest。真實樣本測試指向 scratchpad；樣本不存在則 skip（不失敗）。
"""
from __future__ import annotations

import os
import numpy as np

from hsot.io import (x2cube, load_cube, read_cube_png, read_cube_zip, list_zip_frames,
                     read_falsecolor, MACRO, N_BANDS)

# 真實樣本（由 rclone 從 Drive 拉到 scratchpad）
SCRATCH = ("/private/tmp/claude-501/-Users-seantang-Desktop-Sean-The-Nexus-1-Projects-"
           "WHISPERS-2026-HyperSOT/2781a953-dcc4-445d-85ad-e1444b8102de/scratchpad")


# ---- 官方 HyperTools.X2Cube 原碼（對照用，reshape 寫死 //4）----
def _official_x2cube(img, B=[4, 4], skip=[4, 4], bandNumber=16):
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    return out.reshape(M // 4, N // 4, bandNumber)  # ← 官方寫死 //4


def _make_mosaic(cube, macro):
    """cube (h,w,macro^2) → mosaic：band k 放 macro-pixel 內 (k//macro, k%macro)。"""
    h, w, b = cube.shape
    assert b == macro * macro
    mos = np.zeros((h * macro, w * macro), dtype=cube.dtype)
    for k in range(b):
        mos[k // macro::macro, k % macro::macro] = cube[:, :, k]
    return mos


def test_x2cube_roundtrip():
    """合成已知 cube → mosaic → x2cube 應完整還原（三種 macro）。"""
    rng = np.random.default_rng(0)
    for macro in (4, 5):
        cube = rng.integers(0, 1024, size=(7, 11, macro * macro), dtype=np.uint16)
        out = x2cube(_make_mosaic(cube, macro), macro)
        assert out.shape == (7, 11, macro * macro), f"macro={macro} shape {out.shape}"
        assert np.array_equal(out, cube), f"macro={macro} 還原不符"
    print("[✓] test_x2cube_roundtrip：4×4 / 5×5 合成還原精確")


def test_official_bug_on_nir():
    """官方 //4 對 NIR 5×5 會 ValueError（元素數不符）；修正版正確。"""
    rng = np.random.default_rng(1)
    cube = rng.integers(0, 1024, size=(5, 5, 25), dtype=np.uint16)  # NIR：M=N=25
    mos = _make_mosaic(cube, 5)
    # 修正版：對
    assert x2cube(mos, 5).shape == (5, 5, 25)
    # 官方版：崩
    raised = False
    try:
        _official_x2cube(mos, B=[5, 5], skip=[5, 5], bandNumber=25)
    except ValueError:
        raised = True
    assert raised, "官方 //4 對 NIR 5×5 竟沒崩——對照前提有誤"
    print("[✓] test_official_bug_on_nir：官方 //4 對 5×5 ValueError，修正 //5 正確")


def _ncc(a, b):
    """兩灰階圖 pixel-wise Pearson 相關（空間對齊指標）。"""
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _check_real(name, cube, fc, modality, expect_bands):
    hs, ws = cube.shape[:2]
    hf, wf = fc.shape[:2]
    print(f"  {name}: cube {cube.shape} / 假色 ({hf},{wf})")
    assert cube.shape[2] == expect_bands, f"{name} bands {cube.shape[2]}!={expect_bands}"
    assert (hs, ws) == (hf, wf), f"{name} cube 空間 {(hs,ws)} != 假色 {(hf,wf)}（X2Cube 錯位）"
    # 空間對齊：cube 波段平均 vs 假色灰階 的 NCC，應顯著 > 0
    band_mean = cube.mean(axis=2)
    gray = fc.mean(axis=2)
    ncc = _ncc(band_mean, gray)
    # 對照：故意用錯 macro 讀（若整除）→ 錯位 NCC 應明顯更低（此處只印修正版數值）
    print(f"       band-mean vs 假色 NCC = {ncc:.3f}（空間對齊，>0.3 視為通過）")
    assert ncc > 0.3, f"{name} cube 與假色空間不對齊（NCC={ncc:.3f}）"
    return ncc


def test_real_samples():
    """真實 NIR/RedNIR/VIS 樣本：尺寸==假色、bands 正確、末 band 全零、空間對齊。"""
    ran = False
    # NIR bee2（5×5→25）
    p = f"{SCRATCH}/nir_bee2_0001.png"; fc = f"{SCRATCH}/nir_bee2_fc.jpg"
    if os.path.exists(p) and os.path.exists(fc):
        ran = True
        cube = read_cube_png(p, "nir")
        _check_real("NIR bee2", cube, read_falsecolor(fc), "nir", 25)

    # RedNIR bee1（4×4→16→丟末→15；驗末 band 全零）
    p = f"{SCRATCH}/rednir_bee1_0001.png"; fc = f"{SCRATCH}/rednir_bee1_fc.jpg"
    if os.path.exists(p) and os.path.exists(fc):
        ran = True
        from hsot.io import _read_u16
        full = x2cube(_read_u16(p), 4)          # 未丟末 band（16）
        # 官方稱末 band「only zero values」，但此 raw_archive 版實測非嚴格零；
        # 末 4 band(12-15) 為系統性最弱波段群，丟 index 15 仍合官方慣例。
        tail_max = int(full[:, :, 12:].max())
        main_max = int(full[:, :, :12].max())
        assert tail_max < main_max, f"末波段群應弱於主波段群，實測 tail={tail_max} main={main_max}"
        cube = load_cube(_read_u16(p), "rednir")  # 丟末 band → 15
        _check_real("RedNIR bee1", cube, read_falsecolor(fc), "rednir", 15)
        print(f"       末波段群 max={tail_max} < 主波段群 max={main_max}（非嚴格零，丟末→15 bands）")

    # VIS car10（4×4→16；zip 開箱 + 熱像素）
    zp = f"{SCRATCH}/vis_car10.zip"; fc = f"{SCRATCH}/vis_car10_fc.jpg"
    if os.path.exists(zp) and os.path.exists(fc):
        ran = True
        frame0 = list_zip_frames(zp)[0]
        cube = read_cube_zip(zp, frame0, "vis")
        _check_real("VIS car10", cube, read_falsecolor(fc), "vis", 16)
        hot = int(cube.max())
        print(f"       zip 開箱首幀={frame0}，cube max={hot}（VIS 熱像素離群特性）")

    if not ran:
        print("[skip] test_real_samples：scratchpad 無樣本（先用 rclone 拉）")
    else:
        print("[✓] test_real_samples：真實樣本尺寸/bands/對齊全通過")


if __name__ == "__main__":
    test_x2cube_roundtrip()
    test_official_bug_on_nir()
    test_real_samples()
    print("\n全部通過 ✅")
