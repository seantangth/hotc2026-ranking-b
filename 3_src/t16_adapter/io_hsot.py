"""HSOT HSI mosaic 讀取層 + 修正版 X2Cube（E04 光譜再偵測地基）。

三模態 mosaic → hyperspectral cube：
    VIS    4×4 macro-pixel → 16 bands
    NIR    5×5 macro-pixel → 25 bands
    RedNIR 4×4 macro-pixel → 16 bands，丟末 band（全零）→ 15 bands

官方 HyperTools.X2Cube 的 reshape 寫死 `M//4, N//4`，對 NIR 5×5 會崩/錯位；
本模組唯一修正是空間尺寸改用 `M//macro, N//macro`（im2col 索引與官方逐字等價）。

Drive 存放差異（raw_archive/validation/）：
    HSI-NIR/{seq}/{0001.png..}      解開的目錄，uint16 png
    HSI-RedNIR/{seq}/{0001.png..}   解開的目錄
    HSI-VIS/{seq}.zip               每序列一個 zip（內含 uint16 png）— 需開箱
    HSI-*-FalseColor/{seq}/{0001.jpg..}  官方假色（uint8 RGB），尺寸 == cube 空間尺寸

驗證黃金標準：cube 空間尺寸 (M//macro, N//macro) 必須等於官方假色尺寸。
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

# macro-pixel 邊長；bands 數（RedNIR 丟末全零 band → 15）
MACRO = {"vis": 4, "nir": 5, "rednir": 4}
DROP_LAST = {"rednir"}
N_BANDS = {"vis": 16, "nir": 25, "rednir": 15}


def modality_of(seq: str) -> str:
    """從序列名前綴推模態，如 'nir-bee2' → 'nir'。"""
    m = seq.split("-", 1)[0].lower()
    assert m in MACRO, f"未知模態：{seq!r}"
    return m


def x2cube(mosaic: np.ndarray, macro: int) -> np.ndarray:
    """mosaic (M,N) → cube (M//macro, N//macro, macro*macro)。

    忠實移植官方 HyperTools.X2Cube 的 im2col 索引，唯一修正 reshape 空間尺寸
    用 M//macro, N//macro（官方寫死 //4，NIR 5×5 會崩/錯位）。

    macro-pixel 內 band 排列為 row-major：cube[y,x,k] = mosaic[y*macro + k//macro, x*macro + k%macro]。
    """
    B0 = B1 = macro
    M, N = mosaic.shape
    assert M % macro == 0 and N % macro == 0, f"mosaic {M}×{N} 非 {macro} 整除，模態/尺寸不符"
    band_number = macro * macro
    start_idx = np.arange(B0)[:, None] * N + np.arange(B1)
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B0, B1))
    offset_idx = np.arange(M - B0 + 1)[:, None] * N + np.arange(N - B1 + 1)
    out = np.take(mosaic, start_idx.ravel()[:, None] + offset_idx[::macro, ::macro].ravel())
    out = np.transpose(out)
    return out.reshape(M // macro, N // macro, band_number)  # 修正：//macro（非官方 //4）


def load_cube(mosaic: np.ndarray, modality: str) -> np.ndarray:
    """mosaic uint16 → cube (H,W,bands)。RedNIR 丟末 band（全零）。"""
    cube = x2cube(mosaic, MACRO[modality])
    if modality in DROP_LAST:
        cube = cube[:, :, :-1]
    return cube


def _read_u16(fp) -> np.ndarray:
    arr = np.array(Image.open(fp))
    assert arr.dtype == np.uint16, f"HSI mosaic 應為 uint16，實際 {arr.dtype}（勿抓成假色/8-bit）"
    return arr


def read_cube_png(path: str | Path, modality: str) -> np.ndarray:
    """讀解開目錄下的 mosaic png → cube（NIR/RedNIR）。"""
    return load_cube(_read_u16(path), modality)


def read_cube_zip(zip_path: str | Path, frame: str, modality: str = "vis") -> np.ndarray:
    """從 VIS zip 內讀單幀 mosaic → cube。frame 如 '0001.png'（容忍 zip 內有 seq/ 前綴）。"""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(frame) and not n.endswith("/")]
        assert names, f"{zip_path} 內找不到 {frame}"
        with zf.open(names[0]) as f:
            arr = _read_u16(f)
    return load_cube(arr, modality)


def list_zip_frames(zip_path: str | Path) -> list[str]:
    """列 VIS zip 內的 png 幀名（排序）。"""
    with zipfile.ZipFile(zip_path) as zf:
        frames = [Path(n).name for n in zf.namelist()
                  if n.lower().endswith(".png") and not n.endswith("/")]
    return sorted(frames, key=lambda s: int(Path(s).stem) if Path(s).stem.isdigit() else s)


def read_falsecolor(path: str | Path) -> np.ndarray:
    """讀官方假色 jpg → uint8 (H,W,3) RGB。"""
    return np.array(Image.open(path).convert("RGB"))


# ---------------- E04 地基：光譜簽名 + 光譜角圖 ----------------
def spectral_signature(cube: np.ndarray, box_xywh) -> np.ndarray:
    """box 內像素平均光譜（L2 正規化）。box=[x,y,w,h]，cube 座標系（= 假色/init_rect 座標）。"""
    x, y, w, h = (int(round(float(v))) for v in box_xywh)
    y0, x0 = max(y, 0), max(x, 0)
    patch = cube[y0:y + max(h, 1), x0:x + max(w, 1), :].reshape(-1, cube.shape[2])
    assert patch.size, "box 落在 cube 外或尺寸為零"
    sig = patch.mean(0).astype(np.float32)
    n = float(np.linalg.norm(sig))
    return sig / n if n > 0 else sig


def spectral_angle_map(cube: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """每像素與參考光譜的餘弦相似度圖 (H,W) ∈ [0,1]；ref=首幀目標 spectral_signature。

    E04：空 mask/低信心時，在搜尋區取此圖峰值作為光譜再定位候選。
    """
    H, W, B = cube.shape
    flat = cube.reshape(-1, B).astype(np.float32)
    flat_n = flat / np.clip(np.linalg.norm(flat, axis=1, keepdims=True), 1e-6, None)
    ref_n = ref.astype(np.float32) / max(float(np.linalg.norm(ref)), 1e-6)
    return (flat_n @ ref_n).reshape(H, W).clip(0.0, 1.0)


def peak_in_region(sam: np.ndarray, cx: float, cy: float, R: float, k: int = 3):
    """在 (cx,cy)±R 搜尋區內找光譜角圖 k×k 平滑峰值。回傳 (px, py, smax)。

    E04 再定位核心：tracker 丟失時只在最後可信位置附近局部搜尋（全圖 argmax 會被
    背景淹沒——可行性驗證證 R=2–3×目標尺度最佳）。smax 為落點的平滑相似度（非孤立
    噪點單像素值），與 prompt 落點同指標，供 SIM_MIN 保守閘使用。
    """
    H, W = sam.shape
    y0, y1 = max(int(cy - R), 0), min(int(cy + R) + 1, H)
    x0, x1 = max(int(cx - R), 0), min(int(cx + R) + 1, W)
    sub = sam[y0:y1, x0:x1]
    if sub.size == 0:
        return int(cx), int(cy), 0.0
    if sub.shape[0] < k or sub.shape[1] < k:
        py, px = np.unravel_index(int(np.argmax(sub)), sub.shape)
        smax = float(sub[py, px])
    else:
        win = sliding_window_view(sub, (k, k)).mean(axis=(-1, -2))
        wy, wx = np.unravel_index(int(np.argmax(win)), win.shape)
        py, px = wy + k // 2, wx + k // 2
        smax = float(win[wy, wx])
    return px + x0, py + y0, smax
