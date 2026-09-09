#!/usr/bin/env python3
"""pansharp_v1 — E29：把 mosaic 的「真實解析度」與假色的「顏色」合成到同一張圖。

【為什麼】D060（E26）判準未達結案，但機制解讀是關鍵：mosaic 腿贏的地方
  vis-S_jump2 0.1688 → 0.6747（+0.5059、跟丟率 74.4%→0%）＝ 解析度治好了跟丟；
  輸的地方 兩支 droneshow −0.4872／−0.6229 ＝ **mosaic 被轉成單通道灰階、顏色整個丟掉**，
  而天空背景同型無人機群的唯一線索就是顏色（D058 獨立佐證：跟丟當下假色已足以分辨真目標）。
⇒ D060 自己的結論是「**顏色 vs 解析度的取捨**」——但這個取捨是實作造成的，不是物理必然：
  mosaic 是 4×4／5×5 的 band 排列，官方假色等於只取每個 macro cell 的 3 個像素、丟掉其餘 13。
  Pan-sharpening 是遙測領域處理這件事的標準手段：**亮度細節取自全解析度 pan，色度取自低解析度彩圖**。

【為什麼用 HPF additive 而不是 Brovey】Brovey 是 `low × (pan / I)`，在 I≈0 的像素比值爆炸
  ——而 droneshow 的背景正是暗空，那不是邊角案例是主案例。additive 是
  `low + (pan − blur(pan))`：**色度逐像素原封不動**，只加高頻細節 ⇒ 正好對上假說
  「顏色要保住、解析度要拿到」。Brovey 保留為 --method brovey（附比值 clip）供對照。

【兩個 pan 變體，同機同窗跑，避免賭單一配方】
  raw       ：pan 直接用 mosaic 灰階。E26 的灰階腿沒做均衡化仍贏 +0.51 ⇒ 「SAM3 容忍 4×4 紋樣」
              有一個實測點。
  equalized ：把 macro×macro 每個 sub-position 的子影像各自正規化到共同均值/標準差，
              抹平「不同 band 響應不同」造成的固定紋樣，只留真實空間細節（＝ pan 的定義）。
              理論上更乾淨，但**沒有任何實測點** ⇒ 兩個都跑才有配方層級的歸因。

【白點對齊】假色是 8-bit JPEG、mosaic 原是 10-bit ⇒ 注入前先把 pan 的分佈 match 到
  低解析度圖的亮度分佈（逐窗），否則整體偏亮/偏暗。

用法：
  python3 pansharp_v1.py --e26-root <解開的 e26_frames> --out-root <輸出> \
      --variants raw equalized [--method hpf|brovey] [--preview-dir <PNG 目視>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

EPS = 1e-6


def equalize_subgrid(pan: np.ndarray, macro: int) -> np.ndarray:
    """把 macro×macro 每個 sub-position 的子影像正規化到共同 mean/std。

    mosaic 上相鄰像素屬於**不同波段**，響應不同 ⇒ 原圖帶有 macro 週期的固定紋樣。
    逐 sub-position 正規化＝移除波段間的增益/偏移差，只留空間結構（這就是 panchromatic 的定義）。
    """
    out = pan.copy()
    tgt_m, tgt_s = float(pan.mean()), float(pan.std())
    for dy in range(macro):
        for dx in range(macro):
            sub = pan[dy::macro, dx::macro]
            s = float(sub.std())
            out[dy::macro, dx::macro] = (sub - sub.mean()) / (s + EPS) * tgt_s + tgt_m
    return out


def luma_stretch(img: np.ndarray) -> np.ndarray:
    """對亮度做 p1–p99 線性拉伸，三通道套用**同一條**仿射 ⇒ 通道差只被等比放大，色相不翻轉。

    為什麼需要這一步（08-10 目視發現，且 D054 早已標註為未分離因素）：
    E26 的 mosaic 腿配方是「p1–p99 正規化 ×3ch」，它同時給了 **4× 真實解析度** 與 **對比拉伸**
    兩件事；D054 原文即寫明「只能確認兩者的**組合**救回了它，兩因素未分離」。
    純 pan-sharpen 只補細節、**繼承假色偏暗的色調曲線** ⇒ 目視上 vis-S_jump2 明顯暗於 mosaic 腿。
    ⇒ 對比拉伸必須當成一個獨立可開關的因素來測，否則會把「沒拉對比」誤判成「pan-sharpen 沒用」。
    """
    I = img.mean(axis=2)
    lo, hi = np.percentile(I, 1), np.percentile(I, 99)
    return (img - lo) * (255.0 / max(hi - lo, 1.0))


def pansharpen(fc: np.ndarray, pan: np.ndarray, macro: int,
               variant: str = "raw", method: str = "hpf", stretch: bool = False) -> np.ndarray:
    """fc: (h,w,3) uint8 假色；pan: (H,W) uint8 mosaic 灰階，H=h*macro。回傳 (H,W,3) uint8。"""
    H, W = pan.shape
    low = np.asarray(Image.fromarray(fc).resize((W, H), Image.BICUBIC), np.float32)
    p = pan.astype(np.float32)
    if variant == "equalized":
        p = equalize_subgrid(p, macro)

    I = low.mean(axis=2)                                    # 低解析度圖的亮度
    p = (p - p.mean()) / (p.std() + EPS) * (I.std() + EPS) + I.mean()   # 白點對齊（逐窗）

    if method == "hpf":
        # 低通＝降採樣 macro 倍（BOX＝面積平均）再升回來。這個定義不是隨便挑的：
        # 它移除的**正好是假色已經具備的 macro cell 尺度**，留下的**正好是假色沒有的次 cell 細節**。
        pim = Image.fromarray(p, mode="F")
        blur = np.asarray(pim.resize((W // macro, H // macro), Image.BOX)
                             .resize((W, H), Image.BILINEAR), np.float32)
        out = low + (p - blur)[..., None]                   # 色度逐像素不動，只加細節
    elif method == "brovey":
        ratio = np.clip(p / (I + 1.0), 0.2, 5.0)            # clip：暗背景（droneshow 夜空）防爆
        out = low * ratio[..., None]
    else:
        raise ValueError(method)
    if stretch:
        out = luma_stretch(out)
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e26-root", required=True, help="解開的 e26_frames（含 fc/ mosaic/ prep_meta.json）")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--variants", nargs="+",
                    default=["raw", "raw_stretch", "equalized_stretch"],
                    help="名稱格式 <raw|equalized>[_stretch]；_stretch ＝ 額外做亮度 p1–p99 拉伸")
    ap.add_argument("--method", default="hpf", choices=("hpf", "brovey"))
    ap.add_argument("--preview-dir", help="每序列存 3 張 PNG 供目視（開機前必看）")
    ap.add_argument("--quality", type=int, default=95)
    a = ap.parse_args()

    root, out_root = Path(a.e26_root).resolve(), Path(a.out_root).resolve()
    meta = json.loads((root / "prep_meta.json").read_text())

    for variant in a.variants:
        pan_v, do_stretch = variant.replace("_stretch", ""), variant.endswith("_stretch")
        assert pan_v in ("raw", "equalized"), f"未知變體 {variant}"
        for seq, m in meta.items():
            macro = int(m["macro"])
            d_fc, d_mo = root / "fc" / seq, root / "mosaic" / seq
            d_out = out_root / f"pan_{variant}" / seq
            d_out.mkdir(parents=True, exist_ok=True)
            names = sorted(p.name for p in d_fc.glob("*.jpg"))
            for k, name in enumerate(names):
                fc = np.asarray(Image.open(d_fc / name).convert("RGB"))
                pan = np.asarray(Image.open(d_mo / name).convert("L"))
                assert pan.shape[0] == fc.shape[0] * macro and pan.shape[1] == fc.shape[1] * macro, \
                    f"{seq}/{name}: pan {pan.shape} 不是 fc {fc.shape[:2]} 的 {macro} 倍"
                img = pansharpen(fc, pan, macro, pan_v, a.method, do_stretch)
                Image.fromarray(img).save(d_out / name, quality=a.quality)
                if a.preview_dir and k in (0, len(names) // 2, len(names) - 1):
                    pv = Path(a.preview_dir) / variant
                    pv.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(img).save(pv / f"{seq}_{name.replace('.jpg', '.png')}")
            # init_rect 與 mosaic 腿相同（座標已 ×macro）
            (d_out / "init_rect.txt").write_text((d_mo / "init_rect.txt").read_text())
            print(f"✅ {variant} {seq}: {len(names)} 幀 → {pan.shape[1]}x{pan.shape[0]} (macro={macro})")

    (out_root / "pansharp_meta.json").write_text(json.dumps(
        {"source": "e26_frames", "method": a.method, "variants": a.variants,
         "note": "座標映射與 E26 mosaic 腿相同：box/macro + (x1,y1)"}, indent=1))
    print(f"\n→ {out_root}")


if __name__ == "__main__":
    main()
