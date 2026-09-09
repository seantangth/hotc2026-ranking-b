#!/usr/bin/env python3
"""E-A' band×SAM3（D092）：批次觸發判定＋band-3ch 轉檔 orchestrator。

═══ 合規論證（會進 9/7 交付揭露文件，逐條錨定）═══
1. **band 選擇只用官方首幀 init box**：Fisher 分數與觸發判定的前景/背景遮罩
   全部由 `init_rect.txt`（OPE 協定明文提供的 "initialization from the ground
   truth position in the initial frame"）導出；**首幀之後的任何 GT 皆不觸碰**
   （用了即作弊）。不讀序列名做任何分支（`modality_of` 只決定 X2Cube 的
   macro-pixel 尺寸——那是感測器物理格式，同官方假色管線的既有行為）。
2. **同一套規則對全部 75 支序列適用**（賽規 same model hyper-parameters for
   all sequences）：不是「nir/rednir 用 band、VIS 用假色」的模態特化，而是
   **對每支序列跑同一條資料驅動判定**——首幀 init box 上比較三個 3ch 候選
   的目標-背景 Fisher AUC：{官方假色, top-3 band, Fisher 投影}，最高者須
   ≥ 假色 + MARGIN 才換、否則維持假色（＝維持 v078 結果）。VIS 依 D092
   實測（Δ 中位 −0.001）自然大多不觸發。規則全域、每序列自足（per-seq
   RNG 由 [SEED, crc32(seq)] 導出，與批次順序無關）。
   Fisher 投影候選的依據：D092＋5 支真序列補測——判別力在**波段亮度組合**
   （全譜多變量 AUC 0.87–0.96）而非個別 band（top-3 實測 4/5 輸假色）；
   官方假色本身就是固定線性投影（CIE），投影候選是同一函數族的 per-seq
   資料驅動版，domain 距離不比「挑 3 個生波段」遠。
3. MARGIN=0.02 的依據＝D092 的 21 支獨立 EDA（train 序列）中兩群的分離點
   （VIS 中位 −0.001 vs NIR/RedNIR 中位 +0.032/+0.034），**事前寫死、
   非 LB 搜索**（D033 合規）。
4. 與 v003/v004 條件式規則兩連敗（D037）的差異：那些門檻是从 train/val 分佈
   校準的**學習成分**；本規則的判定統計量只用**該序列自己的首幀**（自足、
   不含任何跨序列學習參數），MARGIN 是常數。誠實限制：首幀可分性對整段
   序列的預測力未經驗證——這正是本次 LB 一發要買的資訊。

核心計算全部重用既有交付件：X2Cube＝`hsot/io.py`（修正版、RedNIR 丟末
全零 band）；Fisher band 分數與 3ch 合成＝`hsot/band_select.py`
（`band_separability`／`normalize_band`，含盒形距離前景/1.5–2.5× 環狀背景）。

兩個模式：
  --decide  對 --seqs 全部序列產觸發判定 json（不寫影像）
  --convert 對 triggered 序列全幀轉檔 → out-root/<seq>/*.jpg ＋ init_rect.txt
            （檔名 stem 與假色目錄逐一 assert 對齊——track_t1 直接可吃）
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 3_src/
from hsot.band_select import band_separability, normalize_band, RING  # noqa: E402
from hsot.io import load_cube, modality_of, _read_u16  # noqa: E402

MARGIN = 0.02       # 觸發判準（D092 分離點；事前寫死，勿改）
FC_CEILING = 0.93   # 假色首幀 AUC ≥ 此值即不換：換輸入的邊際收益趨零、只剩
                    # domain 風險（D092 §三：fc>0.93 的多為 VIS 與已良好序列；
                    # 5 支補測的真觸發候選 fc 全在 0.55–0.80）。事前寫死。
SEED = 42
N_PIX = 2000


def read_init(seq_dir: Path) -> list:
    """讀 init_rect.txt（與 track_t1.read_init 同語意：x y w h，容逗號）。"""
    txt = (seq_dir / "init_rect.txt").read_text().strip().replace(",", " ")
    vals = [float(v) for v in txt.split()[:4]]
    assert len(vals) == 4, f"{seq_dir}/init_rect.txt 欄位不足：{txt!r}"
    return vals


def fg_bg_masks(H: int, W: int, box, ring=RING):
    """前景/背景遮罩——與 band_select.band_separability 完全同口徑
    （盒形距離：d≤1 目標、ring 環狀背景），觸發判定與 band 選擇一家口徑。"""
    x, y, w, h = [float(v) for v in box]
    cx, cy = x + w / 2, y + h / 2
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.maximum(np.abs(xx - cx) / max(w, 1e-6),
                   np.abs(yy - cy) / max(h, 1e-6)) * 2
    return d <= 1.0, (d > ring[0]) & (d <= ring[1])


def _rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(np.float64) + 1
    n_p, n_n = len(pos), len(neg)
    return float((r[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def pixel_fisher_auc(img3: np.ndarray, box, rng: np.random.Generator) -> float:
    """3ch 影像的目標-背景 split-half Fisher AUC（D092 §三同方法）。
    前半 fit Fisher 方向、後半量 AUC——防 3 維上的過擬合樂觀。"""
    H, W = img3.shape[:2]
    fg, bg = fg_bg_masks(H, W, box)
    if fg.sum() < 8 or bg.sum() < 32:
        return 0.5  # 樣本不足＝無資訊＝不觸發方向
    X = img3.reshape(-1, img3.shape[2]).astype(np.float64)

    def sample(mask):
        pts = np.flatnonzero(mask.ravel())
        take = rng.choice(pts, min(N_PIX, len(pts)), replace=False)
        return X[take]

    P, N = sample(fg), sample(bg)
    hp, hn = len(P) // 2, len(N) // 2
    mu_p, mu_n = P[:hp].mean(0), N[:hn].mean(0)
    Sw = np.cov(P[:hp].T) + np.cov(N[:hn].T) + 1e-3 * np.eye(X.shape[1])
    w = np.linalg.solve(Sw, mu_p - mu_n)
    return _rank_auc(P[hp:] @ w, N[hn:] @ w)


def first_mosaic(hsi_seq_dir: Path) -> Path:
    pngs = sorted(hsi_seq_dir.glob("*.png"))
    assert pngs, f"{hsi_seq_dir} 無 mosaic png"
    return pngs[0]


def band3_first_frame(hsi_seq_dir: Path, seq: str, init) -> tuple:
    """首幀 cube → (top3 band indices, band-3ch uint8 影像)。"""
    mod = modality_of(seq)
    cube = load_cube(_read_u16(first_mosaic(hsi_seq_dir)), mod)
    scores = band_separability(cube, init)
    top3 = np.argsort(scores)[::-1][:3].tolist()
    img3 = np.stack([normalize_band(cube[:, :, b]) for b in top3], axis=-1)
    return top3, img3, [round(float(s), 4) for s in scores]


def _fix_sign(w: np.ndarray) -> np.ndarray:
    """特徵向量符號固定（最大絕對分量為正）——確定性的一部分。"""
    return w * np.sign(w[np.argmax(np.abs(w))] or 1.0)


def fisher_proj_matrix(cube: np.ndarray, box) -> np.ndarray:
    """首幀 cube ＋ init box → 投影矩陣 W [B,3]（(a') 形式的核心）。

    ch1 ＝ Fisher 判別方向（`Sw⁻¹(μ_t − μ_b)`，目標/背景遮罩同 band_separability
    的盒形距離口徑）；ch2/ch3 ＝ 對 ch1 正交化後的全樣本 PCA 前 2（保留場景
    整體結構——SAM3 需要的是「判別軸疊在自然影像狀底圖上」而非純判別圖）。
    D092+補測依據：判別力在**波段亮度組合**（全譜多變量 AUC 0.87–0.96）而非
    個別 band（top-3 實測 4/5 輸假色）；官方假色本身就是固定線性投影（CIE），
    本投影是同一函數族的 per-seq 資料驅動版。W 只由首幀決定、全序列固定（因果）。
    """
    H, W_, B = cube.shape
    fg, bg = fg_bg_masks(H, W_, box)
    X = cube.reshape(-1, B).astype(np.float64)
    t, b = X[fg.ravel()], X[bg.ravel()]
    assert len(t) >= 4 and len(b) >= 16, "init box 太小，無法估 Fisher 方向"
    Sw = np.cov(t.T) + np.cov(b.T) + 1e-3 * np.eye(B)
    w1 = _fix_sign(np.linalg.solve(Sw, t.mean(0) - b.mean(0)))
    w1 = w1 / (np.linalg.norm(w1) + 1e-12)
    C = np.cov(X.T)
    C = C - np.outer(w1, w1 @ C) - np.outer(C @ w1, w1) + np.outer(w1, w1) * (w1 @ C @ w1)
    vals, vecs = np.linalg.eigh(C)
    w2, w3 = _fix_sign(vecs[:, -1]), _fix_sign(vecs[:, -2])
    return np.stack([w1, w2, w3], axis=1)  # [B,3]


def proj3_image(cube: np.ndarray, W: np.ndarray) -> np.ndarray:
    """cube [H,W,B] × W [B,3] → uint8 3ch（每通道 p1–p99 normalize，同 band 版）。"""
    proj = cube.astype(np.float64) @ W  # [H,W,3]
    return np.stack([normalize_band(proj[:, :, c]) for c in range(3)], axis=-1)


def seq_rng(seq: str, stream: str) -> np.random.Generator:
    """每序列×每用途的獨立確定性 RNG——觸發判定不得依賴批次處理順序
    （合規主張「每序列自足」的落地；用 crc32 而非 python hash——後者有 salt）。"""
    return np.random.default_rng([SEED, zlib.crc32(seq.encode()),
                                  zlib.crc32(stream.encode())])


def decide_one(seq: str, fc_root: Path, hsi_root: Path) -> dict:
    """三候選觸發判定：{官方假色, top-3 band, Fisher 投影} 中選首幀 AUC 最高者，
    且必須超過假色 + MARGIN 才換（否則維持假色＝維持 v078）。單一規則、全域
    適用、每序列自足；chosen 記進 plan，convert 依它產對應形式。"""
    fc_dir = fc_root / seq
    init = read_init(fc_dir)
    fc_first = sorted(fc_dir.glob("*.jp*g"))
    assert fc_first, f"{fc_dir} 無假色 jpg"
    from PIL import Image

    fc3 = np.array(Image.open(fc_first[0]).convert("RGB"))
    mod = modality_of(seq)
    cube = load_cube(_read_u16(first_mosaic(hsi_root / seq)), mod)
    top3, band3, _ = band3_first_frame(hsi_root / seq, seq, init)
    Wp = fisher_proj_matrix(cube, init)
    proj3 = proj3_image(cube, Wp)
    assert band3.shape[:2] == fc3.shape[:2], \
        f"{seq}: band {band3.shape} vs fc {fc3.shape} 尺寸不合（X2Cube/資料源錯）"
    auc_fc = pixel_fisher_auc(fc3, init, seq_rng(seq, "fc"))
    auc_band = pixel_fisher_auc(band3, init, seq_rng(seq, "band3"))
    auc_proj = pixel_fisher_auc(proj3, init, seq_rng(seq, "proj3"))
    cands = {"band3": auc_band, "proj3": auc_proj}
    best = max(cands, key=cands.get)
    chosen = ("fc" if auc_fc >= FC_CEILING
              else best if cands[best] >= auc_fc + MARGIN else "fc")
    return {"seq": seq, "modality": mod, "top3_bands": top3,
            "auc_fc": round(auc_fc, 4), "auc_band3": round(auc_band, 4),
            "auc_proj3": round(auc_proj, 4), "chosen": chosen,
            "delta_best": round(cands[best] - auc_fc, 4),
            "triggered": chosen != "fc"}


def convert_one(seq: str, fc_root: Path, hsi_root: Path, out_root: Path,
                chosen: str = "band3") -> dict:
    """全幀轉檔。band/proj 選擇只由首幀 init 決定、全序列固定（因果、確定性）。"""
    from PIL import Image

    fc_dir = fc_root / seq
    init = read_init(fc_dir)
    mod = modality_of(seq)
    hsi_dir = hsi_root / seq
    pngs = sorted(hsi_dir.glob("*.png"))
    first_cube = load_cube(_read_u16(pngs[0]), mod)
    if chosen == "band3":
        top3, _, _ = band3_first_frame(hsi_dir, seq, init)
        make = lambda cube: np.stack(  # noqa: E731
            [normalize_band(cube[:, :, b]) for b in top3], axis=-1)
        recipe = {"top3_bands": top3}
    elif chosen == "proj3":
        Wp = fisher_proj_matrix(first_cube, init)  # 首幀固定 W、全幀套用（因果）
        make = lambda cube: proj3_image(cube, Wp)  # noqa: E731
        recipe = {"proj_matrix_shape": list(Wp.shape)}
    else:
        raise ValueError(f"{seq}: 未知 chosen={chosen!r}")

    out = out_root / seq
    out.mkdir(parents=True, exist_ok=True)
    for f in pngs:
        cube = load_cube(_read_u16(f), mod)
        Image.fromarray(make(cube)).save(out / (f.stem + ".jpg"), quality=95)
    (out / "init_rect.txt").write_text((fc_dir / "init_rect.txt").read_text())

    # 對齊 assert（fail-closed）：band 輸出的幀 stem 必須與假色目錄完全一致
    # ——track_t1 以 sorted glob 讀幀，stem 不齊＝幀錯位＝不報錯只做錯事。
    fc_stems = [p.stem for p in sorted(fc_dir.glob("*.jp*g"))]
    band_stems = [p.stem for p in sorted(out.glob("*.jpg"))]
    first_diff = next(((a, b) for a, b in zip(band_stems, fc_stems) if a != b),
                      ("<len>", "<len>"))
    assert band_stems == fc_stems, (
        f"{seq}: band 幀名與假色不齊（band {len(band_stems)} vs fc {len(fc_stems)}；"
        f"首個差異 {first_diff}）")
    return {"seq": seq, "chosen": chosen, **recipe, "n_frames": len(band_stems)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fc-root", required=True, help="假色 75 支 root（含 init_rect.txt）")
    ap.add_argument("--hsi-root", required=True, help="mosaic png root（逐支目錄）")
    ap.add_argument("--seqs", required=True, help="序列清單檔（一行一支）")
    ap.add_argument("--decide-out", help="--decide 模式：觸發判定 json 輸出路徑")
    ap.add_argument("--out-root", help="--convert 模式：band jpg 輸出 root")
    ap.add_argument("--triggered-json", help="--convert 模式：--decide 的輸出（只轉觸發支）")
    args = ap.parse_args(argv)

    seqs = [s.strip() for s in Path(args.seqs).read_text().splitlines() if s.strip()]
    fc_root, hsi_root = Path(args.fc_root), Path(args.hsi_root)

    if args.decide_out:
        rows, failed = [], []
        for seq in seqs:
            try:
                r = decide_one(seq, fc_root, hsi_root)
            except Exception as e:  # noqa: BLE001 —— 單支失敗照實記錄、整體 fail-closed
                r = {"seq": seq, "error": repr(e)[:200]}
                failed.append(seq)
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
        triggered = [r["seq"] for r in rows if r.get("triggered")]
        out = {"margin": MARGIN, "seed": SEED, "n_seqs": len(seqs),
               "n_triggered": len(triggered), "triggered": triggered,
               "failed": failed, "rows": rows}
        Path(args.decide_out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(f"[decide] triggered {len(triggered)}/{len(seqs)}；failed {len(failed)}")
        return 2 if failed else 0

    assert args.out_root and args.triggered_json, "--convert 需要 --out-root 與 --triggered-json"
    plan = json.loads(Path(args.triggered_json).read_text())
    chosen_by_seq = {r["seq"]: r.get("chosen", "band3")
                     for r in plan.get("rows", []) if r.get("triggered")}
    todo = [s for s in seqs if s in set(plan["triggered"])]
    out_root = Path(args.out_root)
    for i, seq in enumerate(todo):
        r = convert_one(seq, fc_root, hsi_root, out_root,
                        chosen=chosen_by_seq.get(seq, "band3"))
        print(f"[convert {i+1}/{len(todo)}] {json.dumps(r, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
