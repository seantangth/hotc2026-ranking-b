#!/usr/bin/env python3
"""E29 判決：crop 窗內 pan-sharpen（顏色＋真實解析度）vs 現行假色。多腿共用同一組 crop 窗。

【假說出處】D060（E26）判準未達結案，但機制解讀是：mosaic 腿在 vis-S_jump2 上
  0.1688 → 0.6747（+0.5059、跟丟率 74.4%→0%），卻在兩支 droneshow 上 −0.4872／−0.6229，
  原因是 **mosaic 被轉成單通道灰階、顏色丟光**。D060 自己下的結論是「顏色 vs 解析度的取捨」
  ——但那是實作造成的取捨，不是物理必然。pan-sharpening 同時給兩者。

【腿】全部共用 E26 的同一組 crop 窗（唯一變因＝窗內像素合成方式）：
  fc                  現行 v008 行為（錨點）
  pan_raw             pan-sharpen，pan 直接用 mosaic 灰階，不拉對比
  pan_raw_stretch     ＋亮度 p1–p99 拉伸
  pan_equalized_stretch  ＋sub-position 均衡（抹平 4×4 固定紋樣）＋拉伸  ← 主候選

【事前判準（Sean 08-10 授權，沿用 E26 原文）】
  主判準：≥2/3 支改善 且 最壞 > −0.05 → 擴到全量（crop 選中序列）後發 LB
         任一支 < −0.10 或改善 < 2 支 → 該腿結案
  健全性錨點：fc 腿 pooled 須重現 E26 實測的 0.4769（偏離 > 0.02 ⇒ rig 有問題，實驗作廢）

【機制讀數（與主判準**分開**記錄，避免錯誤歸因）】
  droneshow 兩支已在 0.68–0.69，本假說對它們的預測不是「更好」而是「不會崩」：
    必要條件：兩支 droneshow Δ > −0.05          ⇒ 顏色保住了
    充分條件：vis-S_jump2 Δ > +0.25（E26 增益的一半）⇒ 解析度增益在 pan-sharpen 下存活
  ⚠️ 兩者皆過但主判準未過（例如 droneshow 微跌 −0.02）⇒ 記為「**機制成立、待調參**」，
     **不是整條線判死**。這行是寫給下一個 session 看的：不要照 Decision Log 慣例直接關線。

【本機零 GPU 前測（08-10，已完成，供對照）】逐幀影像統計顯示 pan_equalized_stretch 是唯一
  在兩支 droneshow 上把色度守在假色基準之上（15.09/29.14 vs fc 的 14.50/32.00）、
  同時把固定紋樣壓到 ≈0.4、且高頻細節達 fc 的 3–12 倍者；pan_raw_stretch 在 droneshow 上
  色度反而掉到 9.39（未均衡的紋樣主導亮度統計）。⇒ 事前預期排序：equalized_stretch > raw_stretch > raw。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

E26_FC_POOLED = 0.4769        # E26 實測的 fc 腿 pooled（同 3 支、同窗）＝ rig 健全性錨點
ANCHOR_TOL = 0.02
SJUMP = "vis-S_jump2"


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[:, 0], b[:, 0]); iy1 = np.maximum(a[:, 1], b[:, 1])
    ix2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    iy2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    it = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return it / np.maximum(a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - it, 1e-9)


def load_leg(csv: Path, meta: dict, scaled: bool, gt_base: dict, gt_n: dict) -> pd.DataFrame:
    """scaled=True ⇒ 該腿的影像是 mosaic 解析度，座標需除回 macro（與 E26 mosaic 腿相同）。"""
    d = pd.read_csv(csv)
    q = d["ID"].str.rsplit("_", n=1, expand=True)
    d["seq"], d["pos"] = q[0], q[1].astype(int)
    out = []
    for s, g in d.groupby("seq"):
        m = meta[s]
        x1, y1 = m["win"][0], m["win"][1]
        sc = m["macro"] if scaled else 1
        # ⚠️ prep 的 frame_ids 是 zip 內編號 1..N，不是 GT 全域幀號（GT 跨序列連續）。
        # 逐支等長時按位置對應：global = pos + gt_min − 1。（D060 的兩個對位 bug 之一）
        assert m["n_frames"] == gt_n[s], f"{s}: prep {m['n_frames']} 幀 != GT {gt_n[s]} 幀"
        g = g.sort_values("pos").copy()
        g["frame"] = g["pos"] + gt_base[s] - 1
        g["x"] = g["x"] / sc + x1
        g["y"] = g["y"] / sc + y1
        g["width"] = g["width"] / sc
        g["height"] = g["height"] / sc
        out.append(g[["seq", "frame", "x", "y", "width", "height"]])
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True, help="含 out_<leg>/submission.csv 與 prep_meta.json")
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--legs", nargs="+",
                    default=["fc", "pan_raw", "pan_raw_stretch", "pan_equalized_stretch"])
    a = ap.parse_args()

    root = Path(a.out_root)
    meta = json.loads((root / "prep_meta.json").read_text())
    gt = pd.read_csv(a.gt_csv)
    p = gt["ID"].str.rsplit("_", n=1, expand=True)
    gt["seq"], gt["frame"] = p[0], p[1].astype(int)
    gt_base = gt.groupby("seq")["frame"].min().to_dict()
    gt_n = gt.groupby("seq").size().to_dict()

    per, pooled = {}, {}
    for leg in a.legs:
        csv = root / f"out_{leg}" / "submission.csv"
        if not csv.exists():
            print(f"⚠️ {leg} 缺 submission.csv，跳過"); continue
        pr = load_leg(csv, meta, leg != "fc", gt_base, gt_n)
        m = gt.merge(pr, on=["seq", "frame"], suffixes=("_g", "_p"))
        m["iou"] = iou(m[["x_g", "y_g", "width_g", "height_g"]].to_numpy(float),
                       m[["x_p", "y_p", "width_p", "height_p"]].to_numpy(float))
        pooled[leg] = float(m["iou"].mean())
        per[leg] = {s: {"auc": float(g["iou"].mean()), "n": len(g),
                        "lost": float((g["iou"] < 0.1).mean())}
                    for s, g in m.groupby("seq")}

    assert "fc" in per, "fc 錨點腿缺失，無法判決"
    seqs = sorted(per["fc"])
    legs = [l for l in a.legs if l in per]

    print(f"\n{'序列':<22}" + "".join(f"{l[:13]:>14}" for l in legs))
    for s in seqs:
        print(f"{s:<22}" + "".join(f"{per[l][s]['auc']:>14.4f}" for l in legs))
    print(f"{'pooled':<22}" + "".join(f"{pooled[l]:>14.4f}" for l in legs))
    print(f"\n{'跟丟率（IoU<0.1）':<22}" + "".join(f"{l[:13]:>14}" for l in legs))
    for s in seqs:
        print(f"{s:<22}" + "".join(f"{per[l][s]['lost']:>13.1%} " for l in legs))

    anchor_off = abs(pooled["fc"] - E26_FC_POOLED)
    print(f"\n健全性錨點：E26 實測 fc pooled {E26_FC_POOLED:.4f} vs 本次 {pooled['fc']:.4f} "
          f"（差 {anchor_off:.4f}）→ "
          f"{'✅ rig 一致' if anchor_off <= ANCHOR_TOL else '🚨 偏離過大，實驗作廢，先查 prep/窗表'}")

    results = {}
    for leg in legs:
        if leg == "fc":
            continue
        dv = {s: per[leg][s]["auc"] - per["fc"][s]["auc"] for s in seqs}
        arr = np.array(list(dv.values()))
        n_better, worst = int((arr > 0).sum()), float(arr.min())
        drone_ok = all(v > -0.05 for s, v in dv.items() if "droneshow" in s)
        jump_ok = dv.get(SJUMP, -9) > 0.25
        if n_better >= 2 and worst > -0.05:
            v = "✅ 主判準達成 → 擴到全量 crop 選中序列，D038 不對稱比過關後發 LB"
        elif drone_ok and jump_ok:
            v = "⚠️ 機制成立、待調參（顏色保住＋解析度增益存活），**不得判死整條線**"
        elif worst < -0.10 or n_better < 2:
            v = "❌ 主判準未達且機制條件未同時成立 → 本配方結案"
        else:
            v = "⚠️ 中間帶 → 依型態決定"
        print(f"\n▶ {leg}: Δ vs fc = " + "  ".join(f"{s.split('-')[-1][:9]} {d:+.4f}" for s, d in dv.items()))
        print(f"   改善 {n_better}/{len(seqs)}｜最壞 {worst:+.4f}｜"
              f"機制必要（droneshow 皆 >−0.05）{'✅' if drone_ok else '❌'}｜"
              f"機制充分（S_jump2 >+0.25）{'✅' if jump_ok else '❌'}")
        print(f"   {v}")
        results[leg] = {"delta": dv, "n_better": n_better, "worst": worst,
                        "drone_ok": drone_ok, "jump_ok": jump_ok, "verdict": v}

    (root / "verdict_e29.json").write_text(json.dumps(
        {"pooled": pooled, "per_seq": per, "anchor_off": anchor_off,
         "anchor_ok": anchor_off <= ANCHOR_TOL, "legs": results},
        indent=1, ensure_ascii=False))
    print(f"\n→ {root}/verdict_e29.json")


if __name__ == "__main__":
    main()
