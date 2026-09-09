#!/usr/bin/env python3
"""E26 判決：crop 窗內「感測器真實像素(mosaic) vs 插值上採樣(假色)」。

兩腿共用同一組 crop 窗 ⇒ 唯一變因＝窗內像素來源。
把兩腿輸出各自映射回**假色原圖座標系**後，對 GT 算逐序列 AUC。

事前判準（寫於執行前，見 launch/setup_e26_mosaic_crop_v1.sh 檔頭）：
  (a) 3 支中 ≥2 支 mosaic > fc 且 (b) 最壞單支 Δ > −0.05 → 擴到全量 test
  任一支 Δ < −0.10 或改善 <2 支 → 結案
  健全性錨點：fc 腿 pooled 須 > E15 無 crop 的 0.1510
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

E15_NO_CROP_POOLED = 0.1510   # 同 3 支、無 crop 的既有基準（本機零成本算出）


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[:, 0], b[:, 0]); iy1 = np.maximum(a[:, 1], b[:, 1])
    ix2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    iy2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    it = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return it / np.maximum(a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - it, 1e-9)


def load_leg(csv: Path, meta: dict, leg: str, gt_base: dict, gt_n: dict) -> pd.DataFrame:
    d = pd.read_csv(csv)
    q = d["ID"].str.rsplit("_", n=1, expand=True)
    d["seq"], d["pos"] = q[0], q[1].astype(int)
    out = []
    for s, g in d.groupby("seq"):
        m = meta[s]
        x1, y1 = m["win"][0], m["win"][1]
        sc = m["macro"] if leg == "mosaic" else 1     # mosaic 腿需除回 macro
        # ⚠️ prep 存的 frame_ids 是 **zip 內編號 1..N**，不是 GT 的全域幀號
        # （GT 是跨序列連續編號，例 vis-droneshow2 ＝ 72186–72635）。
        # 兩者逐支等長時按位置對應：global = pos + gt_min − 1。
        assert m["n_frames"] == gt_n[s], f"{s}: prep {m['n_frames']} 幀 != GT {gt_n[s]} 幀，位置對應不成立"
        g = g.sort_values("pos").copy()
        g["frame"] = g["pos"] + gt_base[s] - 1
        g["x"] = g["x"] / sc + x1
        g["y"] = g["y"] / sc + y1
        g["width"] = g["width"] / sc
        g["height"] = g["height"] / sc
        out.append(g[["seq", "frame", "x", "y", "width", "height"]])
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True, help="含 out_fc/ out_mosaic/ prep_meta.json")
    ap.add_argument("--gt-csv", required=True)
    a = ap.parse_args()

    root = Path(a.out_root)
    meta = json.loads((root / "prep_meta.json").read_text())
    gt = pd.read_csv(a.gt_csv)
    p = gt["ID"].str.rsplit("_", n=1, expand=True)
    gt["seq"], gt["frame"] = p[0], p[1].astype(int)

    gt_base = gt.groupby("seq")["frame"].min().to_dict()
    gt_n = gt.groupby("seq").size().to_dict()

    per, pooled = {}, {}
    for leg in ("fc", "mosaic"):
        pr = load_leg(root / f"out_{leg}" / "submission.csv", meta, leg, gt_base, gt_n)
        m = gt.merge(pr, on=["seq", "frame"], suffixes=("_g", "_p"))
        # ⚠️ merge 後預測欄位帶 _p 後綴（gt 同名欄位造成衝突）——首版誤用無後綴名而 KeyError
        m["iou"] = iou(m[["x_g", "y_g", "width_g", "height_g"]].to_numpy(float),
                       m[["x_p", "y_p", "width_p", "height_p"]].to_numpy(float))
        pooled[leg] = float(m["iou"].mean())
        per[leg] = {s: {"auc": float(g["iou"].mean()), "n": len(g),
                        "lost": float((g["iou"] < 0.1).mean())}
                    for s, g in m.groupby("seq")}

    seqs = sorted(per["fc"])
    print(f"{'序列':<22}{'fc':>9}{'mosaic':>9}{'Δ':>9}   {'跟丟率 fc→mosaic':>20}")
    deltas = []
    for s in seqs:
        f, mo = per["fc"][s]["auc"], per["mosaic"][s]["auc"]
        deltas.append(mo - f)
        print(f"{s:<22}{f:>9.4f}{mo:>9.4f}{mo-f:>+9.4f}   "
              f"{per['fc'][s]['lost']:>8.1%} → {per['mosaic'][s]['lost']:.1%}")
    dv = np.array(deltas)
    print(f"\npooled: fc {pooled['fc']:.4f} | mosaic {pooled['mosaic']:.4f} "
          f"| Δ {pooled['mosaic']-pooled['fc']:+.4f}")
    ok_anchor = pooled["fc"] > E15_NO_CROP_POOLED
    print(f"健全性錨點：E15 無 crop = {E15_NO_CROP_POOLED:.4f} → fc 腿 "
          f"{'✅ 高於（crop 本身有效，實驗成立）' if ok_anchor else '🚨 未高於（實驗不成立，先查 prep）'}")

    n_better, worst = int((dv > 0).sum()), float(dv.min())
    print(f"\n事前判準：≥2/3 支改善 且 最壞 > −0.05")
    print(f"  改善 {n_better}/3 支 ｜ 最壞 {worst:+.4f}")
    if n_better >= 2 and worst > -0.05:
        v = "✅ 判準達成 → 擴到全量 test（21 支 crop 選中序列）後發 LB"
    elif worst < -0.10 or n_better < 2:
        v = "❌ 判準未達 → 結案，記錄逐序列 delta 與型態"
    else:
        v = "⚠️ 中間帶 → 依型態決定，需 Sean 裁示（D037：條件式易失敗）"
    print(f"▶ {v}")
    (root / "verdict_local.json").write_text(json.dumps(
        {"pooled": pooled, "per_seq": per, "n_better": n_better, "worst": worst,
         "anchor_ok": ok_anchor, "verdict": v}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
