#!/usr/bin/env python3
"""E34 前測：跨模態同名場景的 GT 是否同步（$0，train GT）

test 有 13 組跨模態同名場景（26 支＝35% 的 test），幀數完全相同，
其中 10 組 rednir/vis 連影像尺寸都相同（512x272）。
若 GT 在跨模態間同步（同一物體、同一座標系），則「一個模態追丟時借用另一個模態的框」
是因果、無 GT、不認序列名的合法手段——且直接對症 D063 診斷的殘餘跟丟桶。
"""
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

ROOT = "/Users/seantang/Desktop/Sean/The_Nexus/1_Projects/WHISPERS_2026_HyperSOT"
gt = pd.read_csv(f"{ROOT}/1_data/raw/2026training.csv")
gt["seq"] = gt["ID"].str.rsplit("_", n=1).str[0]
gt["frame"] = gt["ID"].str.rsplit("_", n=1).str[1].astype(int)
gt["mod"] = gt["seq"].str.split("-").str[0]
gt["scene"] = gt["seq"].str.split("-", n=1).str[1]

# scene -> mod -> {frame: box}
by = defaultdict(lambda: defaultdict(dict))
for r in gt.itertuples():
    by[r.scene][r.mod][r.frame] = (float(r.x), float(r.y), float(r.width), float(r.height))


def iou(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0


res = defaultdict(list)
for s, mm in by.items():
    mods = sorted(mm)
    if len(mods) < 2:
        continue
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            m1, m2 = mods[i], mods[j]
            fr = sorted(set(mm[m1]) & set(mm[m2]))
            if not fr:
                continue
            a = np.array([mm[m1][f] for f in fr], dtype=float)
            b = np.array([mm[m2][f] for f in fr], dtype=float)
            ious = np.array([iou(a[k], b[k]) for k in range(len(fr))])
            ident = float(np.mean(np.all(a == b, axis=1)))
            res[(m1, m2)].append({
                "scene": s, "n": len(fr), "iou": float(ious.mean()),
                "iou_med": float(np.median(ious)), "ident": ident,
                "dx": float(np.median(b[:, 0] - a[:, 0])), "dy": float(np.median(b[:, 1] - a[:, 1])),
                "dw": float(np.median(b[:, 2] - a[:, 2])), "dh": float(np.median(b[:, 3] - a[:, 3])),
                "sx": float(np.median(b[:, 2] / np.maximum(a[:, 2], 1))),
                "sy": float(np.median(b[:, 3] / np.maximum(a[:, 3], 1))),
                # 幀內時間對齊的檢查：兩模態的框位移序列是否同步變化
                "corr": float(np.corrcoef(np.diff(a[:, 0]), np.diff(b[:, 0]))[0, 1]) if len(fr) > 3 and np.std(np.diff(a[:, 0])) > 0 and np.std(np.diff(b[:, 0])) > 0 else float("nan"),
            })

out = []
for k, v in res.items():
    out.append(f"\n{'='*78}\n=== {k[0]} vs {k[1]}：{len(v)} 組場景 ===")
    mi = np.mean([x["iou"] for x in v])
    mid = np.mean([x["ident"] for x in v])
    corrs = [x["corr"] for x in v if not np.isnan(x["corr"])]
    out.append(f"  平均逐幀 GT-IoU {mi:.4f}｜逐幀框完全相同的比例 {mid:.1%}"
               f"｜位移序列相關 r 中位 {np.median(corrs):.3f} (n={len(corrs)})" if corrs else "")
    out.append(f"  中位 dx: {np.median([x['dx'] for x in v]):+.1f}  dy: {np.median([x['dy'] for x in v]):+.1f}"
               f"  dw: {np.median([x['dw'] for x in v]):+.1f}  dh: {np.median([x['dh'] for x in v]):+.1f}"
               f"  尺度 sx {np.median([x['sx'] for x in v]):.3f} sy {np.median([x['sy'] for x in v]):.3f}")
    out.append(f"  {'場景':<16}{'n':>5}{'IoU均':>8}{'IoU中':>8}{'全同':>7}{'dx':>6}{'dy':>6}{'r':>7}")
    for x in sorted(v, key=lambda z: -z["iou"]):
        out.append(f"  {x['scene']:<16}{x['n']:>5}{x['iou']:>8.4f}{x['iou_med']:>8.4f}"
                   f"{x['ident']:>7.0%}{x['dx']:>+6.0f}{x['dy']:>+6.0f}{x['corr']:>7.3f}")
print("\n".join(out))
