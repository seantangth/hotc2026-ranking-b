#!/usr/bin/env python3
"""E33 前測：mask → box 導出慣例的消融（$0，本機 val SAM2 mask 快取）

判準見 5_outputs/strategy_research_20260812/E33_MASK2BOX_DESIGN_20260814.md（執行前寫死）。
只能「殺」不能「證」（D044：SAM2 mask ≠ SAM3 mask）。
自測（D067(f)）：tight 變體必須位元級重現 npz 內建 boxes，否則整份讀數作廢。
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path("/Users/seantang/Desktop/Sean/The_Nexus/1_Projects/WHISPERS_2026_HyperSOT")
MASKS = ROOT / "5_outputs/t1_rerun_20260805/masks"
GT_CSV = ROOT / "1_data/raw/2026training.csv"
VAL = ROOT / "1_data/val_split_v1.txt"


def iou(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0


def tight(m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1 - xs.min()), float(ys.max() + 1 - ys.min())]


def largest_cc(m):
    """最大連通元件（8-連通）的 tight bbox。孤立雜訊像素被丟棄。"""
    lab, n = ndimage.label(m, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return tight(lab == (int(np.argmax(sizes)) + 1))


def opening(m, k):
    """形態學 opening（k×k），去掉細絲與孤立點後取 tight bbox。"""
    o = ndimage.binary_opening(m, structure=np.ones((k, k), dtype=bool))
    return tight(o) if o.any() else tight(m)


def percentile_box(m, p):
    """沿各軸丟掉最外 p% 的前景像素質量（對稱截尾）。"""
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    x1, x2 = np.percentile(xs, [p, 100 - p])
    y1, y2 = np.percentile(ys, [p, 100 - p])
    w, h = max(1.0, x2 + 1 - x1), max(1.0, y2 + 1 - y1)
    return [float(x1), float(y1), float(w), float(h)]


def dilate_cc(m, k):
    """先 closing 補洞再取最大連通元件——分割破碎時把同一物體重新黏起來。"""
    c = ndimage.binary_closing(m, structure=np.ones((k, k), dtype=bool))
    return largest_cc(c) if c.any() else tight(m)


VARIANTS = {
    "tight(基準)":      lambda m: tight(m),
    "largest_CC":       lambda m: largest_cc(m),
    "open3":            lambda m: opening(m, 3),
    "open5":            lambda m: opening(m, 5),
    "close3+CC":        lambda m: dilate_cc(m, 3),
    "close5+CC":        lambda m: dilate_cc(m, 5),
    "pct1":             lambda m: percentile_box(m, 1),
    "pct2":             lambda m: percentile_box(m, 2),
    "pct5":             lambda m: percentile_box(m, 5),
}

gt = pd.read_csv(GT_CSV)
gt["seq"] = gt["ID"].str.rsplit("_", n=1).str[0]
gt["frame"] = gt["ID"].str.rsplit("_", n=1).str[1].astype(int)
gtmap = {r.ID: (float(r.x), float(r.y), float(r.width), float(r.height)) for r in gt.itertuples()}
val_seqs = [s.strip() for s in VAL.read_text().split() if s.strip()]

ious = {k: [] for k in VARIANTS}
selftest_fail = 0
n_frames = 0
n_empty = 0
cc_differs = 0          # largest_CC 與 tight 不同的幀數＝「mask 有離群前景」的直接證據
per_seq = {k: {} for k in VARIANTS}

for seq in val_seqs:
    f = MASKS / f"{seq}.npz"
    if not f.is_file():
        print(f"[SKIP] {seq}: 無快取", file=sys.stderr)
        continue
    z = np.load(f)
    n, H, W = int(z["n"]), int(z["height"]), int(z["width"])
    bits, boxes, empty = z["bits"], z["boxes"], z["empty"]
    npx = H * W

    def unpack(i):
        """整段 mask 是連續 packbits（不逐幀對齊 byte 邊界）——用 bit offset 取。"""
        s, e = i * npx, (i + 1) * npx
        b0, b1 = s // 8, (e + 7) // 8
        seg = np.unpackbits(bits[b0:b1])
        off = s - b0 * 8
        return seg[off:off + npx].reshape(H, W).astype(bool)
    frames = sorted(gt.loc[gt["seq"] == seq, "frame"].tolist())
    if len(frames) != n:
        print(f"⚠️ {seq}: GT {len(frames)} vs mask {n}", file=sys.stderr)
    prev = {k: None for k in VARIANTS}
    acc = {k: [] for k in VARIANTS}
    for i in range(min(n, len(frames))):
        g = gtmap.get(f"{seq}_{frames[i]}")
        if g is None:
            continue
        n_frames += 1
        if empty[i]:
            n_empty += 1
            for k in VARIANTS:
                b = prev[k] if prev[k] is not None else list(boxes[i])
                ious[k].append(iou(b, g)); acc[k].append(iou(b, g))
            continue
        m = unpack(i)
        base = None
        for k, fn in VARIANTS.items():
            b = fn(m)
            if b is None:
                b = prev[k] if prev[k] is not None else list(boxes[i])
            if k == "tight(基準)":
                base = b
                if [round(v) for v in b] != [round(v) for v in boxes[i]]:
                    selftest_fail += 1
            elif k == "largest_CC" and b != base:
                cc_differs += 1
            prev[k] = b
            ious[k].append(iou(b, g)); acc[k].append(iou(b, g))
    for k in VARIANTS:
        if acc[k]:
            per_seq[k][seq] = float(np.mean(acc[k]))

print(f"\n{'='*72}")
print(f"幀數 {n_frames}（空 mask {n_empty}，{n_empty/max(1,n_frames):.1%}）｜"
      f"自測失配 {selftest_fail}｜largest_CC≠tight 的幀 {cc_differs} ({cc_differs/max(1,n_frames):.2%})")
if selftest_fail:
    print(f"🚨 自測失敗 {selftest_fail} 幀 ⇒ 依 D067(f) 整份讀數作廢")
    sys.exit(2)
print(f"{'='*72}")
base_pool = float(np.mean(ious["tight(基準)"]))
rows = []
for k in VARIANTS:
    pool = float(np.mean(ious[k]))
    d = {s: per_seq[k][s] - per_seq["tight(基準)"][s] for s in per_seq[k]}
    worst = min(d.values()) if d else 0.0
    worst_s = min(d, key=d.get) if d else "-"
    nwin = sum(1 for v in d.values() if v > 1e-6)
    nlose = sum(1 for v in d.values() if v < -1e-6)
    rows.append((k, pool, pool - base_pool, nwin, nlose, worst, worst_s))
    print(f"{k:<14} pooled {pool:.5f}  Δ {pool-base_pool:+.5f}  "
          f"序列 勝{nwin:>2}/敗{nlose:>2}  最壞 {worst:+.4f} ({worst_s})")
json.dump({"n_frames": n_frames, "cc_differs": cc_differs, "base": base_pool,
           "rows": [{"variant": r[0], "pooled": r[1], "delta": r[2], "win": r[3],
                     "lose": r[4], "worst": r[5], "worst_seq": r[6]} for r in rows],
           "per_seq": per_seq},
          open(Path(__file__).parent / "e33_mask2box_result.json", "w"), indent=1, ensure_ascii=False)
print(f"\n結果已寫入 e33_mask2box_result.json")
