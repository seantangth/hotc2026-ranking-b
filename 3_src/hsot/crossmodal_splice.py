#!/usr/bin/env python3
"""E34 跨模態借框拼接（零 GPU，D061 拼接法）

判準見 5_outputs/strategy_research_20260812/E34_CROSSMODAL_DESIGN_20260814.md
（該檔於本腳本產生任何 CSV 之前寫入）。

規則（全部因果，只用當前幀與過去幀；無 GT；不認序列名——只用「同場景名不同模態」這個結構）：
  本側凍結串 ≥K  ∧  對側凍結串 <2  ∧  該配對過去的對齊一致性移動平均 ≥THR（樣本 ≥MIN_HIST）
  ⇒ 以對側框（經座標映射）取代本側框。

座標映射：同尺寸用純平移；不同尺寸用影像尺寸比縮放 ＋ 首幀中心對齊。
平移量只用首幀估、全序列固定 —— 首幀輸出＝官方 init box 原樣（track_t1.py:337），零 GT 洩漏。
"""
import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_csv(p):
    d = defaultdict(dict)
    for r in csv.DictReader(open(p)):
        s, f = r["ID"].rsplit("_", 1)
        d[s][int(f)] = [float(r["x"]), float(r["y"]), float(r["width"]), float(r["height"])]
    return d


def iou(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0


def frozen_runs(fm):
    """{frame: 連續與前一幀完全相同的長度}（0 = 該幀有更新）"""
    ks = sorted(fm)
    out, r = {}, 0
    for i, k in enumerate(ks):
        r = r + 1 if i > 0 and fm[k] == fm[ks[i - 1]] else 0
        out[k] = r
    return out


def make_mapper(sz_a, sz_b, box_a0, box_b0):
    """回傳 f(B 座標系的框) -> A 座標系的框。尺寸相同時退化為純平移。"""
    sx = sz_b["W"] / sz_a["W"]
    sy = sz_b["H"] / sz_a["H"]

    def unscale(b):
        return [b[0] / sx, b[1] / sy, b[2] / sx, b[3] / sy]

    m0 = unscale(box_b0)
    tx = (m0[0] + m0[2] / 2) - (box_a0[0] + box_a0[2] / 2)
    ty = (m0[1] + m0[3] / 2) - (box_a0[1] + box_a0[3] / 2)

    def f(b):
        m = unscale(b)
        return [m[0] - tx, m[1] - ty, m[2], m[3]]

    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="基準提交 CSV（v029）")
    ap.add_argument("--sizes", default=str(ROOT / "1_data/test_sizes_75.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=2, help="本側凍結串門檻")
    ap.add_argument("--thr", type=float, default=0.5, help="配對一致性閘門")
    ap.add_argument("--hist", type=int, default=30, help="一致性移動平均窗長")
    ap.add_argument("--min-hist", type=int, default=10, help="啟用閘門所需最少樣本")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    base = load_csv(a.base)
    sz = json.load(open(a.sizes))
    scene = defaultdict(list)
    for s in base:
        scene[s.split("-", 1)[1]].append(s)
    pairs = [(k, sorted(v)) for k, v in scene.items() if len(v) == 2]
    if not pairs:
        print("🚨 找不到任何跨模態配對 ⇒ 規則不觸發，輸出等同基準", file=sys.stderr)

    runs = {s: frozen_runs(base[s]) for s in base}
    out = {s: {f: list(b) for f, b in fm.items()} for s, fm in base.items()}
    rep = {"pairs": [], "n_replaced": 0, "n_total": sum(len(v) for v in base.values())}

    for k, (A, B) in sorted(pairs):
        fs = sorted(set(base[A]) & set(base[B]))
        if not fs:
            continue
        f0 = fs[0]
        # A<-B 與 B<-A 兩個方向各自的映射
        mBA = make_mapper(sz[A], sz[B], base[A][f0], base[B][f0])
        mAB = make_mapper(sz[B], sz[A], base[B][f0], base[A][f0])
        hist = deque(maxlen=a.hist)
        nA = nB = 0
        gate_on_frames = 0
        for f in fs:
            ra, rb = runs[A][f], runs[B][f]
            # 閘門判定用【過去】的樣本（本幀尚未加入 hist）
            gate = len(hist) >= a.min_hist and (sum(hist) / len(hist)) >= a.thr
            gate_on_frames += int(gate)
            if gate:
                if ra >= a.K and rb < 2:
                    out[A][f] = mBA(base[B][f]); nA += 1
                elif rb >= a.K and ra < 2:
                    out[B][f] = mAB(base[A][f]); nB += 1
            # 只用「兩側都活著」的幀累積一致性證據（因果：本幀之後才入列）
            if ra < 2 and rb < 2:
                hist.append(iou(base[A][f], mBA(base[B][f])))
        rep["pairs"].append({"scene": k, "A": A, "B": B, "n": len(fs),
                             "gate_on_frames": gate_on_frames,
                             "replaced_A": nA, "replaced_B": nB,
                             "final_consistency": round(sum(hist) / len(hist), 4) if hist else None})
        rep["n_replaced"] += nA + nB

    # 寫出（維持基準的列順序 = exact-set 不變）
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ID", "x", "y", "width", "height"])
        for r in csv.DictReader(open(a.base)):
            s, f = r["ID"].rsplit("_", 1)
            b = out[s][int(f)]
            w.writerow([r["ID"]] + [f"{v:g}" for v in b])

    print(f"配對 {len(pairs)} 組｜替換 {rep['n_replaced']} 幀 "
          f"({rep['n_replaced']/rep['n_total']:.2%} of {rep['n_total']})")
    for p in rep["pairs"]:
        print(f"  {p['scene']:<12} n={p['n']:>4} 閘門開 {p['gate_on_frames']:>4} "
              f"一致性 {p['final_consistency']}  換 A{p['replaced_A']:>4} / B{p['replaced_B']:>4}")
    if a.report:
        json.dump(rep, open(a.report, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
