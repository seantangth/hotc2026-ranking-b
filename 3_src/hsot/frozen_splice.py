#!/usr/bin/env python3
"""E35 跨底座借框拼接（零 GPU，D061）

規則（因果、無 GT、不認序列名）：
  基準該幀凍結串 ≥K  ∧  來源該幀凍結串 <2  ⇒  以來源框取代基準框。
兩套 CSV 必須同座標系（同一組校正變換），故不做任何映射。

判準見 5_outputs/strategy_research_20260812/E35_CROSSBASE_DESIGN_20260814.md
（該檔於本腳本產生任何 CSV 之前寫入）。
"""
import argparse
import csv
import json
import sys
from collections import defaultdict, deque


def load(p):
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


def runs(fm):
    ks = sorted(fm)
    o, r = {}, 0
    for i, k in enumerate(ks):
        r = r + 1 if i > 0 and fm[k] == fm[ks[i - 1]] else 0
        o[k] = r
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--src", required=True, help="借框來源 CSV（須與 base 同座標系/同校正）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--src-max-run", type=int, default=2, help="來源凍結串須 < 此值才算「活著」")
    ap.add_argument("--thr", type=float, default=0.0,
                    help=">0 時啟用因果一致性閘門：兩套【過去】都活著的幀上平均 IoU 須 ≥thr 才借框。"
                         "機制＝v016 實證的『兩底座追不同目標時必有一邊全錯』，借框等於換目標。")
    ap.add_argument("--hist", type=int, default=30)
    ap.add_argument("--min-hist", type=int, default=10)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    base, src = load(a.base), load(a.src)
    if set(base) != set(src):
        print(f"🚨 序列集合不同：base {len(base)} vs src {len(src)}", file=sys.stderr)
        sys.exit(2)
    RB = {s: runs(base[s]) for s in base}
    RS = {s: runs(src[s]) for s in src}

    out = {s: {f: list(b) for f, b in fm.items()} for s, fm in base.items()}
    per, blocked_per = {}, {}
    n = tot = nblk = 0
    for s in base:
        c = blk = 0
        hist = deque(maxlen=a.hist)
        for f in sorted(base[s]):
            tot += 1
            if f not in src[s]:
                continue
            # 閘門只看【過去】幀（本幀之後才入列）⇒ 因果
            gate = a.thr <= 0 or (len(hist) >= a.min_hist and sum(hist) / len(hist) >= a.thr)
            if RB[s][f] >= a.K and RS[s][f] < a.src_max_run:
                if gate:
                    out[s][f] = list(src[s][f])
                    c += 1
                else:
                    blk += 1
            if RB[s][f] < 2 and RS[s][f] < 2:
                hist.append(iou(base[s][f], src[s][f]))
        if c:
            per[s] = c
        if blk:
            blocked_per[s] = blk
        n += c
        nblk += blk

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ID", "x", "y", "width", "height"])
        for r in csv.DictReader(open(a.base)):
            s, f = r["ID"].rsplit("_", 1)
            w.writerow([r["ID"]] + [f"{v:g}" for v in out[s][int(f)]])

    print(f"K={a.K} thr={a.thr}｜替換 {n} 幀 ({n/tot:.2%} of {tot})｜涉及 {len(per)} 支序列"
          f"｜閘門擋掉 {nblk} 幀")
    for s, c in sorted(per.items(), key=lambda x: -x[1])[:15]:
        print(f"  換 {s:<26}{c:>5} 幀 ({c/len(base[s]):.1%} of {len(base[s])})")
    for s, c in sorted(blocked_per.items(), key=lambda x: -x[1])[:8]:
        print(f"  擋 {s:<26}{c:>5} 幀")
    if a.report:
        json.dump({"K": a.K, "thr": a.thr, "n_replaced": n, "n_total": tot,
                   "per_seq": per, "blocked_per_seq": blocked_per},
                  open(a.report, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
