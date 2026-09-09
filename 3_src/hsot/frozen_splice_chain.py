#!/usr/bin/env python3
"""E36 多來源鏈式借框（零 GPU，D061）

規則（因果、無 GT、不認序列名）：
  基準該幀凍結串 ≥K ⇒ 依【事前固定的優先序】掃過來源清單，取第一個「活著」（凍結串 <2）的來源框。
來源優先序必須事前寫死並記入判決書——**不得依 LB 分數重排**（那會是 dataset 特化）。
所有 CSV 必須同座標系（同一 +1px 校正）。
"""
import argparse
import csv
import json
from collections import defaultdict


def load(p):
    d = defaultdict(dict)
    for r in csv.DictReader(open(p)):
        s, f = r["ID"].rsplit("_", 1)
        d[s][int(f)] = [float(r["x"]), float(r["y"]), float(r["width"]), float(r["height"])]
    return d


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
    ap.add_argument("--src", action="append", required=True,
                    help="來源 CSV，可重複；順序即優先序（事前固定）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--src-max-run", type=int, default=2)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    base = load(a.base)
    srcs = [(p, load(p)) for p in a.src]
    RB = {s: runs(base[s]) for s in base}
    RS = [{s: runs(d[s]) for s in d} for _, d in srcs]

    out = {s: {f: list(b) for f, b in fm.items()} for s, fm in base.items()}
    layer = defaultdict(int)
    n = tot = miss = 0
    for s in base:
        for f in sorted(base[s]):
            tot += 1
            if RB[s][f] < a.K:
                continue
            for li, (p, d) in enumerate(srcs):
                if f in d.get(s, {}) and RS[li][s][f] < a.src_max_run:
                    out[s][f] = list(d[s][f])
                    layer[p] += 1
                    n += 1
                    break
            else:
                miss += 1

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ID", "x", "y", "width", "height"])
        for r in csv.DictReader(open(a.base)):
            s, f = r["ID"].rsplit("_", 1)
            w.writerow([r["ID"]] + [f"{v:g}" for v in out[s][int(f)]])

    print(f"K={a.K}｜替換 {n} 幀 ({n/tot:.2%})｜全來源皆凍結而放棄 {miss} 幀")
    for i, (p, _) in enumerate(srcs):
        print(f"  第{i+1}層 {p.split('/')[-1]:<32}{layer[p]:>5} 幀")
    if a.report:
        json.dump({"K": a.K, "n_replaced": n, "n_total": tot, "n_missed": miss,
                   "by_layer": {p.split("/")[-1]: layer[p] for p, _ in srcs}},
                  open(a.report, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
