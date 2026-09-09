#!/usr/bin/env python3
"""E37：跨【底座】仲裁的 val 前測（$0）——攻「活著但追錯」的 63.5%

事前判準（執行前寫死，與 E34b 跨模態版**完全相同的三道門檻**，才能直接對照）：
  (i)   不一致幀（兩底座框 IoU<0.3）中「恰好單側跟丟」（一邊 IoU≥0.5、一邊 <0.3）的占比 ≥ 50%
  (ii)  單一主訊號（累積凍結率，因果）的方向準確率 ≥ 65%
  (iii) 對照幀（兩底座一致，IoU≥0.5）的誤觸率 < 5%
三項皆過 ⇒ 設計 test 仲裁版提交；任一不過 ⇒ 仲裁路線整條關閉（跨模態已死、跨底座也死）。

對照組（E34b 讀數，跨模態）：(i) 15.8% ❌ (ii) 69.4% ✅ (iii) 18.9% ❌
本測若 (i) 顯著高於 15.8%，即證明「失敗獨立性」確實把仲裁的前提改變了。
"""
import csv
from collections import defaultdict, deque

import numpy as np
import pandas as pd

ROOT = "/Users/seantang/Desktop/Sean/The_Nexus/1_Projects/WHISPERS_2026_HyperSOT"
A_CSV = f"{ROOT}/5_outputs/e15_sam3_20260806/submission_val65.csv"      # 主線代理 SAM3
B_CSV = f"{ROOT}/5_outputs/e02_kfreset_20260806/submission_val65.csv"   # SAMURAI SAM2.1-L


def load(p):
    d = defaultdict(dict)
    for r in csv.DictReader(open(p)):
        s, f = r["ID"].rsplit("_", 1)
        d[s][int(f)] = (float(r["x"]), float(r["y"]), float(r["width"]), float(r["height"]))
    return d


def iou(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0


gt = pd.read_csv(f"{ROOT}/1_data/raw/2026training.csv")
G = {r.ID: (float(r.x), float(r.y), float(r.width), float(r.height)) for r in gt.itertuples()}
A, B = load(A_CSV), load(B_CSV)
seqs = sorted(set(A) & set(B))

n_incons = n_single = n_both_bad = 0
n_ctrl = n_ctrl_fire = 0
n_arb = n_arb_ok = 0
# 額外診斷（非判準）：仲裁若完美（oracle）能拿多少
oracle_gain = 0.0
n_tot = 0
for s in seqs:
    fs = sorted(set(A[s]) & set(B[s]))
    fa = fb = 0
    pa = pb = None
    for i, f in enumerate(fs):
        key = f"{s}_{f}"
        g = G.get(key)
        if g is None:
            continue
        n_tot += 1
        a, b = A[s][f], B[s][f]
        cons = iou(a, b)
        ia, ib = iou(a, g), iou(b, g)
        rate_a, rate_b = fa / max(i, 1), fb / max(i, 1)
        # 只看「兩邊都活著」的幀——凍結幀已由 E35 的 splice 處理
        alive = (pa is None or a != pa) and (pb is None or b != pb)
        if alive:
            if cons < 0.3:
                n_incons += 1
                good_a, good_b = ia >= 0.5, ib >= 0.5
                if good_a != good_b:
                    n_single += 1
                    if abs(rate_a - rate_b) > 1e-9:
                        n_arb += 1
                        pick_a = rate_a < rate_b
                        n_arb_ok += int((pick_a and good_a) or ((not pick_a) and good_b))
                elif not good_a and not good_b:
                    n_both_bad += 1
                oracle_gain += max(0.0, ib - ia)
            elif cons >= 0.5:
                n_ctrl += 1
                if abs(rate_a - rate_b) > 0.1:
                    n_ctrl_fire += 1
        if pa is not None and a == pa:
            fa += 1
        if pb is not None and b == pb:
            fb += 1
        pa, pb = a, b

c1 = n_single / n_incons if n_incons else float("nan")
c2 = n_arb_ok / n_arb if n_arb else float("nan")
c3 = n_ctrl_fire / n_ctrl if n_ctrl else float("nan")
print(f"val {len(seqs)} 支序列、{n_tot} 幀（只計兩底座皆活著的幀）")
print(f"(i)   不一致幀 {n_incons}｜恰好單側跟丟 {n_single} = {c1:.1%}（門檻 ≥50%；跨模態版 15.8%）"
      f"｜兩側都錯 {n_both_bad}")
print(f"(ii)  可仲裁 {n_arb}｜凍結率訊號方向正確 {n_arb_ok} = {c2:.1%}（門檻 ≥65%）")
print(f"(iii) 對照幀 {n_ctrl}｜誤觸 {n_ctrl_fire} = {c3:.1%}（門檻 <5%）")
print(f"\n診斷（非判準）｜oracle 上界：不一致幀上永遠選對 ⇒ pooled +{oracle_gain/n_tot:.5f}")
ok = (c1 >= 0.5) and (c2 >= 0.65) and (c3 < 0.05)
print(f"⇒ {'✅ 三項全過 ⇒ 設計 test 仲裁版' if ok else '❌ 未全過 ⇒ 仲裁路線整條關閉（跨模態＋跨底座皆死）'}")
