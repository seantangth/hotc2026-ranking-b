#!/usr/bin/env python3
"""Ranking B 座標慣例判別器（無 GT、無 LB、因果）——決定新資料集要不要套 +1px 校正。

## 為什麼需要這支
座標校正（上/左各 +1px）是我方最大的單一增益（**+0.0142**，D067），
但它押的是「新資料集的 GT 標註慣例與 test 相同」這個假設。
若 9/7 的 Ranking B 資料**沒有**該偏移，套用就是**倒扣 0.0142**。
而 Ranking B **無 GT、無 LB 回饋**，D062 已判定原本規劃的判別程序（比 empty/frozen 率）失效。

## 原理
`init_rect.txt`（官方第一幀框）**就是新資料集唯一可得的 GT 樣本**。
若該資料集的 GT 帶上/左各 +1px 外擴，則 init box 會比「tracker 自己分割出的同一物體」
在**寬與高上各大 1px**（外擴一邊 ⇒ 該維度 +1）。
⇒ 比較 `init box` 與該序列**前段穩定期**的 tracker 框尺寸中位數，即可讀出慣例，
   全程只用官方 init box ＋ 我方自己的輸出。

## 已知答案上的校準（見 --calibrate）
| 資料集 | 已知的左緣偏移 | 已知的上緣偏移 | Δw 中位 | Δh 中位 |
|---|---|---|---|---|
| val（train GT） | 無 | 有（弱，+0.0028） | +0.00 | +1.00 |
| test | 有（+0.0051） | 有（強，+0.0091） | +1.00 | +1.00 |
⇒ Δw 能分辨左緣、Δh 能分辨上緣，**方向與已知事實一致**。
"""
import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

ROOT = __file__.rsplit("/3_src/", 1)[0]
HEAD_FRAMES = 20        # 只用前 N 幀＝目標尺寸的穩定期（越後面目標尺寸變化越大，訊號被稀釋）
MIN_FRAMES = 20         # 序列太短則棄權


def load(p):
    d = defaultdict(dict)
    for r in csv.DictReader(open(p)):
        s, f = r["ID"].rsplit("_", 1)
        d[s][int(f)] = np.array([float(r["x"]), float(r["y"]), float(r["width"]), float(r["height"])])
    return d


def probe(D, head=HEAD_FRAMES):
    """回傳 (dw_median, dh_median, 逐序列票數, 棄權數)。逐序列投票比全體中位穩健。"""
    dw, dh = [], []
    skipped = 0
    for s, fm in D.items():
        ks = sorted(fm)
        if len(ks) < MIN_FRAMES:
            skipped += 1
            continue
        init = fm[ks[0]]
        seg = [fm[k] for k in ks[1:head + 1]]
        # 排除凍結幀（與前一幀完全相同）——它們只是 init 的複製，會把訊號歸零
        alive = [b for i, b in enumerate(seg) if i == 0 or not np.array_equal(b, seg[i - 1])]
        if len(alive) < 5:
            skipped += 1
            continue
        a = np.array(alive)
        dw.append(init[2] - np.median(a[:, 2]))
        dh.append(init[3] - np.median(a[:, 3]))
    return np.array(dw), np.array(dh), skipped


def report(name, dw, dh, skipped):
    print(f"\n=== {name} ===（有效 {len(dw)} 支、棄權 {skipped} 支）")
    for nm, v in (("Δw = init.w − median(pred.w)  [左/右緣]", dw),
                  ("Δh = init.h − median(pred.h)  [上/下緣]", dh)):
        pos = float(np.mean(v >= 0.5))
        print(f"  {nm}: 中位 {np.median(v):+.2f}px｜平均 {v.mean():+.2f}"
              f"｜**≥+0.5 的序列 {pos:.0%}**")
    return float(np.median(dw)), float(np.median(dh)), float(np.mean(dw >= 0.5)), float(np.mean(dh >= 0.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", help="待判別資料集的【未校正】tracker 輸出 CSV")
    ap.add_argument("--calibrate", action="store_true", help="在已知答案的 val/test 上跑校準")
    ap.add_argument("--head", type=int, default=HEAD_FRAMES)
    a = ap.parse_args()

    if a.calibrate:
        cal = [("val（train GT：左緣無偏移、上緣弱偏移）",
                f"{ROOT}/5_outputs/e15_sam3_20260806/submission_val65.csv"),
               ("test（左緣有、上緣強）",
                f"{ROOT}/5_outputs/submissions/sub_v023_cropwiden55.csv")]
        out = {}
        for nm, p in cal:
            out[nm] = report(nm, *probe(load(p), a.head))
        print("\n【辨別力】test 相對 val 的位移：")
        (vw, vh, vpw, vph), (tw, th, tpw, tph) = out[cal[0][0]], out[cal[1][0]]
        print(f"  Δw 中位 {vw:+.2f} → {tw:+.2f}（差 {tw-vw:+.2f}）｜票率 {vpw:.0%} → {tpw:.0%}（差 {tpw-vpw:+.0%}）")
        print(f"  Δh 中位 {vh:+.2f} → {th:+.2f}（差 {th-vh:+.2f}）｜票率 {vph:.0%} → {tph:.0%}（差 {tph-vph:+.0%}）")
        print("\n  ⚠️ 若 Δw 的辨別力（左緣，兩資料集已知不同）不明顯，本判別器對【左緣】無效，")
        print("     只能用於上緣，或整體降級為輔助證據。")
        return

    if not a.pred:
        ap.error("需要 --pred 或 --calibrate")
    dw, dh, sk = probe(load(a.pred), a.head)
    mw, mh, pw, ph = report("待判別資料集", dw, dh, sk)
    print("\n【建議】（門檻依 --calibrate 的校準結果設定，見判決書）")
    print(f"  上緣校正：{'建議套用' if mh >= 0.5 else '建議不套'}（Δh 中位 {mh:+.2f}、票率 {ph:.0%}）")
    print(f"  左緣校正：{'建議套用' if mw >= 0.5 else '建議不套'}（Δw 中位 {mw:+.2f}、票率 {pw:.0%}）")
    print("  ⚠️ 這是輔助證據不是判決——決策權在人，且須連同「兩資料集的先驗同源程度」一起判斷。")


if __name__ == "__main__":
    main()
