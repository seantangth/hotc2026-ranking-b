#!/usr/bin/env python3
"""兩份 submission 的逐序列 delta 分佈對照（D033 硬性驗收工具）。

D033(b)：一切方法比較必須看**逐序列 delta 分佈**，不能只看 pooled——
T2 屍檢實證 pooled 純量會掩蓋雙向震盪（nir-turkey +0.537 與 nir-bracelet −0.442 互抵）；
E03 的「廣泛退步」與 E13 的「高方差對症」在 pooled 上長得像，delta 分佈才分得出來。

用法：
  python3.14 -m hsot.compare_runs BASE.csv NEW.csv GT.csv --seqs val_split_v1.txt
  python3.14 -m hsot.compare_runs BASE.csv NEW.csv GT.csv --seqs S.txt --names E02 E15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .eval import evaluate

MODALITIES = ("vis", "nir", "rednir")


def modality(seq: str) -> str:
    for m in ("rednir", "nir", "vis"):  # rednir 須先於 nir
        if seq.startswith(m + "-"):
            return m
    return "unknown"


def bootstrap_delta(common: list[str], d: dict, n_by_seq: dict,
                    n_boot: int = 2000, seed: int = 42):
    """delta 的 **cluster bootstrap**（重抽單位＝序列，不是幀）。

    為何以序列為單位：同一序列內的幀高度相關（追蹤是遞迴的，跟丟會連續數百幀），
    以幀重抽會把有效樣本數灌水數十倍、CI 假性變窄。D040 的「本地 SE ±0.023」
    正是序列級的量測噪聲——要與它對話，重抽單位必須一致。

    回傳 (pooled_delta 分佈, seq_mean_delta 分佈)，各長 n_boot。
    """
    rng = np.random.default_rng(seed)
    dv = np.array([d[s] for s in common], float)
    w = np.array([n_by_seq[s] for s in common], float)
    idx = rng.integers(0, len(common), size=(n_boot, len(common)))
    pooled = (dv[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
    return pooled, dv[idx].mean(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_csv")
    ap.add_argument("new_csv")
    ap.add_argument("gt_csv")
    ap.add_argument("--seqs", help="限定序列清單檔")
    ap.add_argument("--names", nargs=2, default=["BASE", "NEW"])
    ap.add_argument("--stable-thresh", type=float, default=0.80,
                    help="BASE AUC ≥ 此值視為「全程穩定精準」組，該組不得退步")
    ap.add_argument("--hard-thresh", type=float, default=0.50,
                    help="BASE AUC < 此值視為「難序列」組＝核心病灶（跟丟後回不來 21/65、小目標）")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap 重抽次數（0＝關閉）。重抽單位＝序列（cluster bootstrap）")
    args = ap.parse_args()

    seqs = None
    if args.seqs:
        seqs = [s.strip() for s in Path(args.seqs).read_text().split() if s.strip()]

    A_NAME, B_NAME = args.names
    a = evaluate(args.base_csv, args.gt_csv, seqs)
    b = evaluate(args.new_csv, args.gt_csv, seqs)

    common = sorted(set(a["per_seq"]) & set(b["per_seq"]))
    only_a = sorted(set(a["per_seq"]) - set(b["per_seq"]))
    only_b = sorted(set(b["per_seq"]) - set(a["per_seq"]))
    if only_a or only_b:
        print(f"⚠️ 序列集合不一致：只在 {A_NAME} {len(only_a)} 支、只在 {B_NAME} {len(only_b)} 支"
              f"（以下只比對共有的 {len(common)} 支）")
        if only_a[:3]:
            print(f"   只在 {A_NAME}: {only_a[:3]}")
        if only_b[:3]:
            print(f"   只在 {B_NAME}: {only_b[:3]}")

    d = {s: b["per_seq"][s]["auc"] - a["per_seq"][s]["auc"] for s in common}
    dv = np.array(list(d.values()))

    print(f"\n{'':22s}{A_NAME:>10s}{B_NAME:>10s}{'Δ':>10s}")
    print(f"{'pooled AUC':22s}{a['pooled']['auc']:>10.5f}{b['pooled']['auc']:>10.5f}"
          f"{b['pooled']['auc'] - a['pooled']['auc']:>+10.5f}")
    print(f"{'seq-mean AUC':22s}{a['seq_mean_auc']:>10.5f}{b['seq_mean_auc']:>10.5f}"
          f"{b['seq_mean_auc'] - a['seq_mean_auc']:>+10.5f}")
    print(f"{'DP@20':22s}{a['pooled']['dp20']:>10.5f}{b['pooled']['dp20']:>10.5f}"
          f"{b['pooled']['dp20'] - a['pooled']['dp20']:>+10.5f}")
    print(f"{'CLE (越低越好)':22s}{a['pooled']['cle']:>10.2f}{b['pooled']['cle']:>10.2f}"
          f"{b['pooled']['cle'] - a['pooled']['cle']:>+10.2f}")

    win = int((dv > 0.001).sum()); lose = int((dv < -0.001).sum())
    print(f"\n逐序列 delta（n={len(dv)}）：改善 {win} / 退步 {lose} / 持平 {len(dv)-win-lose}")
    print(f"  中位 {np.median(dv):+.4f} | 平均 {dv.mean():+.4f} | 標準差 {dv.std():.4f}"
          f" | 最佳 {dv.max():+.4f} | 最差 {dv.min():+.4f}")

    print("\n按模態：")
    for m in MODALITIES:
        sub = [d[s] for s in common if modality(s) == m]
        if not sub:
            continue
        sv = np.array(sub)
        print(f"  {m:8s} n={len(sv):3d}  Δ中位 {np.median(sv):+.4f}  Δ平均 {sv.mean():+.4f}"
              f"  改善 {int((sv>0.001).sum()):2d} / 退步 {int((sv<-0.001).sum()):2d}")

    order = sorted(common, key=lambda s: d[s])
    print(f"\nTop {args.top} 退步：")
    for s in order[:args.top]:
        print(f"  {s:32s} {a['per_seq'][s]['auc']:.4f} → {b['per_seq'][s]['auc']:.4f}  {d[s]:+.4f}")
    print(f"Top {args.top} 改善：")
    for s in order[::-1][:args.top]:
        print(f"  {s:32s} {a['per_seq'][s]['auc']:.4f} → {b['per_seq'][s]['auc']:.4f}  {d[s]:+.4f}")

    # D033：穩定組硬性檢查——本來就跑得好的序列不該被新方法弄壞
    stable = [s for s in common if a["per_seq"][s]["auc"] >= args.stable_thresh]
    if stable:
        sv = np.array([d[s] for s in stable])
        worst = stable[int(np.argmin(sv))]
        n_bad = int((sv < -0.01).sum())
        print(f"\n穩定組（{A_NAME} AUC ≥ {args.stable_thresh}，n={len(stable)}）："
              f"最壞 {sv.min():+.4f}（{worst}）、退步>0.01 者 {n_bad} 支")
        if n_bad > len(stable) * 0.2:
            print(f"  🚩 穩定組 {n_bad}/{len(stable)} 支明顯退步 = E03 式廣泛污染的特徵，非高方差對症")

    # 難序列組：新方法若有價值，價值應落在這裡（S4 小目標 / S5 跟丟後回不來）。
    # 與穩定組併看即可分辨三種型態：全面贏（真突破）／難序列贏但穩定組賠（需混合或移植）
    # ／普遍小輸但少數大贏撐住 pooled（假象，D033 要擋的就是這種）。
    hard = [s for s in common if a["per_seq"][s]["auc"] < args.hard_thresh]
    if hard:
        hv = np.array([d[s] for s in hard])
        n_frames_hard = sum(a["per_seq"][s]["n"] for s in hard)
        print(f"\n難序列組（{A_NAME} AUC < {args.hard_thresh}，n={len(hard)}，"
              f"{n_frames_hard:,} 幀 = {n_frames_hard/a['pooled']['n_frames']*100:.0f}% 權重）：")
        print(f"  Δ中位 {np.median(hv):+.4f} | Δ平均 {hv.mean():+.4f}"
              f" | 改善 {int((hv>0.001).sum())} / 退步 {int((hv<-0.001).sum())}"
              f" | 最佳 {hv.max():+.4f} | 最差 {hv.min():+.4f}")
        if hv.mean() > 0.01 and len(hard) >= 5:
            print("  💡 難序列組明顯改善 → 若穩定組同時退步，考慮「取兩者之長」而非二選一")

    # bootstrap CI（D040：本地 val 的序列級量測噪聲 SE ±0.023 大於多數方法的效果量
    # → 沒有 CI 的 pooled 差值不可當增益證據）
    if args.n_boot > 0 and len(common) >= 3:
        n_by_seq = {s: a["per_seq"][s]["n"] for s in common}
        bp, bs = bootstrap_delta(common, d, n_by_seq, args.n_boot)
        lo, hi = np.percentile(bp, [2.5, 97.5])
        slo, shi = np.percentile(bs, [2.5, 97.5])
        print(f"\nBootstrap 95% CI（cluster by sequence, n_boot={args.n_boot}）：")
        print(f"  pooled Δ    {np.mean(bp):+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  SE {bp.std():.4f}")
        print(f"  seq-mean Δ  {np.mean(bs):+.4f}  CI [{slo:+.4f}, {shi:+.4f}]  SE {bs.std():.4f}")
        if lo <= 0 <= hi:
            print("  ⚠️ pooled CI 跨 0 ⇒ **與零效果相容**，不可作為增益證據（D040：LB 才是裁判）")
        else:
            print(f"  ✅ pooled CI 不跨 0（方向：{'改善' if lo > 0 else '退步'}）——"
                  "仍須注意這是本地 val，分佈外縮水風險見 D037")

    net = b["pooled"]["auc"] - a["pooled"]["auc"]
    print(f"\n判定：pooled {'淨增' if net > 0 else '淨退'} {net:+.5f}", end="")
    print(f" | 校準外插 LB ≈ {b['pooled']['auc'] - 0.024:.5f}"
          f"（高分段偏移 −0.024，僅對分佈無關方法成立——D037）")


if __name__ == "__main__":
    main()
