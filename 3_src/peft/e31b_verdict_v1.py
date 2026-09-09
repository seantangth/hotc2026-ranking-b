#!/usr/bin/env python3
"""E31b 延長驗證判準——Sean 08-12 選「延長腿 B 再驗一次」時，判準已寫在選項描述內。

【要回答的唯一問題】腿 B（lr=1e-4）在 8 epochs 時的 19.16% 「下降」，
是**真的在學**，還是**大 LR 把起點推高後的擾動回歸**？
E31 的三條讀數指向後者（兩腿終點差 0.2%、19.16% 幾乎全來自 ep0、逐筆秩相關 0.954），
但那是事後分析 ⇒ 本實驗用**事前判準**直接裁決，不再落 WEAK 帶。

【事前判準（Sean 授權時的原文）】
  「若擾動回歸判讀正確，d 會縮到 <10% 且終點仍 ~0.51；若真在學，終點應明顯低於 0.50」

【操作化（執行前寫死）】基準 ＝ 腿 A 8-epoch 的最後 2 epochs 中位數（0.5400），
  它是「已知沒有被優化」的參考點。腿 B 跑 16 epochs 後：

  主判準 — 終點比較（不受起點影響）：
    腿 B 最後 2 epochs 中位數 vs 基準 0.5400，相對下降 g：
      g >= 10%（即 <= 0.4860）  ⇒ ✅ GO：真的在學 ⇒ 訓練線重開，列新決策編號
      g <  10%                  ⇒ 🚨 NO-GO：擾動回歸確認 ⇒ 訓練線定案關閉
  輔助（記錄，不當閘門）：
    · d（前 1/4 vs 後 1/4）——擾動回歸預測 d 會**縮小**（ep0 在 16 epochs 中權重減半）
    · Mann-Whitney U 檢定 p 值（腿 B 末 2 ep vs 腿 A 末 2 ep）
    · 逐 epoch 中位數序列（看有無趨勢）

【取值】一律 instantaneous 中位數（E31 已證均值會被尖峰主導而產生假陽性）。
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

PAT = re.compile(
    r"Train Epoch: \[(\d+)\]\[\s*(\d+)/(\d+)\].*Losses/train_all_loss: ([\d.eE+-]+) \(")

BASELINE = 0.5400      # 腿 A 8-epoch 最後 2 epochs 中位數（E31 實測）
G_PASS = 0.10          # 相對下降 >=10% ⇒ 真在學


def load(path: Path):
    ep, inst = [], []
    for line in path.read_text(errors="replace").splitlines():
        m = PAT.search(line)
        if m:
            ep.append(int(m.group(1)))
            inst.append(float(m.group(4)))
    return np.array(ep), np.array(inst)


def mannwhitney_p(x, y) -> float:
    """雙尾 Mann-Whitney U 的常態近似（避開 scipy 依賴）。"""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    allv = np.concatenate([x, y])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 平手取平均秩
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sd == 0:
        return float("nan")
    z = (u1 - mu) / sd
    return float(math.erfc(abs(z) / math.sqrt(2)))   # 雙尾常態近似


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg-b-ext", required=True, help="腿 B 延長版（16 epochs）console log")
    ap.add_argument("--leg-a-ref", default=None, help="腿 A 8-epoch console log（基準來源）")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    ep, inst = load(Path(a.leg_b_ext))
    if len(ep) == 0:
        raise SystemExit("🚨 腿 B 延長版 log 內找不到 loss 行——實驗未成立")
    n_ep = int(ep.max()) + 1
    per_ep = [float(np.median(inst[ep == e])) for e in range(n_ep)]
    last2 = inst[ep >= n_ep - 2]
    endpoint = float(np.median(last2))

    baseline, ref_last2 = BASELINE, None
    if a.leg_a_ref and Path(a.leg_a_ref).exists():
        ra, ia = load(Path(a.leg_a_ref))
        if len(ra):
            ref_last2 = ia[ra >= ra.max() - 1]
            baseline = float(np.median(ref_last2))

    g = (baseline - endpoint) / baseline
    q = n_ep / 4.0
    first_q = float(np.median(inst[ep < q]))
    last_q = float(np.median(inst[ep >= 3 * q]))
    d = (first_q - last_q) / first_q

    go = g >= G_PASS
    verdict = "GO" if go else "NO-GO"
    detail = (f"腿 B 終點 {endpoint:.4f} vs 基準 {baseline:.4f} ⇒ 相對下降 {100*g:.2f}%"
              + ("（>=10%）⇒ ✅ 真的在學：訓練線重開，列新決策編號（G1/D038/diag19 照舊）"
                 if go else
                 "（<10%）⇒ 🚨 擾動回歸確認：訓練線定案關閉"))

    p = mannwhitney_p(last2, ref_last2) if ref_last2 is not None else float("nan")

    print(f"【E31b 延長驗證】腿 B lr=1e-4，{n_ep} epochs／{len(ep)} 筆\n")
    print("逐 epoch 中位數：")
    for i in range(0, n_ep, 8):
        print("  ep%-2d–%-2d: " % (i, min(i + 7, n_ep - 1))
              + "  ".join(f"{v:.3f}" for v in per_ep[i:i + 8]))
    print(f"\n{'項目':<28}{'值':>12}")
    print(f"{'基準（腿 A 8ep 末 2 ep 中位）':<28}{baseline:>12.4f}")
    print(f"{'腿 B 終點（末 2 ep 中位）':<28}{endpoint:>12.4f}")
    print(f"{'主判準 g（相對下降）':<28}{100*g:>11.2f}%   門檻 >= {100*G_PASS:.0f}%")
    print(f"{'輔助 d（前1/4 vs 後1/4）':<28}{100*d:>11.2f}%   E31 的 8ep 版為 19.16%")
    print(f"{'Mann-Whitney p（末2ep 兩腿）':<28}{p:>12.4f}")
    print(f"\n【判決】{verdict} — {detail}")

    Path(a.json).write_text(json.dumps(
        {"verdict": verdict, "detail": detail, "n_epochs": n_ep,
         "endpoint": endpoint, "baseline": baseline, "g": g,
         "d_quarter": d, "first_quarter": first_q, "last_quarter": last_q,
         "per_epoch_median": per_ep, "mannwhitney_p": p,
         "threshold": {"g_pass": G_PASS, "baseline_default": BASELINE}},
        indent=1, ensure_ascii=False, default=float))
    print(f"產出：{a.json}")


if __name__ == "__main__":
    main()
