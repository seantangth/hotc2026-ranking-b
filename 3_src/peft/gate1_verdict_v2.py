#!/usr/bin/env python3
"""gate1_verdict_v2 — D068 Gate 0-C／Gate 1 判準工具（三模式）。

【事前判準（Sean 08-12 授權選項 A 時已寫入任務 #41；出處 report_audit_training.md §5）】
  mode=floor（G0-C，lr=0 雙腿）：
      median(事件腿 instantaneous) / median(對照腿) >= 5.0 ⇒ PASS
      （單位陷阱：舊 mask-loss 的 2.5/0.5 絕對數字不適用投影 loss ⇒ 對照腿當場實測地板，
        比值判準免疫單位換算——advisor 08-13 修正 #3）
  mode=verdict（Gate 1，4 epochs）：
      末 epoch instantaneous 中位 vs 首 epoch 中位，相對下降 >= 20%
      且 Mann-Whitney（末 epoch vs 首 epoch）p < 0.05  ⇒ GO（目標函數可被優化）
      否則 NO-GO ⇒ 訓練線升級「架構層不可訓」永久關閉。
      （不用 E31b 已證對分段敏感的 d 指標；中位數防重尾——D067(f) 教訓 #1/#2）
  mode=selftest（開機前擋門，D067(f) 鐵律）：
      對已知持平的負例 log 跑 verdict ⇒ 必須 NO-GO；
      對負例 × linspace(1.0, 0.55)（合成 45% 線性下降）⇒ 必須 GO；
      兩者皆中才寫 marker（launch 腳本 preflight 檢查 marker 才准開機）。
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

FLOOR_RATIO_PASS = 5.0
DROP_PASS = 0.20
P_PASS = 0.05


def load(path: Path):
    ep, inst = [], []
    for line in path.read_text(errors="replace").splitlines():
        m = PAT.search(line)
        if m:
            ep.append(int(m.group(1)))
            inst.append(float(m.group(4)))
    return np.array(ep), np.array(inst)


def mannwhitney_p(x, y) -> float:
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    allv = np.concatenate([x, y])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    mu, sd = n1 * n2 / 2, math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sd == 0:
        return float("nan")
    return float(math.erfc(abs((u1 - mu) / sd) / math.sqrt(2)))


def verdict_from(ep: np.ndarray, inst: np.ndarray) -> dict:
    if len(ep) == 0:
        return {"verdict": "INVALID", "detail": "log 無 loss 行"}
    n_ep = int(ep.max()) + 1
    first, last = inst[ep == 0], inst[ep == n_ep - 1]
    m0, mL = float(np.median(first)), float(np.median(last))
    drop = (m0 - mL) / m0 if m0 else float("nan")
    p = mannwhitney_p(last, first)
    go = (drop >= DROP_PASS) and (p < P_PASS)
    per_ep = [float(np.median(inst[ep == e])) for e in range(n_ep)]
    return {"verdict": "GO" if go else "NO-GO", "n_epochs": n_ep,
            "first_ep_median": m0, "last_ep_median": mL,
            "rel_drop": drop, "mannwhitney_p": p, "per_epoch_median": per_ep,
            "thresholds": {"drop_pass": DROP_PASS, "p_pass": P_PASS},
            "detail": (f"首ep中位 {m0:.4f} → 末ep中位 {mL:.4f}（降 {100*drop:.1f}%，"
                       f"門檻 ≥{100*DROP_PASS:.0f}%）；MW p={p:.4g}（門檻 <{P_PASS}）")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("floor", "verdict", "selftest"), required=True)
    ap.add_argument("--log", help="verdict：Gate 1 console log")
    ap.add_argument("--events-log", help="floor：事件腿 console log")
    ap.add_argument("--control-log", help="floor：對照腿 console log")
    ap.add_argument("--neg-log", help="selftest：已知持平的負例 log（如 Phase 1／E31b）")
    ap.add_argument("--marker", help="selftest 通過時寫出的 marker 檔")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    if a.mode == "floor":
        _, ie = load(Path(a.events_log))
        _, ic = load(Path(a.control_log))
        if len(ie) == 0 or len(ic) == 0:
            raise SystemExit("🚨 floor：任一腿 log 無 loss 行")
        me, mc = float(np.median(ie)), float(np.median(ic))
        ratio = me / mc if mc > 0 else float("inf")
        ok = ratio >= FLOOR_RATIO_PASS
        out = {"verdict": "PASS" if ok else "FAIL", "events_median": me,
               "control_median": mc, "ratio": ratio,
               "threshold": FLOOR_RATIO_PASS,
               "detail": f"事件腿中位 {me:.4f} / 對照腿中位 {mc:.4f} = {ratio:.2f}×"
                         f"（門檻 ≥{FLOOR_RATIO_PASS}×）"}
        print(f"【G0-C floor】{out['detail']} → {out['verdict']}")

    elif a.mode == "verdict":
        ep, inst = load(Path(a.log))
        out = verdict_from(ep, inst)
        print(f"【Gate 1】{out['detail']} → {out['verdict']}")
        if out.get("per_epoch_median"):
            print("逐 epoch 中位：" + "  ".join(f"{v:.4f}" for v in out["per_epoch_median"]))

    else:  # selftest
        ep, inst = load(Path(a.neg_log))
        if len(ep) == 0:
            raise SystemExit("🚨 selftest：負例 log 無 loss 行")
        neg = verdict_from(ep, inst)
        synth = inst * np.linspace(1.0, 0.55, len(inst))
        pos = verdict_from(ep, synth)
        ok = neg["verdict"] == "NO-GO" and pos["verdict"] == "GO"
        out = {"verdict": "SELFTEST_OK" if ok else "SELFTEST_FAIL",
               "negative_case": neg, "synthetic_positive": pos}
        print(f"負例（應 NO-GO）：{neg['verdict']} — {neg['detail']}")
        print(f"合成正例（應 GO）：{pos['verdict']} — {pos['detail']}")
        print(f"【selftest】{'✅ 通過' if ok else '🚨 未過——不得開機'}")
        if ok and a.marker:
            Path(a.marker).write_text("selftest ok\n")
        if not ok:
            Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
            raise SystemExit(2)

    Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(f"產出：{a.json}")


if __name__ == "__main__":
    main()
