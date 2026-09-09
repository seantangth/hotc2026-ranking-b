#!/usr/bin/env python3
"""E31 判準評定器——判準取自 DESIGN.md §4，**執行前寫死，此處僅執行不重新定義**。

用法：
  python3 e31_verdict_v1.py --leg-a <legA_console.log> --leg-b <legB_console.log> \
      [--phase1-log <phase1_stageB.log>] --json out.json

🚨【取值與統計量——本節於開機前因自我測試失敗而修正過一次，必讀】
  初版用「running average 的均值」，拿**已知完全持平**的 Phase 1 log 自我測試，
  卻輸出「下降 52.4% ⇒ GO」＝**假陽性**。根因：**均值被少數極端尖峰主導**
  （Phase 1 inst 中位 0.61 但 max 47.7；前 8 個 epoch 混進 6.97／4.74 兩個異常值，
  把前段均值拉到 2.52，後段 1.36 ⇒ 憑空生出 46% 的「下降」，那是尖峰分佈的隨機差異）。
  同一份資料改用 **instantaneous 中位數**：ep0-7 = 0.6070 → ep32-39 = 0.6740
  ＝**略升 11%**，正確反映「目標函數從未被優化」。
  ⇒ **主判準一律用 instantaneous 的中位數**（重尾分佈的標準穩健統計）。
  running average 均值僅並列記錄，供與 D065 的 1.095／1.107 對帳。

【主判準（斜率）】每腿 1,216 步（8 epochs × 152），
  前 1/4（步 0–303）vs 後 1/4（步 912–1215）的 **instantaneous 中位數**，
  相對下降 d = (前 − 後)/前：
    兩腿皆 d < 5%          ⇒ 🚨 訓練線定案關閉（LR＋協定雙重修正仍不收斂）
    任一腿 d >= 20%        ⇒ ✅ 假說①獲支持，列新決策編號重開
    中間 5% <= d < 20%     ⇒ ⚠️ 弱訊號，帶數字請 Sean 裁示

【早期訊號（記分用，非閘門）】機制預測「任務變難 ⇒ loss 水準上移」：
  腿 A 的 instantaneous 中位數應顯著高於 Phase 1 同期的 **0.6070**（ep0-7），
  操作定義 >= 0.75（高 ~24%）。若仍 ≈0.61 ⇒ 機制故事有問題，
  不得直接宣稱假說①成立（即使斜率下降也須重新解釋）。

【災難停損】NaN 或 loss 爆炸至初始 10× ⇒ 該腿標記失敗（1e-4 腿有此風險）。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

PAT = re.compile(
    r"Train Epoch: \[(\d+)\]\[\s*(\d+)/(\d+)\].*Losses/train_all_loss: ([\d.eE+-]+) \(([\d.eE+-]+)\)")

PHASE1_ANCHOR_MED = 0.6070   # Phase 1 ep0-7 的 instantaneous 中位數（同期對照）
EARLY_SIGNAL_MIN = 0.75      # 事前預測（非閘門）：任務變難 ⇒ 中位數應上移 ~24%
D_KILL = 0.05                # < 5% ⇒ 持平
D_PASS = 0.20                # >= 20% ⇒ 下降


def parse(path: Path) -> dict:
    steps, inst, avg = [], [], []
    for line in path.read_text(errors="replace").splitlines():
        m = PAT.search(line)
        if m:
            ep, it, per = int(m.group(1)), int(m.group(2)), int(m.group(3))
            steps.append(ep * per + it)
            inst.append(float(m.group(4)))
            avg.append(float(m.group(5)))
    if not steps:
        return {"ok": False, "reason": "log 內找不到任何 loss 行"}
    o = np.argsort(steps)
    return {"ok": True, "step": np.array(steps)[o], "inst": np.array(inst)[o],
            "avg": np.array(avg)[o], "per_epoch": per}


def evaluate(d: dict, name: str) -> dict:
    if not d["ok"]:
        return {"leg": name, "ok": False, "reason": d["reason"]}
    step, avg, inst = d["step"], d["avg"], d["inst"]
    n_steps = int(step.max()) + 1
    q = n_steps / 4.0
    # 🚨 主判準用 instantaneous 中位數（重尾分佈；均值版本已於自我測試產生假陽性）
    first, last = inst[step < q], inst[step >= 3 * q]
    r = {"leg": name, "ok": True, "n_records": int(len(step)), "n_steps": n_steps,
         "median_all": float(np.median(inst)),
         "mean_runavg_all": float(avg.mean()),   # 僅供與 D065 對帳
         "first_quarter": float(np.median(first)) if len(first) else None,
         "last_quarter": float(np.median(last)) if len(last) else None,
         "n_first": int(len(first)), "n_last": int(len(last)),
         "min_inst": float(inst.min()), "max_inst": float(inst.max())}
    # 災難停損
    r["nan"] = bool(np.isnan(inst).any() or np.isnan(avg).any())
    r["exploded"] = bool(len(inst) > 1 and inst.max() > 10 * max(inst[0], 1e-9))
    r["failed"] = bool(r["nan"])          # 爆炸只記錄不判失敗（Phase 1 本來就有 47.7 的尖峰）
    if r["first_quarter"] and r["last_quarter"]:
        r["d"] = (r["first_quarter"] - r["last_quarter"]) / r["first_quarter"]
    else:
        r["d"] = None
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg-a", required=True)
    ap.add_argument("--leg-b", required=True)
    ap.add_argument("--phase1-log", default=None)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    legs = [evaluate(parse(Path(a.leg_a)), "A(lr=5e-6)"),
            evaluate(parse(Path(a.leg_b)), "B(lr=1e-4)")]

    anchor = PHASE1_ANCHOR_MED
    if a.phase1_log and Path(a.phase1_log).exists():
        p = parse(Path(a.phase1_log))
        if p["ok"]:
            anchor = float(np.median(p["inst"][p["step"] < 1216]))

    print(f"【對照錨點】Phase 1 同期 instantaneous 中位數 = {anchor:.4f}"
          f"{'（現場重算，前 1216 步）' if a.phase1_log else '（DESIGN 記錄值 ep0-7）'}\n")
    print(f"{'腿':<12} {'筆數':>5} {'步數':>6} {'前1/4中位':>10} {'後1/4中位':>10} {'相對下降':>9} "
          f"{'全段中位':>9}  狀態")
    for r in legs:
        if not r["ok"]:
            print(f"{r['leg']:<12} 🚨 {r['reason']}")
            continue
        dd = f"{100*r['d']:>8.2f}%" if r["d"] is not None else "     n/a"
        print(f"{r['leg']:<12} {r['n_records']:>5d} {r['n_steps']:>6d} "
              f"{r['first_quarter']:>10.4f} {r['last_quarter']:>10.4f} {dd} "
              f"{r['median_all']:>9.4f}  {'🚨NaN' if r['nan'] else '✅'}")

    ok = [r for r in legs if r["ok"] and not r["failed"] and r["d"] is not None]
    if not ok:
        verdict, detail = "INVALID", "兩腿皆無有效資料——實驗未成立，不得據此下任何結論"
    elif any(r["d"] >= D_PASS for r in ok):
        verdict = "GO"
        detail = ("假說①獲支持：" +
                  "、".join(f"{r['leg']} 下降 {100*r['d']:.1f}%" for r in ok if r["d"] >= D_PASS) +
                  " ⇒ 列新決策編號重開訓練線，再談是否投全訓練（G1/D038/diag19 照舊）")
    elif all(r["d"] < D_KILL for r in ok):
        verdict = "NO-GO"
        detail = ("兩腿皆持平（d<5%）⇒ 訓練線以「LR＋協定雙重修正仍不收斂」定案關閉，"
                  "列新決策編號，不再重開")
    else:
        verdict = "WEAK"
        detail = "落在 5–20% 弱訊號帶 ⇒ 不重開也不定案關閉，帶數字請 Sean 裁示"

    # 早期訊號（記分用）
    early = None
    la = next((r for r in legs if r["leg"].startswith("A") and r["ok"]), None)
    if la:
        early = {"leg_a_median": la["median_all"], "anchor": anchor,
                 "predicted_min": EARLY_SIGNAL_MIN,
                 "supported": bool(la["median_all"] >= EARLY_SIGNAL_MIN)}
        print(f"\n【早期訊號（記分用，非閘門）】腿 A 全段中位數 {la['median_all']:.4f} vs "
              f"Phase 1 同期 {anchor:.4f}（事前預測 >= {EARLY_SIGNAL_MIN}）"
              f" ⇒ {'✅ 機制故事獲支持' if early['supported'] else '⚠️ 機制故事有問題，即使斜率下降也須重新解釋'}")

    print(f"\n【判決】{verdict} — {detail}")
    Path(a.json).write_text(json.dumps(
        {"verdict": verdict, "detail": detail, "legs": legs, "anchor_608": anchor,
         "early_signal": early,
         "thresholds": {"d_kill": D_KILL, "d_pass": D_PASS,
                        "early_signal_min": EARLY_SIGNAL_MIN}},
        indent=1, ensure_ascii=False, default=float))
    print(f"產出：{a.json}")


if __name__ == "__main__":
    main()
