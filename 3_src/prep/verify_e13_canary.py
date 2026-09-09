#!/usr/bin/env python3
"""E13 canary gate 判定:HiM2SAM 25 序列 vs T1 基準逐序列對照(D032/D033)。

用法:python3.14 3_src/prep/verify_e13_canary.py <e13 submission.csv>
判準:pooled 淨增 且 改善數>退步數 且 穩定組(canary 內 10 支高分序列)最壞 delta > -0.01。
E03 式廣泛退步(改善<退步)→ 砍,備案 SAM2Long。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsot import eval as ev  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "5_outputs/t2_bench/local_T1_all65.csv"
GT = ROOT / "1_data/raw/2026training.csv"
CANARY = ROOT / "1_data/canary_e13.txt"
STABLE_10 = {"nir-redbag", "nir-glass_cup", "nir-balloon", "vis-officefan2", "vis-receipts3",
             "vis-glass2", "vis-mirror_egg", "rednir-officefan2", "rednir-glass2", "rednir-receipts3"}


def main() -> None:
    e13_csv = Path(sys.argv[1])
    seqs = [s.strip() for s in CANARY.read_text().split() if s.strip()]
    r_new = ev.evaluate(e13_csv, GT, seqs)
    r_old = ev.evaluate(BASELINE, GT, seqs)
    p_new, p_old = r_new["pooled"]["auc"], r_old["pooled"]["auc"]
    deltas = {s: r_new["per_seq"][s]["auc"] - r_old["per_seq"][s]["auc"]
              for s in seqs if s in r_new["per_seq"] and s in r_old["per_seq"]}
    imp = sum(1 for d in deltas.values() if d > 0.002)
    reg = sum(1 for d in deltas.values() if d < -0.002)
    worst_stable = min((deltas[s] for s in STABLE_10 if s in deltas), default=0.0)

    print(f"=== E13 canary gate({len(deltas)} 序列)===")
    print(f"pooled:HiM2SAM {p_new:.5f} vs T1 {p_old:.5f} | Δ {p_new - p_old:+.5f}")
    print(f"改善 {imp} / 退步 {reg}(|Δ|>0.002)| 穩定組最壞 Δ {worst_stable:+.5f}")
    print("\n逐序列 Δ(依增益排序):")
    for s, d in sorted(deltas.items(), key=lambda kv: -kv[1]):
        tag = "穩定組" if s in STABLE_10 else "失分組"
        print(f"  {s:<26} {d:+.5f}  [{tag}] (T1 {r_old['per_seq'][s]['auc']:.3f} → {r_new['per_seq'][s]['auc']:.3f})")

    ok = (p_new > p_old) and (imp > reg) and (worst_stable > -0.01)
    print(f"\n{'✅ GATE PASS → 進全量 65 確認' if ok else '❌ GATE FAIL → 砍 HiM2SAM(備案 SAM2Long)或收工'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
