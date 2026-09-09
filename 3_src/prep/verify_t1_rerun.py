#!/usr/bin/env python3
"""track_t1.py 重跑驗收:與 local_T1_all65 基準的回歸不變量檢查(task #2)。

用法:python3.14 3_src/prep/verify_t1_rerun.py <新 submission.csv>
通過標準:pooled 差 <0.002 且逐序列 |delta| 最大 <0.005(CUDA 非確定性容忍;
超標即列出偏差序列供 debug)。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsot import eval as ev  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "5_outputs/t2_bench/local_T1_all65.csv"
GT = ROOT / "1_data/raw/2026training.csv"
SPLIT = ROOT / "1_data/val_split_v1.txt"


def main() -> None:
    new_csv = Path(sys.argv[1])
    seqs = [s.strip() for s in SPLIT.read_text().split() if s.strip()]
    r_new = ev.evaluate(new_csv, GT, seqs)
    r_old = ev.evaluate(BASELINE, GT, seqs)
    p_new, p_old = r_new["pooled"]["auc"], r_old["pooled"]["auc"]
    print(f"新跑 pooled {p_new:.5f} | 基準 pooled {p_old:.5f} | Δ {p_new - p_old:+.5f}")
    deltas = {s: r_new["per_seq"][s]["auc"] - r_old["per_seq"][s]["auc"]
              for s in seqs if s in r_new["per_seq"] and s in r_old["per_seq"]}
    worst = sorted(deltas.items(), key=lambda kv: -abs(kv[1]))[:8]
    print("逐序列 |Δ| 最大 8 筆:")
    for s, d in worst:
        print(f"  {s}: {d:+.5f}")
    ok = abs(p_new - p_old) < 0.002 and all(abs(d) < 0.005 for d in deltas.values())
    print("✅ 驗收通過:track_t1.py 忠實復刻 E02 pipeline" if ok
          else "🚨 驗收未過:偏差超出 CUDA 非確定性容忍,需 debug 上列序列")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
