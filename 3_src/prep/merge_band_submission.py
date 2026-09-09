#!/usr/bin/env python3
"""E-A' band×SAM3：把 band run（觸發支）的結果換入 v078 基準提交，產最終 75 支 CSV。

驗證（與 finalize_submission 的 validator 同語意，全部 fail-closed）：
  exact ID set 且順序＝sample；首幀＝init 照抄（preserve_init）；框全 finite 且 w,h>0；
  觸發支的行全部來自 band CSV、其餘行與 base 位元級相同（字串級比對）。
另產 diff 報告（band vs base 逐支：預測間 IoU 分佈＋中心位移統計）——test 無 GT，
此報告量的是「換輸入改變了多少預測」，供 LB 讀數歸因，不是對錯。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 3_src/
from train_tracker_g1.losses import box_iou_xywh  # noqa: E402


def read_csv_rows(path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows and rows[0][0].strip().lower() == "id", f"{path} 標頭異常 {rows[:1]}"
    return rows


def seq_of(frame_id: str) -> str:
    return frame_id.rsplit("_", 1)[0]


def merge(base_rows, band_rows, triggered: set) -> tuple:
    band_by_id = {r[0]: r for r in band_rows[1:]}
    merged, replaced, kept = [base_rows[0]], 0, 0
    for r in base_rows[1:]:
        if seq_of(r[0]) in triggered:
            assert r[0] in band_by_id, f"band CSV 缺 {r[0]}（觸發支必須整支覆蓋）"
            merged.append(band_by_id[r[0]])
            replaced += 1
        else:
            merged.append(r)
            kept += 1
    return merged, replaced, kept


def validate(merged, sample_rows, fc_root: Path, triggered: set) -> list:
    errs = []
    ids = [r[0] for r in merged[1:]]
    sample_ids = [r[0] for r in sample_rows[1:]]
    if ids != sample_ids:
        errs.append(f"ID 集合/順序與 sample 不同（前差異 "
                    f"{next(((a, b) for a, b in zip(ids, sample_ids) if a != b), 'len')}）")
    # 「每序列第一個出現的列＝首幀」成立的前提：ID 順序已驗與 sample 相同，
    # 且 sample_submisson.csv 實查為每序列幀號升冪（nir-bee2_1,2,3,…）。
    first_seen = set()
    for r in merged[1:]:
        seq = seq_of(r[0])
        vals = [float(v) for v in r[1:5]]
        if not all(v == v and abs(v) != float("inf") for v in vals):
            errs.append(f"{r[0]}: 非有限值 {vals}")
            break
        if vals[2] <= 0 or vals[3] <= 0:
            errs.append(f"{r[0]}: w/h 非正 {vals}")
            break
        if seq not in first_seen:
            first_seen.add(seq)
            init_p = fc_root / seq / "init_rect.txt"
            if init_p.exists():
                init = [float(v) for v in
                        init_p.read_text().strip().replace(",", " ").split()[:4]]
                if any(abs(a - b) > 1e-6 for a, b in zip(vals, init)):
                    errs.append(f"{seq}: 首幀 {vals} ≠ init {init}（preserve_init 違反）")
            else:
                errs.append(f"{seq}: 缺 init_rect.txt，無法驗首幀")
    return errs


def diff_report(base_rows, band_rows, triggered: set) -> dict:
    base_by_id = {r[0]: [float(v) for v in r[1:5]] for r in base_rows[1:]}
    per_seq: dict = {}
    for r in band_rows[1:]:
        seq = seq_of(r[0])
        if seq not in triggered or r[0] not in base_by_id:
            continue
        a = [float(v) for v in r[1:5]]
        b = base_by_id[r[0]]
        d = per_seq.setdefault(seq, {"iou": [], "shift": []})
        d["iou"].append(box_iou_xywh(a, b))
        d["shift"].append(((a[0] + a[2] / 2 - b[0] - b[2] / 2) ** 2 +
                          (a[1] + a[3] / 2 - b[1] - b[3] / 2) ** 2) ** 0.5)
    return {s: {"n": len(d["iou"]),
                "iou_vs_base_median": round(statistics.median(d["iou"]), 4),
                "frac_same_box": round(sum(1 for v in d["iou"] if v > 0.99) / len(d["iou"]), 4),
                "center_shift_median_px": round(statistics.median(d["shift"]), 2)}
            for s, d in sorted(per_seq.items())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-csv", required=True, help="v078 基準提交（75 支）")
    ap.add_argument("--band-csv", required=True, help="band run 的 submission.csv（觸發支）")
    ap.add_argument("--triggered-json", required=True, help="make_band3ch --decide 輸出")
    ap.add_argument("--sample", required=True, help="sample_submisson.csv（75 支 canonical）")
    ap.add_argument("--fc-root", required=True, help="假色 root（init_rect.txt 驗首幀）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args(argv)

    triggered = set(json.loads(Path(args.triggered_json).read_text())["triggered"])
    assert triggered, "觸發清單為空——無事可換，不應呼叫 merge"
    base = read_csv_rows(args.base_csv)
    band = read_csv_rows(args.band_csv)
    sample = read_csv_rows(args.sample)

    merged, replaced, kept = merge(base, band, triggered)
    errs = validate(merged, sample, Path(args.fc_root), triggered)
    if errs:
        for e in errs:
            print(f"VALIDATION FAIL: {e}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(merged)
    tmp.replace(out)  # 原子寫入（半截檔不得存在）

    rep = {"triggered": sorted(triggered), "rows_replaced": replaced,
           "rows_kept": kept, "per_seq_diff_vs_base": diff_report(base, band, triggered)}
    Path(args.report).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"[merge] {out}：換入 {replaced} 行（{len(triggered)} 支）、沿用 {kept} 行；驗證全過")
    return 0


if __name__ == "__main__":
    sys.exit(main())
