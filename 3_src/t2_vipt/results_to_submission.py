r"""T2 推論結果 → Kaggle 提交 CSV。

ViPT 的 `tracking/test_hsi_mgpus_all.py` 每序列輸出一個 txt（逐幀 `x,y,w,h`，`%.2f`
逗號分隔），檔名為 `HSI-{VIS,NIR,RedNIR}-FalseColor-{name}.txt`；Kaggle 要的是單一 CSV，
ID = `{modality}-{name}_{frame}`（模態小寫、幀號 **1-based**），如 `nir-bee2_1`。

本腳本做三件事：對映命名、展平成 CSV、**逐 ID 比對 sample_submission**（多一個少一個
都當錯誤）——提交格式出錯的代價是浪費一發每日額度，寧可在本地擋下。

用法：
    python results_to_submission.py \
        --results <ViPT>/results/HOT23TEST/deep_all \
        --sample  1_data/raw/sample_submisson.csv \
        --out     5_outputs/submissions/sub_vNNN_t2_vipt.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

# 推論輸出的資料夾名 → 提交 ID 的模態前綴
MOD_PREFIX = {
    "HSI-VIS-FalseColor": "vis",
    "HSI-NIR-FalseColor": "nir",
    "HSI-RedNIR-FalseColor": "rednir",
}


def parse_seq_txt(path: Path) -> tuple[str, list[list[float]]]:
    """`HSI-NIR-FalseColor-bee2.txt` → ("nir-bee2", [[x,y,w,h], ...])。"""
    stem = path.stem
    for folder, prefix in MOD_PREFIX.items():
        if stem.startswith(folder + "-"):
            seq = f"{prefix}-{stem[len(folder) + 1:]}"
            break
    else:
        raise ValueError(f"無法判定模態：{path.name}")

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace("\t", ",").split(",")
            rows.append([float(v) for v in parts[:4]])
    return seq, rows


def load_sample_ids(sample_csv: Path) -> list[str]:
    with open(sample_csv) as f:
        r = csv.reader(f)
        next(r)
        return [row[0] for row in r]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="ViPT results/<dataset>/<yaml> 目錄")
    ap.add_argument("--sample", default="", help="sample_submisson.csv（正式提交用；省略＝只輸出實際預測到的 ID，供本地評分）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-ids", default="",
                    help="本地評分用：沿用此 CSV 的 ID 編號（2026training.csv）。"
                         "⚠️ 提交格式的幀號是**每序列從 1** 起算（sample_submisson.csv：nir-bee2_1），"
                         "但訓練 GT 用的是**全域連續編號**（vis-L_basketball_person_84193）→ "
                         "直接拿提交格式去比對訓練 GT 會全部對不上、AUC 歸零。依序列內順序位置對映。")
    ap.add_argument("--allow-missing", action="store_true",
                    help="缺的 ID 用 0,0,0,0 補（僅除錯用；正式提交不該需要）")
    args = ap.parse_args()

    # 本地評分模式：先讀 GT 的 ID 編號（每序列依數字後綴排序）供位置對映
    gt_ids: dict[str, list[str]] = {}
    if args.gt_ids:
        with open(args.gt_ids) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                seq, _fr = row[0].rsplit("_", 1)
                gt_ids.setdefault(seq, []).append(row[0])
        for s in gt_ids:
            gt_ids[s].sort(key=lambda i: int(i.rsplit("_", 1)[1]))

    preds: dict[str, list[float]] = {}
    seq_lens: dict[str, int] = {}
    for txt in sorted(Path(args.results).glob("*.txt")):
        seq, rows = parse_seq_txt(txt)
        seq_lens[seq] = len(rows)
        if args.gt_ids:
            ids = gt_ids.get(seq, [])
            if len(ids) != len(rows):
                print(f"⚠️ {seq}: 預測 {len(rows)} 幀 vs GT {len(ids)} 幀——取較短者對映")
            for gid, box in zip(ids, rows):            # 依序列內順序位置對映
                preds[gid] = box
        else:
            for i, box in enumerate(rows, start=1):    # 提交格式：每序列 1-based
                preds[f"{seq}_{i}"] = box

    if not args.sample:
        # 本地評分模式：沒有 sample 可比對，直接輸出實際預測（依序列名、幀號排序）
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ID", "x", "y", "width", "height"])
            for sid in sorted(preds, key=lambda s: (s.rsplit("_", 1)[0], int(s.rsplit("_", 1)[1]))):
                w.writerow([sid] + [f"{v:.1f}" for v in preds[sid]])
        print(f"✓ {out}（{len(preds)} 列，{len(seq_lens)} 序列；未比對 sample＝本地評分模式）")
        return

    sample_ids = load_sample_ids(Path(args.sample))
    missing = [i for i in sample_ids if i not in preds]
    extra = set(preds) - set(sample_ids)

    print(f"序列 {len(seq_lens)} 支／預測 {len(preds)} 幀；sample {len(sample_ids)} 幀")
    if extra:
        # 多出來的一定是錯（命名對映錯、或跑到不該跑的序列）——直接擋
        sample_seqs = {i.rsplit("_", 1)[0] for i in sample_ids}
        bad_seqs = sorted({i.rsplit("_", 1)[0] for i in extra} - sample_seqs)
        raise SystemExit(f"🚨 有 {len(extra)} 個 ID 不在 sample 內；可疑序列名：{bad_seqs[:10]}")
    if missing:
        miss_seqs = sorted({i.rsplit("_", 1)[0] for i in missing})
        msg = (f"🚨 缺 {len(missing)} 幀，涉及 {len(miss_seqs)} 支序列：{miss_seqs[:10]}\n"
               f"   （常見原因：該序列推論未跑完／幀數與官方不符）")
        if not args.allow_missing:
            raise SystemExit(msg)
        print(msg + "\n   --allow-missing → 以 0,0,0,0 填補")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "x", "y", "width", "height"])
        for sid in sample_ids:                          # 嚴格照 sample 的順序
            box = preds.get(sid, [0.0, 0.0, 0.0, 0.0])
            w.writerow([sid] + [f"{v:.1f}" for v in box])

    print(f"✓ {out}（{len(sample_ids)} 列，順序與 sample 一致）")


if __name__ == "__main__":
    main()
