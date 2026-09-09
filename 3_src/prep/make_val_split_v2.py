"""建立 val_split_v2：跨模態 scene stem 整組保留（修 P0-04 洩漏）。

v1 的缺陷（審核 P0-04，08-08 證實）：group key 為 (modality, stem)，同一 stem 的
其他 sensor 留在 train——實例：rednir-bytheriver1 在 val、vis-bytheriver1 在 train，
兩者 525 幀、首框 (415,60,87,211) vs (418,56,88,215) ＝ 同步拍攝場景洩漏。
v1 檔案保留不動（E15 等既有實驗的基準）；訓練類實驗（E-E 起）一律用 v2。

v2 設計：
- group key = stem（**無模態**）——選中一個 stem，其全部模態的序列整組進 val。
- 挑選：隨機 stem 順序（不再偏小 group——v1 偏稀有小 group 的取樣偏差，審核 §5.1），
  裝到 ~65 支停。deterministic（seed 固定）。
- 產出後報告：模態／長度／小目標比例 vs test 分佈（只報告不迭代，避免 val 特化）。
"""
from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "1_data"
TARGET_N = 65
SEED = 20260808


def main() -> None:
    seqs: set[str] = set()
    first_area: dict[str, float] = {}
    n_frames: Counter = Counter()
    with open(RAW / "raw" / "2026training.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            s = row[0].rsplit("_", 1)[0]
            seqs.add(s)
            n_frames[s] += 1
            if s not in first_area:
                first_area[s] = (float(row[3]) * float(row[4])) ** 0.5

    by_stem: dict[str, list[str]] = defaultdict(list)
    for s in sorted(seqs):
        _mod, name = s.split("-", 1)
        stem = re.sub(r"\d+$", "", name)
        by_stem[stem].append(s)  # 跨模態整組（v1 的 key 是 (mod, stem)——洩漏根源）

    rng = random.Random(SEED)
    stems = sorted(by_stem)
    rng.shuffle(stems)
    held: list[str] = []
    held_stems: list[str] = []
    for st in stems:
        members = by_stem[st]
        if len(held) + len(members) > TARGET_N + 3:
            continue
        held += members
        held_stems.append(st)
        if len(held) >= TARGET_N:
            break

    held = sorted(held)
    out = RAW / "val_split_v2.txt"
    out.write_text("\n".join(held) + "\n")

    # 驗證：val 內任何 stem 不得出現在 train 剩餘（跨模態）
    train_left = seqs - set(held)
    train_stems = {re.sub(r"\d+$", "", s.split("-", 1)[1]) for s in train_left}
    overlap = set(held_stems) & train_stems
    assert not overlap, f"跨模態洩漏仍存在：{overlap}"

    dist = Counter(s.split("-")[0] for s in held)
    lens = sorted(n_frames[s] for s in held)
    small = sum(1 for s in held if first_area[s] < 32)
    print(f"val_split_v2: {len(held)} 序列 / {len(held_stems)} stems → {out}")
    print(f"✅ 跨模態洩漏驗證通過（val stems ∩ train stems = ∅）")
    print(f"模態: {dict(dist)}（test 參考 vis40/nir22/rednir13）")
    print(f"長度: median {lens[len(lens)//2]} mean {sum(lens)/len(lens):.0f}（test 306/358）")
    print(f"小目標(√area<32): {small}/{len(held)} = {small/len(held):.1%}（test 57.3%）")
    print(f"訓練剩餘: {len(train_left)} 序列")


if __name__ == "__main__":
    main()
