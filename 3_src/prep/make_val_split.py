"""建立 val_split_v1：按場景 stem 整組保留的本地驗證集。

設計依據（COMPETITION_STRATEGY §3、D003）：
- test 44 個 stem 中 34 個不在 train → 驗證集以「整個 stem 保留」模擬 unseen scene
- 模態比例對齊 test：VIS 40/75=53%、NIR 22/75=29%、RedNIR 13/75=17%
- 優先挑「該 stem 在 train 中序列數少」者（減少訓練資料損失），deterministic（seed 固定）
"""
from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "1_data"
TARGET = {"vis": 35, "nir": 19, "rednir": 11}  # ~65 序列，比例對齊 test
SEED = 20260803


def main() -> None:
    seqs = set()
    with open(RAW / "raw" / "2026training.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            seqs.add(row[0].rsplit("_", 1)[0])

    by_stem: dict[tuple[str, str], list[str]] = defaultdict(list)
    for s in sorted(seqs):
        mod, name = s.split("-", 1)
        stem = re.sub(r"\d+$", "", name)
        by_stem[(mod, stem)].append(s)

    rng = random.Random(SEED)
    held: list[str] = []
    for mod, quota in TARGET.items():
        stems = [(k, v) for k, v in by_stem.items() if k[0] == mod]
        rng.shuffle(stems)
        stems.sort(key=lambda kv: len(kv[1]))  # 序列數少的 stem 優先（穩定排序保留隨機 tie-break）
        picked: list[str] = []
        for _key, members in stems:
            if len(picked) + len(members) > quota + 2:
                continue
            picked += members
            if len(picked) >= quota:
                break
        held += picked

    held = sorted(held)
    out = RAW / "val_split_v1.txt"
    out.write_text("\n".join(held) + "\n")
    dist = Counter(s.split("-")[0] for s in held)
    print(f"val_split_v1: {len(held)} 序列 → {out}")
    print(f"模態分布: {dict(dist)}（目標 {TARGET}）")
    print(f"訓練剩餘: {len(seqs) - len(held)} 序列")


if __name__ == "__main__":
    main()
