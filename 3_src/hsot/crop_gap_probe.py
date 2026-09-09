#!/usr/bin/env python3
"""D043 重開的零成本幾何前測：被 40% 面積門檻剔除的小目標，到底還能移除多少干擾物場？

**為何重開**（時序證據）：D043（放寬門檻判死，08-06）的依據是「線性放大倍率 2.05→1.68」
＝純**解析度**框架；而 D046（08-07 收盤）以 v013/v014 的完整 2×2 實測推翻解析度說，
確立 **crop 的價值在移除干擾物**。⇒ D043 的否決寫在它的前提被推翻**之前**，
且 D046 自己的結論「主線改向抑制干擾物／縮小有效視野」從未被套用回被剔除組。

**本前測只回答一件事**：被剔除組的窗幾何長什麼樣。
  窗佔 55% ⇒ 仍移除 45% 的干擾物場（有戲）；窗佔 90% ⇒ 等於沒裁（死）。
判準門檻**從幾何事前註冊**，不從 val AUC 掃（D033/D037：v004 就死在 val 推導門檻）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .crop_rerun import AREA_FRAC_MAX, SMALL_T, _window

# D057 失分地圖：跟丟幀數（用於量化候選覆蓋多少主戰場）
LOST_FRAMES = {
    "nir-fake_orange": 742, "nir-pingpong": 661, "vis-S_jump2": 431, "nir-yo_yo": 424,
    "vis-droneshow2": 377, "rednir-droneshow2": 356, "vis-leaf": 269, "vis-L_runner": 262,
    "nir-leaves": 236, "rednir-drone2": 189,
}


def parse(p) -> pd.DataFrame:
    d = pd.read_csv(p)
    q = d["ID"].str.rsplit("_", n=1, expand=True)
    d["seq"], d["frame"] = q[0], q[1].astype(int)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-csv", required=True, help="E02 軌跡（t1_rerun submission.csv）")
    ap.add_argument("--extra-csv", required=True, help="E15 軌跡（聯集用，同 v008 窗定義）")
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--sizes-json", required=True, help="每序列原圖尺寸 {seq: [W,H]}")
    ap.add_argument("--seqs", help="限定序列清單（val_split）")
    ap.add_argument("--remove-frac-min", type=float, default=0.40,
                    help="事前註冊門檻：窗須至少移除這麼多比例的場才算候選")
    a = ap.parse_args()

    base, extra = parse(a.base_csv), parse(a.extra_csv)
    gt = parse(a.gt_csv)
    sizes = json.loads(Path(a.sizes_json).read_text())
    seqs = None
    if a.seqs:
        seqs = {s.strip() for s in Path(a.seqs).read_text().split() if s.strip()}

    rows = []
    for seq, g in gt.groupby("seq"):
        if seqs and seq not in seqs:
            continue
        if seq not in sizes:
            continue
        W, H = sizes[seq][:2]
        g = g.sort_values("frame")
        r0 = g.iloc[0]
        scale = float(np.sqrt(float(r0["width"]) * float(r0["height"])))
        if scale >= SMALL_T:            # 只看小目標（大目標已於 08-07 判死：無標的）
            continue
        tr = [d[d["seq"] == seq][["x", "y", "width", "height"]].to_numpy(float)
              for d in (base, extra)]
        tr = [t for t in tr if len(t)]
        if not tr:
            continue
        boxes = np.vstack(tr)           # 聯集：單一軌跡跟丟時會凍結在錯位置（v008 窗定義）
        x1, y1, x2, y2 = _window(boxes, scale, W, H)
        area_frac = ((x2 - x1) * (y2 - y1)) / (W * H)
        rows.append({
            "seq": seq, "init_scale": round(scale, 1), "orig": [W, H],
            "win": [x1, y1, x2 - x1, y2 - y1],
            "area_frac": round(float(area_frac), 3),
            "removed_frac": round(float(1 - area_frac), 3),
            "linear_zoom": round(float(np.sqrt(1 / max(area_frac, 1e-9))), 2),
            "selected_now": bool(area_frac < AREA_FRAC_MAX),
            "lost_frames": LOST_FRAMES.get(seq, 0),
        })

    df = pd.DataFrame(rows).sort_values("area_frac")
    sel = df[df["selected_now"]]
    exc = df[~df["selected_now"]]
    cand = exc[exc["removed_frac"] >= a.remove_frac_min]

    print(f"小目標（首幀 sqrt(w·h) < {SMALL_T}）共 {len(df)} 支\n")
    print(f"【現行選中】{len(sel)} 支（窗 < {AREA_FRAC_MAX:.0%}）")
    print(sel[["seq", "area_frac", "removed_frac", "linear_zoom", "lost_frames"]].to_string(index=False))
    print(f"\n【被門檻剔除】{len(exc)} 支")
    print(exc[["seq", "area_frac", "removed_frac", "linear_zoom", "lost_frames"]].to_string(index=False))

    tot_lost = sum(LOST_FRAMES.values())
    print(f"\n{'='*70}")
    print(f"事前註冊門檻：仍能移除 ≥{a.remove_frac_min:.0%} 的場 ⇒ 候選")
    print(f"  候選 {len(cand)}/{len(exc)} 支｜覆蓋 top-10 跟丟幀 "
          f"{int(cand['lost_frames'].sum()):,}/{tot_lost:,} = {cand['lost_frames'].sum()/tot_lost:.0%}")
    if len(cand):
        print(cand[["seq", "area_frac", "removed_frac", "linear_zoom", "lost_frames"]].to_string(index=False))
    print(f"\n對照：現行選中組覆蓋 top-10 跟丟幀 "
          f"{int(sel['lost_frames'].sum()):,}/{tot_lost:,} = {sel['lost_frames'].sum()/tot_lost:.0%}"
          f"；被剔除組合計 {int(exc['lost_frames'].sum()):,} = {exc['lost_frames'].sum()/tot_lost:.0%}")
    df.to_json("crop_gap_probe.json", orient="records", indent=1)
    print("\n→ crop_gap_probe.json")


if __name__ == "__main__":
    main()
