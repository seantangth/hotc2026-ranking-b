"""HSOT 追蹤失敗分析（B4）：無 GT 下用 submission box 序列還原失敗型態，並多方法對照。

test set 無 GT，無法直接算逐序列 IoU 失分。但我們的推論 harness 對「空 mask」統一
「沿用前一幀框」（見 exp003/exp004 Cell 5 mask_to_box），因此**連續完全相同的 box =
tracker 在該段吐空 mask（丟失目標）**，這是從 submission CSV 就能還原的高信度失敗指紋，
不需 diagnostics.json。

在此之上疊加尺度抖動 / 中心瞬移 / 小目標 / 漂移等 proxy，逐序列打分並歸類失敗型態；
多個 submission 時做 pairwise 對照（E01↔E02 容量效應、E02↔E03 記憶機制效應），
直接餵 E04（光譜再偵測/否決）與 E05（band-regroup / 小目標放大）的設計。

用法：
    python -m hsot.error_analysis SUB1.csv [SUB2.csv ...] \
        --names E01 E02 [--sample 1_data/raw/sample_submisson.csv] [--out 5_outputs/eda]

輸出：
    <out>/error_analysis.md            人讀報告（模態聚合 + Top-N 失敗序列 + 方法對照）
    <out>/error_analysis_per_seq.csv   逐序列所有指標（供後續程式化使用）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hsot.eval import load_boxes, overlap_ratio

# --- 型態判定閾值（可調；依 EDA 經驗設定，非最佳化超參）---
SMALL_AREA = 400.0        # < 20×20 px 視為小目標
FROZEN_HI = 0.15          # 凍結（丟失）幀比例達此值 → 追蹤失敗序列
DRIFT_LO, DRIFT_HI = 0.3, 3.0   # 末/首面積比超出此範圍 → 漂移/尺度崩壞
JUMP_HI = 2.0             # 單步中心位移 > 2× 目標尺度 → 瞬移（跳 distractor）
ASPECT_HI = 3.0           # 長寬比動態範圍 > 3 → 形變/抓錯物
EARLY_FRAC = 0.2          # 前 20% 幀


def _runs_of_true(flags: np.ndarray) -> list[int]:
    """回傳布林陣列中連續 True 段的長度清單。"""
    runs, cur = [], 0
    for f in flags:
        if f:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    return runs


def seq_proxy_metrics(boxes: np.ndarray) -> dict:
    """單序列 proxy 失敗指標。boxes: (N,4) 已按 frame 排序，[x,y,w,h] float。"""
    n = len(boxes)
    areas = boxes[:, 2] * boxes[:, 3]
    centers = boxes[:, :2] + boxes[:, 2:] / 2.0
    scale = np.sqrt(np.clip(areas, 1.0, None))  # 目標尺度（對角線量級）

    # 凍結 = 與前一幀 box 完全相同（沿用前框 = 空 mask/丟失）
    if n > 1:
        frozen = np.all(boxes[1:] == boxes[:-1], axis=1)
    else:
        frozen = np.zeros(0, dtype=bool)
    frozen_ratio = float(frozen.mean()) if frozen.size else 0.0
    runs = _runs_of_true(frozen)
    max_frozen_run = int(max(runs)) if runs else 0
    n_frozen_runs = int(len(runs))

    # 早期丟失：前 EARLY_FRAC 幀內的凍結比例（init 沒抓好 / 早期跳失）
    k = max(1, int(n * EARLY_FRAC))
    early_frozen = float(frozen[: k - 1].mean()) if k > 1 and frozen.size else 0.0

    # 尺度抖動：相鄰幀 log 面積差的標準差
    log_area = np.log(np.clip(areas, 1.0, None))
    scale_cv = float(np.std(np.diff(log_area))) if n > 1 else 0.0

    # 中心瞬移：單步位移 / 目標尺度 的最大值（跳 distractor proxy）
    if n > 1:
        jumps = np.linalg.norm(np.diff(centers, axis=0), axis=1) / np.clip(scale[1:], 1.0, None)
        max_center_jump = float(np.max(jumps))
    else:
        max_center_jump = 0.0

    init_area = float(areas[0])
    area_drift = float(areas[-1] / max(areas[0], 1.0))
    aspect = boxes[:, 2] / np.clip(boxes[:, 3], 1e-6, None)
    aspect_range = float(np.max(aspect) / max(float(np.min(aspect)), 1e-6))

    return {
        "n": int(n),
        "frozen_ratio": round(frozen_ratio, 4),
        "max_frozen_run": max_frozen_run,
        "n_frozen_runs": n_frozen_runs,
        "early_frozen": round(early_frozen, 4),
        "scale_cv": round(scale_cv, 4),
        "max_center_jump": round(max_center_jump, 3),
        "init_area": round(init_area, 1),
        "area_drift": round(area_drift, 3),
        "aspect_range": round(aspect_range, 3),
    }


def classify(m: dict) -> list[str]:
    """依 proxy 指標貼失敗型態標籤（一序列可多標籤）。"""
    tags = []
    if m["init_area"] < SMALL_AREA:
        tags.append("small_target")
    if m["frozen_ratio"] >= FROZEN_HI:
        if m["early_frozen"] >= FROZEN_HI:
            tags.append("early_loss")
        if m["max_frozen_run"] >= 0.3 * m["n"]:
            tags.append("sustained_loss")
        if m["n_frozen_runs"] >= 3 and m["max_frozen_run"] < 0.3 * m["n"]:
            tags.append("intermittent_loss")
    if not (DRIFT_LO <= m["area_drift"] <= DRIFT_HI) or m["aspect_range"] >= ASPECT_HI:
        tags.append("drift_or_deform")
    if m["max_center_jump"] >= JUMP_HI:
        tags.append("jumpy")
    return tags or ["clean"]


def analyze_one(df: pd.DataFrame) -> pd.DataFrame:
    """單一 submission → 逐序列 proxy 指標 + 型態標籤的 DataFrame。"""
    rows = []
    for seq, grp in df.sort_values("frame").groupby("seq"):
        boxes = grp[["x", "y", "w", "h"]].to_numpy(float)
        m = seq_proxy_metrics(boxes)
        m["seq"] = seq
        m["modality"] = seq.split("-", 1)[0]
        m["tags"] = "|".join(classify(m))
        rows.append(m)
    cols = ["seq", "modality", "n", "frozen_ratio", "max_frozen_run", "n_frozen_runs",
            "early_frozen", "scale_cv", "max_center_jump", "init_area", "area_drift",
            "aspect_range", "tags"]
    return pd.DataFrame(rows)[cols].sort_values("frozen_ratio", ascending=False)


def pairwise_iou(dfa: pd.DataFrame, dfb: pd.DataFrame) -> dict[str, float]:
    """兩 submission 逐序列平均框 IoU（低 = 兩方法在該序列分歧大）。"""
    m = dfa.merge(dfb, on="ID", suffixes=("_a", "_b"))
    a = m[["x_a", "y_a", "w_a", "h_a"]].to_numpy(float)
    b = m[["x_b", "y_b", "w_b", "h_b"]].to_numpy(float)
    m["iou"] = overlap_ratio(a, b)
    return m.groupby("seq_a")["iou"].mean().to_dict()


def build_report(subs: dict[str, pd.DataFrame], per_seq: dict[str, pd.DataFrame],
                 out_dir: Path) -> tuple[Path, Path]:
    names = list(subs)
    # --- 合併逐序列 CSV（每方法一組欄位，前綴方法名）---
    base = per_seq[names[0]][["seq", "modality", "n", "init_area"]].copy()
    merged = base
    for nm in names:
        cols = ["seq", "frozen_ratio", "max_frozen_run", "early_frozen",
                "scale_cv", "max_center_jump", "area_drift", "aspect_range", "tags"]
        r = per_seq[nm][cols].rename(columns={c: f"{nm}_{c}" for c in cols if c != "seq"})
        merged = merged.merge(r, on="seq")
    # --- 方法對照 IoU（相鄰兩方法）---
    for i in range(len(names) - 1):
        a, b = names[i], names[i + 1]
        iou = pairwise_iou(subs[a], subs[b])
        merged[f"iou_{a}_{b}"] = merged["seq"].map(iou).round(4)
        merged[f"dfroz_{a}_{b}"] = (
            merged[f"{b}_frozen_ratio"] - merged[f"{a}_frozen_ratio"]).round(4)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "error_analysis_per_seq.csv"
    merged.to_csv(csv_path, index=False)

    # --- Markdown 報告 ---
    L = []
    L.append("# HSOT 追蹤失敗分析（B4）\n")
    L.append(f"> 方法：{', '.join(names)}；序列數 {len(merged)}；"
             f"指標為無 GT proxy（frozen=空 mask 沿用前框段=丟失指紋）。\n")

    # 模態 × 方法聚合
    L.append("## 模態 × 方法：丟失（frozen_ratio）聚合\n")
    L.append("| 模態 | 序列數 | 小目標數 | " +
             " | ".join(f"{nm} frozen中位/最大" for nm in names) + " |")
    L.append("|---|---|---|" + "---|" * len(names))
    for mod in ["vis", "nir", "rednir"]:
        sub = merged[merged["modality"] == mod]
        if sub.empty:
            continue
        n_small = int((sub["init_area"] < SMALL_AREA).sum())
        cells = []
        for nm in names:
            fr = sub[f"{nm}_frozen_ratio"]
            cells.append(f"{fr.median():.3f} / {fr.max():.3f}")
        L.append(f"| {mod} | {len(sub)} | {n_small} | " + " | ".join(cells) + " |")

    # 每方法 Top-10 失敗序列
    for nm in names:
        L.append(f"\n## {nm} — Top-10 最可疑序列（按 frozen_ratio）\n")
        L.append("| 序列 | n | frozen | 最長丟失段 | early | scale_cv | max_jump | drift | 型態 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        top = per_seq[nm].head(10)
        for _, r in top.iterrows():
            L.append(f"| {r.seq} | {r.n} | {r.frozen_ratio:.3f} | {r.max_frozen_run} | "
                     f"{r.early_frozen:.3f} | {r.scale_cv:.3f} | {r.max_center_jump:.2f} | "
                     f"{r.area_drift:.2f} | {r.tags} |")

    # 方法對照：改善 / 退步最大的序列
    for i in range(len(names) - 1):
        a, b = names[i], names[i + 1]
        col = f"dfroz_{a}_{b}"
        d = merged[["seq", "modality", f"{a}_frozen_ratio", f"{b}_frozen_ratio",
                    col, f"iou_{a}_{b}"]].copy()
        L.append(f"\n## 對照 {a} → {b}：丟失改善/退步（Δfrozen = {b} − {a}）\n")
        L.append(f"- 平均框 IoU（{a} vs {b}）：{merged[f'iou_{a}_{b}'].mean():.3f}"
                 f"（越低代表兩方法分歧越大）")
        L.append(f"- {b} 相對 {a}：改善 {int((d[col] < -0.02).sum())} 序列、"
                 f"退步 {int((d[col] > 0.02).sum())} 序列、持平其餘\n")
        L.append("**改善最多 Top-5**（Δ 最負）：\n")
        L.append("| 序列 | " + f"{a}_frozen | {b}_frozen | Δ | IoU |")
        L.append("|---|---|---|---|---|")
        for _, r in d.nsmallest(5, col).iterrows():
            L.append(f"| {r.seq} | {r[f'{a}_frozen_ratio']:.3f} | {r[f'{b}_frozen_ratio']:.3f} "
                     f"| {r[col]:+.3f} | {r[f'iou_{a}_{b}']:.3f} |")
        L.append("\n**退步最多 Top-5**（Δ 最正）：\n")
        L.append("| 序列 | " + f"{a}_frozen | {b}_frozen | Δ | IoU |")
        L.append("|---|---|---|---|---|")
        for _, r in d.nlargest(5, col).iterrows():
            L.append(f"| {r.seq} | {r[f'{a}_frozen_ratio']:.3f} | {r[f'{b}_frozen_ratio']:.3f} "
                     f"| {r[col]:+.3f} | {r[f'iou_{a}_{b}']:.3f} |")

    # 型態計數（最後一個方法為準）
    last = per_seq[names[-1]]
    L.append(f"\n## 失敗型態計數（{names[-1]}）\n")
    tag_counts: dict[str, int] = {}
    for tags in last["tags"]:
        for t in tags.split("|"):
            tag_counts[t] = tag_counts.get(t, 0) + 1
    L.append("| 型態 | 序列數 |")
    L.append("|---|---|")
    for t, c in sorted(tag_counts.items(), key=lambda kv: -kv[1]):
        L.append(f"| {t} | {c} |")

    L.append("\n---\n")
    L.append("**指標定義**："
             "`frozen_ratio`=與前幀完全相同 box 的幀比例（空 mask 沿用前框=丟失）；"
             "`max_frozen_run`=最長連續丟失幀數；`early_frozen`=前 20% 幀丟失率；"
             "`scale_cv`=相鄰 log 面積差 std（尺度抖動）；`max_center_jump`=單步中心位移/目標尺度 最大值；"
             "`area_drift`=末/首面積比；`aspect_range`=長寬比動態範圍。\n")
    L.append("**型態**：small_target(<20×20)、early_loss(前段就丟)、sustained_loss(長段丟失)、"
             "intermittent_loss(多次短丟)、drift_or_deform(尺度/長寬比崩)、jumpy(瞬移)、clean。\n")

    md_path = out_dir / "error_analysis.md"
    md_path.write_text("\n".join(L), encoding="utf-8")
    return md_path, csv_path


def main() -> None:
    ap = argparse.ArgumentParser(description="HSOT 追蹤失敗 proxy 分析 + 多方法對照")
    ap.add_argument("subs", nargs="+", help="一或多個 submission CSV（依序=時間序，如 E01 E02 E03）")
    ap.add_argument("--names", nargs="*", help="對應方法名（預設用檔名 stem）")
    ap.add_argument("--out", default="5_outputs/eda", help="輸出目錄")
    args = ap.parse_args()

    names = args.names or [Path(p).stem for p in args.subs]
    assert len(names) == len(args.subs), "names 數量需與 subs 一致"

    subs, per_seq = {}, {}
    for nm, p in zip(names, args.subs):
        df = load_boxes(p)
        subs[nm] = df
        per_seq[nm] = analyze_one(df)
        n_fail = int((per_seq[nm]["frozen_ratio"] >= FROZEN_HI).sum())
        print(f"{nm}: {len(per_seq[nm])} 序列，其中 {n_fail} 序列 frozen_ratio≥{FROZEN_HI}")

    md, csv = build_report(subs, per_seq, Path(args.out))
    print(f"報告：{md}")
    print(f"逐序列：{csv}")


if __name__ == "__main__":
    main()
