#!/usr/bin/env python3
"""E30 首幀固定 crop（one-pass 合規變體）——零 GPU 前測。

【為何做】D052a 的 OPE/two-pass 合規風險若被裁定不利，主線 v023 與授權備案 v011 的 crop 窗
**都**來自 base 軌跡 envelope ＝ two-pass ⇒ 需要一個「窗只由首幀決定」的合規變體。
D065（PEFT 結案）後這是唯一未量測的路線，雙重用途：合規避險 ＋ 新資訊。

【唯一變因】窗的來源：
  v023  窗 = 該段 base 軌跡 envelope（＋extra 聯集）外擴 max(2×scale, 32)   ← two-pass
  E30   窗 = 首幀 box 中心 ± R×scale/2 的方形（K=1、不分段）                 ← one-pass
其餘（SMALL_T=32 選序列、面積門檻、merge 規則）與 v023 對齊。
⚠️ 分段（--segments K）在首幀變體下**不可用**：段窗需要該段起始幀的位置＝仍是 two-pass。

【事前判準（Sean 08-11 授權，執行前寫死）】
  存在一個 R 同時滿足：
    (i)   災難檢查：選中序列的 GT 脫窗率 < 1%
    (ii)  面積門檻通過（主報 0.55 ＝ v023 同門檻以隔離單變因；附報 0.40）
    (iii) test75 選中 ≥ 8 支
  ⇒ 開 A100 跑選中序列 → merge → 一發 LB。
  三者任一不滿足 ⇒ 零成本判死，寫入 EXPERIMENT_LOG 並列新決策編號。
  依 D063 上檔條款聲明：**這一發買的是合規避險與資訊，不是分數。**

【附加的硬分析（比脫窗率更有決策價值，主動加報）】
  crop 後 tracker 的輸出恆在窗內 ⇒ 逐幀 IoU 上界 ＝ area(GT ∩ win) / area(GT)。
  故「窗造成的 pooled AUC 上界損失」＝ 1 − mean(clip_iou_ub)，可直接與 crop 已知增益
  （+0.0108 本地 D064 ／ +0.0175 LB）比較。若損失 > 增益 ⇒ 即使判準 (i)(iii) 過也是淨負。

【事前預期（寫在執行前，防事後合理化）】
  1. 脫窗率、選中支數皆隨 R 單調（前者降、後者降）⇒ 存在 Pareto 取捨
  2. R 小（≈v023 等效尺度）時脫窗率**遠高於 1%**——crop_rerun.py:57-59 明文記載
     「跟丟時軌跡凍結在錯位置 → 窗把真目標切在外面」，envelope＋extra 聯集正是為此而加；
     首幀窗連 envelope 都沒有
  3. ⇒ 預期 (i) 與 (iii) 難以同時滿足，**傾向判死**
  ⚠️ 若實測推翻 2/3，以實測為準（D061：否決的前提被推翻時，否決本身必須重驗）

【對照錨點（依 D062 canary 紀律：必須含「現行管線表現良好」的對照）】
  同一支腳本對 val65 重算 v023 的 envelope 窗脫窗率。若該對照本身脫窗率不接近 0，
  表示本量測的絕對值不可信（量到的是別的東西），只能看相對關係。

用法：
  python3 -m hsot.firstframe_crop_probe --json 5_outputs/e30_firstframe_crop_probe.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── 與 crop_rerun.py 對齊的常數（不得改動，改動即引入第二個變因）────────────────
SMALL_T = 32.0        # 首幀 sqrt(w*h) < 32 才算小目標
MARGIN_SCALE = 2.0    # v023 envelope 外擴 = max(2×scale, 32)
MIN_MARGIN = 32.0
AFM_MAIN = 0.55       # v023 實際使用門檻（cropwiden55）
AFM_ALT = 0.40        # crop_rerun.py 預設值，附報

# 掃描的方形窗邊長倍率 R（窗邊長 = R × 首幀 scale）。
# 涵蓋範圍需同時包含「≈v023 等效尺度」與「大到脫窗率必為 0」兩端，否則 Pareto 前緣看不完整。
R_GRID = [4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0]

ROOT = Path(__file__).resolve().parents[2]


def _parse_ids(df: pd.DataFrame) -> pd.DataFrame:
    p = df["ID"].str.rsplit("_", n=1, expand=True)
    df = df.copy()
    df["seq"], df["frame"] = p[0], p[1].astype(int)
    return df


def load_val_gt(split_file: Path) -> dict[str, pd.DataFrame]:
    """val65 逐幀 GT。排除 w≤0 或 h≤0 的幀（官方評分對這些幀給 IoU=-1，D063(b)）。"""
    seqs = [s.strip() for s in split_file.read_text().splitlines() if s.strip()]
    gt = _parse_ids(pd.read_csv(ROOT / "1_data/raw/2026training.csv"))
    gt = gt[gt["seq"].isin(seqs)]
    out = {}
    for seq, g in gt.groupby("seq"):
        g = g.sort_values("frame")
        valid = (g["width"] > 0) & (g["height"] > 0)
        out[seq] = g[valid].reset_index(drop=True)
    missing = set(seqs) - set(out)
    if missing:
        raise SystemExit(f"GT 缺序列：{sorted(missing)}")
    return out


# ── 原圖尺寸 ───────────────────────────────────────────────────────────────
# val65 的影像不在本機（1_data/processed 為空）。做法：候選解析度集匹配 GT 上界，
# 並用「已有真實尺寸的序列」驗證推導法的準確率；再對每支做 min/max 敏感度。
def size_candidates() -> list[tuple[int, int]]:
    cands = set()
    for v in json.loads((ROOT / "1_data/test_sizes_75.json").read_text()).values():
        cands.add((int(v["W"]), int(v["H"])))
    for f in ("5_outputs/crop_windows_val.json", "3_src/crop_gap_probe.json"):
        p = ROOT / f
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows = d.values() if isinstance(d, dict) else d
        for r in rows:
            if "orig" in r:
                cands.add((int(r["orig"][0]), int(r["orig"][1])))
    return sorted(cands, key=lambda wh: wh[0] * wh[1])


def known_val_sizes() -> dict[str, tuple[int, int]]:
    known = {}
    p = ROOT / "5_outputs/crop_windows_val.json"
    if p.exists():
        for seq, r in json.loads(p.read_text()).items():
            known[seq] = (int(r["orig"][0]), int(r["orig"][1]))
    p = ROOT / "3_src/crop_gap_probe.json"
    if p.exists():
        for r in json.loads(p.read_text()):
            known[r["seq"]] = (int(r["orig"][0]), int(r["orig"][1]))
    return known


def infer_size(g: pd.DataFrame, cands: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    """回傳 (最小可容納候選, 最大候選)。前者為推定值，後者供敏感度分析。
    低估 W×H ⇒ 高估 area_frac ⇒ 選中更少序列 ⇒ 對「該不該開機」是保守方向。"""
    mx = float((g["x"] + g["width"]).max())
    my = float((g["y"] + g["height"]).max())
    fit = [c for c in cands if c[0] >= mx and c[1] >= my]
    if not fit:  # GT 超出所有候選（不應發生）⇒ 用 GT 上界本身
        return (int(np.ceil(mx)), int(np.ceil(my))), (int(np.ceil(mx)), int(np.ceil(my)))
    return fit[0], fit[-1]


# ── 窗 ────────────────────────────────────────────────────────────────────
def envelope_window(boxes: np.ndarray, scale: float, W: float, H: float) -> tuple[int, int, int, int]:
    """v023 算式（對照組）。與 crop_rerun._window 位元級同義。"""
    m = max(MARGIN_SCALE * scale, MIN_MARGIN)
    x1 = max(0.0, float(np.min(boxes[:, 0])) - m)
    y1 = max(0.0, float(np.min(boxes[:, 1])) - m)
    x2 = min(float(W), float(np.max(boxes[:, 0] + boxes[:, 2])) + m)
    y2 = min(float(H), float(np.max(boxes[:, 1] + boxes[:, 3])) + m)
    return int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2))


def firstframe_window(box0: np.ndarray, scale: float, R: float,
                      W: float, H: float, clip: bool = True) -> tuple[int, int, int, int]:
    """E30 算式：以首幀 box 中心為心、邊長 R×scale 的方形，夾在原圖內。
    夾取後仍保證涵蓋首幀 box（首幀 box 必在圖內且以其中心展開）。
    clip=False ＝ 假想無限畫布：窗不被邊界夾小 ⇒ 脫窗率的**絕對下界**，
    且完全不依賴原圖尺寸推導（本機無 val 影像，推導只有 8/15 準）。"""
    cx = float(box0[0]) + float(box0[2]) / 2.0
    cy = float(box0[1]) + float(box0[3]) / 2.0
    half = max(R * scale / 2.0, float(box0[2]) / 2.0, float(box0[3]) / 2.0)
    if not clip:
        return int(cx - half), int(cy - half), int(np.ceil(cx + half)), int(np.ceil(cy + half))
    x1 = max(0.0, cx - half)
    y1 = max(0.0, cy - half)
    x2 = min(float(W), cx + half)
    y2 = min(float(H), cy + half)
    return int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2))


def strip_window(box0: np.ndarray, scale: float, R: float, W: float, H: float,
                 axis: str) -> tuple[int, int, int, int]:
    """軸解耦條帶窗（advisor 08-11 指出的第三種 one-pass 變體）。
    axis='h'：**全寬** × 高度 R×scale，以首幀 cy 為心 ⇒ 水平脫窗恆為 0，只剩垂直要量。
    axis='v'：全高 × 寬度 R×scale，以首幀 cx 為心。
    動機：required_R 的極端值由**單軸**主導（vis-taxi 118× ＝ 橫越畫面），
    條帶把該軸的脫窗降為零，且極端長寬比窗已在現行管線驗證過（D062：vis-rccar2 的 fc 窗 512×105）。
    窗完全由首幀決定 ⇒ 與方形窗同屬 one-pass。"""
    if axis == "h":
        cy = float(box0[1]) + float(box0[3]) / 2.0
        half = max(R * scale / 2.0, float(box0[3]) / 2.0)
        return 0, int(max(0.0, cy - half)), int(np.ceil(W)), int(np.ceil(min(H, cy + half)))
    cx = float(box0[0]) + float(box0[2]) / 2.0
    half = max(R * scale / 2.0, float(box0[2]) / 2.0)
    return int(max(0.0, cx - half)), 0, int(np.ceil(min(W, cx + half))), int(np.ceil(H))


def make_window(kind: str, box0, scale, R, W, H):
    if kind == "square":
        return firstframe_window(box0, scale, R, W, H)
    return strip_window(box0, scale, R, W, H, axis=kind.split("_")[1])


WIN_KINDS = ["square", "strip_h", "strip_v"]


def required_R(box0: np.ndarray, scale: float, boxes: np.ndarray) -> float:
    """要讓該序列**零脫窗**，以首幀中心展開的方形窗邊長至少需要幾倍 scale。
    此數不依賴原圖尺寸，是「首幀固定窗物理上需要多大」的直接讀數。"""
    cx = float(box0[0]) + float(box0[2]) / 2.0
    cy = float(box0[1]) + float(box0[3]) / 2.0
    half = max(np.max(np.abs(boxes[:, 0] - cx)),
               np.max(np.abs(boxes[:, 0] + boxes[:, 2] - cx)),
               np.max(np.abs(boxes[:, 1] - cy)),
               np.max(np.abs(boxes[:, 1] + boxes[:, 3] - cy)))
    return float(2.0 * half / scale)


def clip_stats(boxes: np.ndarray, win: tuple[int, int, int, int]) -> dict:
    """GT 被窗裁切後的逐幀 IoU 上界 = area(GT ∩ win)/area(GT)。"""
    x1, y1, x2, y2 = win
    gx1, gy1 = boxes[:, 0], boxes[:, 1]
    gx2, gy2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    iw = np.clip(np.minimum(gx2, x2) - np.maximum(gx1, x1), 0, None)
    ih = np.clip(np.minimum(gy2, y2) - np.maximum(gy1, y1), 0, None)
    area = np.maximum(boxes[:, 2] * boxes[:, 3], 1e-9)
    ub = np.clip(iw * ih / area, 0.0, 1.0)
    return {
        "n": int(len(ub)),
        "escape_rate": float((ub < 1.0).mean()),      # 任何一角被切 = 脫窗（判準 i）
        "severe_rate": float((ub < 0.5).mean()),      # IoU 上界跌破 0.5
        "total_rate": float((ub <= 0.0).mean()),      # 完全出窗
        "iou_ub_mean": float(ub.mean()),
        "ub_loss": float(1.0 - ub.mean()),            # 該序列的 AUC 上界損失
    }


# ── 主流程 ────────────────────────────────────────────────────────────────
def run_val(gt: dict[str, pd.DataFrame], base: pd.DataFrame) -> dict:
    cands, known = size_candidates(), known_val_sizes()
    size_check = {"n_known": 0, "n_correct": 0, "wrong": []}
    seq_meta = {}
    for seq, g in gt.items():
        est, est_max = infer_size(g, cands)
        if seq in known:
            size_check["n_known"] += 1
            size_check["n_correct"] += int(known[seq] == est)
            if known[seq] != est:
                size_check["wrong"].append({"seq": seq, "true": known[seq], "est": est})
        W, H = known.get(seq, est)
        f = g.iloc[0]
        seq_meta[seq] = {
            "W": W, "H": H, "from_known": seq in known,
            "W_max": est_max[0], "H_max": est_max[1],
            "scale": float(np.sqrt(f["width"] * f["height"])),
            "box0": np.array([f["x"], f["y"], f["width"], f["height"]], float),
            "boxes": g[["x", "y", "width", "height"]].to_numpy(float),
            "n": int(len(g)),
        }

    small = {s: m for s, m in seq_meta.items() if m["scale"] < SMALL_T}

    # 對照錨點：v023 envelope 窗（來源＝base 軌跡，two-pass）在 val 上的脫窗率
    ctrl = {}
    bs = _parse_ids(base)
    for seq, m in small.items():
        gb = bs[bs["seq"] == seq].sort_values("frame")
        if not len(gb):
            continue
        win = envelope_window(gb[["x", "y", "width", "height"]].to_numpy(float),
                              m["scale"], m["W"], m["H"])
        af = (win[2] - win[0]) * (win[3] - win[1]) / float(m["W"] * m["H"])
        st = clip_stats(m["boxes"], win)
        st.update({"area_frac": af, "selected_055": af < AFM_MAIN, "selected_040": af < AFM_ALT})
        ctrl[seq] = st

    # E30：逐 R 掃描 × 三種窗型（含 no-clip 敏感度腿：脫窗率下界，不依賴尺寸推導）
    sweep = {k: {} for k in WIN_KINDS}
    for kind in WIN_KINDS:
        for R in R_GRID:
            per = {}
            for seq, m in small.items():
                win = make_window(kind, m["box0"], m["scale"], R, m["W"], m["H"])
                af = (win[2] - win[0]) * (win[3] - win[1]) / float(m["W"] * m["H"])
                # 敏感度：用候選集中最大的解析度重算 area_frac（推導尺寸不確定時的另一端）
                af_max = (win[2] - win[0]) * (win[3] - win[1]) / float(m["W_max"] * m["H_max"])
                st = clip_stats(m["boxes"], win)
                st.update({"area_frac": af, "area_frac_altsize": af_max,
                           "selected_055": af < AFM_MAIN, "selected_040": af < AFM_ALT})
                if kind == "square":
                    nc = clip_stats(m["boxes"], firstframe_window(
                        m["box0"], m["scale"], R, m["W"], m["H"], clip=False))
                    st.update({"escape_rate_noclip": nc["escape_rate"],
                               "ub_loss_noclip": nc["ub_loss"]})
                else:  # 條帶：被夾軸恆全幅，脫窗只由另一軸決定，夾取同樣不改變判定
                    st.update({"escape_rate_noclip": st["escape_rate"],
                               "ub_loss_noclip": st["ub_loss"]})
                per[seq] = st
            sweep[kind][f"{R:g}"] = per

    # 「零脫窗所需的窗邊長倍率」分佈——不依賴尺寸，直接回答窗必須多大
    req = {s: required_R(m["box0"], m["scale"], m["boxes"]) for s, m in small.items()}
    return {"seq_meta": {s: {k: v for k, v in m.items() if k not in ("box0", "boxes")}
                         for s, m in seq_meta.items()},
            "size_check": size_check, "control_envelope": ctrl, "sweep": sweep,
            "required_R": req, "n_small": len(small)}


def run_test(v023: pd.DataFrame) -> dict:
    """test75：首幀 box 取自 v023 提交的第 1 幀（OPE 首幀輸出 ＝ init GT）。
    無 GT ⇒ 只能算選中支數與替換幀比例（判準 iii），脫窗率由 val 代表。"""
    sizes = json.loads((ROOT / "1_data/test_sizes_75.json").read_text())
    d = _parse_ids(v023)
    out = {k: {} for k in WIN_KINDS}
    total_frames = int(len(d))
    for kind in WIN_KINDS:
        for R in R_GRID:
            sel, frames = [], 0
            for seq, g in d.groupby("seq"):
                g = g.sort_values("frame")
                f = g.iloc[0]
                scale = float(np.sqrt(f["width"] * f["height"]))
                if scale >= SMALL_T:
                    continue
                W, H = sizes[seq]["W"], sizes[seq]["H"]
                box0 = np.array([f["x"], f["y"], f["width"], f["height"]], float)
                x1, y1, x2, y2 = make_window(kind, box0, scale, R, W, H)
                af = (x2 - x1) * (y2 - y1) / float(W * H)
                if af < AFM_MAIN:
                    sel.append({"seq": seq, "area_frac": round(af, 4),
                                "zoom": round(W * H / max((x2 - x1) * (y2 - y1), 1), 2),
                                "n": int(len(g))})
                    frames += int(len(g))
            out[kind][f"{R:g}"] = {"n_selected": len(sel), "frames": frames,
                                   "frac_frames": frames / total_frames, "seqs": sel}
    n_small = sum(1 for _, g in d.groupby("seq")
                  if np.sqrt(g.sort_values("frame").iloc[0]["width"]
                             * g.sort_values("frame").iloc[0]["height"]) < SMALL_T)
    return {"n_small": n_small, "total_frames": total_frames, "by_R": out}


def verdict(val: dict, test: dict, kind: str = "square") -> dict:
    """事前判準（檔頭定義，此處僅執行、不重新定義）。"""
    rows, passing = [], []
    for R in R_GRID:
        k = f"{R:g}"
        per = val["sweep"][kind][k]
        sel = {s: v for s, v in per.items() if v["selected_055"]}
        n_f = sum(v["n"] for v in sel.values())
        esc = (sum(v["escape_rate"] * v["n"] for v in sel.values()) / n_f) if n_f else 0.0
        ubl = (sum(v["ub_loss"] * v["n"] for v in sel.values()) / n_f) if n_f else 0.0
        # no-clip 腿：脫窗率的絕對下界（窗不被邊界夾小），不依賴尺寸推導
        esc_nc = (sum(v["escape_rate_noclip"] * v["n"] for v in sel.values()) / n_f) if n_f else 0.0
        ubl_nc = (sum(v["ub_loss_noclip"] * v["n"] for v in sel.values()) / n_f) if n_f else 0.0
        t = test["by_R"][kind][k]
        row = {
            "kind": kind, "R": R, "val_selected": len(sel), "val_frames": n_f,
            "val_escape_rate": esc, "val_ub_loss_selected": ubl,
            "val_escape_rate_noclip": esc_nc, "val_ub_loss_noclip": ubl_nc,
            # 換算到全 test 的 pooled 上界損失（＝選中幀比例 × 選中集平均損失）
            "test_selected": t["n_selected"], "test_frac_frames": t["frac_frames"],
            "pooled_ub_loss_est": ubl * t["frac_frames"],
            "gate_i_escape_lt_1pct": esc < 0.01,
            "gate_iii_ge_8_seqs": t["n_selected"] >= 8,
        }
        row["all_gates"] = bool(row["gate_i_escape_lt_1pct"] and row["gate_iii_ge_8_seqs"])
        rows.append(row)
        if row["all_gates"]:
            passing.append(row)
    return {"table": rows, "passing_R": [r["R"] for r in passing],
            "GO": bool(passing),
            "best": max(passing, key=lambda r: r["test_selected"]) if passing else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(ROOT / "1_data/val_split_v1.txt"))
    ap.add_argument("--base", default=str(ROOT / "5_outputs/blend_val_v1.csv"))
    ap.add_argument("--v023", default=str(ROOT / "5_outputs/submissions/sub_v023_cropwiden55.csv"))
    ap.add_argument("--json", default=str(ROOT / "5_outputs/e30_firstframe_crop_probe.json"))
    a = ap.parse_args()

    gt = load_val_gt(Path(a.split))
    val = run_val(gt, pd.read_csv(a.base))
    test = run_test(pd.read_csv(a.v023))
    verdicts = {k: verdict(val, test, k) for k in WIN_KINDS}
    v = verdicts["square"]

    sc = val["size_check"]
    print(f"[尺寸推導自檢] 已知真實尺寸 {sc['n_known']} 支，推導正確 {sc['n_correct']} 支"
          + (f"；誤判：{sc['wrong']}" if sc["wrong"] else ""))
    print(f"[val] 小目標序列 {val['n_small']}/65 支；[test] 小目標序列 {test['n_small']}/75 支\n")

    ctrl = val["control_envelope"]
    cs = {s: v2 for s, v2 in ctrl.items() if v2["selected_055"]}
    if cs:
        nf = sum(v2["n"] for v2 in cs.values())
        print("【對照錨點 v023 envelope 窗（two-pass）】選中 %d 支／%d 幀，"
              "脫窗率 %.4f%%，IoU 上界損失 %.5f"
              % (len(cs), nf,
                 100 * sum(v2["escape_rate"] * v2["n"] for v2 in cs.values()) / nf,
                 sum(v2["ub_loss"] * v2["n"] for v2 in cs.values()) / nf))
    label = {"square": "方形窗（首幀中心 ± R×scale/2）",
             "strip_h": "水平條帶（全寬 × 高 R×scale，水平脫窗恆 0）",
             "strip_v": "垂直條帶（全高 × 寬 R×scale，垂直脫窗恆 0）"}
    for kind in WIN_KINDS:
        print(f"\n【E30-{kind}】{label[kind]}"
              "  (val 脫窗率 = 判準 i；test 選中 = 判準 iii)")
        print(f"{'R':>5} {'val選中':>7} {'val脫窗率':>10} {'val上界損失':>11} "
              f"{'test選中':>8} {'test幀佔比':>10} {'pooled上界損失':>13}  判準")
        for r in verdicts[kind]["table"]:
            print(f"{r['R']:>5.0f} {r['val_selected']:>7d} {100*r['val_escape_rate']:>9.3f}% "
                  f"{r['val_ub_loss_selected']:>11.5f} {r['test_selected']:>8d} "
                  f"{100*r['test_frac_frames']:>9.2f}% {r['pooled_ub_loss_est']:>13.5f}  "
                  f"{'✅GO' if r['all_gates'] else ('i✗' if not r['gate_i_escape_lt_1pct'] else '') + ('iii✗' if not r['gate_iii_ge_8_seqs'] else '')}")

    req = np.array(sorted(val["required_R"].values()))
    print(f"\n【零脫窗所需窗邊長 ÷ 首幀尺度】(val 30 支小目標，不依賴尺寸推導)")
    print(f"  中位 {np.median(req):.1f}×｜75th {np.percentile(req,75):.1f}×｜"
          f"90th {np.percentile(req,90):.1f}×｜最大 {req.max():.1f}×"
          f"｜≤8× 的序列 {int((req<=8).sum())}/{len(req)} 支")
    print("\n【判決】")
    any_go = False
    for kind in WIN_KINDS:
        vk = verdicts[kind]
        any_go |= vk["GO"]
        print(f"  {kind:>8}：" + ("✅ GO — 通過的 R = " + str(vk["passing_R"]) if vk["GO"]
                                  else "NO-GO（無任何 R 同時滿足判準 (i)(iii)）"))
    print("  ⇒ " + ("有窗型過閘 ⇒ 帶數字回報 Sean 走同一閘門鏈決定是否開機"
                    if any_go else
                    "三種 one-pass 窗型全數 NO-GO ⇒ 依事前判準零成本判死，不開 GPU、不發 LB"))

    Path(a.json).write_text(json.dumps(
        {"verdicts": verdicts, "val": val, "test": test,
         "constants": {"SMALL_T": SMALL_T, "AFM_MAIN": AFM_MAIN, "AFM_ALT": AFM_ALT,
                       "R_GRID": R_GRID, "WIN_KINDS": WIN_KINDS}},
        indent=1, ensure_ascii=False, default=float))
    print(f"\n產出：{a.json}")


if __name__ == "__main__":
    main()
