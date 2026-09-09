#!/usr/bin/env python3
"""E14 小目標 crop-zoom(v1a:envelope 固定窗兩段式,吃 S4 槓桿)。

原理:SAM2 內部把輸入 resize 到 1024——裁掉無關區域=把 1024 解析度分配給更小視野,
小目標有效解析度提升。逐幀動窗與 video predictor 架構衝突,v1a 用「base 軌跡全程
外包框 + margin」的固定窗:記憶穩定、目標恆在窗內(base 軌跡覆蓋)。

選序列規則(確定性、無 GT、不認序列名——Ranking B 合法):
  首幀 box sqrt(w*h) < 32(小目標)且 固定窗面積 < 原圖 40%(放大倍率足夠)。

兩段用法:
  1) prep:生成裁切資料集 + 轉換後 init + 窗口表
     python3.14 -m hsot.crop_rerun prep --frames-root <原圖根> --base-csv <blend.csv> \
         --out-root <裁切根> --meta <windows.json>
     # 實驗性的高保真裁切（預設仍是歷史 legacy-q95）：
     python3.14 -m hsot.crop_rerun prep ... --jpeg-profile q100-444
  2)(外部)track_t1.py --frames-root <裁切根> 跑選中序列
  3) merge:把裁切座標輸出映射回原圖,未選序列保留 base
     python3.14 -m hsot.crop_rerun merge --base-csv <blend.csv> --crop-csv <裁切輸出> \
         --meta <windows.json> --out <最終.csv>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SMALL_T = 32.0
AREA_FRAC_MAX = 0.40
MARGIN_SCALE = 2.0   # 窗 = 軌跡 envelope 外擴 2×目標尺度(呼應 E04 R=2–3× 域內證據)
MIN_MARGIN = 32.0
JPEG_PROFILE_LEGACY = "legacy-q95"
JPEG_PROFILE_Q100_444 = "q100-444"
JPEG_PROFILES = (JPEG_PROFILE_LEGACY, JPEG_PROFILE_Q100_444)


def _jpeg_encoding(profile: str) -> tuple[dict, dict]:
    """回傳 (metadata, Pillow save kwargs)。

    legacy profile 刻意只傳 ``quality=95``：多傳任何 encoder 選項都可能
    改變過去 crop 的 bytes，因此預設路徑不顯式指定 subsampling。
    q100-444 則依 Pillow JPEG 介面顯式傳 ``subsampling=0``（4:4:4）。
    """
    if profile == JPEG_PROFILE_LEGACY:
        kwargs = {"quality": 95}
        metadata = {
            "profile": profile,
            "format": "JPEG",
            "quality": 95,
            "subsampling": "pillow-default",
        }
    elif profile == JPEG_PROFILE_Q100_444:
        kwargs = {"quality": 100, "subsampling": 0}
        metadata = {
            "profile": profile,
            "format": "JPEG",
            "quality": 100,
            "subsampling": 0,
            "chroma_subsampling": "4:4:4",
        }
    else:
        raise ValueError(f"未知 JPEG profile: {profile!r}")
    return metadata, kwargs


def parse(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    p = df["ID"].str.rsplit("_", n=1, expand=True)
    df["seq"], df["frame"] = p[0], p[1].astype(int)
    return df


def _window(boxes: np.ndarray, scale: float, W: float, H: float) -> tuple[int, int, int, int]:
    """軌跡 envelope 外擴 margin 後夾在原圖內，回傳整數窗 (x1,y1,x2,y2)。"""
    m = max(MARGIN_SCALE * scale, MIN_MARGIN)
    x1 = max(0.0, float(np.min(boxes[:, 0])) - m)
    y1 = max(0.0, float(np.min(boxes[:, 1])) - m)
    x2 = min(float(W), float(np.max(boxes[:, 0] + boxes[:, 2])) + m)
    y2 = min(float(H), float(np.max(boxes[:, 1] + boxes[:, 3])) + m)
    return int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2))


def cmd_prep(args) -> None:
    from PIL import Image

    base = parse(args.base_csv)
    jpeg_profile = getattr(args, "jpeg_profile", JPEG_PROFILE_LEGACY)
    jpeg_metadata, jpeg_save_kwargs = _jpeg_encoding(jpeg_profile)
    # --envelope-extra：額外軌跡，窗口取聯集。
    # 為何需要：envelope 來自 base 軌跡，但**跟丟時軌跡會凍結在錯的位置**，
    # 窗口就會把真目標切在外面（凍結軌跡的 envelope 又特別小，還會通過 40% 面積檢查）。
    # 取兩條獨立軌跡的聯集可大幅降低此風險，且仍是確定性、不認序列名（Ranking B 合法）。
    extra = parse(args.envelope_extra) if getattr(args, "envelope_extra", None) else None
    frames_root = Path(args.frames_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    # --segments K：把序列切成 K 段、每段各算窗（K=1 即 E19 原行為）。
    # 為何分段嚴格優於「放寬面積門檻」（08-06 crop_sweep 實測 75 支 test）：
    #   k=1 門檻 0.40 → 選中 21 支/30.0% 幀/線性放大中位 2.05
    #   k=1 門檻 0.70 → 選中 35 支/49.4% 幀/放大 1.68  ← 放寬納入的都是大窗，把品質拉低
    #   k=4 門檻 0.40 → 選中 38 支/51.1% 幀/放大 2.50  ← 覆蓋與放大同時變好
    # 短段內目標移動範圍小 → 窗更小 → 放大更大，且更多序列的窗通得過原門檻。
    # 門檻一律不動，維持 E19 已在 test 驗證過的規則（少一個變因）。
    K = max(1, int(getattr(args, "segments", 1) or 1))
    # --area-frac-max：門檻參數化（預設 = AREA_FRAC_MAX = 0.40，**既有行為位元級不變**）。
    # E27（08-09）：上面那段「放寬納入的都是大窗」的否決是 08-06 寫的，依據為
    # 「線性放大中位 2.05→1.68」＝**解析度框架**；而 D046（08-07）已用 v013/v014 的完整 2×2
    # 推翻解析度說、確立 crop 的價值在**移除干擾物** ⇒ 否決寫在其前提被推翻之前。
    # 08-09 幾何前測：test 被剔除的 22 支中 11 支的窗僅 40–60%（仍移除 40%+ 的場），
    # 屬「中等窗」而非「大窗」；sweep 當年測過 0.70（35 支）卻**從未測過 0.60**。
    afm = float(getattr(args, "area_frac_max", None) or AREA_FRAC_MAX)
    only = set(getattr(args, "only_seqs", None) or [])
    meta = {}
    n_small = n_area_reject = n_seg_reject = 0
    for seq, g in base.groupby("seq"):
        if only and seq not in only:
            continue
        g = g.sort_values("frame")
        first = g.iloc[0]
        scale = float(np.sqrt(first["width"] * first["height"]))
        if scale >= SMALL_T:
            continue
        n_small += 1
        seq_dir = frames_root / seq
        jpgs = sorted(seq_dir.glob("*.jpg"), key=lambda q: int(q.stem))
        if not jpgs:
            continue
        # Frame value -> image file, by position within this sequence.
        #
        # This used to be `jpgs[fr - 1]`, which assumed base["frame"] is 1-based
        # and contiguous.  That holds for the Ranking-A test set but NOT for the
        # train405 contract, whose frame values are global GT row numbers
        # (nir-rider15 runs 16305..16459), so the old index blew past the end of
        # a 155-image list.  It hid this long because test75 uses K=1, where
        # merge's `f_lo + f - 1` degenerates to the identity and both readings
        # agree -- the v049 bit-exact selftest cannot discriminate them.
        # Deriving the position from the data is convention-agnostic.
        if len(g) != len(jpgs):
            raise SystemExit(
                f"{seq}: base 有 {len(g)} 列但目錄有 {len(jpgs)} 張影像；"
                "幀與影像不對齊，拒絕裁切")
        frame_to_jpg = dict(zip(g["frame"].tolist(), jpgs, strict=True))
        W, H = Image.open(jpgs[0]).size
        e_all = extra[extra["seq"] == seq] if extra is not None else None

        # 段邊界以「序列內位置」切分；輸出幀號沿用 base 的 frame 值（可為任意編號）
        n = len(g)
        bounds = np.linspace(0, n, K + 1).astype(int)
        seg_ok = 0
        for si in range(K):
            lo, hi = int(bounds[si]), int(bounds[si + 1])
            if hi <= lo:
                continue
            gs = g.iloc[lo:hi]
            b = gs[["x", "y", "width", "height"]].to_numpy(float)
            if e_all is not None and len(e_all):
                # 以 frame 值對齊（非位置切片）——兩條軌跡的幀集合不保證等長
                es = e_all[e_all["frame"].isin(gs["frame"].values)]
                if len(es):
                    b = np.vstack([b, es[["x", "y", "width", "height"]].to_numpy(float)])
            x1i, y1i, x2i, y2i = _window(b, scale, W, H)
            if (x2i - x1i) * (y2i - y1i) >= afm * W * H:
                if K == 1:
                    n_area_reject += 1
                else:
                    n_seg_reject += 1
                continue  # 放大倍率不足，該段保留 base
            name = seq if K == 1 else f"{seq}__seg{si:02d}"
            dest = out_root / name
            dest.mkdir(exist_ok=True)
            frames = gs["frame"].tolist()
            # 段內幀 → 檔案：以序列內位置對應，不假設幀號的起點或連續性
            for fr in frames:
                f = frame_to_jpg[fr]
                Image.open(f).crop((x1i, y1i, x2i, y2i)).save(
                    dest / f.name, **jpeg_save_kwargs)
            f0 = gs.iloc[0]
            init = [float(f0["x"]) - x1i, float(f0["y"]) - y1i,
                    float(f0["width"]), float(f0["height"])]
            (dest / "init_rect.txt").write_text(" ".join(str(v) for v in init))
            meta[name] = {"x1": x1i, "y1": y1i, "w": x2i - x1i, "h": y2i - y1i,
                          "orig": [W, H], "seq": seq, "frames": [frames[0], frames[-1]],
                          "zoom": round(W * H / ((x2i - x1i) * (y2i - y1i)), 2),
                          # 放在每個 window 內，保持頂層仍為 seq/segment 名稱映射，
                          # 舊 merge 與產生 seq-list 的工具不會把 schema header 誤當序列。
                          "crop_image_encoding": dict(jpeg_metadata)}
            seg_ok += 1
        if seg_ok:
            z = [meta[k]["zoom"] for k in meta if meta[k]["seq"] == seq]
            print(f"{seq}: {seg_ok}/{K} 段選中，放大 {min(z):.1f}–{max(z):.1f}x")
    Path(args.meta).write_text(json.dumps(meta, indent=1))
    zooms = [m["zoom"] for m in meta.values()]
    n_seq = len({m["seq"] for m in meta.values()})
    print(f"\n小目標(<{SMALL_T:.0f}px) {n_small} 支；K={K}；"
          f"面積過大剔除 {n_area_reject} 支 / {n_seg_reject} 段；"
          f"**選中 {n_seq} 支、{len(meta)} 段** → {args.out_root}")
    if zooms:
        print(f"放大倍率(面積) 中位 {np.median(zooms):.1f}x  範圍 {min(zooms):.1f}–{max(zooms):.1f}x"
              f"（線性中位 {np.sqrt(np.median(zooms)):.2f}x）")
    print(f"窗口表 → {args.meta}")
    print(f"裁切 JPEG 編碼 → {jpeg_profile}")


def cmd_merge(args) -> None:
    """把裁切座標映射回原圖；未選中的序列/段一律沿用 base。

    分段版（meta 帶 `seq`/`frames`）的關鍵差異：一個原序列可能只有部分段被裁切，
    故以「原序列的 frame」為主鍵逐幀覆蓋，而非整支替換。裁切子序列的輸出幀號是
    段內 1-based（track_t1 無 --gt-csv 時的行為），需加上該段起始幀還原。
    """
    base = parse(args.base_csv)
    crop = parse(args.crop_csv)
    meta = json.loads(Path(args.meta).read_text())
    done = set(crop["seq"])

    out = base.set_index("ID")[["x", "y", "width", "height"]].copy()
    n_seq, n_seg, n_frame = set(), 0, 0
    for name, w in meta.items():
        if name not in done:
            continue
        seq = w.get("seq", name)
        c = crop[crop["seq"] == name].sort_values("frame")
        f_lo = w.get("frames", [1, None])[0]
        # 段內 1-based → 原序列幀號
        ids = [f"{seq}_{f_lo + int(f) - 1}" for f in c["frame"]]
        miss = [i for i in ids if i not in out.index]
        assert not miss, f"{name}: {len(miss)} 個幀號不存在於 base（首個 {miss[0]}）"
        out.loc[ids, "x"] = c["x"].to_numpy() + w["x1"]
        out.loc[ids, "y"] = c["y"].to_numpy() + w["y1"]
        out.loc[ids, "width"] = c["width"].to_numpy()
        out.loc[ids, "height"] = c["height"].to_numpy()
        n_seq.add(seq); n_seg += 1; n_frame += len(ids)

    out = out.reset_index()[["ID", "x", "y", "width", "height"]]
    assert len(out) == len(base), f"列數不符 {len(out)} vs {len(base)}"
    assert (out["ID"].values == base["ID"].values).all(), "ID 順序被改變"
    out.to_csv(args.out, index=False)
    print(f"合併完成：{len(n_seq)} 支 / {n_seg} 段 / {n_frame} 幀 "
          f"({n_frame/len(base):.1%}) 採裁切重跑 → {args.out}")


def cmd_check(args) -> None:
    """分段 merge 後的無 GT 災難檢查（D040：test 無 GT，只能驗完整性與診斷指紋）。

    分段特有的風險是 K=1 不存在的：每段各自 init，**段邊界可能接不上**。
    偵測法不需要 GT——若銜接失敗，邊界那一步的中心位移會遠大於該序列平時的
    逐幀位移。以「邊界位移 / 序列內位移中位數」為指標，比絕對閾值更耐得住
    不同序列的運動速度差異（D024：每序列解析度與運動尺度都不同）。
    """
    base = parse(args.base_csv)
    new = parse(args.new_csv)
    meta = json.loads(Path(args.meta).read_text())

    assert len(new) == len(base), f"列數不符 {len(new)} vs {len(base)}"
    assert (new["ID"].values == base["ID"].values).all(), "ID 順序不符"
    assert new.isna().sum().sum() == 0, "有 NaN"
    assert (new["width"] > 0).all() and (new["height"] > 0).all(), "有非正的 w/h"
    ch = (new[["x", "y", "width", "height"]].to_numpy()
          != base[["x", "y", "width", "height"]].to_numpy()).any(1)
    print(f"完整性 ✅｜變動 {ch.sum():,} 幀（{ch.mean():.1%}）")

    # 每個原序列的段邊界（各段的結束幀；最後一段的結尾不算邊界）
    bnd: dict[str, list[int]] = {}
    for w in meta.values():
        seq = w.get("seq")
        if seq and "frames" in w:
            bnd.setdefault(seq, []).append(int(w["frames"][1]))

    print(f"\n段邊界銜接檢查（{len(bnd)} 支）：")
    flagged = []
    for seq, ends in sorted(bnd.items()):
        g = new[new["seq"] == seq].sort_values("frame")
        cx = g["x"].to_numpy() + g["width"].to_numpy() / 2
        cy = g["y"].to_numpy() + g["height"].to_numpy() / 2
        step = np.hypot(np.diff(cx), np.diff(cy))
        if len(step) < 3:
            continue
        typical = float(np.median(step)) or 1e-6
        fr = g["frame"].to_numpy()
        last = int(fr[-1])
        for e in sorted(set(ends)):
            if e >= last:
                continue  # 序列尾端不是銜接點
            i = int(np.searchsorted(fr, e)) - 1
            if not (0 <= i < len(step)):
                continue
            jump, ratio = float(step[i]), float(step[i] / max(typical, 1e-6))
            # 兩個條件同時成立才算可疑：相對於該序列平時位移異常大、且絕對值夠大
            if ratio > 5 and jump > 15:
                flagged.append((seq, e, jump, ratio, typical))
    if flagged:
        print(f"  ⚠️ {len(flagged)} 個邊界可疑（相對倍率>5 且 絕對位移>15px）：")
        for seq, e, jump, ratio, typ in sorted(flagged, key=lambda r: -r[3])[:12]:
            print(f"    {seq} @frame {e}: 跳 {jump:.1f}px = 該序列平時的 {ratio:.1f}×"
                  f"（平時 {typ:.1f}px）")
    else:
        print("  ✅ 無可疑邊界")

    # 與 base 的逐序列偏離（量級參考，非增益判準——本地無 GT 且 D040 誤差 ±0.023）
    d = new.copy(); b = base.copy()
    cd = np.hypot((d["x"] + d["width"] / 2) - (b["x"] + b["width"] / 2),
                  (d["y"] + d["height"] / 2) - (b["y"] + b["height"] / 2))
    per = pd.DataFrame({"seq": d["seq"], "cd": cd}).groupby("seq").cd.median().sort_values()
    moved = per[per > 0.5]
    print(f"\n與 base 的中心位移：全體中位 {np.median(cd):.1f}px；"
          f"有實質改動的序列 {len(moved)}/{per.size}")
    print(f"  偏離最大 8 支：{moved.tail(8).round(1).to_dict()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prep")
    p1.add_argument("--frames-root", required=True)
    p1.add_argument("--base-csv", required=True)
    p1.add_argument("--envelope-extra", help="額外軌跡 CSV，窗口取聯集（防跟丟軌跡產生錯誤窗口）")
    p1.add_argument("--out-root", required=True)
    p1.add_argument("--meta", required=True)
    p1.add_argument("--area-frac-max", type=float, default=None,
                    help=f"窗面積上限（預設 {AREA_FRAC_MAX}＝E19 已驗證規則，不給即行為不變）。"
                         "E27：0.60 是 08-06 sweep 未測過的點，且當年的否決理由（放大倍率）"
                         "已被 D046 推翻——crop 的價值在移除干擾物，不在解析度")
    p1.add_argument("--only-seqs", nargs="*", default=None,
                    help="只處理這些序列（用於「只跑新增序列、不動 v008 已驗證的 21 支」）")
    p1.add_argument("--segments", type=int, default=1,
                    help="把序列切成 K 段、每段各算窗（預設 1 ＝ E19 原行為）。"
                         "短段軌跡範圍小 → 窗更小 → 放大更大，且更多序列通得過面積門檻。")
    p1.add_argument(
        "--jpeg-profile", choices=JPEG_PROFILES, default=JPEG_PROFILE_LEGACY,
        help=("裁切圖 JPEG 編碼：legacy-q95（預設）完全保留歷史 "
              "Image.save(quality=95) 路徑；q100-444 顯式使用 "
              "quality=100, subsampling=0（4:4:4）。"),
    )
    p2 = sub.add_parser("merge")
    p2.add_argument("--base-csv", required=True)
    p2.add_argument("--crop-csv", required=True)
    p2.add_argument("--meta", required=True)
    p2.add_argument("--out", required=True)
    p3 = sub.add_parser("check", help="分段 merge 後的無 GT 災難檢查（含段邊界銜接）")
    p3.add_argument("--base-csv", required=True, help="對照基準（如 v008）")
    p3.add_argument("--new-csv", required=True, help="待檢查的 merge 產出")
    p3.add_argument("--meta", required=True)
    args = ap.parse_args()
    {"prep": cmd_prep, "merge": cmd_merge, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
