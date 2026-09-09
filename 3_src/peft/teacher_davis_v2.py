#!/usr/bin/env python3
"""teacher_davis_v2 — 逐幀 teacher mask ＋ 直接物化成 DAVIS/MOSE 目錄樹（E-F video PEFT 的資料層）。

設計依據：5_outputs/peft_phase0_20260810/DESIGN.md §3。與 v1（hsot/teacher_gen.py）的三處差異：

  1. **stride 1（逐幀）**——v1 的 stride 5 使訓練時的記憶跨度是推論時的 5 倍，正踩 D050 已證有害的
     變因（記憶的時間密度 > 跨度）。逐幀讓訓練與推論的時間密度一致。
  2. **低 tightness 不丟棄影格，只是不寫標註 PNG**——v1 的丟棄會把幀距打成不規則（實測 nir-toy_car
     變成 0,15,25,30,35…）。本版讓該幀照常參與 propagation 但不計 loss，且幀距嚴格連續。
  3. **直接輸出 DAVIS 樹**，不經 npz 中轉——SAM2 官方 training/README 明文支援 DAVIS-style 資料集，
     是風險最低的接點（不需猜 VOSRawDataset 內部 API）。

⚠️ 對位規則逐字沿用 v1（AUDIT.md §1 已實測釘死，錯了不會報錯只會靜默學錯）：
     jpgs = sorted(glob("*.jpg"))                       # 實測 105/105 支皆 0001..N 連續
     rows = gt[gt.seq==seq].sort_values("frame").reset_index(drop=True)
     第 pos 幀 ⇔ jpgs[pos] ⇔ rows.iloc[pos]            # 位置對位，非 GT ID 後綴
   GT ID 的數字後綴是**跨序列流水號且各模態範圍重疊**，絕不可當幀號用。

⚠️ 呼叫端須先排除 n_jpg != n_gt 的序列（GT 雙區塊 → 對位損毀，AUDIT.md §2）。本腳本會自行複檢並拒跑。

⚠️ **JPEGImages 是 symlink**（省磁碟與 I/O）⇒ 這棵樹只在「假色 tar 已解開的同一台機」上有效。
   打包回傳時 `tar -ch`（dereference）否則只存到斷鏈的符號連結；rclone 回傳同理需 `--copy-links`。

輸出（out-dir）：
  JPEGImages/<seq>/00000.jpg …      全部影格（symlink，不複製）
  Annotations/<seq>/00000.png …     僅 tightness ≥ floor 的幀（palette PNG，物件 id=1）
  meta/<seq>.json                   逐幀 tightness／是否標註／GT 框（診斷與事後加權用）

用法（已裝 samurai/sam2 的機器）：
  python3 teacher_davis_v2.py --frames-root <假色根> --gt-csv 2026training.csv \
      --seq-list peft_nir_train.txt --samurai-dir ~/samurai --ckpt <sam2.1-L.pt> \
      --out-dir ~/davis_nir --tightness-floor 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# DAVIS palette：index 0 = 背景(黑)、index 1 = 物件(紅)，其餘留白。
_PALETTE = [0, 0, 0, 128, 0, 0] + [0] * (256 * 3 - 6)


def box_iou(a, b) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    import torch
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--seq-list", required=True)
    ap.add_argument("--samurai-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tightness-floor", type=float, default=0.5,
                    help="固定常數，非 val 掃出來的門檻（D033/D037）")
    args = ap.parse_args()

    # S-02(D) 教訓：chdir 之前先把所有路徑解析成絕對路徑
    frames_root = Path(args.frames_root).resolve()
    gt_csv = Path(args.gt_csv).resolve()
    seq_list = Path(args.seq_list).resolve()
    ckpt = Path(args.ckpt).resolve()
    out_dir = Path(args.out_dir).resolve()
    samurai = Path(args.samurai_dir).resolve()

    os.chdir(samurai / "sam2")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor(
        build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(ckpt), device="cuda:0"))

    gt = pd.read_csv(gt_csv)
    gt.columns = ["ID", "x", "y", "w", "h"]
    parts = gt["ID"].str.rsplit("_", n=1, expand=True)
    gt["seq"], gt["frame"] = parts[0], parts[1].astype(int)

    (out_dir / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (out_dir / "Annotations").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    seqs = [s.strip() for s in seq_list.read_text().split() if s.strip()]
    failed: list[str] = []
    for si, seq in enumerate(seqs):
        meta_f = out_dir / "meta" / f"{seq}.json"
        if meta_f.exists():                       # resume：以 meta 落地為完成標記（最後才寫）
            continue
        jpgs = sorted((frames_root / seq).glob("*.jpg"))
        rows = gt[gt["seq"] == seq].sort_values("frame").reset_index(drop=True)

        # fail-closed：GT 雙區塊序列的 image↔box 配對必錯，拒跑而非默默截斷（AUDIT.md §2）
        if len(jpgs) == 0:
            print(f"[SKIP] {seq}: 無影格", file=sys.stderr); failed.append(seq); continue
        if len(jpgs) != len(rows):
            print(f"[REFUSE] {seq}: n_jpg={len(jpgs)} != n_gt={len(rows)}（GT 雙區塊，對位不可信）",
                  file=sys.stderr)
            failed.append(seq); continue

        img_d = out_dir / "JPEGImages" / seq
        ann_d = out_dir / "Annotations" / seq
        img_d.mkdir(exist_ok=True); ann_d.mkdir(exist_ok=True)

        recs, n_ann = [], 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for pos in range(len(jpgs)):
                src = jpgs[pos]
                dst = img_d / f"{pos:05d}.jpg"
                if not dst.exists():
                    dst.symlink_to(src)           # symlink：不複製，省磁碟與 I/O
                r = rows.iloc[pos]
                g = [float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])]
                rec = {"pos": pos, "box": g, "tightness": None, "annotated": False}
                if min(g) <= 0:                   # 官方無效幀（遮擋／出視野）：留影格、不標註
                    recs.append(rec); continue
                img = np.array(Image.open(src).convert("RGB"))
                predictor.set_image(img)
                masks, _, _ = predictor.predict(
                    box=np.array([g[0], g[1], g[0] + g[2], g[1] + g[3]]), multimask_output=False)
                m = masks[0].astype(bool)
                ys, xs = np.nonzero(m)
                if len(xs) == 0:
                    recs.append(rec); continue
                pb = [float(xs.min()), float(ys.min()),
                      float(xs.max() + 1 - xs.min()), float(ys.max() + 1 - ys.min())]
                t = box_iou(pb, g)
                rec["tightness"] = t
                if t >= args.tightness_floor:
                    p = Image.fromarray(m.astype(np.uint8), mode="P")
                    p.putpalette(_PALETTE)
                    p.save(ann_d / f"{pos:05d}.png")
                    rec["annotated"] = True; n_ann += 1
                recs.append(rec)

        tmp = meta_f.with_suffix(".json.part")     # D059：.part + fsync + atomic rename
        with open(tmp, "w") as fh:
            json.dump({"seq": seq, "n_frames": len(jpgs), "n_annotated": n_ann,
                       "tightness_floor": args.tightness_floor, "frames": recs}, fh)
            fh.flush(); os.fsync(fh.fileno())
        tmp.rename(meta_f)
        print(f"[{si+1}/{len(seqs)}] {seq}: {len(jpgs)} 幀 → 標註 {n_ann} "
              f"({n_ann/len(jpgs):.0%})", flush=True)

    if failed:
        print(f"FAILED_SEQS={len(failed)}: {failed}", file=sys.stderr)
        sys.exit(2)                               # fail-closed（D059 ③）
    print("TEACHER-DAVIS-V2-DONE")


if __name__ == "__main__":
    main()
