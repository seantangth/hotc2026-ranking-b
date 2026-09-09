#!/usr/bin/env python3
"""teacher_rect_v1 — D068 Gate 1 的 rect-target DAVIS 樹生成器（零 GPU、零 SAM2 依賴）。

【與 teacher_davis_v2 的本質差異】target 不再是 teacher（SAM2.1-L image-mode）的 mask，
而是 **GT box 填滿的矩形 mask**。配合 loss_box_proj_v1（純軸投影 loss），數學上等價於
box 監督：矩形的 x/y 軸投影恰為 box 區間，loss 從不看矩形內部形狀 ⇒ 不會教模型輸出矩形。
自蒸餾恆等式天花板（teacher=student 同權重）與 teacher 品質上限（tightness 中位 0.892）同時消失；
GT≤0（遮擋/出視野）幀不寫 PNG ⇒ 後續 prune_unannotated_v1 剔除（v2 同款流程）。

【三種輸出】
  1. 基礎序列   {seq}          全幀（有效 GT 幀皆有 rect PNG）
  2. 速度增強   {seq}__x{k}    每 k 幀取 1（--speed-strides "4"；假說②解藥：慢速池 ×k 倍速）
  3. 失敗段 clip {seq}__e{i:03d} 由 --events-json 指定的事件窗（Gate 0 產出）

【承襲 v2 的 fail-closed】n_jpg != n_gt（GT 雙區塊）⇒ REFUSE 整支拒跑；meta 用 .part+fsync+rename。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_PALETTE = [0, 0, 0, 128, 0, 0] + [0] * (256 * 3 - 6)


def write_variant(name: str, jpg_paths: list, boxes: list, sizes: tuple, out_dir: Path) -> dict:
    """把 (jpg, box) 序列以 0-based 連續編號寫成一個 DAVIS 序列。box=None ⇒ 該幀不標註。"""
    H, W = sizes
    img_d = out_dir / "JPEGImages" / name
    ann_d = out_dir / "Annotations" / name
    img_d.mkdir(parents=True, exist_ok=True)
    ann_d.mkdir(parents=True, exist_ok=True)
    n_ann = 0
    for pos, (src, g) in enumerate(zip(jpg_paths, boxes)):
        dst = img_d / f"{pos:05d}.jpg"
        if not dst.exists():
            dst.symlink_to(src)
        if g is None:
            continue
        x0 = max(0, int(round(g[0])));  y0 = max(0, int(round(g[1])))
        x1 = min(W, int(round(g[0] + g[2])));  y1 = min(H, int(round(g[1] + g[3])))
        if x1 <= x0 or y1 <= y0:
            continue                                   # 退化框：不標註（與 GT<=0 同路徑）
        m = np.zeros((H, W), dtype=np.uint8)
        m[y0:y1, x0:x1] = 1
        p = Image.fromarray(m, mode="P")
        p.putpalette(_PALETTE)
        p.save(ann_d / f"{pos:05d}.png")
        n_ann += 1
    return {"name": name, "n_frames": len(jpg_paths), "n_annotated": n_ann}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-csv", required=True, help="2026training.csv")
    ap.add_argument("--frames-root", required=True, help="每序列一個子目錄（jpg 幀）")
    ap.add_argument("--seq-list", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--speed-strides", default="", help='逗號分隔，如 "4"；空字串=不生成')
    ap.add_argument("--events-json", default="", help="gate0_events_v1 產出；給了才生成事件 clip")
    ap.add_argument("--clip-pre", type=int, default=2, help="事件 clip 往事件起點前多取幀數")
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    frames_root = Path(a.frames_root)
    gt = pd.read_csv(a.gt_csv)
    gt.columns = ["ID", "x", "y", "w", "h"]
    parts = gt["ID"].str.rsplit("_", n=1, expand=True)
    gt["seq"], gt["frame"] = parts[0], parts[1].astype(int)

    seqs = [s.strip() for s in Path(a.seq_list).read_text().split() if s.strip()]
    strides = [int(s) for s in a.speed_strides.split(",") if s.strip()]
    events = []
    if a.events_json:
        events = json.loads(Path(a.events_json).read_text())["events"]

    manifest, failed = [], []
    for si, seq in enumerate(seqs):
        jpgs = sorted((frames_root / seq).glob("*.jpg"))
        rows = gt[gt["seq"] == seq].sort_values("frame").reset_index(drop=True)
        if len(jpgs) == 0:
            print(f"[SKIP] {seq}: 無影格", file=sys.stderr); failed.append(seq); continue
        if len(jpgs) != len(rows):
            print(f"[REFUSE] {seq}: n_jpg={len(jpgs)} != n_gt={len(rows)}（GT 雙區塊，對位不可信）",
                  file=sys.stderr)
            failed.append(seq); continue
        with Image.open(jpgs[0]) as im:
            W, H = im.size
        boxes = []
        for pos in range(len(jpgs)):
            r = rows.iloc[pos]
            g = [float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])]
            boxes.append(None if min(g) <= 0 else g)    # 官方無效幀：留影格、不標註

        manifest.append(write_variant(seq, jpgs, boxes, (H, W), out_dir))
        for k in strides:
            idx = list(range(0, len(jpgs), k))
            if len(idx) < 8:
                continue                                 # 短於 num_frames 的變體無意義
            manifest.append(write_variant(
                f"{seq}__x{k}", [jpgs[i] for i in idx], [boxes[i] for i in idx], (H, W), out_dir))
        print(f"[{si+1}/{len(seqs)}] {seq}: {len(jpgs)} 幀 ok", flush=True)

    n_evt = 0
    for i, ev in enumerate(events):
        seq = ev["seq"]
        if seq in failed:
            continue
        jpgs = sorted((frames_root / seq).glob("*.jpg"))
        rows = gt[gt["seq"] == seq].sort_values("frame").reset_index(drop=True)
        if len(jpgs) != len(rows):
            continue
        with Image.open(jpgs[0]) as im:
            W, H = im.size
        s = max(0, int(ev["start"]) - a.clip_pre)
        e = min(len(jpgs), s + a.clip_len)
        if e - s < 8:
            continue
        boxes = []
        for pos in range(s, e):
            r = rows.iloc[pos]
            g = [float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])]
            boxes.append(None if min(g) <= 0 else g)
        if boxes[0] is None:
            continue                                     # clip 首幀必須有有效 GT（條件幀）
        manifest.append(write_variant(
            f"{seq}__e{i:03d}", jpgs[s:e], boxes, (H, W), out_dir))
        n_evt += 1

    tmp = Path(a.manifest).with_suffix(".json.part")
    with open(tmp, "w") as fh:
        json.dump({"variants": manifest, "n_events_written": n_evt,
                   "failed_seqs": failed}, fh, indent=1)
        fh.flush(); os.fsync(fh.fileno())
    tmp.rename(a.manifest)
    print(f"[rect_v1] 變體 {len(manifest)}（事件 clip {n_evt}）；REFUSE {len(failed)}: {failed}")
    if failed and len(failed) > 4:                       # v2 已知雙區塊 4 支；更多=資料異常
        sys.exit(2)
    print("TEACHER-RECT-V1-DONE")


if __name__ == "__main__":
    main()
