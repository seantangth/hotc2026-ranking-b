#!/usr/bin/env python3
"""prune_unannotated_v1 — DESIGN.md §3 已定 fallback 的實作（2026-08-11 上游源碼判定後採用）。

判定依據（本機逐字讀 facebookresearch/sam2@2b90b9f 訓練原始碼，非猜測）：
  - PNGRawDataset.get_video 的影格清單來自 JPEGImages 的 **全部 jpg**；
  - PalettisedPNGSegmentLoader.load 對無 PNG 的幀是 `self.frame_id_to_png_filename[frame_id]`
    → **KeyError 直接崩潰**，不是「進 clip 不計 loss」。
  ⇒ DESIGN §3 的未驗證假設不成立，依既定 fallback：**只保留有標註 PNG 的影格**（不重新設計）。

動作：對 DAVIS 樹逐序列 unlink 沒有對應 PNG 的 jpg symlink；序列剩餘幀 < min-frames 者
從訓練清單剔除。輸出修剪後清單與統計。
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis-root", required=True)
    ap.add_argument("--seq-list", required=True)
    ap.add_argument("--min-frames", type=int, default=8, help="= scratch.num_frames")
    ap.add_argument("--out-list", required=True)
    ap.add_argument("--stats-json", required=True)
    a = ap.parse_args()

    root = Path(a.davis_root)
    seqs = [s.strip() for s in Path(a.seq_list).read_text().split() if s.strip()]
    kept_seqs, stats = [], {}
    for seq in seqs:
        img_d = root / "JPEGImages" / seq
        ann_d = root / "Annotations" / seq
        if not img_d.is_dir():
            stats[seq] = {"error": "no JPEGImages dir"}
            continue
        pngs = {p.stem for p in ann_d.glob("*.png")} if ann_d.is_dir() else set()
        jpgs = sorted(img_d.glob("*.jpg"))
        removed = 0
        for j in jpgs:
            if j.stem not in pngs:
                j.unlink()
                removed += 1
        n_left = len(jpgs) - removed
        # 樹不變量：剩下的每個 jpg 都必須有 png（loader 的 KeyError 前提徹底消除）
        left = {p.stem for p in img_d.glob("*.jpg")}
        assert left <= pngs, f"{seq}: 修剪後仍有無標註影格 {sorted(left - pngs)[:3]}"
        stats[seq] = {"n_total": len(jpgs), "n_kept": n_left, "n_removed": removed}
        if n_left >= a.min_frames:
            kept_seqs.append(seq)
        else:
            stats[seq]["excluded"] = f"kept {n_left} < min {a.min_frames}"

    Path(a.out_list).write_text("\n".join(kept_seqs) + "\n")
    tot = sum(s.get("n_total", 0) for s in stats.values())
    kept = sum(s.get("n_kept", 0) for s in stats.values())
    summary = {"n_seqs_in": len(seqs), "n_seqs_kept": len(kept_seqs),
               "n_frames_total": tot, "n_frames_annotated": kept,
               "coverage": round(kept / tot, 4) if tot else 0.0, "per_seq": stats}
    Path(a.stats_json).write_text(json.dumps(summary, indent=1))
    print(f"[prune_v1] 序列 {len(seqs)}→{len(kept_seqs)}｜幀 {tot}→{kept}"
          f"（標註覆蓋率 {summary['coverage']:.1%}）")
    excluded = [s for s in seqs if s not in kept_seqs]
    if excluded:
        print(f"[prune_v1] 剔除（幀數不足）：{excluded}")


if __name__ == "__main__":
    main()
