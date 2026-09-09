#!/usr/bin/env python3
"""slide_rerun 的座標往返回歸測試（合成資料，純 CPU、秒級）。

為什麼需要：prep 把原圖座標搬進「逐段平移的窗座標系」，merge 再搬回來。
這條往返若有 off-by-one 或段對齊錯誤，**不會報錯，只會產生錯位的框**——
而 test 無 GT，錯位在提交前看不出來（D040：本地只能做完整性檢查）。
故用「已知答案的合成序列」把往返鎖死：

  原圖真值軌跡 → prep 產生窗表 → 模擬 tracker 在裁切座標系輸出「完美」框
  → merge 搬回原圖 → **應與原圖真值逐幀相等**

同時驗證 E19 序列被 --exclude-meta 排除（D038 範圍鐵律，不靠人記得）。

執行：python3 -m hsot.test_slide_rerun
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

W, H, N = 400, 240, 120
BOX_W, BOX_H = 12, 10


def build_truth() -> pd.DataFrame:
    """三支序列，各驗一條規則：

      moving-tiny   小目標、橫掃全圖（全序列窗必然過大）→ **應被滑動窗選中**
      excluded-tiny 與上者同型，但列在 --exclude-meta 裡 → **應被範圍鐵律擋下**
                    （必須也是小目標，否則會先被 SMALL_T 濾掉，就測不到 exclude 了）
      static-big    大目標 → 應被 SMALL_T 濾掉
    """
    rows = []
    for i in range(N):
        f = i + 1
        x = 20 + (W - 60) * i / (N - 1)          # 橫掃全圖 → 全序列窗必然過大
        y = 100 + 40 * np.sin(2 * np.pi * i / N)
        rows.append(("moving-tiny", f, x, y, BOX_W, BOX_H))
        rows.append(("excluded-tiny", f, x, y, BOX_W, BOX_H))
        rows.append(("static-big", f, 100.0, 100.0, 60.0, 60.0))
    return pd.DataFrame(rows, columns=["seq", "frame", "x", "y", "width", "height"])


def to_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out["ID"] = out["seq"] + "_" + out["frame"].astype(str)
    out[["ID", "x", "y", "width", "height"]].to_csv(path, index=False)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="slide_test_"))
    truth = build_truth()
    frames_root = tmp / "frames"
    for seq in truth["seq"].unique():
        d = frames_root / seq
        d.mkdir(parents=True)
        for f in range(1, N + 1):
            Image.new("RGB", (W, H), (30, 30, 30)).save(d / f"{f:06d}.jpg", quality=90)
        g = truth[(truth.seq == seq) & (truth.frame == 1)].iloc[0]
        (d / "init_rect.txt").write_text(f"{g.x} {g.y} {g.width} {g.height}")

    base_csv = tmp / "base.csv"
    to_csv(truth, base_csv)
    # 排除表：假裝 excluded-tiny 已被 E19 選中。它與 moving-tiny 逐幀相同，
    # 唯一差別就是列在這張表裡 → 若它也被選中，就證明範圍鐵律沒生效。
    excl = tmp / "excl.json"
    excl.write_text(json.dumps({"excluded-tiny": {"seq": "excluded-tiny"}}))

    meta = tmp / "slide_meta.json"
    out_root = tmp / "slide_fc"
    r = subprocess.run(
        [sys.executable, "-m", "hsot.slide_rerun", "prep",
         "--frames-root", str(frames_root), "--base-csv", str(base_csv),
         "--exclude-meta", str(excl), "--segments", "8",
         "--out-root", str(out_root), "--meta", str(meta)],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        print(f"❌ prep 失敗\n{r.stderr}")
        return 1

    m = json.loads(meta.read_text())
    fails = []
    if "excluded-tiny" in m:
        fails.append("excluded-tiny 未被 --exclude-meta 排除（D038 範圍鐵律失效）")
    if "static-big" in m:
        fails.append("static-big 未被 SMALL_T 濾掉（選序列規則失效）")
    if "moving-tiny" not in m:
        print(f"❌ moving-tiny 未被選中；窗表 = {m}")
        return 1

    w = m["moving-tiny"]
    cw, ch = w["win"]
    # 所有裁切幀必須同尺寸——這是「可拼成單一連續影片」的前提
    sizes = {Image.open(p).size for p in sorted((out_root / "moving-tiny").glob("*.jpg"))}
    if sizes != {(cw, ch)}:
        fails.append(f"裁切幀尺寸不一致：{sizes}（應全為 {(cw, ch)}）")
    n_crop = len(list((out_root / "moving-tiny").glob("*.jpg")))
    if n_crop != N:
        fails.append(f"裁切幀數 {n_crop} != {N}")

    # 模擬 tracker：在裁切座標系輸出「完美」框（＝真值減去該幀所屬段的偏移）
    gt = truth[truth.seq == "moving-tiny"].sort_values("frame").reset_index(drop=True)
    ox = np.zeros(N); oy = np.zeros(N)
    for s in w["segments"]:
        sel = (gt.frame >= s["lo"]) & (gt.frame <= s["hi"])
        ox[sel.to_numpy()], oy[sel.to_numpy()] = s["ox"], s["oy"]
    sim = pd.DataFrame({"seq": "moving-tiny", "frame": np.arange(1, N + 1),
                        "x": gt.x.to_numpy() - ox, "y": gt.y.to_numpy() - oy,
                        "width": gt.width, "height": gt.height})
    # 裁切座標必須全部落在窗內，否則 prep 的窗規劃有誤
    if (sim.x < 0).any() or (sim.y < 0).any() or \
       (sim.x + sim.width > cw).any() or (sim.y + sim.height > ch).any():
        fails.append("真值在裁切座標系中落到窗外 → 窗規劃有誤")

    crop_csv = tmp / "crop.csv"
    to_csv(sim, crop_csv)
    final = tmp / "final.csv"
    r = subprocess.run(
        [sys.executable, "-m", "hsot.slide_rerun", "merge",
         "--base-csv", str(base_csv), "--crop-csv", str(crop_csv),
         "--meta", str(meta), "--out", str(final)],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        print(f"❌ merge 失敗\n{r.stderr}")
        return 1

    got = pd.read_csv(final)
    got["seq"] = got.ID.str.rsplit("_", n=1).str[0]
    got["frame"] = got.ID.str.rsplit("_", n=1).str[1].astype(int)
    mv = got[got.seq == "moving-tiny"].sort_values("frame")
    err = np.abs(mv[["x", "y", "width", "height"]].to_numpy()
                 - gt[["x", "y", "width", "height"]].to_numpy()).max()
    if err > 1e-6:
        fails.append(f"往返誤差 {err:.4f}px（應為 0）—— 座標映射有 bug")

    # 未選中的序列（無論是被 exclude 擋下還是被 SMALL_T 濾掉）必須逐位元沿用 base
    for seq in ("excluded-tiny", "static-big"):
        sb = got[got.seq == seq].sort_values("frame")
        bb = truth[truth.seq == seq].sort_values("frame")
        if not np.allclose(sb[["x", "y", "width", "height"]].to_numpy(),
                           bb[["x", "y", "width", "height"]].to_numpy()):
            fails.append(f"{seq} 遭改動（merge 汙染了未選中序列）")

    print()
    if fails:
        for f in fails:
            print(f"❌ {f}")
        return 1
    print(f"✅ 全部通過：往返誤差 0px｜{N} 幀同尺寸 {cw}×{ch}｜"
          f"放大 {w['zoom']:.1f}x｜排除規則生效｜base 未受汙染")
    return 0


if __name__ == "__main__":
    sys.exit(main())
