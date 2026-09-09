#!/usr/bin/env python3
"""HSOT T1 推論腳本 — E02 pipeline（SAMURAI + SAM2.1-Large 假色 zero-shot）腳本化。

v2 路線（D034）：本腳本同時是 E12–E14 的載體與 Ranking B 一鍵重現交付物。
與 E02（LB 0.66608）嚴格對齊的固定行為：
  首幀=init 照抄、空 mask 沿用前框、mask→box=nonzero 外接矩形、
  缺幀用最後可信框補滿、SEED 42、torch.inference_mode + bfloat16 autocast、
  offload_video_to_cpu=True。

E15（--backend sam3）：換 SAM3/SAM3.1 底座，其餘行為與 E02 逐項共用
（單變因隔離＝只換 tracker，mask→box／空 mask 沿用／首幀 init 全同）。
**--backend samurai 為預設，且每序列預設隔離 Kalman 狀態**。只有明確加上
``--samurai-legacy-cross-seq-kf`` 才會重現 E02 的跨序列污染行為。

用法（train/val，GT 全域幀號模式——輸出可直接餵 hsot/eval.py 對 2026training.csv）：
  python track_t1.py --frames-root DATA --seq-list val_split_v1.txt \
      --gt-csv 2026training.csv --out-dir out_t1 --mask-cache
用法（test 75 序列，每序列 1-based 幀號 = sample_submission 慣例）：
  python track_t1.py --frames-root DATA --out-dir out_t1

frames-root 佈局：<root>/<seq>/*.jpg（SAM2 video loader 依檔名數字排序）。
init 來源：<seq>/init_rect.txt 優先；否則 --gt-csv 該序列排序後首列（x,y,w,h）。
mask 快取（--mask-cache）：out/masks/<seq>.npz — packbits 二值 mask + 原始 box +
  空 mask 旗標，供 E12 mask→box 精修在 CPU 上離線消融。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODALITIES = ("nir", "rednir", "vis")


def detect_modality(seq: str) -> str:
    for m in ("rednir", "nir", "vis"):  # rednir 須先於 nir 比對
        if seq.startswith(m + "-"):
            return m
    return "unknown"


def read_init(seq_dir: Path, seq: str, gt: pd.DataFrame | None) -> list[float]:
    f = seq_dir / "init_rect.txt"
    if f.exists():
        # 分隔符容錯：官方 Ranking A 為空白分隔，但 VOT 慣例是逗號——9/7 新資料兩種都可能
        # （冷啟動演練 #2 於 08-11 實測抓到：逗號格式會讓舊版 float() 直接炸掉 65/65 支）
        v = [t for t in re.split(r"[,\s]+", f.read_text().strip()) if t]
        if len(v) < 4:
            raise ValueError(f"{seq}: init_rect.txt 欄位不足 4（內容={f.read_text()!r}）")
        return [float(x) for x in v[:4]]
    if gt is not None:
        rows = gt[gt["seq"] == seq].sort_values("frame")
        if len(rows):
            r = rows.iloc[0]
            return [float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])]
    raise FileNotFoundError(f"{seq}: 無 init_rect.txt 且 GT 無此序列")


def mask_to_box(mask: np.ndarray, prev_box: list[float]):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return prev_box, True
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], False


def load_sample_frame_ids(path: str | Path) -> dict[str, list[int]]:
    """讀取 sample/contract 的逐序列 frame ID，並保留檔案內順序。

    Ranking 資料的 ID 是每支 1-based，但 train405 使用全域 frame ID；因此
    ``--sample-csv`` 不只是一個終局 validator，也是位置到輸出 ID 的正式 contract。
    這可避免已知的四支短幀 train 序列被錯誤映射到另一個 GT 區塊。
    """
    sample = pd.read_csv(path)
    if list(sample.columns) != SUB_COLS:
        raise ValueError(f"sample 欄位 {list(sample.columns)} != {SUB_COLS}")
    if sample["ID"].duplicated().any():
        dup = sample.loc[sample["ID"].duplicated(), "ID"].iloc[0]
        raise ValueError(f"sample 含重複 ID：{dup}")
    parts = sample["ID"].astype(str).str.rsplit("_", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("sample ID 必須能拆成 <seq>_<int>")
    try:
        frames = parts[1].astype(int)
    except ValueError as exc:
        raise ValueError("sample frame ID 必須是整數") from exc
    groups: dict[str, list[int]] = {}
    for seq, frame in zip(parts[0], frames, strict=True):
        groups.setdefault(str(seq), []).append(int(frame))
    return groups


def frame_ids_for(seq: str, n_expect: int, gt: pd.DataFrame | None,
                  sample_frame_ids: dict[str, list[int]] | None = None) -> list[int]:
    """位置到輸出 ID：sample contract 優先，其次 GT，最後才是 1-based。"""
    if sample_frame_ids is not None:
        if seq not in sample_frame_ids:
            raise ValueError(f"{seq}: sample contract 缺此序列")
        frames = sample_frame_ids[seq]
        if len(frames) != n_expect:
            raise ValueError(
                f"{seq}: sample contract {len(frames)} 幀 != 實際影像 {n_expect} 幀")
        # 09-04 稽核 Q02：檔內列序即為輸出順序（rows_from_boxes 用 enumerate 綁位置）。
        # 非遞增 ⇒ 每一列綁到錯的幀，而下游 finalize 的時序運算又用 sorted()。
        # 兩端契約不一致且所有既有驗證都抓不到 ⇒ 這裡 fail closed。**不要自動排序**
        # （train405 那種非 1-based 全域 frame ID 的情形，自動排序會做錯）。
        if any(b <= a for a, b in zip(frames, frames[1:])):
            bad = next(i for i, (a, b) in enumerate(zip(frames, frames[1:])) if b <= a)
            raise ValueError(
                f"{seq}: sample contract 的 frame ID 未嚴格遞增"
                f"（第 {bad + 1}→{bad + 2} 個：{frames[bad]} → {frames[bad + 1]}）"
                "——檔內列序即為輸出順序，非遞增會讓每一列綁到錯的幀")
        return list(frames)
    if gt is not None:
        frames = sorted(gt.loc[gt["seq"] == seq, "frame"].tolist())
        if frames:
            if len(frames) != n_expect:
                raise ValueError(
                    f"{seq}: GT {len(frames)} 幀 != 實際影像 {n_expect} 幀；"
                    "短幀／多區塊序列必須提供 --sample-csv contract，禁止猜測映射")
            return frames
    return list(range(1, n_expect + 1))


def _iter_masks_samurai(predictor, seq_dir: Path, init: list[float], reset_kf: bool = True):
    """SAMURAI/SAM2.1 路徑；預設在每序列開始前清空 Kalman 狀態。

    reset_kf=True 修上游 bug：SAMURAI 的 kf_mean/kf_covariance/stable_frames 掛在 model 物件上、
    只在 __init__ 設定一次，sam2_base.py 無 reset 方法、reset_state() 也不碰 → 單一 predictor
    連跑 65 序列時，前一支的運動狀態會污染下一支開頭（追蹤是遞迴的，開頭選錯毀整段）。
    reset_kf=False 只供 ``--samurai-legacy-cross-seq-kf`` 重現舊 E02 行為。
    """
    if reset_kf:
        predictor.kf_mean = None
        predictor.kf_covariance = None
        predictor.stable_frames = 0
    state = predictor.init_state(video_path=str(seq_dir), offload_video_to_cpu=True)
    box_xyxy = [init[0], init[1], init[0] + init[2], init[1] + init[3]]
    predictor.add_new_points_or_box(state, box=box_xyxy, frame_idx=0, obj_id=0)
    try:
        for fi, obj_ids, masks in predictor.propagate_in_video(state):
            yield fi, (masks[0][0] > 0.0).cpu().numpy()
    finally:
        predictor.reset_state(state)


def _iter_masks_sam3(predictor, seq_dir: Path, init: list[float], async_load: bool = True,
                     samurai: bool = False, gate_cfg: dict | None = None,
                     gate_stats: dict | None = None, score_sink: list | None = None):
    """E15：SAM3/SAM3.1。與 SAM2 的三處介面差異，每一處錯了都不會報錯只做錯事——

    1. box 須為**正規化**座標 [[x1/W, y1/H, x2/W, y2/H]]（內部再乘 image_size）；
       且 D024 每序列解析度不同 → 必須逐序列讀原圖尺寸，不可寫死。
    2. propagate_in_video 的 start/max/reverse **無預設值**，官方範例寫 max=240
       → 照抄會讓 nir（平均 892 幀）只追前 240 幀。傳 None = 追到底（原始碼確認）。
    3. 回傳 5 值（多 obj_scores，SAM2 是 3 值），mask 取 video_res_masks。

    async_load：推論路徑無 DataLoader，故無 num_workers；載入是單執行緒 for 迴圈逐張
    解碼+resize 到 1008²（sam2_utils.load_video_frames_from_jpg_images）。開 async 會起
    一條背景 thread 讓載入與 GPU forward 重疊（非平行化載入本身），實測前置阻塞約佔
    每序列 7%。若懷疑 async 造成幀錯位，用 --sam3-no-async-load 排除此變因。
    """
    from PIL import Image

    if samurai:
        # 每序列必重置 Kalman：上游 bug——SAMURAI 的 kf_mean/stable_frames 掛在 model 物件上、
        # 只在 __init__ 設定一次，reset_state() 不碰它們。單一 predictor 連跑多序列時，
        # 前一支的運動狀態會帶進下一支開頭，而追蹤是遞迴的（詳 prep/patch_sam3_samurai.py）。
        from sam3.model.sam3_tracker_base import reset_kalman
        reset_kalman(predictor)

    first = sorted(seq_dir.glob("*.jp*g"))[0]
    with Image.open(first) as im:
        W, H = im.size
    x, y, w, h = init
    rel_box = np.array([[x / W, y / H, (x + w) / W, (y + h) / H]], dtype=np.float32)
    assert rel_box.max() <= 1.5, f"box 正規化失敗 {rel_box.tolist()}（原圖 {W}x{H}，init {init}）"

    state = predictor.init_state(video_path=str(seq_dir), offload_video_to_cpu=True,
                                 async_loading_frames=async_load)
    predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=0, box=rel_box)
    g_prev, g_anchor, g_susp = None, None, 0  # E25 gate 狀態（gate_cfg=None 時恆不動）
    for fi, obj_ids, low_res, video_res, obj_scores in predictor.propagate_in_video(
        state, start_frame_idx=0, max_frame_num_to_track=None, reverse=False,
        propagate_preflight=True, tqdm_disable=True,
    ):
        m = np.squeeze((video_res[0] > 0.0).cpu().numpy())
        assert m.ndim == 2, f"mask 維度異常 {m.shape}"
        if score_sink is not None:  # E-A 診斷：obj_score logit（審核 §3.3——上游回傳、先前被丟棄）
            v = obj_scores.float().cpu().numpy() if hasattr(obj_scores, "cpu") else obj_scores
            score_sink.append(round(float(np.ravel(v)[0]), 4))
        if gate_cfg is not None:
            # yield 時 propagate 停在「本幀已存入 memory、下一幀尚未 forward」之間，
            # 此處 pop 恰好落在該窗口（sam3_tracking_predictor.py:860 存入 → :873 yield）
            g_prev, g_anchor, g_susp = _memory_gate_step(
                state, fi, m, g_prev, g_anchor, g_susp, gate_cfg, gate_stats)
        yield fi, m
    if hasattr(predictor, "reset_state"):
        predictor.reset_state(state)


def _memory_gate_step(state, fi: int, mask: np.ndarray, prev, anchor, susp: int,
                      cfg: dict, stats: dict):
    """E25 anti-switch gate 單步（回傳更新後的 prev/anchor/susp）。

    診斷依據（E15/E18）：identity switch = 單一時刻輸出中心跳 80–110px（GT 真實運動
    最大僅 2.7px/幀），之後 memory 6 幀內被錯誤目標填滿而「穩住」。離線前測（E15 val65
    輸出）：k=2 命中 2/2 跳變型病灶；穩定組誤觸發 29 幀/17 支，每次代價僅一幀不入記憶。

    機制：單幀中心位移 > max(k×前幀長邊, floor) → 從 inference_state pop 該幀的
    non-cond memory（sam3 的 frame_filter 掃描與讀取端對缺幀皆 .get+continue 容錯，
    已逐字核對 sam3_tracker_base.py:544-547/644-653——pop 幀被自動跳過並遞補更早幀）。
    解凍二擇一：輸出中心回到觸發前框 return_r×長邊內（模型跳回，該幀恢復寫入）、
    或凍結滿 suspend_max 幀（超時放行，視同誤觸發）。
    gate 只動 memory、從不改輸出框——確定性、因果、不認序列名（Ranking B 三維＋因果全過）。
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return prev, anchor, susp  # 空 mask：無位移訊號
    box = (float(xs.min()), float(ys.min()),
           float(xs.max() + 1 - xs.min()), float(ys.max() + 1 - ys.min()))
    if fi >= 1 and prev is not None:
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        d = float(np.hypot(cx - (prev[0] + prev[2] / 2), cy - (prev[1] + prev[3] / 2)))
        thr = max(cfg["k"] * max(prev[2], prev[3]), cfg["floor"])
        if susp == 0:
            if d > thr:
                susp, anchor = cfg["suspend_max"], prev
                stats["triggers"].append(int(fi))
        elif susp == cfg["suspend_max"] - 1 and d > cfg["cont_frac"] * thr:
            # v2（canary #1 的 nir-leaves −0.106 教訓）：觸發後第一幀位移仍大
            # ＝「持續運動」（葉子飄動類真實快速移動），非 E15 診斷的 switch 簽名
            # 「單一時刻跳過去之後穩住」→ 判誤觸發、立即解凍，傷害縮到 1 幀。
            # 真 switch（跳後黏住新位置）此幀位移小 → 不會走進這個分支。
            susp = 0
            stats["aborts"].append(int(fi))
        else:
            ar = cfg["return_r"] * max(anchor[2], anchor[3], cfg["floor"])
            if np.hypot(cx - (anchor[0] + anchor[2] / 2),
                        cy - (anchor[1] + anchor[3] / 2)) < ar:
                susp = 0  # 跳回觸發前位置：解凍，本幀正常寫入
                stats["returns"].append(int(fi))
        if susp > 0:
            state["output_dict"]["non_cond_frame_outputs"].pop(fi, None)
            for od in state["output_dict_per_obj"].values():
                od["non_cond_frame_outputs"].pop(fi, None)
            stats["frozen"] += 1
            susp -= 1
    return box, anchor, susp


def _resize_sam3_input(predictor, new_size: int, device: str) -> None:
    """E20：把 SAM3 的輸入解析度從預設 1008 改成 new_size（同時重建 RoPE）。

    機制：SAM3 內部把整張圖 resize 到 image_size²（stride 14 → 72×72 patch）。原圖約
    409×216 時，5.5px 的目標只佔約 1 個 patch，模型根本沒有東西可看。E19 的 crop-zoom
    （線性 2.05x）就是靠提高目標的 patch 預算換到 LB +0.0175；本函式是它的全域版。

    為何不能只設 `predictor.image_size`：backbone 的 RoPE 頻率張量 `freqs_cis` 是依
    `input_size` 預先算好的 buffer，改 image_size 而不重建它，forward 會撞
    `assert freqs_cis.shape == (x.shape[-2], x.shape[-1])`（實測 08-06）。

    32 個帶 RoPE 的 attention 分兩類（實測）：28 個 window attention 的 input_size
    ＝window 尺寸 (24,24)，與影像解析度無關、不可動；4 個 global attention 的
    input_size ＝全圖 patch 網格 (72,72)，才是要重建的。重建走官方原生的
    `rope_interp` 分支（scale_pos = rope_pt_size / input_size），把座標插值回預訓練的
    24×24 範圍——是設計好的行為，不是外掛。另 rel_pos=False、cls_token=False，
    故沒有可學的位置權重需要一併插值。

    兩個整除條件缺一不可：new_size 必須整除於 stride（否則 patch 切不齊），
    且新網格必須整除於 window 尺寸（否則 window partition 需要 padding，行為改變）。
    合法值例：1344（網格 96＝24×4）、1680（網格 120＝24×5）。
    """
    stride = getattr(predictor, "backbone_stride", 14)
    old_size = predictor.image_size
    assert new_size % stride == 0, f"image_size {new_size} 不是 backbone_stride {stride} 的倍數"
    old_grid, new_grid = old_size // stride, new_size // stride

    rope_mods = [m for m in predictor.backbone.modules() if hasattr(m, "_setup_rope_freqs")]
    win = {tuple(m.input_size) for m in rope_mods if tuple(m.input_size) != (old_grid, old_grid)}
    assert len(win) <= 1, f"window 尺寸不唯一 {win}，配置與實測不符，停止"
    if win:
        w = win.pop()[0]
        assert new_grid % w == 0, \
            f"新網格 {new_grid} 不是 window 尺寸 {w} 的倍數 —— window partition 會需要 padding"

    n = 0
    for m in rope_mods:
        if tuple(m.input_size) != (old_grid, old_grid):
            continue  # window attention：input_size 是 window 尺寸，不隨影像變
        m.input_size = (new_grid, new_grid)
        m._setup_rope_freqs()          # 重算 freqs_cis 並 register_buffer（可能落在 CPU）
        n += 1
    assert n > 0, f"沒有任何 input_size=({old_grid},{old_grid}) 的 global attention 被重建"
    predictor.backbone.to(device)      # 新 buffer 搬回 GPU

    # image_size 在建構時衍生出一串尺寸常數，漏改任何一個都會撞 assert（逐個實測補齊）：
    #   low_res_mask_size / input_mask_size —— mask 的解析度契約
    #   sam_image_embedding_size —— mask decoder 對 backbone 特徵的尺寸斷言
    #   prompt encoder 的三個屬性 —— dense PE 網格與點座標正規化的基準
    # prompt encoder 的 pe_layer 是與尺寸無關的隨機高斯矩陣、執行期才依屬性產生網格，
    # 故只需改屬性、不必重建權重。
    for attr in ("low_res_mask_size", "input_mask_size", "sam_image_embedding_size"):
        assert hasattr(predictor, attr), f"predictor 缺 {attr}，SAM3 版本與實測不符"
    predictor.image_size = new_size
    predictor.low_res_mask_size = new_grid * 4
    predictor.input_mask_size = predictor.low_res_mask_size * 4
    predictor.sam_image_embedding_size = new_grid
    pe = predictor.sam_prompt_encoder
    pe.image_embedding_size = (new_grid, new_grid)
    pe.input_image_size = (new_size, new_size)
    pe.mask_input_size = (new_grid * 4, new_grid * 4)

    # memory encoder（08-07 解開的第 4 關，原以為要動手術，實際只是一個常數）：
    #   `MemoryEncoder.forward` 做 `x = x + masks`，x 來自 pix_feat（網格 new_grid），
    #   masks 來自 mask_downsampler。而 `SimpleMaskDownSampler` 會先把 mask 插值到
    #   建構期寫死的 `interpol_size`（官方 [1152,1152]）再用 4 層 stride-2 conv 降 16 倍
    #   → 1152/16 ＝ 72，正好等於預設的 1008/14。改了 image_size 而不改它，
    #   就撞 new_grid vs 72（1680 時是 120 vs 72，即 exp009 卡住的地方）。
    # 關鍵：`interpol_size` 是**單一真相源**——上游 `sam3_video_base` 直接讀它決定把
    # low_res_masks 插值到多大（註解明寫 "Avoid an extra interpolation step"），
    # downsampler 內部再比對尺寸、相符就跳過。故改這一個屬性，整條 mask 路徑即一致。
    # 以屬性搜尋而非寫死路徑或 import 類別：predictor 可能是 tracker 本身或其包裝，
    # 且 `interpol_size` 這個屬性名夠特定，不會誤中其他模組。
    mds = [m for m in predictor.modules() if hasattr(m, "interpol_size")]
    assert len(mds) == 1, f"找到 {len(mds)} 個帶 interpol_size 的模組，配置與實測不符，停止"
    old_interp = list(mds[0].interpol_size or [])
    assert old_interp == [old_size // 14 * 16] * 2 or not old_interp, \
        f"interpol_size {old_interp} 與預設網格推導不符，SAM3 版本可能已變，停止"
    # total_stride=16（4 層 stride-2 conv）→ 要讓輸出等於 new_grid，輸入須為 new_grid*16
    mds[0].interpol_size = [new_grid * 16, new_grid * 16]

    # 以下兩處**已查證確認不需要動**（記下來，免得日後又重查一遍）：
    #  · position encoding 的 `precompute_resolution=1008`：只是建構時預先填 `self.cache`
    #    以避開 torch.compile 的 symbolic shape tracing，forward 遇未快取尺寸會現算並補上。
    #  · backbone 的 absolute position embedding（`use_abs_pos=True, tile_abs_pos=True`）：
    #    `pos_embed` 是依 `pretrain_img_size=336`（＝24×24 網格）建的預訓練參數，而
    #    `get_abs_pos()` 在 **forward 時**依實際 token 網格動態 tile
    #    （`tile(h//24 + 1)` 後切片到 :h），故與 image_size 無關。
    #    **附帶收穫**：這給了「新網格須整除 24」的第二個理由——除了 window partition 不需
    #    padding 之外，整除時每個 24×24 塊都是一份完整的預訓練位置模式，tile 的語意才乾淨。

    print(f"  image_size {old_size} → {new_size}（patch 網格 {old_grid}² → {new_grid}²，"
          f"線性 ×{new_size/old_size:.2f}、計算量 ×{(new_size/old_size)**2:.2f}；"
          f"重建 {n} 個 global attention 的 RoPE；"
          f"mask interpol {old_interp} → {mds[0].interpol_size}）")


def track_sequence(predictor, seq_dir: Path, init: list[float], want_masks: bool,
                   backend: str = "samurai", sam3_async: bool = True,
                   sam3_samurai: bool = False, samurai_reset_kf: bool = True,
                   gate_cfg: dict | None = None):
    """回傳 (boxes_by_pos, raw, diag)。boxes_by_pos: 序列內 0-based 位置 → box（位置 0 = init）。

    backend 之後的一切（mask→box、空 mask 沿用前框、首幀 init、缺幀補滿）逐項共用
    ＝單變因隔離，兩個 backend 的差異只有 tracker 本身。
    """
    import torch

    t0 = time.time()
    n_avail = len(list(seq_dir.glob("*.jpg"))) + len(list(seq_dir.glob("*.jpeg")))
    boxes_by_pos = {0: init}
    raw = {"boxes": [], "empty": [], "masks": []} if want_masks else None
    empty_cnt = 0
    gate_stats = {"triggers": [], "returns": [], "aborts": [], "frozen": 0} if gate_cfg else None
    scores = [] if backend != "samurai" else None
    if n_avail >= 2:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            stream = (_iter_masks_samurai(predictor, seq_dir, init, samurai_reset_kf)
                      if backend == "samurai"
                      else _iter_masks_sam3(predictor, seq_dir, init, sam3_async, sam3_samurai,
                                            gate_cfg, gate_stats, scores))
            prev = init
            for fi, mask in stream:
                box, was_empty = mask_to_box(mask, prev)
                empty_cnt += int(was_empty)
                if fi >= 1:  # 首幀固定 init，tracker 結果從第 2 幀（位置 1）採用
                    boxes_by_pos[fi] = box
                if raw is not None:
                    raw["boxes"].append(box)
                    raw["empty"].append(was_empty)
                    raw["masks"].append(np.asarray(mask, dtype=bool))
                prev = box
        torch.cuda.empty_cache()
    dt = time.time() - t0
    diag = {"n_avail": n_avail, "empty_mask_frames": empty_cnt,
            "elapsed_s": round(dt, 1), "fps": round(n_avail / max(dt, 0.1), 1)}
    if gate_stats is not None:
        diag["gate"] = {"n_triggers": len(gate_stats["triggers"]),
                        "triggers": gate_stats["triggers"][:20],
                        "returns": gate_stats["returns"][:20],
                        "aborts": gate_stats["aborts"][:20],
                        "frozen_frames": gate_stats["frozen"]}
    if scores:
        diag["obj_scores"] = scores  # 與 propagate yield 對齊的逐幀 logit（含第 0 幀）
    return boxes_by_pos, raw, diag


def write_scores_atomic(path: Path, ids: list, scores: list) -> None:
    """ID 與 submission 同一套；缺分補空字串，不改框。"""
    tmp = path.with_suffix(path.suffix + ".part")
    n = len(ids)
    sc = list(scores)[:n] + [""] * max(0, n - len(scores))
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ID", "obj_score"])
        for i, sid in enumerate(ids):
            w.writerow([sid, sc[i]])
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def save_mask_cache(path: Path, raw: dict) -> None:
    if not raw["masks"]:
        np.savez_compressed(path, n=0)
        return
    m = np.stack(raw["masks"])  # (n, H, W) bool；逐序列解析度不同（D024）
    np.savez_compressed(
        path,
        n=m.shape[0], height=m.shape[1], width=m.shape[2],
        bits=np.packbits(m, axis=None),
        boxes=np.asarray(raw["boxes"], dtype=np.float32),
        empty=np.asarray(raw["empty"], dtype=bool),
    )


SUB_COLS = ["ID", "x", "y", "width", "height"]
STATUS_SCHEMA_VERSION = 2


def resolve_samurai_reset_kf(reset_flag: bool, legacy_cross_seq_flag: bool) -> bool:
    """解析新舊 CLI 開關。新預設一律 reset；舊 ``--samurai-reset-kf`` 保留相容。

    同時指定「要 reset」與「要跨序列污染」是無法解釋的衝突，必須明確拒絕。
    """
    if reset_flag and legacy_cross_seq_flag:
        raise ValueError("--samurai-reset-kf 與 --samurai-legacy-cross-seq-kf 不可同時指定")
    return not legacy_cross_seq_flag


def rows_from_boxes(seq: str, frame_ids: list[int], boxes_by_pos: dict[int, list[float]]) -> list[tuple]:
    """將 tracker 位置映射成 CSV 列；缺位置只能使用當時已知的最近前框。

    不可先取整段最後一框再回填中間空洞；那會把未來 tracker 結果借給過去幀。
    """
    if frame_ids and 0 not in boxes_by_pos:
        raise ValueError("缺少位置 0 的 init 框，無法因果補幀")
    rows = []
    latest = boxes_by_pos.get(0)
    for pos, fid in enumerate(frame_ids):
        if pos in boxes_by_pos:
            latest = boxes_by_pos[pos]
        if latest is None:
            raise ValueError(f"位置 {pos} 之前沒有任何已知框")
        rows.append((f"{seq}_{fid}", *latest))
    return rows


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _path_identity(path: str | Path | None) -> dict:
    """以內容而非 mtime 固定檔案身分，讓同一 artifact 可安全跨 VM resume。"""
    if not path:
        return {"exists": False}
    p = Path(path).expanduser().resolve()
    ident = {"exists": p.is_file()}
    if p.is_file():
        ident.update({"size": p.stat().st_size, "sha256": _file_sha256(p)})
    return ident


def _small_file_content_identity(path: str | Path | None) -> dict:
    """給 CSV/contract 用的跨機穩定身分；不對多 GB checkpoint 做內容雜湊。"""
    if not path:
        return {"exists": False}
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return {"exists": False}
    return {
        "exists": True,
        "size": p.stat().st_size,
        "sha256": _file_sha256(p),
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_run_signature(seq_dir: Path, run_config: dict) -> dict:
    """建立每序列 run signature：綁定 backend/checkpoint/config 與輸入幀 manifest。"""
    frames = sorted([*seq_dir.glob("*.jpg"), *seq_dir.glob("*.jpeg")])
    manifest = []
    for f in frames:
        st = f.stat()
        # 解同一官方 zip 到新 VM 時檔案 mtime 會改變；把 mtime 簽進去會讓已同步至
        # gDrive 的逐序列 artifact 永遠無法跨機 resume。內容來源另由 launch report 的
        # zip/checkpoint SHA256 固定；此處以 name+size 綁定實際 frame manifest。
        manifest.append({"name": f.name, "size": st.st_size})
    manifest_sha = hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()
    init_f = seq_dir / "init_rect.txt"
    init_sha = hashlib.sha256(init_f.read_bytes()).hexdigest() if init_f.is_file() else None
    signed = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_config": run_config,
        "frame_manifest_sha256": manifest_sha,
        "n_frames": len(frames),
        "init_rect_sha256": init_sha,
    }
    return {**signed, "digest": hashlib.sha256(_canonical_json(signed).encode()).hexdigest()}


def status_path_for(csv_f: Path) -> Path:
    return csv_f.with_suffix(".status.json")


def write_sequence_status(csv_f: Path, seq: str, status: str, run_signature: dict,
                          rows: int, details: dict | None = None) -> None:
    """原子寫入每序列狀態；status 是 complete/fallback/running/failed。"""
    path = status_path_for(csv_f)
    tmp = path.with_suffix(path.suffix + ".part")
    payload = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "sequence": seq,
        "status": status,
        "rows": int(rows),
        "run_signature": run_signature,
        "artifact_sha256": (_file_sha256(csv_f)
                             if status in {"complete", "fallback"} and csv_f.is_file() else None),
        "details": details or {},
    }
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def count_frames(seq_dir: Path) -> int:
    """與 track_sequence 同一套數法（單一真相源，避免 resume 驗證與實跑對不上）。"""
    return len(list(seq_dir.glob("*.jpg"))) + len(list(seq_dir.glob("*.jpeg")))


def write_csv_atomic(path: Path, rows: list) -> None:
    """.part → fsync → atomic rename。

    護欄來源：E02 參考實作 `official_reference/samurai_notebook_full.txt:783-786` 本來就有，
    腳本化重寫時遺失（D052b）。少了它，跑到一半被 kill（自毀計時器、OOM）會留下半截 CSV，
    而 resume 只看「檔案存在」就當成完成品 → 殘缺結果靜默進提交。
    """
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", newline="") as fh:
        pd.DataFrame(rows, columns=SUB_COLS).to_csv(fh, index=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _csv_structure_ok(csv_f: Path, seq: str, n_expect: int) -> tuple[bool, str]:
    if not csv_f.exists():
        return False, "missing"
    try:
        d = pd.read_csv(csv_f)
    except Exception as e:  # noqa: BLE001 —— 半截/損壞檔
        return False, f"unreadable:{type(e).__name__}"
    if list(d.columns) != SUB_COLS:
        return False, f"bad_columns:{list(d.columns)}"
    if n_expect and len(d) != n_expect:
        return False, f"row_mismatch:{len(d)}!={n_expect}"
    bad_prefix = ~d["ID"].astype(str).str.startswith(f"{seq}_")
    if bad_prefix.any():
        return False, f"bad_id_prefix:{d.loc[bad_prefix, 'ID'].iloc[0]}"
    return True, "ok"


def _artifact_ok(csv_f: Path, seq: str, n_expect: int, expected_signature: dict | None,
                 allowed_statuses: set[str]) -> tuple[bool, str]:
    ok, why = _csv_structure_ok(csv_f, seq, n_expect)
    if not ok:
        return ok, why
    if expected_signature is None:
        return False, "missing_expected_run_signature"
    status_f = status_path_for(csv_f)
    if not status_f.exists():
        return False, "missing_status"
    try:
        status = json.loads(status_f.read_text())
    except Exception as e:  # noqa: BLE001
        return False, f"bad_status:{type(e).__name__}"
    if status.get("schema_version") != STATUS_SCHEMA_VERSION:
        return False, f"bad_status_schema:{status.get('schema_version')!r}"
    if status.get("sequence") != seq:
        return False, f"bad_status_sequence:{status.get('sequence')!r}"
    state = status.get("status")
    if state not in allowed_statuses:
        return False, f"status:{state}"
    if status.get("rows") != n_expect:
        return False, f"status_row_mismatch:{status.get('rows')}!={n_expect}"
    if status.get("run_signature") != expected_signature:
        got = status.get("run_signature") or {}
        return False, f"run_signature_mismatch:{got.get('digest', 'missing')}"
    expected_artifact_sha = status.get("artifact_sha256")
    if not expected_artifact_sha:
        return False, "missing_artifact_sha256"
    if _file_sha256(csv_f) != expected_artifact_sha:
        return False, "artifact_sha256_mismatch"
    return True, state


def resume_ok(csv_f: Path, seq: str, n_expect: int,
              expected_signature: dict | None = None) -> tuple[bool, str]:
    """既有 CSV 是否可 resume。只接受同 run signature 的 complete sidecar。

    fallback/running/failed 都必須重跑；沒傳 expected_signature 也不得 fail-open。
    """
    ok, why = _artifact_ok(csv_f, seq, n_expect, expected_signature, {"complete"})
    return (True, "ok") if ok else (False, why)


def score_cache_ok(score_f: Path, csv_f: Path, n_expect: int) -> bool:
    """SAM3 score sidecar 必須與 complete status 的 hash、列數及 ID 完全一致。"""
    if not score_f.is_file() or not csv_f.is_file():
        return False
    try:
        status = json.loads(status_path_for(csv_f).read_text())
        expected_sha = status.get("details", {}).get("obj_scores_sha256")
        if not expected_sha or _file_sha256(score_f) != expected_sha:
            return False
        scores = pd.read_csv(score_f)
        boxes = pd.read_csv(csv_f, usecols=["ID"])
        return (list(scores.columns) == ["ID", "obj_score"] and len(scores) == n_expect
                and scores["ID"].astype(str).tolist() == boxes["ID"].astype(str).tolist())
    except Exception:  # noqa: BLE001 —— 壞/半截 cache 一律重跑該序列
        return False


def mask_cache_ok(npz_f: Path) -> bool:
    """壞快取要能自我修復：讀不開就當不存在（參考實作 848-852 的 unlink 護欄）。"""
    if not npz_f.exists():
        return False
    try:
        with np.load(npz_f) as z:
            _ = int(z["n"])
        return True
    except Exception:  # noqa: BLE001
        try:
            npz_f.unlink()
            print(f"⚠️ 壞 mask 快取已移除，將重跑：{npz_f.name}", file=sys.stderr)
        except OSError:
            pass
        return False


def validate_submission(sub: pd.DataFrame, sample_csv: Path | None,
                        fallback_sequences: list[str] | None = None,
                        allow_fallback: bool = False) -> list[str]:
    """對 sample submission 做 exact-set 比對（P0-09 缺口）。回傳問題清單，空＝通過。"""
    errs = []
    if list(sub.columns) != SUB_COLS:
        errs.append(f"欄位不符：{list(sub.columns)} != {SUB_COLS}")
    if sub["ID"].duplicated().any():
        d = sub.loc[sub["ID"].duplicated(), "ID"].head(3).tolist()
        errs.append(f"重複 ID {int(sub['ID'].duplicated().sum())} 筆，例：{d}")
    box_cols = ["x", "y", "width", "height"]
    if sub[box_cols].isna().any().any():
        errs.append("含 NaN 座標")
    try:
        finite = np.isfinite(sub[box_cols].to_numpy(dtype=float)).all()
    except (TypeError, ValueError):
        finite = False
    if not finite:
        errs.append("含非 finite（NaN/Inf）座標")
    negative_xy = (sub[["x", "y"]] < 0).any(axis=1).sum()
    if negative_xy:
        errs.append(f"負 x/y {int(negative_xy)} 列")
    nonpos = (sub[["width", "height"]] <= 0).any(axis=1).sum()
    if nonpos:
        errs.append(f"非正 w/h {int(nonpos)} 列")
    if sample_csv and Path(sample_csv).exists():
        smp = pd.read_csv(sample_csv)
        if smp["ID"].duplicated().any():
            errs.append(f"sample 含重複 ID {int(smp['ID'].duplicated().sum())} 筆")
        want, got = set(smp["ID"]), set(sub["ID"])
        if want != got:
            miss, extra = want - got, got - want
            if miss:
                errs.append(f"缺 {len(miss)} 個 sample ID，例：{sorted(miss)[:3]}")
            if extra:
                errs.append(f"多出 {len(extra)} 個非 sample ID，例：{sorted(extra)[:3]}")
        elif len(smp) != len(sub):
            errs.append(f"sample/output row count 不符：{len(smp)} != {len(sub)}")
        elif smp["ID"].astype(str).tolist() != sub["ID"].astype(str).tolist():
            errs.append("輸出 ID 順序與 sample 不同")
    fallback_sequences = fallback_sequences or []
    if fallback_sequences and not allow_fallback:
        errs.append(f"含 {len(fallback_sequences)} 支 fallback 序列（預設拒絕）："
                    f"{fallback_sequences[:5]}")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-root", required=True, help="每序列一個子目錄（jpg 幀）")
    ap.add_argument("--seq-list", help="只跑清單內序列（一行一序列名）")
    ap.add_argument("--gt-csv", help="2026training.csv —— 提供則 init 後援 + 全域幀號輸出")
    ap.add_argument("--sample-csv",
                    help="sample_submisson.csv —— 提供則對其做 exact-set 比對驗證（缺列/多列即 exit 2）。"
                         "Ranking B 當天務必帶上：它是『殘缺 CSV 靜默上傳』的唯一自動防線")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="明確允許 submission 含 init-copy fallback 並 exit 0；預設為驗證失敗/exit 2")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--backend", choices=("samurai", "sam3"), default="samurai",
                    help="samurai=SAM2.1/SAMURAI（預設隔離每序列 Kalman）；"
                         "sam3=E15 SAM3/3.1 底座")
    ap.add_argument("--sam3-version", choices=("sam3", "sam3.1"), default="sam3",
                    help="sam3=官方 box-prompt 範例唯一背書；sam3.1=VOS 7 benchmark 改進 6（multiplex builder）")
    ap.add_argument("--sam3-ckpt", default="", help="SAM3 本地 .pt 路徑；留空則自 HF 下載（gated，需 hf auth login）")
    ap.add_argument("--source-revision", default="",
                    help="tracker source 的固定 commit/revision；寫入 run signature 供跨機 resume 驗證")
    ap.add_argument("--sam3-image-size", type=int, default=0,
                    help="E20：覆寫 SAM3 內部輸入解析度（預設 0＝用模型預設 1008）。"
                         "必須是 backbone_stride=14 的倍數（1176/1400/1568）。"
                         "機制：E19 的全域版——1008² 下 5.5px 的目標只佔約 1 個 patch，"
                         "提高解析度＝提高目標的 patch 預算。計算量 ~ size²。")
    ap.add_argument("--memory-stride", type=int, default=0,
                    help="E24：memory_temporal_stride_for_eval（預設 0＝不動，維持位元級複現）。"
                         "**兩 backend 同名同預設 r=1，但走不同程式路徑**（已逐字核對上游原始碼）："
                         "[SAM3] use_memory_selection 預設 True（build_sam3_video_model 的 "
                         "apply_temporal_disambiguation=True）→ r 傳進 frame_filter 當 step=-r，"
                         "自 t-1 往回以 r 為間隔掃描、收集 eff_iou_score>mf_threshold(0.01) 的幀（上限 15），"
                         "6 個 maskmem 槽取其中最新 6 個；must_include 保證 t-1 恆在。"
                         "[SAM2] 無 use_memory_selection → 走 else 分支的 stride 公式 "
                         "prev_frame_idx=((frame_idx-2)//r)*r-(t_rel-2)*r。"
                         "機制：r=1 時 6 槽只覆蓋 t-1..t-6，identity switch 後約 6 幀整個記憶庫就被錯的目標"
                         "填滿（＝E15 診斷的『單一時刻跳變後穩住、切換後 IoU 仍高』）；"
                         "r=5 使 6 槽覆蓋 t-1,t-6,...,t-26 ＝跨度 6→26 幀（×4.3），"
                         "正確身分的證據要 26 幀才被沖乾淨。槽的位置編碼索引不變，只改哪些幀填進槽。"
                         "文獻預設值 r=5（XMem/Cutie），依 D033 不網格搜索。"
                         "⚠️ num_maskmem 不可比照辦理——它被烘進 maskmem_tpos_enc 的參數形狀")
    ap.add_argument("--samurai-reset-kf", action="store_true",
                    help="舊參數相容：每序列重置 Kalman。現在已是預設，保留此 flag 不讓舊命令失效")
    ap.add_argument("--samurai-legacy-cross-seq-kf", action="store_true",
                    help="明確 opt-in 舊 E02 的跨序列 Kalman 狀態（會污染下一支序列）")
    ap.add_argument("--memory-gate", action="store_true",
                    help="E25：運動異常觸發的 memory 凍結（anti-switch gate，僅 sam3 backend；"
                         "預設關閉＝位元級複現不受影響）。機制與離線前測見 _memory_gate_step docstring。"
                         "介入層=memory 寫入（病灶所在層），非 E18 的 mask 選擇層；"
                         "條件性罕見觸發，非 D050 的無條件全域稀疏化")
    ap.add_argument("--gate-k", type=float, default=2.0,
                    help="觸發閾值倍率（單幀位移 > k×前幀長邊；物理先驗：GT 真實運動最大 "
                         "2.7px/幀 vs switch 跳變 80-110px，非 val 搜索）")
    ap.add_argument("--gate-floor", type=float, default=15.0, help="觸發閾值下限 px（極小目標保護）")
    ap.add_argument("--gate-suspend-max", type=int, default=45,
                    help="凍結超時幀數（25FPS≈1.8s；超時=視同誤觸發、恢復寫入）")
    ap.add_argument("--gate-return-r", type=float, default=2.0,
                    help="解凍回歸半徑（輸出中心回到觸發前框 r×長邊內＝模型跳回）")
    ap.add_argument("--gate-cont-frac", type=float, default=0.5,
                    help="v2 持續運動判別：觸發後第一幀位移 > cont_frac×thr ＝真實快速運動"
                         "（非跳後穩住的 switch 簽名）→ 立即解凍。0.5＝半閾值中點（物理先驗）")
    ap.add_argument("--sam3-samurai", action="store_true",
                    help="E18：啟用 SAMURAI Kalman 運動先驗（需先跑 prep/patch_sam3_samurai.py）"
                         "——攔截 identity switch，E15 診斷出的 3 支無人機群崩潰即此病因")
    ap.add_argument("--sam3-no-async-load", action="store_true",
                    help="關閉背景載入 thread（載入與 GPU forward 就不重疊，約慢 7%%）；懷疑幀錯位時用它排除變因")
    ap.add_argument("--sam3-eval", action="store_true",
                    help="E26：顯式 model.eval()。上游 build_sam3_video_model 從不呼叫 .eval()"
                         "（對照同檔 image/multiplex builder 都有——唯獨 video 漏掉，已逐字核對），"
                         "nn.Module 預設 training=True → tracker memory attention 的 dropout=0.1 "
                         "推定在推論路徑生效（transformer.py: dropout_p = self.dropout_p if "
                         "self.training else 0.0）。預設關閉＝保 E15/v008 可複現；開啟即單變因量測")
    ap.add_argument("--samurai-dir", default=os.environ.get("SAMURAI_DIR", ""), help="SAM2 fork clone 路徑(samurai 或 HiM2SAM,需已 pip install -e <fork>/sam2)")
    ap.add_argument("--model-cfg", default="configs/samurai/sam2.1_hiera_l.yaml",
                    help="E13 HiM2SAM 用 configs/him2sam/lasot/sam2.1_hiera_l.yaml(D033:用官方預設超參)")
    ap.add_argument("--ckpt", default="sam2.1_hiera_large.pt")
    ap.add_argument("--mask-cache", action="store_true", help="輸出 SAM2 二值 mask 快取（E12 用）")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    try:
        samurai_reset_kf = resolve_samurai_reset_kf(
            args.samurai_reset_kf, args.samurai_legacy_cross_seq_kf)
    except ValueError as e:
        ap.error(str(e))

    # S-02(D)：**所有路徑必須在任何 chdir 之前解析成絕對路徑**。
    # samurai backend 會 `os.chdir(samurai/"sam2")`（fork 的 config 需要相對路徑），
    # 而該行之後才使用 frames_root／ckpt ⇒ 任何相對路徑都會相對於**錯的目錄**解析。
    # 現行腳本一律傳絕對路徑故未爆，但 Ranking B 當天換人用相對路徑就會炸 ⇒ fail-closed 先修。
    for _a in ("frames_root", "out_dir", "seq_list", "gt_csv", "sample_csv",
               "ckpt", "sam3_ckpt", "samurai_dir"):
        _v = getattr(args, _a, None)
        if _v:
            setattr(args, _a, str(Path(_v).expanduser().resolve()))

    random.seed(42); np.random.seed(42)

    frames_root = Path(args.frames_root)
    out = Path(args.out_dir); seq_csv_dir = out / "seq_csv"; mask_dir = out / "masks"
    seq_csv_dir.mkdir(parents=True, exist_ok=True)
    if args.mask_cache:
        mask_dir.mkdir(parents=True, exist_ok=True)

    gt = None
    if args.gt_csv:
        gt = pd.read_csv(args.gt_csv)
        gt.columns = ["ID", "x", "y", "w", "h"]
        parts = gt["ID"].str.rsplit("_", n=1, expand=True)
        gt["seq"], gt["frame"] = parts[0], parts[1].astype(int)
    sample_frame_ids = load_sample_frame_ids(args.sample_csv) if args.sample_csv else None

    seqs = sorted(d.name for d in frames_root.iterdir() if d.is_dir() and any(d.glob("*.jp*g")))
    if args.seq_list:
        want = {s.strip() for s in Path(args.seq_list).read_text().split() if s.strip()}
        missing = want - set(seqs)
        if missing:
            print(f"⚠️ 清單中 {len(missing)} 序列在 frames-root 缺資料：{sorted(missing)[:5]}...", file=sys.stderr)
        seqs = [s for s in seqs if s in want]
    if not seqs:
        sys.exit("沒有可跑的序列")
    by_mod = {m: sum(1 for s in seqs if detect_modality(s) == m) for m in MODALITIES}
    tag = args.backend if args.backend == "samurai" else args.sam3_version
    print(f"待跑 {len(seqs)} 序列 {by_mod} | backend={tag}")

    model_cfg_path = Path(args.model_cfg)
    if args.backend == "samurai" and not model_cfg_path.is_absolute() and args.samurai_dir:
        model_cfg_path = Path(args.samurai_dir) / "sam2" / model_cfg_path
    checkpoint_identity = (_path_identity(args.ckpt) if args.backend == "samurai" else
                           (_path_identity(args.sam3_ckpt) if args.sam3_ckpt else
                            {"source": "huggingface_default", "model": args.sam3_version}))
    run_config = {
        "backend": tag,
        "checkpoint": checkpoint_identity,
        "model_config": _path_identity(model_cfg_path) if args.backend == "samurai" else {
            "builder": args.sam3_version},
        "gt_csv": _path_identity(args.gt_csv),
        "sample_contract": _small_file_content_identity(args.sample_csv),
        "source_revision": args.source_revision,
        "tracker_entrypoint_sha256": _file_sha256(Path(__file__).resolve()),
        "samurai_reset_kf": samurai_reset_kf,
        "sam3_samurai": bool(args.sam3_samurai),
        "sam3_async_load": not args.sam3_no_async_load,
        "sam3_eval": bool(args.sam3_eval),
        "sam3_image_size": args.sam3_image_size,
        "memory_stride": args.memory_stride,
        "memory_gate": ({"k": args.gate_k, "floor": args.gate_floor,
                         "suspend_max": args.gate_suspend_max, "return_r": args.gate_return_r,
                         "cont_frac": args.gate_cont_frac} if args.memory_gate else False),
        "mask_cache": bool(args.mask_cache),
        "device": args.device,
    }

    import torch
    torch.manual_seed(42)
    if args.backend == "samurai":
        # --- SAMURAI predictor（E02 Cell 3 對齊）---
        samurai = Path(args.samurai_dir) if args.samurai_dir else None
        if not samurai or not samurai.exists():
            sys.exit("--samurai-dir 未指定或不存在（需 yangchris11/samurai clone，已 pip install -e sam2）")
        os.chdir(samurai / "sam2")  # fork config 相對路徑需要
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(args.model_cfg, str(Path(args.ckpt).resolve()), device=args.device)
        n_params = sum(p.numel() for p in predictor.parameters())
        assert n_params > 150e6, f"參數量 {n_params/1e6:.0f}M —— 不是 Large（~224M）"
        print(f"predictor OK（samurai mode）| {n_params/1e6:.0f}M params")
    else:
        # --- E15：SAM3 / SAM3.1（獨立 venv，py3.12+/torch2.7+，與 samurai 環境不共存）---
        from sam3.model_builder import build_sam3_multiplex_video_model, build_sam3_video_model
        builder = build_sam3_multiplex_video_model if args.sam3_version == "sam3.1" else build_sam3_video_model
        kw = {"checkpoint_path": args.sam3_ckpt} if args.sam3_ckpt else {}
        sam3_model = builder(device=args.device, **kw)
        predictor = sam3_model.tracker
        predictor.backbone = sam3_model.detector.backbone  # 官方範例必要的一行，漏了會錯
        assert hasattr(predictor, "add_new_points_or_box"), \
            f"{args.sam3_version} 的 tracker 無 box prompt 介面（此版本只支援 text/session API）"
        n_params = sum(p.numel() for p in sam3_model.parameters())
        assert n_params > 500e6, f"參數量 {n_params/1e6:.0f}M —— 不是 SAM3（~848M），權重可能載錯"
        print(f"  S-04 探針：sam3_model.training={sam3_model.training} "
              f"tracker.training={predictor.training}")
        if args.sam3_eval:
            sam3_model.eval()
            assert not predictor.training, "eval() 未傳導到 tracker"
            print("  E26 顯式 eval() 已套用（memory attention dropout 0.1 → 0）")
        if args.sam3_image_size:
            _resize_sam3_input(predictor, args.sam3_image_size, args.device)
        if args.sam3_samurai:
            from sam3.model.sam3_tracker_base import enable_samurai  # patch 後才存在
            enable_samurai(predictor)  # 超參用 SAMURAI 官方預設（D033）
            assert getattr(predictor, "samurai_mode", False), "samurai_mode 未生效"
            print(f"  E18 SAMURAI Kalman 已啟用（kf_w={predictor.kf_score_weight}, "
                  f"stable_thr={predictor.stable_frames_threshold}）")
        print(f"predictor OK（{args.sam3_version}）| {n_params/1e6:.0f}M params | ckpt={args.sam3_ckpt or 'HF'}")

    # --- E24：memory 時間跨度（兩 backend 共用；預設 0 = 不動，位元級複現不受影響）---
    if args.memory_stride:
        attr = "memory_temporal_stride_for_eval"
        # 上游若改名要立刻炸，不能靜默無效（E18 的教訓：機制沒生效卻以為在跑）
        assert hasattr(predictor, attr), f"{args.backend} 的 predictor 無 {attr}——上游已改名，停工確認"
        old = getattr(predictor, attr)
        setattr(predictor, attr, args.memory_stride)
        assert getattr(predictor, attr) == args.memory_stride, f"{attr} 設定未生效"
        span = args.memory_stride * (getattr(predictor, "num_maskmem", 7) - 1)
        print(f"  E24 {attr}: {old} → {args.memory_stride}"
              f"（{getattr(predictor, 'num_maskmem', 7) - 1} 個非條件記憶槽最舊者 ≈ t-{1 + args.memory_stride * (getattr(predictor, 'num_maskmem', 7) - 2)} 幀）")

    gate_cfg = None
    if args.memory_gate:
        assert args.backend == "sam3", "--memory-gate 僅支援 sam3 backend（E25）"
        gate_cfg = {"k": args.gate_k, "floor": args.gate_floor,
                    "suspend_max": args.gate_suspend_max, "return_r": args.gate_return_r,
                    "cont_frac": args.gate_cont_frac}
        print(f"  E25 memory gate 啟用：k={args.gate_k} floor={args.gate_floor}px "
              f"suspend_max={args.gate_suspend_max} return_r={args.gate_return_r}")

    diagnostics = {}
    run_signatures = {}
    for i, seq in enumerate(seqs):
        csv_f = seq_csv_dir / f"{seq}.csv"
        score_f = seq_csv_dir / f"{seq}.scores.csv"
        npz_f = mask_dir / f"{seq}.npz"
        seq_dir = frames_root / seq
        # resume 護欄（P0-02）：不只看檔案存在，還驗列數／欄位／ID 前綴；
        # 不合格就重跑（並移除壞快取），避免半截檔被當完成品靜默進提交。
        n_expect = count_frames(seq_dir)
        signature = build_run_signature(seq_dir, run_config)
        run_signatures[seq] = signature
        ok, why = resume_ok(csv_f, seq, n_expect, signature)
        scores_ok = score_cache_ok(score_f, csv_f, n_expect)
        if ok and (not args.mask_cache or mask_cache_ok(npz_f)) and (
                args.backend == "samurai" or scores_ok):
            diagnostics[seq] = {"skipped": "already_done", "rows": n_expect,
                                "status_sidecar": str(status_path_for(csv_f)),
                                "run_signature": signature["digest"]}
            continue
        if csv_f.exists() and not ok:
            print(f"⚠️ {seq}: 既有 CSV 不可信（{why}）→ 重跑", file=sys.stderr)
        # 先把舊 complete 狀態原子覆寫；若進程中斷，下次 resume 也不會誤用舊 CSV。
        write_sequence_status(csv_f, seq, "running", signature, 0, {"resume_rejected": why})
        try:
            init = read_init(seq_dir, seq, gt)
            boxes_by_pos, raw, diag = track_sequence(predictor, seq_dir, init, args.mask_cache,
                                                     args.backend, not args.sam3_no_async_load,
                                                     args.sam3_samurai, samurai_reset_kf,
                                                     gate_cfg)
            fids = frame_ids_for(seq, diag["n_avail"], gt, sample_frame_ids)
            rows = rows_from_boxes(seq, fids, boxes_by_pos)
            write_csv_atomic(csv_f, rows)
            if args.backend != "samurai":
                write_scores_atomic(score_f, [r[0] for r in rows], diag.get("obj_scores") or [])
                diag["obj_scores_sha256"] = _file_sha256(score_f)
            if args.mask_cache and raw is not None:
                save_mask_cache(npz_f, raw)
            diag["modality"] = detect_modality(seq)
            diag["status"] = "complete"
            diag["status_sidecar"] = str(status_path_for(csv_f))
            diag["run_signature"] = signature["digest"]
            status_details = {k: v for k, v in diag.items() if k != "obj_scores"}
            write_sequence_status(csv_f, seq, "complete", signature, len(rows), status_details)
            diagnostics[seq] = diag
            print(f"[{i+1}/{len(seqs)}] {seq}: {diag['n_avail']}f {diag['fps']}fps empty={diag['empty_mask_frames']}")
        except Exception as e:  # 任何序列失敗不 crash 全局（Ranking B §6 fallback 精神）
            diagnostics[seq] = {"error": repr(e)[:200]}
            print(f"FAIL {seq}: {repr(e)[:120]}", file=sys.stderr)
            # §6 第 2 條規格：失敗序列**退化為 init 框複製**，而非靜默缺列。
            # 舊版只記 diagnostics 就往下走，最後 concat 只取存在的檔 ⇒ 殘缺 CSV 照樣上傳且 exit 0（D052b）。
            try:
                init_fb = read_init(seq_dir, seq, gt)
                fids_fb = frame_ids_for(seq, n_expect, gt, sample_frame_ids)
                fallback_rows = [(f"{seq}_{fid}", *init_fb) for fid in fids_fb]
                write_csv_atomic(csv_f, fallback_rows)
                diagnostics[seq]["fallback"] = f"init_copy x{len(fids_fb)}"
                diagnostics[seq]["status"] = "fallback"
                diagnostics[seq]["status_sidecar"] = str(status_path_for(csv_f))
                diagnostics[seq]["run_signature"] = signature["digest"]
                write_sequence_status(csv_f, seq, "fallback", signature, len(fallback_rows), {
                    "error": diagnostics[seq]["error"],
                    "fallback": diagnostics[seq]["fallback"],
                })
                print(f"  ↳ fallback：init 框複製 {len(fids_fb)} 列已寫入", file=sys.stderr)
            except Exception as e2:  # noqa: BLE001 —— 連 init 都拿不到＝真的無法產出該序列
                diagnostics[seq]["fallback_error"] = repr(e2)[:200]
                diagnostics[seq]["status"] = "failed"
                diagnostics[seq]["status_sidecar"] = str(status_path_for(csv_f))
                diagnostics[seq]["run_signature"] = signature["digest"]
                write_sequence_status(csv_f, seq, "failed", signature, 0, diagnostics[seq])
                print(f"  ↳ fallback 也失敗：{repr(e2)[:120]}", file=sys.stderr)

    # 合併前再驗一次 sidecar+signature，禁止上一次 run 的 CSV 混入本次提交。
    have, invalid, artifact_states = [], [], {}
    for seq in seqs:
        csv_f = seq_csv_dir / f"{seq}.csv"
        ok, state = _artifact_ok(csv_f, seq, count_frames(frames_root / seq),
                                 run_signatures[seq], {"complete", "fallback"})
        if ok:
            have.append(seq)
            artifact_states[seq] = state
        else:
            invalid.append((seq, state))
            diagnostics.setdefault(seq, {})["artifact_error"] = state
    sub = (pd.concat([pd.read_csv(seq_csv_dir / f"{s}.csv") for s in have], ignore_index=True)
           if have else pd.DataFrame(columns=SUB_COLS))
    # 總檔也要原子寫入；否則中止時可留下「檔案存在但只有半截」的假成品。
    write_csv_atomic(out / "submission.csv", list(sub[SUB_COLS].itertuples(index=False, name=None)))
    score_parts = [seq_csv_dir / f"{s}.scores.csv" for s in have
                   if artifact_states[s] == "complete" and
                   (seq_csv_dir / f"{s}.scores.csv").exists()]
    if score_parts:
        pd.concat([pd.read_csv(p) for p in score_parts], ignore_index=True).to_csv(
            out / "obj_scores.csv", index=False)
        print(f"obj_scores {sum(1 for _ in open(out / 'obj_scores.csv'))-1} 列 → {out / 'obj_scores.csv'}")
    meta = {"_meta": {"backend": tag,
                      "ckpt": (args.ckpt if args.backend == "samurai" else
                               (args.sam3_ckpt or "HF-default")),
                      "checkpoint_identity": checkpoint_identity, "n_seqs": len(seqs),
                      "sam3_samurai": bool(args.sam3_samurai),
                      "sam3_image_size": args.sam3_image_size or "default",
                      "samurai_reset_kf": samurai_reset_kf,
                      "samurai_legacy_cross_seq_kf": bool(args.samurai_legacy_cross_seq_kf),
                      "memory_gate": gate_cfg or False,
                      "sam3_eval": bool(args.sam3_eval),
                      "allow_fallback": bool(args.allow_fallback)}}
    # formal validator（P0-09）：對 sample 做 exact-set 比對，問題寫進 diagnostics 供事後追溯
    fallback_seqs = [s for s in have if artifact_states[s] == "fallback"]
    errs = validate_submission(sub, Path(args.sample_csv) if args.sample_csv else None,
                               fallback_seqs, args.allow_fallback)
    if invalid:
        errs.append(f"無有效 artifact 的序列 {len(invalid)} 支：{invalid[:5]}")
    meta["_meta"]["validation"] = errs or "pass"
    (out / "diagnostics.json").write_text(json.dumps({**meta, **diagnostics}, indent=1, ensure_ascii=False))
    n_fail = sum(1 for d in diagnostics.values() if "error" in d)
    n_fb = len(fallback_seqs)
    print(f"完成 {len(seqs)-n_fail}/{len(seqs)}，失敗 {n_fail}"
          f"（其中 {n_fb} 支已 fallback 補滿）；submission {len(sub)} 列 → {out}")
    if errs:
        print("❌ 提交檔驗證未過：", file=sys.stderr)
        for e in errs:
            print(f"   - {e}", file=sys.stderr)
        # fail-closed：fallback 也是正式驗證錯誤；只有 --allow-fallback 能明確放行。
        sys.exit(2)
    print("✅ 提交檔驗證通過"
          + (f"（{n_fail} 支序列曾失敗但已由 init 框複製補滿，見 diagnostics）" if n_fail else ""))


if __name__ == "__main__":
    main()
