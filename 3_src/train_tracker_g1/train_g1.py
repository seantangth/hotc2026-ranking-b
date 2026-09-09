#!/usr/bin/env python3
"""G1 = 10-clip overfit sanity。存在意義：排除 D065 的死亡簽名
（「loss 40 epochs 完全持平＝目標函數從未被優化」）。

═══ 事前判準（寫死於 2026-08-29，跑前不得改；自動判定，不靠人眼）═══
全部基於「每 EVAL_EVERY 步的全量 eval（10 clips、tracker.eval()、no_grad）」
的 per-clip loss / IoU 序列——**不用** per-step 訓練 loss（round-robin 下
每步是不同 clip，混 clip 窗口均值會把「有在學」誤判成 flat；D065 量的
也是 eval 序列）。

ABORT（D065 簽名）：完成 ABORT_CHECK_STEP 步時，pooled eval loss 相對
  step0 的下降比例 < ABORT_MIN_REL_DROP ⇒ 立即停、verdict=ABORT。
PASS（兩條都要）：
  (i)  L_final ≤ PASS_LOSS_RATIO × L_step0，且 eval loss 序列大致單調：
       相鄰點滿足 next ≤ prev×(1+MONO_TOL) 的比例 ≥ MONO_MIN_FRAC。
  (ii) mean IoU（mask→tight-box vs GT box，E33 慣例、空 mask 沿用前框）
       final − step0 ≥ PASS_IOU_MIN_GAIN。
FAIL：非 PASS 非 ABORT。

eval 用 tracker.eval()（dropout off，判準低噪聲）；與部署的已知偏差
（上游從不 .eval()，部署預設 training-mode dropout active，E26）記錄於
交付回報，不影響 overfit sanity 的判定力。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 3_src/
from train_tracker_g1 import clip_dataset as cd  # noqa: E402
from train_tracker_g1 import losses as L  # noqa: E402

# ═══ 事前判準常數（勿改）═══
EVAL_EVERY = 50
ABORT_CHECK_STEP = 100
ABORT_MIN_REL_DROP = 0.02  # team-lead 指令口徑：前 100 步 loss 持平 <2% ＝ D065 簽名
PASS_LOSS_RATIO = 0.50
PASS_IOU_MIN_GAIN = 0.10
MONO_TOL = 0.05
MONO_MIN_FRAC = 0.70
GRAD_CLIP_NORM = 1.0

# 第一次 backward 才能證實的兩個 in-place 嫌疑點（sot_trainable.py docstring 的
# [G1-V1]/[G1-V2]）；backward 撞 in-place RuntimeError 時印此提示後以 rc=4 退出
# ——rc=4 是工程錯誤、不是判決，launch 腳本會視為真失敗攔下。
INPLACE_HINT = """\
[G1] backward 撞 in-place autograd 錯誤。兩個已知嫌疑點（G0 報告第二節 #5/#6）：
  [G1-V1] RoPEAttention 的 `q, k[:, :, :num_k_rope] = apply_rotary_enc(...)`
          （sam3/sam/transformer.py:321-337）→ mitigation：呼叫端先 k = k.clone()。
  [G1-V2] `maskmem_features += (1-p)*no_obj_embed_spatial`
          （sam3_tracker_base.py:845-847）→ mitigation：改 out-of-place 加法。
套用對應的一行修改後重跑；勿改其他上游程式。"""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips-json", required=True)
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--frames-root", help="prep_oof405_frames.py 的輸出根目錄")
    ap.add_argument("--sam3-ckpt", default="", help="本地 sam3.pt；留空從 HF 下載（gated）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n-clips", type=int, default=10)
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=1008)
    ap.add_argument("--lr", type=float, default=1e-4, help="文獻預設（D033：不網格搜索）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--detach-memory", action="store_true",
                    help="VRAM 逃生口：截斷 BPTT（G1 預設不用）")
    ap.add_argument("--print-sequences", action="store_true",
                    help="只列出選中 clip 的序列名後退出（launch 腳本選擇性拉資料用）")
    return ap.parse_args(argv)


def select_and_load(args):
    index = cd.load_clip_index(args.clips_json)
    gt = cd.load_gt_csv(args.gt_csv)
    chosen = cd.select_g1_clips(index["clips"], gt, n=args.n_clips,
                                t=args.t, stride=args.stride)
    return chosen, gt


def evaluate(model, samples, device):
    """全量 eval：per-clip loss 與 per-clip mean IoU（valid 幀）。

    IoU 語意對照 track_t1：mask>0 的 tight bbox（E33），空 mask 沿用前一幀
    的框（首幀前框=GT init）。回傳 (per_clip_loss, per_clip_iou)。
    """
    import torch

    tracker = model.tracker
    was_training = tracker.training
    tracker.eval()  # 判準去噪；部署偏差已記錄於模組 docstring
    per_loss, per_iou = [], []
    with torch.no_grad():
        for sample in samples:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.startswith("cuda")):
                out = model.forward_clip(sample["images"].to(device),
                                         sample["init_box_rel"])
            gt_boxes = sample["gt_boxes_model"].to(device)
            valid = sample["valid"]
            total, _ = L.g1_total_loss(out["high_res_logits"], out["iou_scores"],
                                       gt_boxes, valid)
            per_loss.append(float(total))

            ious, prev_box = [], tuple(float(v) for v in gt_boxes[0])
            for j in range(gt_boxes.shape[0]):
                box = L.mask_to_box_xywh(out["high_res_logits"][j, 0] > 0)
                if box is None:
                    box = prev_box  # 空 mask 沿用前框（track_t1.mask_to_box 慣例）
                prev_box = box
                if bool(valid[j]):
                    ious.append(L.box_iou_xywh(box, tuple(float(v) for v in gt_boxes[j])))
            per_iou.append(statistics.fmean(ious) if ious else 0.0)
    if was_training:
        from train_tracker_g1.sot_trainable import set_train_mode
        set_train_mode(tracker)
    return per_loss, per_iou


def decide_verdict(eval_log, final_step):
    """依模組 docstring 的事前判準回傳 (verdict, detail dict)。eval_log =
    [{"step": s, "loss": [per-clip], "iou": [per-clip]}, ...]（step 升冪含 0）。"""
    pooled = [(e["step"], statistics.fmean(e["loss"]), statistics.fmean(e["iou"]))
              for e in eval_log]
    l0, iou0 = pooled[0][1], pooled[0][2]
    lf, iouf = pooled[-1][1], pooled[-1][2]
    detail = {
        "loss_step0": l0, "loss_final": lf,
        "iou_step0": iou0, "iou_final": iouf,
        "rel_drop_at_abort_check": None, "monotone_frac": None,
        "criteria": {
            "eval_every": EVAL_EVERY, "abort_check_step": ABORT_CHECK_STEP,
            "abort_min_rel_drop": ABORT_MIN_REL_DROP,
            "pass_loss_ratio": PASS_LOSS_RATIO,
            "pass_iou_min_gain": PASS_IOU_MIN_GAIN,
            "mono_tol": MONO_TOL, "mono_min_frac": MONO_MIN_FRAC,
        },
    }

    abort_pts = [p for p in pooled if p[0] >= ABORT_CHECK_STEP]
    if abort_pts:
        rel_drop = (l0 - abort_pts[0][1]) / max(l0, 1e-8)
        detail["rel_drop_at_abort_check"] = rel_drop
        if rel_drop < ABORT_MIN_REL_DROP:
            return "ABORT", detail

    if len(pooled) >= 3:
        ok = sum(1 for (_, a, _), (_, b, _) in zip(pooled, pooled[1:])
                 if b <= a * (1 + MONO_TOL))
        detail["monotone_frac"] = ok / (len(pooled) - 1)
    else:
        detail["monotone_frac"] = 1.0

    passed = (
        lf <= PASS_LOSS_RATIO * l0
        and detail["monotone_frac"] >= MONO_MIN_FRAC
        and (iouf - iou0) >= PASS_IOU_MIN_GAIN
        and final_step >= ABORT_CHECK_STEP
    )
    return ("PASS" if passed else "FAIL"), detail


def main(argv=None):
    args = parse_args(argv)
    chosen, gt = select_and_load(args)

    if args.print_sequences:
        for c in chosen:
            print(c["sequence"])
        return 0

    import torch

    torch.manual_seed(args.seed)
    import random

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    assert args.frames_root, "--frames-root 必填（非 --print-sequences 模式）"
    ds = cd.ClipDataset(chosen, gt, frames_root=args.frames_root,
                        t=args.t, stride=args.stride, image_size=args.image_size)
    print(f"[g1] 載入 {len(ds)} clips 進 RAM …", flush=True)
    samples = [ds[i] for i in range(len(ds))]
    for s in samples:
        print(f"  {s['meta']['sequence']} @{s['meta']['start_position']} "
              f"valid={int(s['valid'].sum())}/{args.t}")

    from train_tracker_g1.sot_trainable import build_trainable_tracker, TrainableSOT

    sam3_model, tracker, n_trainable = build_trainable_tracker(
        args.sam3_ckpt or None, device=args.device)
    print(f"[g1] trainable = {n_trainable/1e6:.2f}M（tracker.*）", flush=True)
    model = TrainableSOT(tracker, detach_memory=args.detach_memory)

    trainable_params = [p for p in tracker.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.lr)

    eval_log = []

    def run_eval(step):
        pl, pi = evaluate(model, samples, args.device)
        entry = {"kind": "eval", "step": step, "loss": pl, "iou": pi,
                 "loss_mean": statistics.fmean(pl), "iou_mean": statistics.fmean(pi)}
        eval_log.append({"step": step, "loss": pl, "iou": pi})
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[g1] eval@{step}: loss={entry['loss_mean']:.4f} iou={entry['iou_mean']:.4f}",
              flush=True)
        return entry

    run_eval(0)
    t_start = time.time()
    aborted = False
    step = 0
    for step in range(1, args.steps + 1):
        sample = samples[(step - 1) % len(samples)]  # round-robin
        optim.zero_grad(set_to_none=True)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=args.device.startswith("cuda")):
                out = model.forward_clip(sample["images"].to(args.device),
                                         sample["init_box_rel"])
            # loss 在 autocast 外、fp32（losses.py 內部亦 .float()）
            total, parts = L.g1_total_loss(out["high_res_logits"], out["iou_scores"],
                                           sample["gt_boxes_model"].to(args.device),
                                           sample["valid"])
            total.backward()
        except RuntimeError as e:
            if "inplace" not in str(e).lower():
                raise
            print(f"[g1] ERROR_INPLACE@{step}: {e}\n{INPLACE_HINT}", flush=True)
            (out_dir / "g1_verdict.json").write_text(json.dumps(
                {"verdict": "ERROR_INPLACE", "step": step, "error": str(e),
                 "hint": INPLACE_HINT}, indent=2, ensure_ascii=False))
            return 4
        torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP_NORM)
        optim.step()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "train", "step": step,
                                "clip": sample["meta"]["sequence"], **parts}) + "\n")

        if step % EVAL_EVERY == 0:
            run_eval(step)
            verdict_now, _ = decide_verdict(eval_log, step)
            if verdict_now == "ABORT" and step >= ABORT_CHECK_STEP:
                print(f"[g1] ABORT@{step}：D065 簽名（eval loss 未動）", flush=True)
                aborted = True
                break

    if not aborted and eval_log[-1]["step"] != step:
        run_eval(step)

    verdict, detail = decide_verdict(eval_log, step)
    detail.update({"steps_run": step, "wall_sec": round(time.time() - t_start, 1),
                   "n_trainable": n_trainable,
                   "clips": [c["sequence"] for c in chosen],
                   "args": {k: v for k, v in vars(args).items()}})
    (out_dir / "g1_verdict.json").write_text(
        json.dumps({"verdict": verdict, **detail}, indent=2, ensure_ascii=False))
    print(f"[g1] verdict = {verdict}", flush=True)

    # ckpt 只存 tracker.*，且過濾掉 attach 上去的 backbone（1.24GB → ~47MB）。
    sd = {k: v for k, v in tracker.state_dict().items()
          if not k.startswith("backbone.")}
    torch.save({"tracker_state_dict": sd, "step": step, "verdict": verdict,
                "n_params": sum(v.numel() for v in sd.values())},
               out_dir / "tracker_g1_final.pt")
    return 0 if verdict == "PASS" else (3 if verdict == "ABORT" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
