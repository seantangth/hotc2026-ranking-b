#!/usr/bin/env python3
"""P1 精度線（D105，2026-09-02）：uniform 取樣的 tracker 微調 ＋ Stage A 事前判準。

與 G2（train_g2.py）的刻意差異，**且只有這些**：
* 資料：`build_uniform_clips.py` 的均勻 clip 索引（全 405、三模態、16 幀非重疊窗），
  不是 failure-centric hard clips——這是 D105 要測的單一變因。
* trainable scope（arm）：`--trainable-scope decoder`（只訓 tracker.sam_mask_decoder，4.2M）
  或 `tracker`（全 tracker.*，11.7M，同 G2）。兩 arm **只差這一維**。
* lr：1e-4 ＋ linear warmup 100 步 ＋ cosine 衰減至 0.1×（D095 註記「2400 步起震盪、
  延長訓練前先解 lr schedule」的處置；兩 arm 皆加、非搜索）。
* holdout eval 上限 120 clips（均勻 clips 基線 IoU 高、方差小，多抽一點）。
loss／T／stride／image_size／seed／grad clip 全同 G2。

═══ Stage A 事前判準（寫死於 2026-09-02，跑前不得改；全文見 STRATEGY D105）═══
g ≡ holdout best mean IoU − step0 mean IoU（clip 級，mask→tight box，同 G1 evaluate）。
ABORT   ⇔ 首個 ≥100 步的 holdout eval 點相對 step0 的 loss 下降 < 2%（D065 簽名）。
PASS_A  ⇔ g ≥ +0.01 且跑滿 ≥100 步 ⇒ 該 arm 進 Stage B（full-sequence smoke，見 precision_compare.py）。
FAIL_A  ⇔ 其餘。
rc：PASS_A=0／FAIL_A=2／ABORT=3／ERROR_INPLACE=4。

`--smoke-plan-out` 模式：產 Stage B 的代表性序列計畫（fold 4、每模態 min(4, 合格數) 支、字典序等距；
合格＝GT 行數＝jpg 數 ∧ 有效 GT ≥16）後退出。這是 D062 意義下的**代表性樣本**，不是病灶挑選。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 3_src/
from train_tracker_g1 import clip_dataset as cd  # noqa: E402
from train_tracker_g1 import losses as L  # noqa: E402
from train_tracker_g1.train_g1 import (  # noqa: E402 —— G1 定案的共用件
    GRAD_CLIP_NORM, INPLACE_HINT, evaluate)
from train_tracker_g1.train_g2 import (  # noqa: E402 —— G2 定案的共用件
    SMOKE_MIN_VALID_GT, parse_folds, sample_holdout, save_tracker_ckpt, split_by_folds)

# ═══ P1 事前判準常數（勿改）═══
P1_MIN_CLIP_GAIN = 0.01
ABORT_CHECK_STEP = 100
ABORT_MIN_REL_DROP = 0.02
HOLDOUT_MAX = 120
SMOKE_PER_MODALITY = 4
MODALITIES = ("vis", "nir", "rednir")
SCOPES = ("decoder", "tracker")
LR_WARMUP_STEPS = 100
LR_FINAL_FRAC = 0.1


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips-json", required=True, help="build_uniform_clips.py 的輸出")
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--frames-root")
    ap.add_argument("--sam3-ckpt", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm", default="A", help="標籤（A/B），只影響檔名")
    ap.add_argument("--trainable-scope", default="decoder", choices=SCOPES)
    ap.add_argument("--train-folds", default="0,1,2,3")
    ap.add_argument("--holdout-folds", default="4")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--holdout-max", type=int, default=HOLDOUT_MAX)
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=1008)
    ap.add_argument("--lr", type=float, default=1e-4, help="同 G1/G2（D033：不網格搜索）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--detach-memory", action="store_true")
    ap.add_argument("--smoke-plan-out",
                    help="只產 Stage B 代表性序列計畫後退出（json ＋ <path>.seqlist.txt）；需 --frames-root")
    return ap.parse_args(argv)


def cosine_lr(step: int, total: int, base_lr: float,
              final_frac: float = LR_FINAL_FRAC, warmup: int = LR_WARMUP_STEPS) -> float:
    """step 1-based。warmup 內線性升到 base_lr；之後 cosine 衰減到 final_frac·base_lr。"""
    if step <= warmup:
        return base_lr * step / max(warmup, 1)
    progress = min(1.0, (step - warmup) / max(total - warmup, 1))
    return base_lr * (final_frac + (1.0 - final_frac) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def decide_stage_a(eval_log, best_iou: float, best_step: int, final_step: int):
    """eval_log = [{"step","loss","iou"}...]（per-clip 列表，step 升冪含 0）。"""
    pooled = [(e["step"], statistics.fmean(e["loss"]), statistics.fmean(e["iou"]))
              for e in eval_log]
    l0, iou0 = pooled[0][1], pooled[0][2]
    detail = {"loss_step0": l0, "iou_step0": iou0,
              "best_iou": best_iou, "best_step": best_step,
              "clip_gain": best_iou - iou0,
              "rel_drop_at_abort_check": None,
              "criteria": {"min_clip_gain": P1_MIN_CLIP_GAIN,
                           "abort_check_step": ABORT_CHECK_STEP,
                           "abort_min_rel_drop": ABORT_MIN_REL_DROP}}
    abort_pts = [p for p in pooled if p[0] >= ABORT_CHECK_STEP]
    if abort_pts:
        rel_drop = (l0 - abort_pts[0][1]) / max(l0, 1e-8)
        detail["rel_drop_at_abort_check"] = rel_drop
        if rel_drop < ABORT_MIN_REL_DROP:
            return "ABORT", detail
    ok = (best_iou - iou0) >= P1_MIN_CLIP_GAIN and final_step >= ABORT_CHECK_STEP
    return ("PASS_A" if ok else "FAIL_A"), detail


def evenly_spaced(items: list, k: int) -> list:
    """字典序排序後等距取 k 個（k ≥ len 則全取）。確定性、無隨機。"""
    items = sorted(items)
    if k <= 0:
        return []
    if k >= len(items):
        return items
    if k == 1:
        return [items[0]]
    idx = [round(i * (len(items) - 1) / (k - 1)) for i in range(k)]
    return [items[i] for i in idx]


def build_precision_smoke_plan(clips, frames_root, gt_by_id, holdout_folds: set,
                               per_modality: int = SMOKE_PER_MODALITY) -> dict:
    """Stage B 代表性序列：holdout fold 內、每模態 min(per_modality, 合格數) 支、字典序等距。

    合格（同 G2 build_smoke_plan 的兩道約束）：(a) GT 行數 == frames_root/<seq> 的 jpg 張數
    （否則 track_t1 --gt-csv 會 fail-closed）；(b) 有效 GT 幀 ≥ SMOKE_MIN_VALID_GT。
    不合格者記入 substituted，並由等距選取在合格集合上重算（而非同位替補）。
    """
    from train_tracker_g1.smoke_compare import seq_of

    gt_rows, gt_valid = {}, {}
    for fid, box in gt_by_id.items():
        s = seq_of(fid)
        gt_rows[s] = gt_rows.get(s, 0) + 1
        if cd.frame_valid(box):
            gt_valid[s] = gt_valid.get(s, 0) + 1

    cand_by_mod = {m: set() for m in MODALITIES}
    for c in clips:
        if c["fold"] in holdout_folds and c["modality"] in cand_by_mod:
            cand_by_mod[c["modality"]].add(c["sequence"])

    substituted, chosen_by_mod = [], {}
    for mod in MODALITIES:
        eligible = []
        for seq in sorted(cand_by_mod[mod]):
            n_jpg = len(list((Path(frames_root) / seq).glob("*.jp*g")))
            if gt_rows.get(seq, 0) != n_jpg:
                substituted.append({"sequence": seq,
                                    "reason": f"gt_rows={gt_rows.get(seq, 0)} != jpgs={n_jpg}"})
                continue
            if gt_valid.get(seq, 0) < SMOKE_MIN_VALID_GT:
                substituted.append({"sequence": seq,
                                    "reason": f"valid_gt={gt_valid.get(seq, 0)} < {SMOKE_MIN_VALID_GT}"})
                continue
            eligible.append(seq)
        chosen_by_mod[mod] = evenly_spaced(eligible, per_modality)
    all_seqs = [s for m in MODALITIES for s in chosen_by_mod[m]]
    assert all_seqs, "Stage B 計畫為空：holdout fold 內無合格序列"
    return {"holdout_folds": sorted(holdout_folds), "per_modality": per_modality,
            "by_modality": chosen_by_mod, "all": all_seqs,
            "substituted": substituted,
            "constraints": {"gt_rows_eq_jpgs": True, "min_valid_gt": SMOKE_MIN_VALID_GT,
                            "selection": "lexicographic evenly spaced within eligible"}}


def group_breakdown_p1(holdout_clips, per_iou_frozen, per_iou_best) -> dict:
    """依 modality 分組；identity 分組只在所有 clip 都有 min_iou 時才算（uniform 索引可能無統計）。"""
    out = {}

    def add(name, idxs):
        if idxs:
            out[name] = {"n": len(idxs),
                         "frozen": statistics.fmean(per_iou_frozen[i] for i in idxs),
                         "best": statistics.fmean(per_iou_best[i] for i in idxs)}

    for mod in sorted({c["modality"] for c in holdout_clips}):
        add(f"modality:{mod}", [i for i, c in enumerate(holdout_clips) if c["modality"] == mod])
    if all(c.get("min_iou") is not None for c in holdout_clips):
        add("identity", [i for i, c in enumerate(holdout_clips)
                         if c["min_iou"] >= cd.IDENTITY_MIN_IOU])
        add("non_identity", [i for i, c in enumerate(holdout_clips)
                             if c["min_iou"] < cd.IDENTITY_MIN_IOU])
    else:
        out["identity_grouping"] = "skipped (index has no pred stats)"
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    index = cd.load_clip_index(args.clips_json)
    holdout_folds = parse_folds(args.holdout_folds)

    if args.smoke_plan_out:
        assert args.frames_root, "--smoke-plan-out 需要 --frames-root"
        plan = build_precision_smoke_plan(index["clips"], args.frames_root,
                                          cd.load_gt_csv(args.gt_csv), holdout_folds)
        out = Path(args.smoke_plan_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        Path(str(out) + ".seqlist.txt").write_text("\n".join(plan["all"]) + "\n")
        print(json.dumps(plan, ensure_ascii=False))
        return 0

    gt = cd.load_gt_csv(args.gt_csv)
    train_clips, holdout_all = split_by_folds(
        index["clips"], gt, parse_folds(args.train_folds), holdout_folds, args.t, args.stride)
    holdout_clips = sample_holdout(holdout_all, args.seed, cap=args.holdout_max)
    tag = f"p1_{args.arm}_{args.trainable_scope}"
    print(f"[{tag}] train={len(train_clips)} clips（folds {args.train_folds}，uniform）"
          f" holdout={len(holdout_clips)}/{len(holdout_all)}（folds {args.holdout_folds}）",
          flush=True)

    import torch

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{tag}_train_log.jsonl"

    assert args.frames_root, "--frames-root 必填"
    train_ds = cd.ClipDataset(train_clips, gt, frames_root=args.frames_root,
                              t=args.t, stride=args.stride, image_size=args.image_size)
    holdout_ds = cd.ClipDataset(holdout_clips, gt, frames_root=args.frames_root,
                                t=args.t, stride=args.stride, image_size=args.image_size)
    print(f"[{tag}] 預載 holdout {len(holdout_ds)} clips 進 RAM（訓練集 lazy）…", flush=True)
    holdout_samples = [holdout_ds[i] for i in range(len(holdout_ds))]

    from train_tracker_g1.sot_trainable import (
        TrainableSOT, build_trainable_tracker_scoped, set_train_mode_scope)

    sam3_model, tracker, n_trainable = build_trainable_tracker_scoped(
        args.sam3_ckpt or None, device=args.device, scope=args.trainable_scope)
    print(f"[{tag}] trainable = {n_trainable/1e6:.2f}M（scope={args.trainable_scope}）", flush=True)
    model = TrainableSOT(tracker, detach_memory=args.detach_memory)
    trainable_params = [p for p in tracker.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.lr)

    eval_log = []
    best = {"iou": -1.0, "step": -1, "per_iou": None}

    def run_eval(step):
        pl, pi = evaluate(model, holdout_samples, args.device)
        set_train_mode_scope(tracker, args.trainable_scope)  # evaluate() 只會還原到 G1 的全 train mode
        eval_log.append({"step": step, "loss": pl, "iou": pi})
        entry = {"kind": "eval", "step": step,
                 "loss_mean": statistics.fmean(pl), "iou_mean": statistics.fmean(pi)}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**entry, "loss": pl, "iou": pi}) + "\n")
        print(f"[{tag}] eval@{step}: loss={entry['loss_mean']:.4f} iou={entry['iou_mean']:.4f}",
              flush=True)
        if entry["iou_mean"] > best["iou"]:
            best.update(iou=entry["iou_mean"], step=step, per_iou=pi)
            save_tracker_ckpt(tracker, out_dir / f"tracker_{tag}_best.pt",
                              step=step, holdout_iou=entry["iou_mean"],
                              scope=args.trainable_scope, arm=args.arm)
            print(f"[{tag}]   new best（iou={entry['iou_mean']:.4f}）→ tracker_{tag}_best.pt", flush=True)
        return pl, pi

    _, frozen_per_iou = run_eval(0)
    t_start = time.time()
    order, cursor = [], 0
    step = 0
    aborted = False
    for step in range(1, args.steps + 1):
        if cursor >= len(order):
            order = list(range(len(train_clips)))
            rng.shuffle(order)
            cursor = 0
        sample = train_ds[order[cursor]]
        cursor += 1
        lr_now = cosine_lr(step, args.steps, args.lr)
        for g in optim.param_groups:
            g["lr"] = lr_now
        optim.zero_grad(set_to_none=True)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=args.device.startswith("cuda")):
                out = model.forward_clip(sample["images"].to(args.device), sample["init_box_rel"])
            total, parts = L.g1_total_loss(out["high_res_logits"], out["iou_scores"],
                                           sample["gt_boxes_model"].to(args.device), sample["valid"])
            total.backward()
        except RuntimeError as e:
            if "inplace" not in str(e).lower():
                raise
            print(f"[{tag}] ERROR_INPLACE@{step}: {e}\n{INPLACE_HINT}", flush=True)
            (out_dir / f"{tag}_stageA.json").write_text(json.dumps(
                {"verdict": "ERROR_INPLACE", "step": step, "error": str(e), "hint": INPLACE_HINT},
                indent=2, ensure_ascii=False))
            return 4
        torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP_NORM)
        optim.step()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "train", "step": step, "lr": lr_now,
                                "clip": sample["meta"]["sequence"], **parts}) + "\n")
        if step % args.eval_every == 0:
            run_eval(step)
            verdict_now, _ = decide_stage_a(eval_log, best["iou"], best["step"], step)
            if verdict_now == "ABORT":
                print(f"[{tag}] ABORT@{step}：D065 簽名（holdout loss 未動）", flush=True)
                aborted = True
                break

    if not aborted and eval_log[-1]["step"] != step:
        run_eval(step)

    verdict, detail = decide_stage_a(eval_log, best["iou"], best["step"], step)
    detail.update({
        "arm": args.arm, "trainable_scope": args.trainable_scope,
        "steps_run": step, "wall_sec": round(time.time() - t_start, 1),
        "n_trainable": n_trainable,
        "n_train_clips": len(train_clips), "n_holdout_clips": len(holdout_clips),
        "train_folds": sorted(parse_folds(args.train_folds)),
        "holdout_folds": sorted(holdout_folds),
        "lr_schedule": {"base": args.lr, "warmup": LR_WARMUP_STEPS, "final_frac": LR_FINAL_FRAC},
        "breakdown": group_breakdown_p1(
            holdout_clips, frozen_per_iou,
            best["per_iou"] if best["per_iou"] is not None else frozen_per_iou),
        "args": {k: v for k, v in vars(args).items()},
    })
    (out_dir / f"{tag}_stageA.json").write_text(
        json.dumps({"verdict": verdict, **detail}, indent=2, ensure_ascii=False))
    print(f"[{tag}] Stage A verdict = {verdict}（best iou {best['iou']:.4f}@{best['step']}，"
          f"gain {detail['clip_gain']:+.4f}）", flush=True)
    save_tracker_ckpt(tracker, out_dir / f"tracker_{tag}_final.pt",
                      step=step, verdict=verdict, scope=args.trainable_scope, arm=args.arm)
    return 0 if verdict == "PASS_A" else (3 if verdict == "ABORT" else 2)


if __name__ == "__main__":
    sys.exit(main())
