#!/usr/bin/env python3
"""G2 = 真訓練＋災難檢查（G1 PASS 後的下一關；D090 授權鏈）。

與 G1 的三個刻意差異：
* 訓練集＝train folds 內**全部** usable clips，**不排除 identity clips**
  （21.6% identity 是 D087 設計的配比＝天然 rehearsal「別忘掉已會的」；
  G1 只挑非 identity 是 overfit sanity 的需要）。usable 的 GT 規則同 G1
  （採樣後首幀 valid ＋ ≥6 幀 valid）。
* fold 切分（clips JSON 的 fold 欄位，值域 {0..4}，對齊 physical capture
  groups、天然防洩漏）：--train-folds 訓練、--holdout-folds 只做 eval。
* 訓練集太大不可預載 RAM（~1800 clips ×8 幀 float32 ≈ 170GB）⇒ 訓練
  lazy 逐 step 讀圖；holdout eval 集（抽樣上限 HOLDOUT_MAX、確定性 seed）
  預載 RAM（80×8 幀 ≈ 7.8GB，Lambda 機無虞）。

═══ 事前判準（寫死於 2026-08-29，跑前不得改）═══
GO    ⇔ holdout best mean IoU − frozen(step0) mean IoU ≥ GO_MIN_IOU_GAIN(+0.02)。
ABORT ⇔ 第一個 ≥100 步的 holdout eval 點相對 step0 的 loss 下降 < 2%
        （D065 簽名；G1 實測 96.7% drop，理論上不會觸發，保留防線）。
ERROR_INPLACE ⇔ backward 撞 in-place autograd 錯誤（提示 [G1-V1]/[G1-V2]）。
keep-best 依 holdout mean IoU；verdict 含 per-modality 與 identity/非 identity
分組、best_step。rc：GO=0／NO_GO=2／ABORT=3／ERROR_INPLACE=4。
"""
from __future__ import annotations

import argparse
import json
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

# ═══ G2 事前判準常數（勿改）═══
GO_MIN_IOU_GAIN = 0.02
ABORT_CHECK_STEP = 100
ABORT_MIN_REL_DROP = 0.02
HOLDOUT_MAX = 80
VALID_FOLDS = {0, 1, 2, 3, 4}  # 實查 hard_clips_vis_rednir.json 的 fold 值域


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips-json", required=True)
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--frames-root")
    ap.add_argument("--sam3-ckpt", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-folds", default="0,1,2,3")
    ap.add_argument("--holdout-folds", default="4")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=1008)
    ap.add_argument("--lr", type=float, default=1e-4, help="同 G1（D033：不網格搜索）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--detach-memory", action="store_true")
    ap.add_argument("--smoke-plan-out",
                    help="只產 smoke 計畫（3 病灶＋3 健康）後退出：寫 json 至此路徑、"
                         "序列清單至 <path>.seqlist.txt（track_t1 --seq-list 用）。"
                         "需要 --frames-root 取 405 序列全集")
    return ap.parse_args(argv)


def parse_folds(s: str) -> set:
    folds = {int(v) for v in s.split(",") if v.strip() != ""}
    assert folds <= VALID_FOLDS, f"fold 值域外：{folds - VALID_FOLDS}"
    return folds


def usable_clip(clip: dict, gt_by_id: dict, t: int, stride: int) -> bool:
    """GT 規則同 G1 的 select（首幀 valid＋≥6 valid），**不看 identity**。"""
    pos = cd.sample_positions(0, clip["length"], t, stride)
    boxes = [gt_by_id.get(clip["frame_ids"][p]) for p in pos]
    return cd.frame_valid(boxes[0]) and sum(1 for b in boxes if cd.frame_valid(b)) >= 6


def split_by_folds(clips, gt_by_id, train_folds: set, holdout_folds: set,
                   t: int, stride: int):
    assert train_folds and holdout_folds, "兩側 folds 都不可為空"
    assert not (train_folds & holdout_folds), \
        f"train/holdout folds 相交：{train_folds & holdout_folds}"
    usable = [c for c in clips if usable_clip(c, gt_by_id, t, stride)]
    train = [c for c in usable if c["fold"] in train_folds]
    holdout = [c for c in usable if c["fold"] in holdout_folds]
    assert train and holdout, f"空集合：train={len(train)} holdout={len(holdout)}"
    return train, holdout


def sample_holdout(holdout, seed: int, cap: int = HOLDOUT_MAX):
    """確定性抽樣：排序後 seeded sample、再排序（控 eval 成本 ~cap 支）。"""
    ordered = sorted(holdout, key=lambda c: (c["sequence"], c["start_position"]))
    if len(ordered) <= cap:
        return ordered
    picked = random.Random(seed).sample(ordered, cap)
    return sorted(picked, key=lambda c: (c["sequence"], c["start_position"]))


SMOKE_MIN_VALID_GT = 16  # 健康對照至少要有這麼多有效 GT 幀才夠格當 canary


def build_smoke_plan(clips, frames_root, gt_by_id, n_failing: int = 3) -> dict:
    """3 病灶＋3 健康（D062：對照組必須含現行管線已良好的序列）。

    病灶＝clips 按 sequence 聚合 failing_frames 總和 top-3（tie 字典序）。
    健康＝405 目錄中 vis/rednir 且**零 clips** 的序列，**2 vis＋1 rednir**
    （各 modality 內字典序取前）——病灶實查全 vis（worker/L_person/pedestrian2），
    健康對照若純按字典序會全 rednir（"rednir-"<"vis-"），災難閘就測不到
    「vis 訓練毀掉健康 vis 序列」這個最可能的災難型——modality 平衡是判準
    的一部分，非美觀。

    兩道 eligibility 約束（每個候選、病灶與健康都檢，違反則同排序順位替補、
    記入 plan json 的 "substituted"）：
    (a) GT 行數 == frames_root/<seq> 的 jpg 張數——2026training.csv 有 2,316 行
        不在 167,174 行 contract 內（duplicate GT blocks；prep_oof405_frames.py
        的 choose_observed_block 就是為此存在）。行數不合的序列會讓 track_t1
        帶 --gt-csv 時 fail-closed（track_t1.py:118-120）＝訓練完成後 smoke 段
        才炸；即使不炸，smoke_compare 逐 GT 行迭代會把孤兒行算成雙側 0.0、
        對稱稀釋災難 delta。
    (b) 有效 GT 幀 ≥ SMOKE_MIN_VALID_GT——零 clips 也可能是「GT 幾乎全無效」
        造成的，那不是健康序列，且會撞 per_seq_mean_iou 的空集 assert。
    """
    gt_rows, gt_valid = {}, {}
    from train_tracker_g1.smoke_compare import seq_of

    for fid, box in gt_by_id.items():
        s = seq_of(fid)
        gt_rows[s] = gt_rows.get(s, 0) + 1
        if cd.frame_valid(box):
            gt_valid[s] = gt_valid.get(s, 0) + 1

    substituted = []

    def eligible(seq: str) -> bool:
        n_jpg = len(list((Path(frames_root) / seq).glob("*.jp*g")))
        if gt_rows.get(seq, 0) != n_jpg:
            substituted.append({"sequence": seq,
                                "reason": f"gt_rows={gt_rows.get(seq, 0)} != jpgs={n_jpg}"})
            return False
        if gt_valid.get(seq, 0) < SMOKE_MIN_VALID_GT:
            substituted.append({"sequence": seq,
                                "reason": f"valid_gt={gt_valid.get(seq, 0)} < {SMOKE_MIN_VALID_GT}"})
            return False
        return True

    def take(ranked, n):
        out = []
        for s in ranked:
            if len(out) == n:
                break
            if eligible(s):
                out.append(s)
        assert len(out) == n, f"合格候選不足 {n}：{out}（跳過 {len(substituted)}）"
        return out

    agg = {}
    for c in clips:
        agg[c["sequence"]] = agg.get(c["sequence"], 0) + c["failing_frames"]
    failing = take([s for s, _ in sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))],
                   n_failing)

    clip_seqs = set(agg)
    all_seqs = sorted(p.name for p in Path(frames_root).iterdir() if p.is_dir())
    zero_vis = [s for s in all_seqs if s.startswith("vis-") and s not in clip_seqs]
    zero_rednir = [s for s in all_seqs if s.startswith("rednir-") and s not in clip_seqs]
    healthy = take(zero_vis, 2) + take(zero_rednir, 1)
    return {"failing": failing, "healthy": healthy, "all": failing + healthy,
            "substituted": substituted,
            "constraints": {"gt_rows_eq_jpgs": True,
                            "min_valid_gt": SMOKE_MIN_VALID_GT}}


def decide_g2(eval_log, best_iou: float, best_step: int, final_step: int):
    """eval_log = [{"step","loss","iou"}...]（per-clip 列表，step 升冪含 0）。"""
    pooled = [(e["step"], statistics.fmean(e["loss"]), statistics.fmean(e["iou"]))
              for e in eval_log]
    l0, iou0 = pooled[0][1], pooled[0][2]
    detail = {"loss_step0": l0, "iou_step0": iou0,
              "best_iou": best_iou, "best_step": best_step,
              "iou_gain": best_iou - iou0,
              "rel_drop_at_abort_check": None,
              "criteria": {"go_min_iou_gain": GO_MIN_IOU_GAIN,
                           "abort_check_step": ABORT_CHECK_STEP,
                           "abort_min_rel_drop": ABORT_MIN_REL_DROP,
                           "holdout_max": HOLDOUT_MAX}}
    abort_pts = [p for p in pooled if p[0] >= ABORT_CHECK_STEP]
    if abort_pts:
        rel_drop = (l0 - abort_pts[0][1]) / max(l0, 1e-8)
        detail["rel_drop_at_abort_check"] = rel_drop
        if rel_drop < ABORT_MIN_REL_DROP:
            return "ABORT", detail
    go = (best_iou - iou0) >= GO_MIN_IOU_GAIN and final_step >= ABORT_CHECK_STEP
    return ("GO" if go else "NO_GO"), detail


def group_breakdown(holdout_clips, per_iou_frozen, per_iou_best) -> dict:
    """holdout per-clip IoU 依 modality 與 identity（min_iou>=0.5）分組。"""
    out = {}
    def add(name, idxs):
        if idxs:
            out[name] = {
                "n": len(idxs),
                "frozen": statistics.fmean(per_iou_frozen[i] for i in idxs),
                "best": statistics.fmean(per_iou_best[i] for i in idxs),
            }
    for mod in sorted({c["modality"] for c in holdout_clips}):
        add(f"modality:{mod}",
            [i for i, c in enumerate(holdout_clips) if c["modality"] == mod])
    add("identity",
        [i for i, c in enumerate(holdout_clips) if c["min_iou"] >= cd.IDENTITY_MIN_IOU])
    add("non_identity",
        [i for i, c in enumerate(holdout_clips) if c["min_iou"] < cd.IDENTITY_MIN_IOU])
    return out


def save_tracker_ckpt(tracker, path, **meta):
    import torch

    sd = {k: v for k, v in tracker.state_dict().items()
          if not k.startswith("backbone.")}
    torch.save({"tracker_state_dict": sd,
                "n_params": sum(v.numel() for v in sd.values()), **meta}, path)


def main(argv=None) -> int:
    args = parse_args(argv)
    index = cd.load_clip_index(args.clips_json)

    if args.smoke_plan_out:
        assert args.frames_root, "--smoke-plan-out 需要 --frames-root"
        plan = build_smoke_plan(index["clips"], args.frames_root,
                                cd.load_gt_csv(args.gt_csv))
        out = Path(args.smoke_plan_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        Path(str(out) + ".seqlist.txt").write_text("\n".join(plan["all"]) + "\n")
        print(json.dumps(plan, ensure_ascii=False))
        return 0

    gt = cd.load_gt_csv(args.gt_csv)
    train_clips, holdout_all = split_by_folds(
        index["clips"], gt, parse_folds(args.train_folds),
        parse_folds(args.holdout_folds), args.t, args.stride)
    holdout_clips = sample_holdout(holdout_all, args.seed)
    print(f"[g2] train={len(train_clips)} clips（folds {args.train_folds}，含 identity）"
          f" holdout={len(holdout_clips)}/{len(holdout_all)}（folds {args.holdout_folds}）",
          flush=True)

    import torch

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    assert args.frames_root, "--frames-root 必填"
    train_ds = cd.ClipDataset(train_clips, gt, frames_root=args.frames_root,
                              t=args.t, stride=args.stride, image_size=args.image_size)
    holdout_ds = cd.ClipDataset(holdout_clips, gt, frames_root=args.frames_root,
                                t=args.t, stride=args.stride, image_size=args.image_size)
    print(f"[g2] 預載 holdout {len(holdout_ds)} clips 進 RAM（訓練集 lazy）…", flush=True)
    holdout_samples = [holdout_ds[i] for i in range(len(holdout_ds))]

    from train_tracker_g1.sot_trainable import build_trainable_tracker, TrainableSOT

    sam3_model, tracker, n_trainable = build_trainable_tracker(
        args.sam3_ckpt or None, device=args.device)
    print(f"[g2] trainable = {n_trainable/1e6:.2f}M（tracker.*）", flush=True)
    model = TrainableSOT(tracker, detach_memory=args.detach_memory)
    trainable_params = [p for p in tracker.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.lr)

    eval_log = []
    best = {"iou": -1.0, "step": -1, "per_iou": None}

    def run_eval(step):
        pl, pi = evaluate(model, holdout_samples, args.device)
        eval_log.append({"step": step, "loss": pl, "iou": pi})
        entry = {"kind": "eval", "step": step,
                 "loss_mean": statistics.fmean(pl), "iou_mean": statistics.fmean(pi)}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**entry, "loss": pl, "iou": pi}) + "\n")
        print(f"[g2] eval@{step}: loss={entry['loss_mean']:.4f} "
              f"iou={entry['iou_mean']:.4f}", flush=True)
        if entry["iou_mean"] > best["iou"]:
            best.update(iou=entry["iou_mean"], step=step, per_iou=pi)
            save_tracker_ckpt(tracker, out_dir / "tracker_g2_best.pt",
                              step=step, holdout_iou=entry["iou_mean"])
            print(f"[g2]   new best（iou={entry['iou_mean']:.4f}）→ tracker_g2_best.pt",
                  flush=True)
        return pl, pi

    _, frozen_per_iou = run_eval(0)
    t_start = time.time()
    order, cursor = [], 0
    step = 0
    aborted = False
    for step in range(1, args.steps + 1):
        if cursor >= len(order):  # epoch 邊界：確定性重洗
            order = list(range(len(train_clips)))
            rng.shuffle(order)
            cursor = 0
        sample = train_ds[order[cursor]]
        cursor += 1
        optim.zero_grad(set_to_none=True)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=args.device.startswith("cuda")):
                out = model.forward_clip(sample["images"].to(args.device),
                                         sample["init_box_rel"])
            total, parts = L.g1_total_loss(out["high_res_logits"], out["iou_scores"],
                                           sample["gt_boxes_model"].to(args.device),
                                           sample["valid"])
            total.backward()
        except RuntimeError as e:
            if "inplace" not in str(e).lower():
                raise
            print(f"[g2] ERROR_INPLACE@{step}: {e}\n{INPLACE_HINT}", flush=True)
            (out_dir / "g2_verdict.json").write_text(json.dumps(
                {"verdict": "ERROR_INPLACE", "step": step, "error": str(e),
                 "hint": INPLACE_HINT}, indent=2, ensure_ascii=False))
            return 4
        torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP_NORM)
        optim.step()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "train", "step": step,
                                "clip": sample["meta"]["sequence"], **parts}) + "\n")
        if step % args.eval_every == 0:
            run_eval(step)
            verdict_now, _ = decide_g2(eval_log, best["iou"], best["step"], step)
            if verdict_now == "ABORT":
                print(f"[g2] ABORT@{step}：D065 簽名（holdout loss 未動）", flush=True)
                aborted = True
                break

    if not aborted and eval_log[-1]["step"] != step:
        run_eval(step)

    verdict, detail = decide_g2(eval_log, best["iou"], best["step"], step)
    detail.update({
        "steps_run": step, "wall_sec": round(time.time() - t_start, 1),
        "n_trainable": n_trainable,
        "n_train_clips": len(train_clips), "n_holdout_clips": len(holdout_clips),
        "train_folds": sorted(parse_folds(args.train_folds)),
        "holdout_folds": sorted(parse_folds(args.holdout_folds)),
        "breakdown": group_breakdown(
            holdout_clips, frozen_per_iou,
            best["per_iou"] if best["per_iou"] is not None else frozen_per_iou),
        "args": {k: v for k, v in vars(args).items()},
    })
    (out_dir / "g2_verdict.json").write_text(
        json.dumps({"verdict": verdict, **detail}, indent=2, ensure_ascii=False))
    print(f"[g2] verdict = {verdict}（best iou {best['iou']:.4f}@{best['step']}，"
          f"gain {detail['iou_gain']:+.4f}）", flush=True)
    save_tracker_ckpt(tracker, out_dir / "tracker_g2_final.pt",
                      step=step, verdict=verdict)
    return 0 if verdict == "GO" else (3 if verdict == "ABORT" else 2)


if __name__ == "__main__":
    sys.exit(main())
