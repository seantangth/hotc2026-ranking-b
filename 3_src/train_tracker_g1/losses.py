#!/usr/bin/env python3
"""G1 losses：box-only 監督（我們沒有 mask GT——405 訓練集只有逐幀 bbox）。

主案 = projection loss（BoxInst 型）：GT box 在 x/y 軸的 0/1 投影 vs
pred mask logits 沿軸 max 的 BCE + dice。對 logits 取軸向 max 再 sigmoid
等價於對 prob 取 max（sigmoid 單調），可微、自包含、不需 teacher。

次要 = IoU head 監督。⚠️ 這是 **proxy**，不是 SAM2 原版 iou_loss：
sam3/train/loss/loss_fns.py 的 iou_loss 需要 GT mask（targets dim=4，
mask-vs-mask IoU），我們沒有 mask GT，簽名不合 ⇒ 自寫 MSE，target 用
「pred mask 的 tight-bbox 對 GT box 的 box-IoU」代替 mask-IoU。
mask→tight-box IoU ≠ mask IoU（已知偏差），G1 權重壓在 0.5。

損失一律以 fp32 計算（呼叫端的 autocast 不應包住本模組；輸入先 .float()）。

GT 有效性：405 GT 含無效幀（x,y,w,h 任一 ≤0，見 build_hard_clips.py 的
``valid = np.all(gt > 0, axis=1)``）；無效幀不進 loss（per-frame valid mask）。
"""
from __future__ import annotations

from typing import Optional, Sequence

# 權重與數值常數（G1 事前定案 2026-08-29；跑前不得改）
W_PROJECTION = 1.0
W_IOU_HEAD = 0.5
DICE_EPS = 1.0


def box_iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    """xywh box IoU，純 python（無 torch 依賴，供 CPU 測試與 eval hook 共用）。"""
    ax1, ay1, aw, ah = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bw, bh = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def mask_to_box_xywh(mask) -> Optional[tuple]:
    """bool/0-1 2D mask → tight bbox xywh；空 mask → None。

    語意逐項對照 track_t1.py:69-73 的 mask_to_box（E33：nonzero 外接矩形，
    x2 = xs.max()+1），空 mask 的「沿用前框」由呼叫端處理（eval hook 同
    track_t1 慣例）。接受 torch tensor 或 numpy array。
    """
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    import numpy as np

    ys, xs = np.nonzero(np.asarray(mask))
    if len(xs) == 0:
        return None
    x1, y1 = float(xs.min()), float(ys.min())
    x2, y2 = float(xs.max() + 1), float(ys.max() + 1)
    return (x1, y1, x2 - x1, y2 - y1)


def projection_targets(gt_boxes_xywh, size: int):
    """GT xywh（模型座標）→ x/y 軸 0/1 投影 targets。

    像素 i 屬於 box ⟺ [i, i+1) 與 [x1, x2) 相交 ⟺ ceil 前的 floor(x1) ≤ i < ceil(x2)。
    回傳 (tx [N, size], ty [N, size])，float32。
    """
    import torch

    n = gt_boxes_xywh.shape[0]
    tx = torch.zeros(n, size, dtype=torch.float32)
    ty = torch.zeros(n, size, dtype=torch.float32)
    for i in range(n):
        x, y, w, h = (float(v) for v in gt_boxes_xywh[i])
        if w <= 0 or h <= 0:
            continue  # 無效 GT：留全零（呼叫端應已用 valid mask 排除）
        import math

        x1 = min(max(int(math.floor(x)), 0), size)
        x2 = min(max(int(math.ceil(x + w)), 0), size)
        y1 = min(max(int(math.floor(y)), 0), size)
        y2 = min(max(int(math.ceil(y + h)), 0), size)
        tx[i, x1:x2] = 1.0
        ty[i, y1:y2] = 1.0
    return tx, ty


def _dice_1d(prob, target):
    """1D 投影 dice（per-sample 平均）。自寫而非 import sam3 的 dice_loss：
    後者是 2D mask 版、含 num_boxes 正規化與 multimask 分支，簽名不合。"""
    import torch

    num = 2.0 * (prob * target).sum(dim=-1) + DICE_EPS
    den = prob.sum(dim=-1) + target.sum(dim=-1) + DICE_EPS
    return (1.0 - num / den).mean()


def projection_loss(mask_logits, gt_boxes_xywh, valid):
    """主 loss。mask_logits [N,1,S,S]、gt_boxes_xywh [N,4]（模型座標）、valid [N] bool。

    回傳 (loss_tensor, parts dict)。valid 全 False → 零 loss（保持 graph 連通）。
    """
    import torch
    import torch.nn.functional as F

    assert mask_logits.dim() == 4 and mask_logits.shape[1] == 1, mask_logits.shape
    size = mask_logits.shape[-1]
    assert mask_logits.shape[-2] == size, "非正方形 mask 不在 G1 範圍"

    logits = mask_logits.float().squeeze(1)  # [N,S,S]；fp32（autocast 外）
    keep = torch.as_tensor(valid, dtype=torch.bool, device=logits.device)
    if not bool(keep.any()):
        zero = logits.sum() * 0.0
        return zero, {"proj_bce": 0.0, "proj_dice": 0.0, "n_valid": 0}

    logits = logits[keep]
    boxes = gt_boxes_xywh[keep] if hasattr(gt_boxes_xywh, "shape") else [
        b for b, k in zip(gt_boxes_xywh, valid) if k
    ]
    tx, ty = projection_targets(boxes, size)
    tx, ty = tx.to(logits.device), ty.to(logits.device)

    # 軸向 max 投影：x 投影 = 沿 H（dim=-2）取 max；y 投影 = 沿 W（dim=-1）。
    x_proj = logits.max(dim=-2).values  # [n,S]
    y_proj = logits.max(dim=-1).values  # [n,S]

    bce = F.binary_cross_entropy_with_logits(x_proj, tx) + F.binary_cross_entropy_with_logits(y_proj, ty)
    dice = _dice_1d(torch.sigmoid(x_proj), tx) + _dice_1d(torch.sigmoid(y_proj), ty)
    loss = bce + dice
    return loss, {
        "proj_bce": float(bce.detach()),
        "proj_dice": float(dice.detach()),
        "n_valid": int(keep.sum()),
    }


def iou_head_proxy_loss(iou_scores, mask_logits, gt_boxes_xywh, valid):
    """IoU head 的 proxy 監督（見模組 docstring 的偏差說明）。

    iou_scores [N]（track_step 在 use_memory_selection=True 下輸出的
    iou_score = 被選 mask 的 IoU 預測）；target = tight-box(pred mask) 對
    GT box 的 box-IoU（no_grad 幾何量）。回傳 (loss, parts)。
    """
    import torch
    import torch.nn.functional as F

    scores = iou_scores.float().reshape(-1)
    keep = torch.as_tensor(valid, dtype=torch.bool, device=scores.device)
    if not bool(keep.any()):
        return scores.sum() * 0.0, {"iou_head_mse": 0.0, "n_valid": 0}

    with torch.no_grad():
        targets = torch.zeros_like(scores)
        for i in range(scores.shape[0]):
            if not bool(keep[i]):
                continue
            box = mask_to_box_xywh(mask_logits[i, 0] > 0)
            gt = tuple(float(v) for v in gt_boxes_xywh[i])
            targets[i] = box_iou_xywh(box, gt) if box is not None else 0.0

    mse = F.mse_loss(scores[keep], targets[keep])
    return mse, {"iou_head_mse": float(mse.detach()), "n_valid": int(keep.sum())}


def g1_total_loss(mask_logits, iou_scores, gt_boxes_xywh, valid):
    """G1 總 loss = W_PROJECTION·projection + W_IOU_HEAD·iou_head_proxy。

    iou_scores 可為 None（防禦：上游 use_memory_selection=False 時無此輸出），
    此時只算 projection。回傳 (total, parts)。
    """
    proj, parts = projection_loss(mask_logits, gt_boxes_xywh, valid)
    total = W_PROJECTION * proj
    if iou_scores is not None:
        iou_l, iou_parts = iou_head_proxy_loss(iou_scores, mask_logits, gt_boxes_xywh, valid)
        total = total + W_IOU_HEAD * iou_l
        parts.update(iou_parts)
    parts["total"] = float(total.detach())
    return total, parts
