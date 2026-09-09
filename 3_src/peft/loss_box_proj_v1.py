"""loss_box_proj_v1 — D068 Gate 1 的 BoxInst 型軸投影損失（配 teacher_rect_v1 的矩形 target）。

【機制】pred mask logits 沿 H 取 amax ⇒ x 軸投影 logits；沿 W 取 amax ⇒ y 軸投影。
矩形 target 的同軸投影恰為 GT box 的 0/1 區間 ⇒ 監督「mask 的外接範圍」而非形狀
＝ mask-free box 監督（BoxInst projection loss 的單物件版）。

【與官方 MultiStepMultiMasksAndIous 的介面相容性】
  - forward(outs_batch, targets_batch) 簽名相同；CORE_LOSS_KEY 聚合相同（trainer 零改動）。
  - weight_dict 沿用官方四鍵名（cfg 結構不變）：
      loss_mask → 投影 BCE（按投影長度取 mean ⇒ 無官方 focal 除全圖 1M 像素的小目標稀釋問題）
      loss_dice → 投影 dice
      loss_iou  → 恆 0（pred mask 對矩形的像素 IoU 被填充率封頂、無意義；權重應設 0）
      loss_class→ 官方 obj-score focal 原樣保留
  - multimask（M>1）：以投影 bce+dice 加權組合的 argmin 選最佳 mask（鏡射官方 pattern）。
  - 接受並忽略官方 cfg 的多餘 kwargs（focal_alpha 等），cfg 可少改。

【幾何前提】RandomAffine 的 rotation/shear 必須關閉（make_train_cfg_v3 --no-rotation）——
投影恆等式僅在軸對齊變換下成立；25° 旋轉會讓矩形投影膨脹 ≈×1.33 ＝系統性教「框太大」。
"""
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.loss_fns import CORE_LOSS_KEY, sigmoid_focal_loss
from training.utils.distributed import get_world_size, is_dist_avail_and_initialized


def _proj_losses(src_masks: torch.Tensor, target_masks: torch.Tensor):
    """src [N,M,H,W] logits；target [N,M,H,W] 0/1。回傳 (bce, dice) 各 [N,M]。"""
    bce_terms, dice_terms = [], []
    for dim in (-2, -1):                           # -2=沿 H 收 ⇒ x 投影；-1=沿 W 收 ⇒ y 投影
        p_logit = src_masks.amax(dim=dim)          # [N,M,L]
        t = target_masks.amax(dim=dim).float()     # [N,M,L]
        bce = F.binary_cross_entropy_with_logits(p_logit, t, reduction="none").mean(-1)
        p = torch.sigmoid(p_logit)
        inter = (p * t).sum(-1)
        dice = 1.0 - (2.0 * inter + 1.0) / (p.sum(-1) + t.sum(-1) + 1.0)
        bce_terms.append(bce)
        dice_terms.append(dice)
    return (bce_terms[0] + bce_terms[1]) / 2.0, (dice_terms[0] + dice_terms[1]) / 2.0


class BoxProjectionLoss(nn.Module):
    def __init__(self, weight_dict, pred_obj_scores=True,
                 focal_gamma_obj_score=0.0, focal_alpha_obj_score=-1, **_ignored):
        super().__init__()
        self.weight_dict = dict(weight_dict)
        for k in ("loss_mask", "loss_dice", "loss_iou"):
            assert k in self.weight_dict, f"weight_dict 缺 {k}"
        self.weight_dict.setdefault("loss_class", 0.0)
        self.pred_obj_scores = pred_obj_scores
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.focal_alpha_obj_score = focal_alpha_obj_score

    # ── 與官方相同的批次聚合 ────────────────────────────────────────────────
    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur = self._forward(outs, targets, num_objects)
            for k, v in cur.items():
                losses[k] += v
        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        target_masks = targets.unsqueeze(1).float()          # [N,1,H,W]
        assert target_masks.dim() == 4
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, object_score_logits in zip(src_masks_list, object_score_logits_list):
            self._update_losses(losses, src_masks, target_masks,
                                num_objects, object_score_logits)
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(self, losses, src_masks, target_masks, num_objects,
                       object_score_logits):
        target_masks = target_masks.expand_as(src_masks)     # [N,M,H,W]
        bce, dice = _proj_losses(src_masks, target_masks)    # [N,M] × 2

        # obj-score 監督：官方原樣（矩形 target>0 ⇒ present；退化/空 target ⇒ absent）
        if not self.pred_obj_scores:
            loss_class = torch.tensor(0.0, dtype=bce.dtype, device=bce.device)
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[..., None].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits, target_obj, num_objects,
                alpha=self.focal_alpha_obj_score, gamma=self.focal_gamma_obj_score)

        if bce.size(1) > 1:                                  # multimask：投影組合 argmin
            combo = (bce * self.weight_dict["loss_mask"]
                     + dice * self.weight_dict["loss_dice"])
            best = torch.argmin(combo, dim=-1)
            rows = torch.arange(combo.size(0), device=combo.device)
            bce, dice = bce[rows, best], dice[rows, best]
        else:
            bce, dice = bce[:, 0], dice[:, 0]

        losses["loss_mask"] += bce.sum() / num_objects
        losses["loss_dice"] += dice.sum() / num_objects
        losses["loss_iou"] += torch.zeros((), dtype=bce.dtype, device=bce.device)
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        reduced = 0
        for k, w in self.weight_dict.items():
            if k in losses and w != 0:
                reduced = reduced + losses[k] * w
        return reduced
