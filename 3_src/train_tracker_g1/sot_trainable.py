#!/usr/bin/env python3
"""策略 (b)：繞過 Sam3TrackerPredictor 的 demo/推論管理層，自寫精簡 clip forward，
直接呼叫 Sam3TrackerBase.track_step。計算核心（tracker.transformer /
sam_mask_decoder / maskmem_backbone）一行不改——與推論共用同一段程式碼。

依據 G0 調查（D090）的手術點，全部有 file:line 錨：
* predictor 8 個入口掛 @torch.inference_mode()（sam3_tracking_predictor.py:56/179/
  342/672/789/906/978/1181）⇒ 訓練不走 predictor 方法。
* Sam3TrackerPredictor.__init__ 進入**永不退出的全域 bf16 autocast**
  （sam3_tracking_predictor.py:50-51）⇒ build 完立刻退出（exit_global_bf16_autocast）。
* forward_image 把 no-grad backbone 與可訓練的 conv_s0/s1（tracker.sam_mask_decoder
  參數）揉在同一函式（sam3_tracker_base.py:446-463）⇒ 這裡拆開：backbone 在
  torch.no_grad() 下、conv_s0/s1 在 grad 下。
* 訓練模式三個未定義屬性（sam3_tracker_base.py:333/684-688）⇒ set_missing_train_attrs。
* 首幀 box prompt 編碼逐行對照 sam3_tracking_predictor.py:216-238
  （rel→abs ×image_size、box→2 點 labels [2,3]、concat_points）。
* memory bank（output_dict）在基底類不 detach（sam3_tracker_base.py:656/731/
  1039-1040）⇒ BPTT 原生成立，本檔不需（也不得）動 memory 傳遞。

G1 驗證清單（第一次 backward 才能證實的兩個 in-place 點；撞錯才動，勿預先改上游）：
* [G1-V1] RoPEAttention 的 `q, k[:, :, :num_k_rope] = apply_rotary_enc(...)`
  （sam3/sam/transformer.py:321-337）——預期 index_put autograd 可處理；
  若 backward 撞 "modified by an inplace operation"，在呼叫端 clone k。
* [G1-V2] `maskmem_features += (1-p)*no_obj_embed_spatial`
  （sam3_tracker_base.py:845-847）——預期無事；撞了改 out-of-place。
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

TRAINABLE_PARAM_MIN = 10_000_000   # tracker.* 實測 11.7M；範圍外＝凍結範圍錯了
TRAINABLE_PARAM_MAX = 14_000_000


def exit_global_bf16_autocast(tracker) -> bool:
    """退出 Sam3TrackerPredictor.__init__ 進入的全域 bf16 autocast。

    從不同呼叫深度 __exit__ 一個 torch.autocast 是合法的狀態還原；失敗則
    fallback：保持其開啟、依賴訓練迴圈自己的同 dtype autocast（同 dtype 巢狀
    autocast 是 no-op），loss 端已用 .float() 保證 fp32。回傳是否成功退出。
    """
    ctx = getattr(tracker, "bf16_context", None)
    if ctx is None:
        return True
    try:
        ctx.__exit__(None, None, None)
        assert not torch.is_autocast_enabled(), "autocast 仍開啟"
        return True
    except Exception as e:  # noqa: BLE001 —— fallback 路徑要留活口
        logging.warning("bf16_context.__exit__ 失敗（%s）；改走巢狀 autocast fallback", e)
        return False


def set_missing_train_attrs(tracker) -> None:
    """補上 OSS release 被截斷的訓練屬性（G0 報告第二節 #7）。"""
    tracker.teacher_force_obj_scores_for_mem = False
    tracker.prob_to_dropout_spatial_mem = 0.0
    if not hasattr(tracker, "frames_to_add_correction_pt"):
        tracker.frames_to_add_correction_pt = []


def freeze_all_but_tracker(model) -> int:
    """整模型凍結 → 只解凍 tracker.* → 再凍 tracker.backbone。

    ⚠️ 順序不可換：track_t1 慣例會把 detector.backbone 掛到 tracker.backbone，
    掛上後它**可經 tracker.parameters() 到達**——漏了第三步，optimizer 會收下
    310M backbone 參數。回傳 trainable 參數量並 hard-fail 範圍檢查。
    """
    model.requires_grad_(False)
    model.tracker.requires_grad_(True)
    if getattr(model.tracker, "backbone", None) is not None:
        model.tracker.backbone.requires_grad_(False)
    n = sum(p.numel() for p in model.tracker.parameters() if p.requires_grad)
    assert TRAINABLE_PARAM_MIN < n < TRAINABLE_PARAM_MAX, (
        f"trainable 參數 {n/1e6:.1f}M 不在 tracker-only 預期範圍 "
        f"({TRAINABLE_PARAM_MIN/1e6:.0f}M–{TRAINABLE_PARAM_MAX/1e6:.0f}M)——凍結範圍錯了"
    )
    return n


def set_train_mode(tracker) -> None:
    """tracker 進 train mode（dropout 0.1 與原始訓練一致），backbone 維持 eval
    ——backbone 前向在 no_grad 下且凍結，train mode 只會多出 ViT drop_path 噪聲。"""
    tracker.train()
    if getattr(tracker, "backbone", None) is not None:
        tracker.backbone.eval()


def build_trainable_tracker(sam3_ckpt: Optional[str], device: str = "cuda"):
    """與 track_t1.py:875-880 同構的組裝（同一 builder、同一 attach），再套訓練手術。

    回傳 (sam3_model, tracker, n_trainable)。sam3_model 要留著（state_dict 載入
    的宿主）；訓練只碰 tracker。
    """
    from sam3.model_builder import build_sam3_video_model

    kw = {"checkpoint_path": sam3_ckpt} if sam3_ckpt else {}
    sam3_model = build_sam3_video_model(device=device, **kw)
    tracker = sam3_model.tracker
    tracker.backbone = sam3_model.detector.backbone  # track_t1.py:880 同款 attach
    exit_global_bf16_autocast(tracker)
    set_missing_train_attrs(tracker)
    n = freeze_all_but_tracker(sam3_model)
    set_train_mode(tracker)
    return sam3_model, tracker, n


def encode_first_frame_box_prompt(tracker, init_box_rel: torch.Tensor, device) -> dict:
    """首幀 box → point_inputs。逐行對照 sam3_tracking_predictor.py:216-238：
    rel 座標 ×image_size 轉絕對（:219-221）、box reshape 成 2 點＋labels [2,3]
    （:234-236，"consistent with how SAM 2 is trained"）、concat_points（:247）。"""
    from sam3.model.sam3_tracker_base import concat_points

    box = init_box_rel.to(device=device, dtype=torch.float32) * tracker.image_size
    box_coords = box.reshape(1, 2, 2)
    box_labels = torch.tensor([2, 3], dtype=torch.int32, device=device).reshape(1, 2)
    return concat_points(None, box_coords, box_labels)


# ═══ P1 精度線（D105，2026-09-02）：trainable scope ═══
# 既有 freeze_all_but_tracker／set_train_mode／build_trainable_tracker 一字不改（G1/G2 selftest 依賴）。
# scope "tracker" ＝ 原行為；scope "decoder" ＝ 只解凍 tracker.sam_mask_decoder（含 conv_s0/s1，
# sam/mask_decoder.py:77-80）。參數量範圍依 scope 帶（D088 實測：tracker.* 11.7M、sam_mask_decoder 4.2M）。
SCOPES = ("decoder", "tracker")
SCOPE_PARAM_RANGES = {
    "tracker": (TRAINABLE_PARAM_MIN, TRAINABLE_PARAM_MAX),
    "decoder": (3_000_000, 6_000_000),
}


def freeze_to_scope(model, scope: str) -> int:
    """依 scope 凍結，回傳 trainable 參數量並 hard-fail 範圍檢查。"""
    if scope not in SCOPES:
        raise ValueError(f"未知 scope {scope!r}；可用 {SCOPES}")
    if scope == "tracker":
        return freeze_all_but_tracker(model)
    model.requires_grad_(False)
    model.tracker.sam_mask_decoder.requires_grad_(True)
    n = sum(p.numel() for p in model.tracker.parameters() if p.requires_grad)
    lo, hi = SCOPE_PARAM_RANGES["decoder"]
    assert lo < n < hi, (
        f"trainable 參數 {n/1e6:.2f}M 不在 decoder-only 預期範圍 ({lo/1e6:.0f}M–{hi/1e6:.0f}M)——凍結範圍錯了"
    )
    return n


def set_train_mode_scope(tracker, scope: str) -> None:
    """train mode 只給可訓子模組；凍結子模組維持 eval（dropout 關）。
    decoder scope 下 transformer／maskmem 的 dropout 若仍開著，會給訓練訊號多一層與部署無關的噪聲。"""
    if scope not in SCOPES:
        raise ValueError(f"未知 scope {scope!r}；可用 {SCOPES}")
    if scope == "tracker":
        set_train_mode(tracker)
        return
    tracker.eval()
    tracker.sam_mask_decoder.train()
    if getattr(tracker, "backbone", None) is not None:
        tracker.backbone.eval()


def build_trainable_tracker_scoped(sam3_ckpt: Optional[str], device: str = "cuda",
                                   scope: str = "tracker"):
    """與 build_trainable_tracker 同構，只差凍結範圍與 train-mode 範圍依 scope 決定。"""
    from sam3.model_builder import build_sam3_video_model

    kw = {"checkpoint_path": sam3_ckpt} if sam3_ckpt else {}
    sam3_model = build_sam3_video_model(device=device, **kw)
    tracker = sam3_model.tracker
    tracker.backbone = sam3_model.detector.backbone  # track_t1.py:880 同款 attach
    exit_global_bf16_autocast(tracker)
    set_missing_train_attrs(tracker)
    n = freeze_to_scope(sam3_model, scope)
    set_train_mode_scope(tracker, scope)
    return sam3_model, tracker, n


class TrainableSOT(nn.Module):
    """單物件 clip forward（BPTT）。呼叫端負責 autocast 與 optimizer。"""

    def __init__(self, tracker, detach_memory: bool = False):
        super().__init__()
        self.tracker = tracker
        # G1 預設全程 BPTT（SAM2 官方訓練同款）；detach_memory 是 VRAM 逃生口。
        self.detach_memory = detach_memory

    def _backbone_features_one_frame(self, image_1chw: torch.Tensor):
        """手術點（G0 第二節 #4）：backbone no-grad、conv_s0/s1 在 grad 下。
        對照 sam3_tracker_base.py:446-463 的 forward_image，但拆開兩段。"""
        tracker = self.tracker
        with torch.no_grad():
            raw = tracker.backbone.forward_image(image_1chw)["sam2_backbone_out"]
        fpn = list(raw["backbone_fpn"])
        # conv_s0/s1 是 tracker.sam_mask_decoder 的參數（sam/mask_decoder.py:77-80）
        # ——必須在 grad 下跑，它們是訓練對象的一部分。
        fpn[0] = tracker.sam_mask_decoder.conv_s0(fpn[0])
        fpn[1] = tracker.sam_mask_decoder.conv_s1(fpn[1])
        backbone_out = {"backbone_fpn": fpn, "vision_pos_enc": list(raw["vision_pos_enc"])}
        (_, vision_feats, vision_pos_embeds, feat_sizes,
         ) = tracker._prepare_backbone_features(backbone_out)
        return vision_feats, vision_pos_embeds, feat_sizes

    def forward_clip(self, images: torch.Tensor, init_box_rel: torch.Tensor) -> dict:
        """images [T,3,S,S]（已 normalize）、init_box_rel [1,4] xyxy 正規化。

        回傳 dict：high_res_logits [T,1,S,S]（fp32，_forward_sam_heads:351 已 .float()）、
        iou_scores [T] 或 None、obj_score_logits [T]。
        """
        tracker = self.tracker
        device = images.device
        t_total = images.shape[0]
        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        point_inputs = encode_first_frame_box_prompt(tracker, init_box_rel, device)

        high_res, iou_scores, obj_scores = [], [], []
        for t in range(t_total):
            vision_feats, vision_pos, feat_sizes = self._backbone_features_one_frame(
                images[t:t + 1])
            is_init = t == 0
            # 與推論路徑的已知偏差（記錄於 G1 交付回報）：推論在首幀
            # add_new_points_or_box 時 run_mem_encoder=False、memory 由
            # propagate_in_video_preflight 的 consolidation 補encode；單物件下
            # 兩者輸出等價，這裡直接在首幀 encode（run_mem_encoder=True 全程）。
            out = tracker.track_step(
                frame_idx=t,
                is_init_cond_frame=is_init,
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos,
                feat_sizes=feat_sizes,
                image=None,  # SimpleMaskEncoder 分支不用 image（sam3_tracker_base.py:832-836）
                point_inputs=point_inputs if is_init else None,
                mask_inputs=None,
                output_dict=output_dict,
                num_frames=t_total,
                run_mem_encoder=True,
            )
            if self.detach_memory and out.get("maskmem_features") is not None:
                out["maskmem_features"] = out["maskmem_features"].detach()
                out["obj_ptr"] = out["obj_ptr"].detach()
            key = "cond_frame_outputs" if is_init else "non_cond_frame_outputs"
            output_dict[key][t] = out

            high_res.append(out["pred_masks_high_res"])
            obj_scores.append(out["object_score_logits"].reshape(-1)[0])
            if "iou_score" in out:  # use_memory_selection=True（部署預設）才有
                iou_scores.append(out["iou_score"].reshape(-1)[0])

        return {
            "high_res_logits": torch.cat(high_res, dim=0),  # [T,1,S,S]
            "iou_scores": torch.stack(iou_scores) if len(iou_scores) == t_total else None,
            "obj_score_logits": torch.stack(obj_scores),
        }
