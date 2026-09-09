#!/usr/bin/env python3
"""make_train_cfg_v3 — D068 Gate 1 config 產生器（v2 的嚴格超集）。

不給新旗標時行為與 v2 位元級一致（同模板、同四處改動、同 --base-lr/--align-protocol）。

【v3 新增】
  --loss-box-proj   trainer.loss.all 換成 training.loss_box_proj_v1.BoxProjectionLoss，
                    weight_dict → {loss_mask:10, loss_dice:1, loss_iou:0, loss_class:1}
                    （loss_iou 必為 0：pred mask 對矩形 target 的像素 IoU 被填充率封頂）
  --no-rotation     遞迴找 RandomAffine 節點 → degrees=0、shear=None
                    （投影恆等式僅軸對齊變換成立；25° 旋轉會系統性教「框太大」）

搭配：teacher_rect_v1（矩形 target 樹）＋ loss_box_proj_v1（須先複製進 $SAM2REPO/training/）。
"""
import argparse
from omegaconf import DictConfig, ListConfig, OmegaConf

ALIGNED_PROTOCOL = {
    "prob_to_use_pt_input_for_train": 1.0,
    "prob_to_use_box_input_for_train": 1.0,
    "num_init_cond_frames_for_train": 1,
    "rand_init_cond_frames_for_train": False,
    "num_frames_to_correct_for_train": 1,
    "rand_frames_to_correct_for_train": False,
    "num_correction_pt_per_frame": 1,          # 🚨 不可 0（L472/L541 UnboundLocalError）
}

BOX_PROJ_WEIGHTS = {"loss_mask": 10, "loss_dice": 1, "loss_iou": 0, "loss_class": 1}


def _walk(node, fn):
    if isinstance(node, DictConfig):
        fn(node)
        for v in node.values():
            _walk(v, fn)
    elif isinstance(node, ListConfig):
        for v in node:
            _walk(v, fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam2-repo", required=True)
    ap.add_argument("--img-folder", required=True)
    ap.add_argument("--gt-folder", required=True)
    ap.add_argument("--file-list", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-epochs", type=int, required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--save-freq", type=int, default=0)
    ap.add_argument("--base-lr", type=float, default=None)
    ap.add_argument("--align-protocol", action="store_true")
    ap.add_argument("--loss-box-proj", action="store_true")
    ap.add_argument("--no-rotation", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = OmegaConf.load(
        f"{a.sam2_repo}/sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml")
    infer_l = OmegaConf.load(f"{a.sam2_repo}/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")

    # ── v1/v2 原樣的標準改動 ────────────────────────────────────────────────
    base.trainer.model.image_encoder = infer_l.model.image_encoder
    base.trainer.model.image_encoder.trunk.drop_path_rate = 0.0
    base.trainer.model._target_ = "freeze_spec_v1.SAM2TrainFrozen"
    base.dataset.img_folder = a.img_folder
    base.dataset.gt_folder = a.gt_folder
    base.dataset.file_list_txt = a.file_list
    base.scratch.num_epochs = a.num_epochs
    base.scratch.num_train_workers = a.num_workers
    base.trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path = a.ckpt
    base.launcher.gpus_per_node = 1
    base.launcher.experiment_log_dir = a.log_dir
    base.trainer.checkpoint.save_freq = a.save_freq

    if a.base_lr is not None:
        print(f"[cfg_v3] base_lr {base.scratch.base_lr} → {a.base_lr}")
        base.scratch.base_lr = a.base_lr

    if a.align_protocol:
        m = base.trainer.model
        for k, v in ALIGNED_PROTOCOL.items():
            if k not in m:
                raise SystemExit(f"🚨 模板缺鍵 {k}——官方 config 結構已變，停止")
            print(f"[cfg_v3] {k}: {m[k]} → {v}")
            m[k] = v
        assert m.num_frames_to_correct_for_train >= m.num_init_cond_frames_for_train
        assert m.num_correction_pt_per_frame >= 1

    if a.loss_box_proj:
        loss = base.trainer.loss.all
        assert str(loss._target_).endswith("MultiStepMultiMasksAndIous"), \
            f"🚨 loss 節點不是官方類（{loss._target_}）——模板結構已變，停止"
        loss._target_ = "training.loss_box_proj_v1.BoxProjectionLoss"
        for k, v in BOX_PROJ_WEIGHTS.items():
            loss.weight_dict[k] = v
        print(f"[cfg_v3] loss → BoxProjectionLoss, weights={BOX_PROJ_WEIGHTS}")

    if a.no_rotation:
        hits = []
        def fix(node):
            t = node.get("_target_", "") if hasattr(node, "get") else ""
            if isinstance(t, str) and t.endswith("RandomAffine"):
                if "degrees" in node:
                    node.degrees = 0
                if "shear" in node:
                    node.shear = None
                hits.append(t)
        _walk(base, fix)
        assert hits, "🚨 找不到 RandomAffine 節點——模板結構已變，停止"
        print(f"[cfg_v3] RandomAffine ×{len(hits)}：degrees=0、shear=None")

    with open(a.out, "w") as fh:
        fh.write("# @package _global_\n")
        fh.write(OmegaConf.to_yaml(base))
    print(f"[cfg_v3] 寫出 {a.out}（epochs={a.num_epochs}, lr={base.scratch.base_lr}, "
          f"aligned={a.align_protocol}, box_proj={a.loss_box_proj}, no_rot={a.no_rotation}）")


if __name__ == "__main__":
    main()
