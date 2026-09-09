#!/usr/bin/env python3
"""make_train_cfg_v2 — E31 協定對齊 micro-probe 的 config 產生器。

v1（Phase 1 用）的**唯一差異**：新增 `--base-lr` 與 `--align-protocol`。
v1 的行為在不給這兩個新旗標時**位元級不變**（同樣載入官方模板、同樣四處改動）。

【--align-protocol 做什麼】把訓練協定改成與我方推論一致（設計依據見
`5_outputs/e31_protocol_probe_20260812/DESIGN.md` §2，每一項都經控制流查證）：

  prob_to_use_pt_input_for_train   0.5 → 1.0   L189 use_pt_input 恆 True，不再餵 GT mask
  prob_to_use_box_input_for_train  0.5 → 1.0   L229 恆走 box（原僅 25%）
  num_init_cond_frames_for_train   2   → 1     L208 ⇒ init_cond_frames=[0]，只有首幀
  rand_init_cond_frames_for_train  T   → F     num=1 時 L190 已無作用，明確化
  num_frames_to_correct_for_train  2   → 1     L84 硬 assert 要求 >= num_init_cond_frames
  rand_frames_to_correct_for_train T   → F     L195-199 條件已不成立，明確化
  num_correction_pt_per_frame      7   → 1     🚨 不可設 0

🚨 **num_correction_pt_per_frame 不可設 0 的理由**（開機前查證，避免機上才炸）：
   `training/model/sam2.py` L472 `for _ in range(self.num_correction_pt_per_frame):`
   迴圈不執行 ⇒ L541 `return point_inputs, sam_outputs` 的 `sam_outputs` 從未賦值
   ⇒ **UnboundLocalError**。要真正歸零須改官方碼並自造 fallback（需多傳 obj_ptr）
   ＝引入我方補丁的新風險，不做。**「首幀 1 個修正點」是已知且有界的殘餘不匹配。**

⇒ 對齊後：8 幀 clip 中只有第 1 幀是條件幀（box ＋1 修正點），其餘 7 幀純傳播。
"""
import argparse
from omegaconf import OmegaConf

# 協定對齊的完整規格（單一真相源，判決書直接引用此表）
ALIGNED_PROTOCOL = {
    "prob_to_use_pt_input_for_train": 1.0,
    "prob_to_use_box_input_for_train": 1.0,
    "num_init_cond_frames_for_train": 1,
    "rand_init_cond_frames_for_train": False,
    "num_frames_to_correct_for_train": 1,
    "rand_frames_to_correct_for_train": False,
    "num_correction_pt_per_frame": 1,
}


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
    ap.add_argument("--base-lr", type=float, default=None,
                    help="覆寫 scratch.base_lr；不給則沿用官方 5.0e-6")
    ap.add_argument("--align-protocol", action="store_true",
                    help="套用 ALIGNED_PROTOCOL（E31 的唯一實質變因）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = OmegaConf.load(
        f"{a.sam2_repo}/sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml")
    infer_l = OmegaConf.load(f"{a.sam2_repo}/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")

    # ── 以下四處與 v1 完全相同 ──────────────────────────────────────────────
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

    # ── v2 新增 ─────────────────────────────────────────────────────────────
    if a.base_lr is not None:
        old = base.scratch.base_lr
        base.scratch.base_lr = a.base_lr
        print(f"[cfg_v2] base_lr {old} → {a.base_lr}")

    if a.align_protocol:
        m = base.trainer.model
        for k, v in ALIGNED_PROTOCOL.items():
            if k not in m:
                raise SystemExit(f"🚨 模板缺鍵 {k}——官方 config 結構已變，停止（不得靜默新增）")
            print(f"[cfg_v2] {k}: {m[k]} → {v}")
            m[k] = v
        # 開機前已驗證的硬 assert（training/model/sam2.py L84）：提前在此失敗而非機上
        assert m.num_frames_to_correct_for_train >= m.num_init_cond_frames_for_train, \
            "num_frames_to_correct_for_train 必須 >= num_init_cond_frames_for_train（L84 assert）"
        assert m.num_correction_pt_per_frame >= 1, \
            "num_correction_pt_per_frame 不可為 0（L472/L541 UnboundLocalError）"

    with open(a.out, "w") as fh:
        fh.write("# @package _global_\n")
        fh.write(OmegaConf.to_yaml(base))
    print(f"[cfg_v2] 寫出 {a.out}（epochs={a.num_epochs}, lr={base.scratch.base_lr}, "
          f"aligned={a.align_protocol}）")


if __name__ == "__main__":
    main()
