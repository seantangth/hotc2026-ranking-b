#!/usr/bin/env python3
"""make_train_cfg_v1 — 由官方 MOSE finetune 模板產生 E-F Phase 1 訓練 config（DESIGN.md §2/§3）。

不手刻 YAML：載入官方 sam2.1_hiera_b+_MOSE_finetune.yaml，只動下列幾處，其餘逐字保留（D033）：
  1. trainer.model.image_encoder ← 換成 sam2.1_hiera_l.yaml（推論 config）的 L 版骨幹子樹
     （b+ 的 embed_dim 112 → L 的 144/stages[2,6,36,4]/global_att[23,33,43]…）
  2. trunk.drop_path_rate = 0.0 —— image encoder 全凍結（freeze_spec_v1），殘留 stochastic depth
     只會對凍結特徵注入雜訊，無任何訓練意義
  3. trainer.model._target_ = freeze_spec_v1.SAM2TrainFrozen（建構後立即套凍結規格並印報告）
  4. dataset 三路徑、checkpoint_path、num_epochs、launcher（單 GPU、明確 log dir）

⚠️ 輸出檔必須以 '# @package _global_' 開頭（hydra 指令，OmegaConf 存檔會弄丟，故手動補）。
⚠️ ${times:}/${divide:} 等自訂 resolver 以未解析字串原樣保留（to_yaml 預設 resolve=False）。
"""
import argparse
from omegaconf import OmegaConf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam2-repo", required=True, help="官方 sam2 repo 根目錄")
    ap.add_argument("--img-folder", required=True)
    ap.add_argument("--gt-folder", required=True)
    ap.add_argument("--file-list", required=True)
    ap.add_argument("--ckpt", required=True, help="sam2.1_hiera_large.pt")
    ap.add_argument("--num-epochs", type=int, required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--save-freq", type=int, default=0,
                    help="每 N epoch 存 ckpt；主訓練用 1（中途保全，D016），smoke 用 0")
    ap.add_argument("--out", required=True, help="輸出 yaml 路徑（須在 <repo>/sam2/configs/ 下）")
    a = ap.parse_args()

    base = OmegaConf.load(
        f"{a.sam2_repo}/sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml")
    infer_l = OmegaConf.load(f"{a.sam2_repo}/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")

    # 1) b+ → L 骨幹（整棵 image_encoder 子樹取自官方 L 推論 config，非手抄參數）
    base.trainer.model.image_encoder = infer_l.model.image_encoder
    # 2) 凍結骨幹 ⇒ 關掉 stochastic depth
    base.trainer.model.image_encoder.trunk.drop_path_rate = 0.0
    # 3) 凍結規格（建構後立即 requires_grad_ 手術 + 報告；core 前綴對不到會硬失敗）
    base.trainer.model._target_ = "freeze_spec_v1.SAM2TrainFrozen"
    # 4) 資料與訓練長度
    base.dataset.img_folder = a.img_folder
    base.dataset.gt_folder = a.gt_folder
    base.dataset.file_list_txt = a.file_list
    base.scratch.num_epochs = a.num_epochs
    base.scratch.num_train_workers = a.num_workers
    base.trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path = a.ckpt
    base.launcher.gpus_per_node = 1
    base.launcher.experiment_log_dir = a.log_dir
    base.trainer.checkpoint.save_freq = a.save_freq

    with open(a.out, "w") as fh:
        fh.write("# @package _global_\n")
        fh.write(OmegaConf.to_yaml(base))
    print(f"[make_train_cfg_v1] 寫出 {a.out}  (epochs={a.num_epochs}, workers={a.num_workers})")


if __name__ == "__main__":
    main()
