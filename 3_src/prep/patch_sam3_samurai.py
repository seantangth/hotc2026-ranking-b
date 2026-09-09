#!/usr/bin/env python3
"""E18：把 SAMURAI 的 Kalman 運動先驗移植到 SAM3 tracker（原始碼 patch，冪等）。

## 為什麼

E15 實測（EXPERIMENT_LOG exp006）：SAM3 在 62/65 支贏 E02 **+0.0129**、CLE 10.44→8.78，
但 3 支無人機群場景全崩（vis/rednir-droneshow2、rednir-drone2）。軌跡診斷顯示凍結率僅
3–9%、瞬移少、CLE 中位卻達 107px（E02 為 1.6–1.8px）＝**identity switch 而非跟丟**：
SAM3 表徵強但無運動先驗，密集相似目標靠外觀無法維持身分。

SAM3 與 SAM2 在此處結構完全相同（`multimask_output_in_sam/for_tracking=True`、
`num_multimask_outputs=3`、以最高 IoU estimate 挑 best mask），故 SAMURAI 的做法可直接移植：
**不改架構，只把挑 mask 的評分從 `ious` 換成 `0.15*kf_ious + 0.85*ious`。**

運動連續性是物理約束、零學習成分、不認場景 → 不吃 D037 分佈外縮水，Ranking B 合法（D038）。

## 修掉的上游 bug（連 SAMURAI 自己都有）

SAMURAI 的 `kf_mean`/`kf_covariance`/`stable_frames` 只在 `__init__` 設定一次，
`sam2_base.py` 無任何 reset 方法、`reset_state()` 也不碰它們。官方 demo 單序列跑碰不到，
但**一個 predictor 連跑多序列時，前一支的運動狀態會帶進下一支的開頭**，而追蹤是遞迴的。
本 patch 提供 `reset_kalman()`，`track_t1.py` 每序列開始前必呼叫。

## 用法

  python3 patch_sam3_samurai.py --samurai-dir ~/samurai          # 套用
  python3 patch_sam3_samurai.py --samurai-dir ~/samurai --revert # 還原
  python3 patch_sam3_samurai.py --check                          # 只檢查狀態

超參用 SAMURAI 論文/官方 config 預設值，不網格搜索（D033）：
  kf_score_weight=0.15、stable_frames_threshold=15、stable_ious_threshold=0.3
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "# === E18 SAMURAI-on-SAM3 (patched) ==="

# 目標：sam3/model/sam3_tracker_base.py 中挑 best mask 的區塊（與 SAM2 原版逐字相同）
ANCHOR = """        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
"""

REPLACEMENT = '''        {marker}
        # 移植自 yangchris11/samurai sam2_base.py:419-499（三階段狀態機）。
        # samurai_mode 未開啟時走 else 分支＝與原生 SAM3 位元級相同，可做單變因對照。
        if multimask_output and getattr(self, "samurai_mode", False):
            assert B == 1, f"SAMURAI Kalman 僅支援單物件追蹤，實得 B={{B}}"
            batch_inds = torch.arange(B, device=device)

            def _bbox_of(mask_2d):
                nz = torch.argwhere(mask_2d > 0.0)
                if len(nz) == 0:
                    return [0, 0, 0, 0]
                y_min, x_min = nz.min(dim=0).values
                y_max, x_max = nz.max(dim=0).values
                return [x_min.item(), y_min.item(), x_max.item(), y_max.item()]

            if self.kf_mean is None or self.stable_frames == 0:
                # 階段一：Kalman 尚未建立（或剛失穩重置）→ 用原生 IoU 選，並初始化 Kalman
                best_iou_inds = torch.argmax(ious, dim=-1)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                self.kf_mean, self.kf_covariance = self.kf.initiate(
                    self.kf.xyxy_to_xyah(_bbox_of(high_res_masks[0][0]))
                )
                self.stable_frames += 1
            elif self.stable_frames < self.stable_frames_threshold:
                # 階段二：暖機——Kalman 只 predict/update，選 mask 仍純靠 IoU（避免用未收斂的運動模型干預）
                self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
                best_iou_inds = torch.argmax(ious, dim=-1)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                if ious[0][best_iou_inds] > self.stable_ious_threshold:
                    self.kf_mean, self.kf_covariance = self.kf.update(
                        self.kf_mean, self.kf_covariance,
                        self.kf.xyxy_to_xyah(_bbox_of(high_res_masks[0][0])),
                    )
                    self.stable_frames += 1
                else:
                    self.stable_frames = 0
            else:
                # 階段三：運動模型已穩定 → 用 Kalman 預測框與各 candidate 的 IoU 加權重選
                # 這一步就是 identity switch 的攔截點：外觀分數相近時由運動連續性決勝。
                self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
                multibboxes = [
                    _bbox_of(high_res_multimasks[batch_inds, i].unsqueeze(1)[0][0])
                    for i in range(ious.shape[1])
                ]
                kf_ious = torch.tensor(
                    self.kf.compute_iou(self.kf_mean[:4], multibboxes), device=device
                )
                weighted_ious = self.kf_score_weight * kf_ious + (1 - self.kf_score_weight) * ious
                best_iou_inds = torch.argmax(weighted_ious, dim=-1)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                if ious[0][best_iou_inds] < self.stable_ious_threshold:
                    self.stable_frames = 0  # 外觀信心崩了 → 放棄運動模型，下一幀重新初始化
                else:
                    self.kf_mean, self.kf_covariance = self.kf.update(
                        self.kf_mean, self.kf_covariance,
                        self.kf.xyxy_to_xyah(multibboxes[best_iou_inds]),
                    )
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        elif multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
'''.format(marker=MARKER)

HELPER = f'''

{MARKER} helpers
def enable_samurai(tracker, kf_score_weight=0.15, stable_frames_threshold=15,
                   stable_ious_threshold=0.3):
    """在已建好的 SAM3 tracker 上啟用 Kalman 運動先驗。超參為 SAMURAI 官方預設（D033）。"""
    from sam3.model.kalman_filter import KalmanFilter

    tracker.kf = KalmanFilter()
    tracker.samurai_mode = True
    tracker.kf_score_weight = kf_score_weight
    tracker.stable_frames_threshold = stable_frames_threshold
    tracker.stable_ious_threshold = stable_ious_threshold
    reset_kalman(tracker)
    return tracker


def reset_kalman(tracker):
    """每個序列開始前必須呼叫。

    上游 bug：SAMURAI 的 Kalman 狀態掛在 model 物件上、只在 __init__ 設定一次，
    reset_state() 不碰它 → 一個 predictor 連跑多序列時前一支的運動狀態會污染下一支開頭。
    """
    tracker.kf_mean = None
    tracker.kf_covariance = None
    tracker.stable_frames = 0
'''


def find_sam3_tracker_base() -> Path:
    try:
        import sam3  # noqa: F401
    except ImportError:
        sys.exit("找不到 sam3 套件——請在裝好 SAM3 的環境執行")
    p = Path(sam3.__file__).parent / "model" / "sam3_tracker_base.py"
    if not p.exists():
        sys.exit(f"找不到 {p}")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samurai-dir", help="samurai clone 路徑（取其 kalman_filter.py）")
    ap.add_argument("--revert", action="store_true", help="還原為原始檔")
    ap.add_argument("--check", action="store_true", help="只檢查是否已 patch")
    args = ap.parse_args()

    target = find_sam3_tracker_base()
    backup = target.with_suffix(".py.orig")
    src = target.read_text()
    patched = MARKER in src

    if args.check:
        print(f"{target}\n  已 patch: {patched}\n  備份存在: {backup.exists()}")
        kf = target.parent / "kalman_filter.py"
        print(f"  kalman_filter.py: {kf.exists()}")
        return 0

    if args.revert:
        if not backup.exists():
            sys.exit("找不到備份，無法還原")
        shutil.copy(backup, target)
        print(f"✓ 已還原 {target}")
        return 0

    if patched:
        print("✓ 已是 patched 狀態（冪等，不重複套用）")
        return 0

    if not args.samurai_dir:
        sys.exit("需要 --samurai-dir（取 SAMURAI 的 kalman_filter.py）")
    kf_src = Path(args.samurai_dir) / "sam2" / "sam2" / "utils" / "kalman_filter.py"
    if not kf_src.exists():
        sys.exit(f"找不到 {kf_src}")

    if ANCHOR not in src:
        sys.exit("🚨 找不到錨點程式碼——SAM3 版本可能已變動，patch 需重新對位（勿盲目套用）")
    if src.count(ANCHOR) != 1:
        sys.exit(f"🚨 錨點出現 {src.count(ANCHOR)} 次，預期 1 次")

    shutil.copy(target, backup)
    shutil.copy(kf_src, target.parent / "kalman_filter.py")  # 零改動複製，只依賴 numpy+scipy
    target.write_text(src.replace(ANCHOR, REPLACEMENT) + HELPER)

    print(f"✓ 已 patch {target}")
    print(f"  備份 → {backup}")
    print(f"  kalman_filter.py → {target.parent / 'kalman_filter.py'}")
    print("  啟用：from sam3.model.sam3_tracker_base import enable_samurai, reset_kalman")
    return 0


if __name__ == "__main__":
    sys.exit(main())
