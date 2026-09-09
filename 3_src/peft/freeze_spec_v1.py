#!/usr/bin/env python3
"""freeze_spec_v1 — E-F video PEFT 的凍結規格（機制對齊層，DESIGN.md §2）。

只練「病灶所在的三塊」，image encoder 全凍：
  memory attention   ← E18 實測 identity switch 發生在此層
  memory encoder     ← D050：記憶的時間密度 > 跨度，寫入表徵是同一病灶
  mask decoder       ← 隨新記憶特徵一起適配

刻意**不用 LoRA/adapter**：直接解凍這三塊已在文獻「<5% 參數」量級，且 state_dict 鍵名由構造保證不變
⇒ 免 merge，`track_t1.py:568` 的 build_sam2_video_predictor(cfg, ckpt) 可原樣載入（DESIGN.md §2）。

用法 A（推薦，YAML 只需把 trainer.model._target_ 指向本模組的子類）：
    _target_: freeze_spec_v1.SAM2TrainFrozen
用法 B（任何已建好的模型物件）：
    from freeze_spec_v1 import apply_freeze, report
    apply_freeze(model); report(model)

⚠️ G0 smoke 必須跑 report() 並肉眼確認：可訓練參數落在 5–10% 且**三塊都非零**。
   若某塊為 0 ⇒ 前綴與該版 SAM2 不符，**停下來對前綴**，不要硬train（E18 型「機制沒接上」的預防）。
"""
from __future__ import annotations

# 前綴以 SAM2.1 的 named_parameters() 命名為準；G0 會實測驗證（見檔頭警告）。
# 核心三塊：對不到參數 ⇒ 前綴與此版 SAM2 不符，硬失敗。
CORE_PREFIXES: tuple[str, ...] = (
    "memory_attention.",
    "memory_encoder.",
    "sam_mask_decoder.",
)
# 輔助兩塊：在某些 config 下可能是 nn.Identity（零參數）或不存在 ⇒ 對不到只警告，不硬失敗。
AUX_PREFIXES: tuple[str, ...] = (
    "mask_downsample",          # memory 路徑上的下採樣層
    "obj_ptr_proj",             # object pointer 投影，屬記憶通路
)
TRAINABLE_PREFIXES: tuple[str, ...] = CORE_PREFIXES + AUX_PREFIXES
FROZEN_PREFIXES: tuple[str, ...] = (
    "image_encoder.",
)


def is_trainable(name: str) -> bool:
    if any(name.startswith(p) for p in FROZEN_PREFIXES):
        return False
    return any(name.startswith(p) for p in TRAINABLE_PREFIXES)


def apply_freeze(model) -> dict[str, int]:
    """就地設定 requires_grad，回傳各群參數量。"""
    stats: dict[str, int] = {}
    for name, p in model.named_parameters():
        t = is_trainable(name)
        p.requires_grad_(t)
        key = next((pre for pre in TRAINABLE_PREFIXES if name.startswith(pre)), None) if t else "FROZEN"
        stats[key or "OTHER_FROZEN"] = stats.get(key or "OTHER_FROZEN", 0) + p.numel()
    return stats


def report(model) -> None:
    stats = apply_freeze(model)
    total = sum(stats.values())
    train = sum(v for k, v in stats.items() if k not in ("FROZEN", "OTHER_FROZEN"))
    print(f"[freeze_spec_v1] 總參數 {total/1e6:.1f}M｜可訓練 {train/1e6:.1f}M ({train/total:.1%})")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"    {k:24s} {v/1e6:8.2f}M {'(凍結)' if k.endswith('FROZEN') else ''}")
    aux_missing = [p for p in AUX_PREFIXES if p not in stats]
    if aux_missing:                     # 可能是 nn.Identity／該版不存在，非錯誤
        print(f"[freeze_spec_v1] ⚠️ 輔助前綴未對到參數（可忽略）：{aux_missing}")
    core_missing = [p for p in CORE_PREFIXES if p not in stats]
    if core_missing:
        raise SystemExit(f"❌ 核心前綴一個參數都沒對到，前綴與此版 SAM2 不符：{core_missing}")


try:                                    # 讓 YAML 可以直接 _target_ 到這裡；缺 sam2 時仍可 import 上面的函式
    from training.model.sam2 import SAM2Train  # type: ignore

    class SAM2TrainFrozen(SAM2Train):
        """SAM2Train ＋ 建構後立即套用凍結規格。"""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            report(self)
except Exception:                       # noqa: BLE001 — 本機無 sam2 時靜默略過
    pass
