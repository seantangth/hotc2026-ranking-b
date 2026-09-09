r"""把三輪（vis / nir / rednir）各自訓好的 prompt 權重合併成單一 ckpt。

**為什麼要合併**：ViPT 一次只能訓一個模態（`vit_ce_prompt_all.py` 的 forward 依
`train_data_type` 三選一，其餘 `raise ValueError()`），但推論端會依序列名自動判模態走對應
分支——所以測試集 75 支（三模態混合）需要一個同時含三組 prompt 的 ckpt。deep_all 架構
本來就同時容納三組，合併只是把各輪訓好的那組搬進同一份。

**命名空間**（實測 ViPT_all.pth.tar 參數名）：
    vis    → 含 "vis"                  110 張量
    rednir → 含 "rednir"               110 張量
    nir    → 含 "nir" 但不含 "rednir"   110 張量
⚠️ 訓練 nir 那輪，凍結的字串比對（`train_data_type in n`）會把 rednir 也解凍（"nir" 是
"rednir" 的子字串），但 forward 只走 nir 分支 → rednir 拿不到梯度、實際不變。本腳本仍
**按命名空間取用**而非「取所有變動的張量」，確保那輪即使有非預期變動也不會污染 rednir。

驗收：合併後逐模態比對「與 base 不同的張量數」，vis/nir/rednir 應各為 110（未跑的模態為 0）。

用法：
    python merge_prompts.py --base ViPT_all.pth.tar \
        --vis run_a/ViPTrack_ep0020.pth.tar \
        --nir run_b/ViPTrack_ep0020.pth.tar \
        --rednir run_c/ViPTrack_ep0020.pth.tar \
        --out ViPT_hsot_merged.pth.tar
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# ViPT 存的 ckpt 內含 pickle 的 `lib.train.admin.settings` 等物件 → 必須讓 `lib` 可 import，
# 否則 torch.load 直接 ModuleNotFoundError。腳本可能不在 repo 內，故手動把 repo 根塞進 path。
# （合併輸出只存純張量 `{"net": ...}`，不帶任何 ViPT 物件——推論/Ranking B 端因此無此依賴。）
sys.path.insert(0, os.environ.get("VIPT_REPO", os.getcwd()))


def modality_of(name: str) -> str | None:
    """參數名 → 所屬模態命名空間（非 prompt 參數回 None）。"""
    if "prompt" not in name:
        return None
    if "rednir" in name:
        return "rednir"
    if "nir" in name:
        return "nir"
    if "vis" in name:
        return "vis"
    return None


def load_net(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["net"] if isinstance(ck, dict) and "net" in ck else ck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="ViPT_all.pth.tar（凍結 backbone 的來源）")
    ap.add_argument("--vis"), ap.add_argument("--nir"), ap.add_argument("--rednir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_ck = torch.load(args.base, map_location="cpu", weights_only=False)
    base_sd = base_ck["net"] if "net" in base_ck else base_ck
    merged = {k: v.clone() for k, v in base_sd.items()}

    for mod in ("vis", "nir", "rednir"):
        p = getattr(args, mod)
        if not p:
            print(f"{mod:7s}: 略過（未提供）")
            continue
        sd = load_net(Path(p))
        if set(sd.keys()) != set(base_sd.keys()):
            raise SystemExit(f"🚨 {mod} 的鍵集合與 base 不符——架構不一致，不可合併")
        names = [k for k in sd if modality_of(k) == mod]
        changed = sum(1 for k in names if not torch.equal(sd[k].float(), base_sd[k].float()))
        for k in names:
            merged[k] = sd[k].clone()
        print(f"{mod:7s}: 搬入 {len(names)} 張量（其中 {changed} 個與 base 不同）← {Path(p).name}")
        if changed == 0:
            print(f"   ⚠️ {mod} 沒有任何張量變動——那輪可能根本沒訓到（檢查 DATATYPE 與 log）")

    # 驗收：逐模態統計最終差異
    print("\n=== 合併結果（與 base 不同的張量數，各模態應為 110 或 0）===")
    for mod in ("vis", "nir", "rednir"):
        n = sum(1 for k in merged if modality_of(k) == mod
                and not torch.equal(merged[k].float(), base_sd[k].float()))
        print(f"  {mod:7s}: {n}")
    others = [k for k in merged if modality_of(k) is None
              and not torch.equal(merged[k].float(), base_sd[k].float())]
    print(f"  非 prompt 參數變動: {len(others)}（應為 0——backbone 必須維持凍結）")
    if others:
        raise SystemExit(f"🚨 backbone 被動到了：{others[:5]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": merged}, out)
    print(f"\n✓ {out}（{len(merged)} 張量）")


if __name__ == "__main__":
    main()
