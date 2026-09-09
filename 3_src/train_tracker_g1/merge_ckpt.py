#!/usr/bin/env python3
"""把 tracker-only state_dict（train_g1/train_g2 的 ckpt 格式，鍵不含 "backbone."）
合併回完整 sam3.pt 檢查點，輸出可直接餵 `track_t1.py --sam3-ckpt` 的檔案。

sam3.pt 結構依據（本機無 1.6GB gated 權重，結構以載入端程式碼為準，
Lambda 上由 --verify 落地驗證）：`model_builder.py:799-807` ——
`torch.load(weights_only=True)` → 若頂層是 `{"model": {...}}` 則取內層 →
`model.load_state_dict(ckpt, strict=True)`，model =
`Sam3VideoInferenceWithInstanceInteractivity(detector=…, tracker=…)`
⇒ flat dict、鍵帶 `tracker.` / `detector.` 前綴。tracker 建構時
`with_backbone=False`（`build_tracker`，model_builder.py:445-497）
⇒ 完整 ckpt 內**沒有** `tracker.backbone.*` 鍵，與我們過濾後的
tracker-only sd 恰好同構：映射就是加 `tracker.` 前綴，無例外。

輸出紀律（strict=True 陷阱）：輸出 ckpt ＝ 輸入的鍵結構原封不動、只換
tracker.* 的 tensor——**不得**加任何 metadata 鍵（flat 格式下會變成
unexpected key、載入直接炸）。provenance 一律寫 sidecar json。

三道 assert：
  (1) tracker.* 鍵 100% 覆蓋（base 的 tracker 鍵集合 == 前綴後的 sd 鍵集合）；
  (2) 零 unexpected（同一集合等式的另一半）；
  (3) 合併前後非 tracker 鍵位元級相同——由 --verify（預設開）落地：
      重載輸出檔逐鍵 torch.equal 對照（非 tracker 鍵 vs base、tracker 鍵 vs sd）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

TRACKER_PREFIX = "tracker."


def load_full_ckpt(path):
    """回傳 (flat_state_dict, has_model_wrapper)。容錯 model_builder.py:802-803
    的兩種格式：`{"model": {...}}` wrapper 或 flat。"""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        # 「同構」要是被檢查的事實：wrapper 格式下不得有兄弟頂層鍵，
        # 否則輸出（只寫 {"model": merged}）會靜默丟掉它們。
        assert set(ckpt) == {"model"}, f"wrapper 旁有未知頂層鍵：{sorted(set(ckpt) - {'model'})[:5]}"
        return ckpt["model"], True
    return ckpt, False


def load_tracker_sd(path):
    """train_g1/train_g2 的 ckpt 格式：{"tracker_state_dict": sd, ...}。"""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert "tracker_state_dict" in ckpt, f"{path} 非 G1/G2 tracker ckpt 格式"
    sd = ckpt["tracker_state_dict"]
    bad = [k for k in sd if k.startswith("backbone.")]
    assert not bad, f"tracker sd 含 backbone.* 鍵（應已在存檔時過濾）：{bad[:3]}"
    return sd, {k: v for k, v in ckpt.items() if k != "tracker_state_dict"}


def merge_state_dicts(base_flat: dict, tracker_sd: dict) -> dict:
    """核心合併（純 dict 操作，CPU 可測）。回傳新 flat dict。"""
    base_tracker_keys = {k for k in base_flat if k.startswith(TRACKER_PREFIX)}
    mapped = {TRACKER_PREFIX + k: v for k, v in tracker_sd.items()}
    missing = base_tracker_keys - set(mapped)
    unexpected = set(mapped) - base_tracker_keys
    assert not missing, f"tracker.* 覆蓋不全，缺 {len(missing)} 鍵，如 {sorted(missing)[:3]}"
    assert not unexpected, f"unexpected tracker 鍵 {len(unexpected)}，如 {sorted(unexpected)[:3]}"
    merged = dict(base_flat)
    merged.update(mapped)
    return merged


def verify_output(out_path, base_flat: dict, tracker_sd: dict, wrapped: bool) -> None:
    """重載輸出檔、逐鍵 torch.equal——把「位元級相同」變成被檢查的事實。"""
    import torch

    reloaded, rewrapped = load_full_ckpt(out_path)
    assert rewrapped == wrapped, "輸出 wrapper 格式與輸入不同構"
    assert set(reloaded) == set(base_flat), "輸出鍵集合與輸入不同"
    for k, v in reloaded.items():
        expect = (tracker_sd[k[len(TRACKER_PREFIX):]]
                  if k.startswith(TRACKER_PREFIX) else base_flat[k])
        assert torch.equal(v, expect), f"驗證失敗：{k} 與預期不符"


def _sha256(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="完整 sam3.pt")
    ap.add_argument("--tracker-ckpt", required=True, help="tracker_g*_best/final.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-verify", action="store_true",
                    help="跳過重載驗證（僅限測試；Lambda 上必須驗）")
    args = ap.parse_args(argv)

    import torch

    base_flat, wrapped = load_full_ckpt(args.base)
    tracker_sd, meta = load_tracker_sd(args.tracker_ckpt)
    merged = merge_state_dicts(base_flat, tracker_sd)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": merged} if wrapped else merged, out)
    if not args.no_verify:
        verify_output(out, base_flat, tracker_sd, wrapped)

    sidecar = {
        "base": str(Path(args.base).resolve()), "base_sha256": _sha256(args.base),
        "tracker_ckpt": str(Path(args.tracker_ckpt).resolve()),
        "tracker_ckpt_meta": {k: v for k, v in meta.items()
                              if isinstance(v, (str, int, float, bool))},
        "out_sha256": _sha256(out),
        "n_tracker_keys": sum(1 for k in merged if k.startswith(TRACKER_PREFIX)),
        "wrapped": wrapped, "verified": not args.no_verify,
    }
    Path(str(out) + ".provenance.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False))
    print(f"[merge] OK → {out}（tracker 鍵 {sidecar['n_tracker_keys']}，"
          f"verified={sidecar['verified']}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
