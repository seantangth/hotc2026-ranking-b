#!/usr/bin/env python3
"""E16 adapter 部署:訓好的 adapter 權重 → 對序列 cube 生成 3ch jpg(track_t1 可直接吃)。

用法:
  python3 adapter_apply.py --adapter <adapter_xx.pt> --hsi-root ~/hsi_data \
      --seq-list <清單> --init-json ~/e05_init.json --out-root ~/adapter3ch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    import torch
    import torch.nn as nn
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from io_hsot import load_cube, _read_u16, modality_of, MACRO  # noqa

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--hsi-root", required=True)
    ap.add_argument("--seq-list", required=True)
    ap.add_argument("--init-json", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    ck = torch.load(args.adapter, map_location="cpu", weights_only=True)
    mod = ck["modality"]
    arch = ck.get("arch", "linear")
    if arch == "mlp":
        sd = ck["adapter"]
        n_bands = sd["linear.weight"].shape[1]
        hidden = sd["res.0.weight"].shape[0]

        class ResidualAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Conv2d(n_bands, 3, 1, bias=True)
                self.res = nn.Sequential(nn.Conv2d(n_bands, hidden, 1), nn.ReLU(),
                                         nn.Conv2d(hidden, 3, 1, bias=True))

            def forward(self, x):
                return self.linear(x) + self.res(x)

        adapter = ResidualAdapter()
    else:
        n_bands = ck["adapter"]["weight"].shape[1]
        adapter = nn.Conv2d(n_bands, 3, 1, bias=True)
    adapter.load_state_dict(ck["adapter"])
    adapter.eval()

    init = json.loads(Path(args.init_json).read_text())
    seqs = [s.strip() for s in Path(args.seq_list).read_text().split() if s.strip()]
    for seq in seqs:
        assert modality_of(seq) == mod, f"{seq} 模態 ≠ adapter({mod})"
        pngs = sorted((Path(args.hsi_root) / seq).glob("*.png"))
        out = Path(args.out_root) / seq
        out.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            for f in pngs:
                cube = load_cube(_read_u16(f), mod)
                lo = np.percentile(cube, 1, axis=(0, 1), keepdims=True)
                hi = np.percentile(cube, 99, axis=(0, 1), keepdims=True)
                cube = np.clip((cube - lo) / np.maximum(hi - lo, 1e-6), 0, 1).astype(np.float32)
                x = torch.from_numpy(cube).permute(2, 0, 1)[None]
                y = torch.clamp(adapter(x), 0, 1)[0].permute(1, 2, 0).numpy()
                Image.fromarray((y * 255).astype(np.uint8)).save(out / (f.stem + ".jpg"), quality=95)
        (out / "init_rect.txt").write_text(init[seq])
        print(f"{seq}: {len(pngs)} 幀 → {out}")
    print("ADAPTER-APPLY-DONE")


if __name__ == "__main__":
    main()
