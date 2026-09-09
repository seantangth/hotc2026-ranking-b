#!/usr/bin/env python3
"""E16 可學光譜 adapter 訓練 v1(D032 條件線;「可學 adapter 注入 SAM2」= 文獻空白)。

架構:SpectralAdapter(per-modality conv1x1,N band → 3ch)→ 凍結 SAM2.1-L
      (forward_image → prompt_encoder(GT box)→ mask_decoder)→ low-res mask。
損失:L = dice(pred, teacher) + λ_proj · projection_loss(pred, gt_box)
  - teacher(假色域 GT-box 偽 mask,tightness≥0.8)只當形狀先驗;
  - projection loss(mask 的 x/y 投影 vs box 投影,box-supervised 標準法)提供
    「超越假色」的監督——否則學生天花板=假色(關鍵設計決策)。
初始化:adapter 以「cube→官方假色」最小二乘閉式解起步(保底=假色性能)。
紀律(D033/屍檢教訓):明示初始化語意、低 LR+warmup、逐 epoch 存 ckpt、
  訓練期指標僅當收斂訊號——真驗收必跑 track_t1 完整追蹤(D029)。

用法(DRY,單模態 nir 8 支):
  python3 train_e16_adapter.py --modality nir --hsi-root ~/hsi_data \
      --fc-root ~/t1_data --teacher-dir ~/teacher_masks --seq-list ~/e05_seqs.txt \
      --samurai-dir ~/samurai --ckpt ~/ckpt/sam2.1_hiera_large.pt \
      --epochs 3 --out-dir ~/e16_runs/dry1
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

N_BANDS = {"vis": 16, "nir": 25, "rednir": 15}
SAM_MEAN = [0.485, 0.456, 0.406]
SAM_STD = [0.229, 0.224, 0.225]


def build_dataset_index(args, mod):
    """回傳樣本清單:[(seq, pos, gt_box, teacher_idx)]——teacher npz 内合格幀。"""
    idx = []
    seqs = [s.strip() for s in Path(args.seq_list).read_text().split() if s.strip()]
    for seq in seqs:
        if not seq.startswith(mod + "-"):
            continue
        tz = Path(args.teacher_dir) / f"{seq}.npz"
        if not tz.exists():
            continue
        z = np.load(tz)
        for k, pos in enumerate(z["frames"].tolist()):
            idx.append((seq, int(pos), z["boxes"][k].tolist(), k))
    return idx


class FrameLoader:
    """逐樣本載入 cube(mosaic png 現場 X2Cube)+ teacher mask。"""

    def __init__(self, args, mod):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from io_hsot import load_cube, _read_u16  # 機器上與 io_hsot.py 同目錄
        self._load_cube, self._read_u16 = load_cube, _read_u16
        self.args, self.mod = args, mod
        self._png_cache: dict[str, list] = {}
        self._teacher: dict[str, dict] = {}

    def pngs(self, seq):
        if seq not in self._png_cache:
            self._png_cache[seq] = sorted((Path(self.args.hsi_root) / seq).glob("*.png"))
        return self._png_cache[seq]

    def teacher(self, seq):
        if seq not in self._teacher:
            z = np.load(Path(self.args.teacher_dir) / f"{seq}.npz")
            n, h, w = int(z["n"]), int(z["height"]), int(z["width"])
            masks = np.unpackbits(z["bits"], count=n * h * w).reshape(n, h, w)
            self._teacher[seq] = {"masks": masks}
        return self._teacher[seq]

    def get(self, seq, pos, teacher_idx):
        cube = self._load_cube(self._read_u16(self.pngs(seq)[pos]), self.mod)  # (H,W,B) float
        # per-band percentile normalize(D024)
        lo = np.percentile(cube, 1, axis=(0, 1), keepdims=True)
        hi = np.percentile(cube, 99, axis=(0, 1), keepdims=True)
        cube = np.clip((cube - lo) / np.maximum(hi - lo, 1e-6), 0, 1).astype(np.float32)
        tmask = self.teacher(seq)["masks"][teacher_idx].astype(np.float32)
        return cube, tmask


def fit_falsecolor_init(loader, index, fc_root, n_sample=150):
    """閉式解 A(3×N)+ b:min ||A·cube + b − falsecolor||²——adapter 保底起點。"""
    from PIL import Image
    rng = random.Random(42)
    xs, ys = [], []
    for seq, pos, _, ti in rng.sample(index, min(n_sample, len(index))):
        cube, _ = loader.get(seq, pos, ti)
        jpgs = sorted((Path(fc_root) / seq).glob("*.jpg"))
        fc = np.asarray(Image.open(jpgs[pos]).convert("RGB"), np.float32) / 255.0
        if fc.shape[:2] != cube.shape[:2]:
            continue
        sel = rng.sample(range(cube.shape[0] * cube.shape[1]), 400)
        xs.append(cube.reshape(-1, cube.shape[2])[sel])
        ys.append(fc.reshape(-1, 3)[sel])
    X = np.concatenate(xs)   # (M, N)
    Y = np.concatenate(ys)   # (M, 3)
    X1 = np.concatenate([X, np.ones((len(X), 1), np.float32)], axis=1)
    W, *_ = np.linalg.lstsq(X1, Y, rcond=None)  # (N+1, 3)
    A, b = W[:-1].T, W[-1]   # A: (3, N)
    resid = float(np.abs(X1 @ W - Y).mean())
    print(f"假色擬合初始化:mean|resid|={resid:.4f}(0.05 以下代表假色≈cube 線性投影)")
    return A.astype(np.float32), b.astype(np.float32)


def main() -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=list(N_BANDS))
    ap.add_argument("--hsi-root", required=True)
    ap.add_argument("--fc-root", required=True)
    ap.add_argument("--teacher-dir", required=True)
    ap.add_argument("--seq-list", required=True)
    ap.add_argument("--samurai-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-proj", type=float, default=0.5)
    ap.add_argument("--arch", default="linear", choices=["linear", "mlp"],
                    help="mlp = 線性(假色擬合起點)+ 零初始化殘差 MLP(N→16→3)——非線性容量、嚴格保底")
    ap.add_argument("--batch", type=int, default=3, help="同序列幀 batch(v3:凍結 forward 批次化,3-5x 提速)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    mod = args.modality
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    os.chdir(Path(args.samurai_dir) / "sam2")
    from sam2.build_sam import build_sam2
    model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(Path(args.ckpt).resolve()), device="cuda:0")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    index = build_dataset_index(args, mod)
    print(f"訓練樣本:{len(index)} 幀({mod})")
    loader = FrameLoader(args, mod)
    A0, b0 = fit_falsecolor_init(loader, index, args.fc_root)

    class ResidualAdapter(nn.Module):
        """線性(假色擬合起點)+ 零初始化殘差 MLP:forward = linear(x) + res(x)。"""

        def __init__(self, n_bands, hidden=16):
            super().__init__()
            self.linear = nn.Conv2d(n_bands, 3, 1, bias=True)
            self.res = nn.Sequential(nn.Conv2d(n_bands, hidden, 1), nn.ReLU(),
                                     nn.Conv2d(hidden, 3, 1, bias=True))
            nn.init.zeros_(self.res[-1].weight); nn.init.zeros_(self.res[-1].bias)

        def forward(self, x):
            return self.linear(x) + self.res(x)

    if args.arch == "mlp":
        adapter = ResidualAdapter(N_BANDS[mod]).cuda()
        lin = adapter.linear
    else:
        adapter = nn.Conv2d(N_BANDS[mod], 3, 1, bias=True).cuda()
        lin = adapter
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(A0).view(3, N_BANDS[mod], 1, 1))
        lin.bias.copy_(torch.from_numpy(b0))
    n_train = sum(p.numel() for p in adapter.parameters())
    print(f"初始化語意:假色最小二乘起點+{'零初始化殘差 MLP' if args.arch=='mlp' else '純線性'}(續訓保底);可訓參數 {n_train}(全模型其餘凍結)")

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = 20
    mean = torch.tensor(SAM_MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(SAM_STD, device="cuda").view(1, 3, 1, 1)

    def forward_batch(cube_t, boxes):
        """cube_t: (B,N,H,W) 0..1;boxes: (B,4) xywh 原圖座標 → (B,256,256) logits。"""
        B = cube_t.shape[0]
        x = torch.clamp(adapter(cube_t), 0, 1)
        x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        backbone_out = model.forward_image(x)
        _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]  # 1024 輸入固定尺寸(predictor 同款)
        feats = [f.permute(1, 2, 0).view(B, -1, *fs) for f, fs in
                 zip(vision_feats[::-1], bb_feat_sizes[::-1])][::-1]
        H, W = cube_t.shape[-2:]
        sx, sy = 1024.0 / W, 1024.0 / H
        bx = torch.stack([torch.tensor([b[0] * sx, b[1] * sy, (b[0] + b[2]) * sx, (b[1] + b[3]) * sy],
                                       device="cuda") for b in boxes]).reshape(B, 2, 2)
        labels = torch.tensor([[2, 3]], device="cuda").expand(B, 2)
        sparse, dense = model.sam_prompt_encoder(points=(bx, labels), boxes=None, masks=None)
        low_res, iou_pred, _, _ = model.sam_mask_decoder(
            image_embeddings=feats[-1], image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False, repeat_image=False, high_res_features=feats[:-1])
        return low_res[:, 0]  # (B, 256, 256) logits

    def losses_batch(logits, tmasks, boxes, hw):
        B = logits.shape[0]
        prob = torch.sigmoid(logits)
        t = F.interpolate(tmasks[:, None], size=logits.shape[-2:], mode="bilinear")[:, 0]
        inter = (prob * t).sum(dim=(1, 2))
        dice = (1 - (2 * inter + 1) / (prob.sum(dim=(1, 2)) + t.sum(dim=(1, 2)) + 1)).mean()
        H, W = hw
        gx = torch.zeros(B, logits.shape[-1], device="cuda")
        gy = torch.zeros(B, logits.shape[-2], device="cuda")
        for i, box in enumerate(boxes):
            x1 = int(box[0] / W * logits.shape[-1]); x2 = int(np.ceil((box[0] + box[2]) / W * logits.shape[-1]))
            y1 = int(box[1] / H * logits.shape[-2]); y2 = int(np.ceil((box[1] + box[3]) / H * logits.shape[-2]))
            gx[i, x1:max(x2, x1 + 1)] = 1; gy[i, y1:max(y2, y1 + 1)] = 1
        px = prob.max(dim=-2).values; py = prob.max(dim=-1).values
        proj = F.binary_cross_entropy(px.clamp(1e-5, 1 - 1e-5), gx) + \
               F.binary_cross_entropy(py.clamp(1e-5, 1 - 1e-5), gy)
        return dice, proj

    # v3:按序列分桶(同序列同解析度)→ 序列內切 batch → 桶間打散
    by_seq: dict[str, list] = {}
    for seq, pos, box, ti in index:
        by_seq.setdefault(seq, []).append((pos, box, ti))
    step = 0
    log_f = open(out / "train.log", "w")
    for ep in range(1, args.epochs + 1):
        batches = []
        for seq, items in by_seq.items():
            random.shuffle(items)
            for i in range(0, len(items), args.batch):
                batches.append((seq, items[i:i + args.batch]))
        random.shuffle(batches)
        ep_dice = ep_proj = 0.0
        for bi, (seq, items) in enumerate(batches):
            cubes, tmasks, boxes = [], [], []
            for pos, box, ti in items:
                c, tm = loader.get(seq, pos, ti)
                cubes.append(c); tmasks.append(tm); boxes.append(box)
            cube_t = torch.from_numpy(np.stack(cubes)).permute(0, 3, 1, 2).cuda()
            tmask_t = torch.from_numpy(np.stack(tmasks)).cuda()
            for g in opt.param_groups:
                g["lr"] = args.lr * min(1.0, (step + 1) / warmup)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_batch(cube_t, boxes)
            logits = logits.float()
            dice, proj = losses_batch(logits, tmask_t, boxes, cubes[0].shape[:2])
            loss = dice + args.lambda_proj * proj
            opt.zero_grad(); loss.backward(); opt.step()
            ep_dice += float(dice); ep_proj += float(proj); step += 1
            if step % 10 == 0:
                msg = f"ep{ep} step{step} dice={ep_dice/(bi+1):.4f} proj={ep_proj/(bi+1):.4f}"
                print(msg); log_f.write(msg + "\n"); log_f.flush()
        torch.save({"adapter": adapter.state_dict(), "modality": mod, "epoch": ep,
                    "arch": args.arch, "init": "falsecolor-lstsq", "lr": args.lr},
                   out / f"adapter_{mod}_ep{ep:02d}.pt")
        print(f"=== epoch {ep} 完:mean dice {ep_dice/len(batches):.4f} proj {ep_proj/len(batches):.4f}(ckpt 已存)===")
    log_f.close()
    print("E16-TRAIN-DONE")


if __name__ == "__main__":
    main()
