#!/usr/bin/env python3
"""LoRAT zero-shot 探針（09-04）：現代 SOT tracker 裸機打不打得贏 SAM3？

**這不是交付候選。** 它回答一個問題：08-03 列在候選表、因 ViPT 失敗連坐而從未被測的
`LoRAT`（DINOv2 ＋ LoRA，Apache-2.0），在我方假色資料上 zero-shot 的分數是多少。
對照組是同一份 val65 上的既有讀數：E02 SAMURAI 0.68985、E15 SAM3 0.68843、
E46 crop-SAM3 0.70039、交付鏈 0.71096（test75 口徑）。

═══ 為何是獨立 runner，而不是跑官方 run.sh ═══
官方入口 `./run.sh LoRAT <variant>` 會啟動整套 benchmark 框架（Hydra 式 config、
分散式 evaluator、dataset registry、result collector），要求 `consts.yaml` 指到
LaSOT/GOT-10k/TrackingNet 的目錄結構，並沒有「給首幀框追一個資料夾」的介面
（repo issue 有人問「Will any demo be uploaded?」，未實作）。
本檔採與 D090 G0 策略 (b) 相同的手法：**繞過框架、直接呼叫核心計算路徑**，
把 `OneStreamTrackerPipeline.initialize/track` 的邏輯攤平成單序列迴圈。
計算核心（backbone、head、SiamFC 裁切、post-process）全部 import 自上游，一行不改
⇒ 與官方 eval 的數值語意一致，差別只在批次與資料來源。

═══ 逐行對照（改動這支腳本前先讀）═══
上游 `trackit/runners/evaluation/distributed/tracker_evaluator/default/pipelines/one_stream/__init__.py`
* initialize: `template_cache.put(curated_image)`、`cropping_params_provider.initialize(gt_bbox)`
* track: `provider.get(search_size)` → `apply_siamfc_cropping(image, search_size, params, mode, align, z_mean)`
         → `x/255` → normalize → `model({'z','x','z_feat_mask'})` → `post_process`
         → `apply_siamfc_cropping_to_boxes(box, reverse_siamfc_cropping_params(params))`
         → `bbox_clip_to_image_boundary_` → `provider.update(score, box, image_size)`
`trackit/data/methods/siamese_tracker_eval/transform/default.py`
* 模板：`get_siamfc_cropping_params(gt_bbox, template_area_factor, template_size)`
         → `apply_siamfc_cropping(...)` → `/255` → normalize；同時取得 `image_mean`
         （search region 的 padding 用模板的均值填，不是 0——漏了會在目標靠邊界時失真）
`.../pipelines/one_stream/plugins/template_foreground_indicating_mask_generation.py`
* `z_feat_mask`：template_feat 網格上，GT 框覆蓋處為 1、其餘 0（token_type_embed 索引）

═══ 參數來源（config/LoRAT/，勿憑印象改）═══
B-224/config.yaml：template 112²、search 224²、template_feat 8²、search_feat 16²、
                   interpolation bilinear/align_corners False、normalization imagenet
run.yaml：template_area_factor 2.0、search area_factor 4.0、min_object_size 10、
          post_process box_with_score_map、window_penalty 0.45
_mixin/_378.yaml：378 變體改 template 196²、search 378²、feat 14²/27²、**search area_factor 5.0**

用法：
  PYTHONPATH=<LoRAT repo> python run_lorat_val65.py \
      --lorat-root /path/to/LoRAT --variant L-224 --weight /path/to/large.bin \
      --frames-root /path/to/val65_fc --gt-csv 2026training.csv \
      --seqs 1_data/val_split_v1.txt --out sub_lorat_L224_val65.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

# ═══ 變體表（config/LoRAT/*/config.yaml ＋ _mixin/*.yaml 的攤平版）═══
VARIANTS = {
    "B-224": dict(backbone="ViT-B/14", template_size=(112, 112), search_size=(224, 224),
                  template_feat=(8, 8), search_feat=(16, 16), search_area_factor=4.0,
                  weight_name="base.bin"),
    "L-224": dict(backbone="ViT-L/14", template_size=(112, 112), search_size=(224, 224),
                  template_feat=(8, 8), search_feat=(16, 16), search_area_factor=4.0,
                  weight_name="large.bin"),
    "g-224": dict(backbone="ViT-g/14", template_size=(112, 112), search_size=(224, 224),
                  template_feat=(8, 8), search_feat=(16, 16), search_area_factor=4.0,
                  weight_name="giant.bin"),
    "B-378": dict(backbone="ViT-B/14", template_size=(196, 196), search_size=(378, 378),
                  template_feat=(14, 14), search_feat=(27, 27), search_area_factor=5.0,
                  weight_name="base-378.bin"),
    "L-378": dict(backbone="ViT-L/14", template_size=(196, 196), search_size=(378, 378),
                  template_feat=(14, 14), search_feat=(27, 27), search_area_factor=5.0,
                  weight_name="large-378.bin"),
    "g-378": dict(backbone="ViT-g/14", template_size=(196, 196), search_size=(378, 378),
                  template_feat=(14, 14), search_feat=(27, 27), search_area_factor=5.0,
                  weight_name="giant-378.bin"),
}
TEMPLATE_AREA_FACTOR = 2.0      # run.yaml: data.eval.transform.template_area_factor
WINDOW_PENALTY = 0.45           # run.yaml: runner.test.evaluator.pipeline.post_process
MIN_OBJECT_SIZE = 10            # run.yaml: ...search_region_cropping.min_object_size
NORM_STATS = "imagenet"         # B-224/config.yaml: common.normalization
INTERP_MODE = "bilinear"
INTERP_ALIGN_CORNERS = False
LORA = dict(r=64, alpha=64, dropout=0.0, use_rslora=False)


def build_model(variant: str, weight_path: str, device, dtype):
    """照 trackit/models/methods/LoRAT/builder.py 建模，再載入官方權重。

    ⚠️ **官方 .bin 的兩個反直覺事實（09-04 在機上實查，勿憑副檔名推測）**：
    1. **副檔名是 .bin，格式卻是 safetensors**（magic `a0 72 00 …{"blocks`）。
       `torch.load` 會炸在 `UnpicklingError: Unsupported operand 160`。
       上游 `checkpoint/load.py:load_model_weight` 的預設就是 `use_safetensors=True`。
    2. **檔案只含 LoRA 增量，不含 backbone**：large.bin 301 鍵／32.5M 參數
       ＝ blocks.*.{attn,mlp}.*.lora.{A,B} 288 ＋ head.* 12 ＋ token_type_embed 1。
       patch_embed／norm／pos_embed／blocks 的基礎權重**都不在裡面**
       ⇒ **`load_pretrained=True` 是必要的**，DINOv2 主幹要從官方下載當基底。

    `optimize_for_inference=True` ⇒ `attach_lora_state_dict_hooks_`：載入時對每個
    Linear 取 `module.weight.data`（＝剛載好的 DINOv2 權重）當 base，把 lora.A/B
    併進去再寫回 state_dict（`vit_lora_utils.py:linear_hook`）。
    checkpoint 未帶 `lora_alpha` ⇒ hook 的 alpha=None ⇒ `_lora_delta` 取 scaling=1.0，
    與 config 的 alpha/r ＝ 64/64 ＝ 1.0 **數值完全相同**，故此路徑無語意偏差。
    """
    import torch
    from trackit.models import ModelImplementationSuggestions
    from trackit.models.methods.LoRAT.builder import build_LoRAT_model

    v = VARIANTS[variant]
    config = {
        "model": {"type": "dinov2",
                  "backbone": {"type": "DINOv2",
                               "parameters": {"name": v["backbone"], "acc": "default"}},
                  "lora": LORA},
        "common": {"template_size": list(v["template_size"]),
                   "search_region_size": list(v["search_size"]),
                   "template_feat_size": list(v["template_feat"]),
                   "search_region_feat_size": list(v["search_feat"]),
                   "response_map_size": list(v["search_feat"]),
                   "interpolation_mode": INTERP_MODE,
                   "interpolation_align_corners": INTERP_ALIGN_CORNERS,
                   "normalization": NORM_STATS},
    }
    sugg = ModelImplementationSuggestions(device=device, dtype=dtype,
                                          optimize_for_inference=True, load_pretrained=True)
    model = build_LoRAT_model(config, sugg)

    # 事前快照：用來證明 LoRA 真的併進去了（見下方 fail-closed 三）
    before = model.blocks[0].attn.qkv.weight.detach().float().clone()

    from safetensors.torch import load_file
    state = load_file(weight_path)          # ⚠️ 不是 torch.load，見 docstring
    n_ckpt = len(state)
    ckpt_keys = set(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    after = model.blocks[0].attn.qkv.weight.detach().float()
    delta = float((after - before).abs().max())

    missing_set = set(missing)
    report = {"n_ckpt_keys": n_ckpt, "n_missing": len(missing), "n_unexpected": len(unexpected),
              "qkv_merge_delta": delta,
              "missing_sample": sorted(missing)[:8], "unexpected_sample": sorted(unexpected)[:8]}
    print(f"[lorat] ckpt={n_ckpt} missing={len(missing)} unexpected={len(unexpected)} "
          f"qkv_merge_delta={delta:.3e}", flush=True)

    # ═══ fail-closed 三道（D051 的 SAM 3.1 就是「載得進去但記憶層是空的」）═══
    # 一：checkpoint 的每一個鍵都要被消化（lora.* 由 hook pop 掉並換成合併後的 weight）
    assert not unexpected, f"checkpoint 有 {len(unexpected)} 個鍵沒被使用：{sorted(unexpected)[:5]}"
    # 二：checkpoint 應該提供的部分不得落在 missing（backbone 落在 missing 是預期的，
    #     它由 load_pretrained=True 的 DINOv2 提供）
    must_have = {k for k in ("token_type_embed",) if k in ckpt_keys} | \
                {k for k in ckpt_keys if k.startswith("head.")}
    bad = sorted(must_have & missing_set)
    assert not bad, f"checkpoint 提供的鍵竟然 missing：{bad[:5]}"
    # 三：LoRA 必須真的改變了 backbone 權重——否則等於在跑純 DINOv2，分數無意義
    assert delta > 1e-6, f"blocks.0.attn.qkv.weight 未被 LoRA 改動（delta={delta:.3e}）"

    model.eval().to(device=device, dtype=dtype)
    return model, report


class LoRATTracker:
    """單序列 SOT。攤平自 OneStreamTrackerPipeline（batch=1）。box 一律 xyxy。"""

    def __init__(self, model, variant: str, device, dtype):
        import torch
        from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
        from trackit.runners.evaluation.distributed.tracker_evaluator.components.post_process.box_with_score_map \
            import PostProcessing_BoxWithScoreMap

        v = VARIANTS[variant]
        self.torch = torch
        self.model = model
        self.device = device
        self.dtype = dtype
        self.template_size = np.array(v["template_size"])
        self.search_size = np.array(v["search_size"])
        self.template_feat = v["template_feat"]
        self.search_area_factor = v["search_area_factor"]
        self.normalize_ = get_dataset_norm_stats_transform(NORM_STATS, inplace=True)
        self.post = PostProcessing_BoxWithScoreMap(device, tuple(v["search_feat"]),
                                                   tuple(v["search_size"]), WINDOW_PENALTY)
        self.post.start()

    def _to_chw(self, img_hwc_uint8: np.ndarray):
        t = self.torch.from_numpy(img_hwc_uint8).permute(2, 0, 1)
        return t.to(self.device).to(self.torch.float32)

    def init(self, img_hwc_uint8: np.ndarray, box_xyxy: np.ndarray):
        from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
        from trackit.core.utils.siamfc_cropping import (apply_siamfc_cropping,
                                                        get_siamfc_cropping_params)
        from trackit.runners.evaluation.common.siamfc_search_region_cropping_params_provider.simple \
            import SiamFCCroppingParameterSimpleProvider
        from trackit.runners.evaluation.distributed.tracker_evaluator.default.pipelines.utils.bbox_mask_gen \
            import get_foreground_bounding_box

        z_image = self._to_chw(img_hwc_uint8)
        params = get_siamfc_cropping_params(box_xyxy, TEMPLATE_AREA_FACTOR, self.template_size)
        z, z_mean, params_adj = apply_siamfc_cropping(
            z_image, self.template_size, params, INTERP_MODE, INTERP_ALIGN_CORNERS)
        z.div_(255.0)
        self.normalize_(z)
        self.z = z.to(self.dtype).unsqueeze(0)
        self.z_mean = z_mean

        # 模板前景遮罩：template_feat 網格上 GT 框覆蓋處為 1
        fw, fh = self.template_feat
        stride = (self.template_size[0] / fw, self.template_size[1] / fh)
        mask = self.torch.zeros((fh, fw), dtype=self.torch.long)
        fg = get_foreground_bounding_box(box_xyxy, params_adj, stride)
        bbox_clip_to_image_boundary_(fg, np.array([fw, fh]))
        if fg[2] > fg[0] and fg[3] > fg[1]:
            mask[fg[1]:fg[3], fg[0]:fg[2]] = 1
        else:  # 目標小於一個 patch：至少標一格，否則模板無前景 token
            cx = min(max(int((fg[0] + fg[2]) // 2), 0), fw - 1)
            cy = min(max(int((fg[1] + fg[3]) // 2), 0), fh - 1)
            mask[cy, cx] = 1
        self.z_feat_mask = mask.to(self.device).unsqueeze(0)

        self.provider = SiamFCCroppingParameterSimpleProvider(self.search_area_factor, MIN_OBJECT_SIZE)
        self.provider.initialize(box_xyxy)
        self.last_box = box_xyxy.astype(np.float64).copy()

    def track(self, img_hwc_uint8: np.ndarray):
        from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
        from trackit.core.utils.siamfc_cropping import (apply_siamfc_cropping,
                                                        apply_siamfc_cropping_to_boxes,
                                                        reverse_siamfc_cropping_params)

        x_image = self._to_chw(img_hwc_uint8)
        H, W = x_image.shape[-2:]
        image_size = np.array((W, H), dtype=np.int32)
        params = self.provider.get(self.search_size)
        x, _, params_adj = apply_siamfc_cropping(
            x_image, self.search_size, params, INTERP_MODE, INTERP_ALIGN_CORNERS, self.z_mean)
        x = x / 255.0
        self.normalize_(x)
        x = x.to(self.dtype).unsqueeze(0)

        with self.torch.inference_mode():
            out = self.model(self.z, x, self.z_feat_mask)
        processed = self.post(out)
        score = float(processed["confidence"][0].item())
        box = processed["box"][0].to(self.torch.float64).cpu().numpy()

        # 非批次形式（box (4,)、params (2,2)）——與 get_foreground_bounding_box 同一條
        # 已驗證路徑；上游兩種形狀都支援，這裡取分支較少的那個。
        box_full = apply_siamfc_cropping_to_boxes(box, reverse_siamfc_cropping_params(params_adj))
        bbox_clip_to_image_boundary_(box_full, image_size)
        self.provider.update(score, box_full, image_size)
        self.last_box = box_full
        return box_full, score


def load_gt(gt_csv: Path, seqs: set) -> dict:
    """→ {seq: [(ID, x, y, w, h) …]}，序列內依全域幀號排序（同 track_t1 慣例）。"""
    by_seq: dict[str, list] = {}
    with gt_csv.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            ident = r["ID"]
            seq = ident.rsplit("_", 1)[0]
            if seq not in seqs:
                continue
            by_seq.setdefault(seq, []).append(
                (ident, int(ident.rsplit("_", 1)[1]),
                 float(r["x"]), float(r["y"]), float(r["width"]), float(r["height"])))
    for seq in by_seq:
        by_seq[seq].sort(key=lambda t: t[1])
    return by_seq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lorat-root", required=True, help="LoRAT repo 根目錄（會加進 sys.path）")
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--weight", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--seqs", required=True, help="序列清單檔（val_split_v1.txt）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=("float32", "float16", "bfloat16"))
    ap.add_argument("--limit-seqs", type=int, help="只跑前 N 支（冒煙用）")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(args.lorat_root).resolve()))
    import torch
    from PIL import Image

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    seqs = set(Path(args.seqs).read_text().split())
    gt = load_gt(Path(args.gt_csv), seqs)
    assert gt, "GT 內找不到任何指定序列"
    seq_names = sorted(gt)
    if args.limit_seqs:
        seq_names = seq_names[: args.limit_seqs]

    model, load_report = build_model(args.variant, args.weight, device, dtype)
    tracker_factory = lambda: LoRATTracker(model, args.variant, device, dtype)  # noqa: E731

    rows, diag = [], {"variant": args.variant, "weight": args.weight,
                      "dtype": args.dtype, "load_report": load_report, "sequences": {}}
    t_start = time.time()
    total_frames = 0
    for i, seq in enumerate(seq_names, 1):
        seq_dir = Path(args.frames_root) / seq
        frames = sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
        entries = gt[seq]
        assert len(frames) == len(entries), \
            f"{seq}: 影像 {len(frames)} 張 != GT {len(entries)} 列（frames_root 與 GT 版本不符）"

        t0 = time.time()
        tk = tracker_factory()
        x, y, w, h = entries[0][2:6]
        init_xyxy = np.array([x, y, x + w, y + h], dtype=np.float64)
        first_img = np.array(Image.open(frames[0]).convert("RGB"))
        tk.init(first_img, init_xyxy)
        rows.append((entries[0][0], x, y, w, h))          # 首幀照抄 init（OPE 慣例）

        for idx in range(1, len(frames)):
            img = np.array(Image.open(frames[idx]).convert("RGB"))
            box, _ = tk.track(img)
            rows.append((entries[idx][0], float(box[0]), float(box[1]),
                         float(box[2] - box[0]), float(box[3] - box[1])))
        dt = time.time() - t0
        total_frames += len(frames)
        diag["sequences"][seq] = {"n_frames": len(frames), "sec": round(dt, 1),
                                  "fps": round(len(frames) / max(dt, 1e-6), 2)}
        print(f"[lorat] ({i}/{len(seq_names)}) {seq}: {len(frames)} 幀 "
              f"{dt:.1f}s {len(frames)/max(dt,1e-6):.1f} fps", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["ID", "x", "y", "width", "height"])
        wtr.writerows(rows)
    wall = time.time() - t_start
    diag.update({"n_sequences": len(seq_names), "n_frames": total_frames,
                 "wall_sec": round(wall, 1), "fps": round(total_frames / max(wall, 1e-6), 2)})
    Path(str(out) + ".diagnostics.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False))
    print(f"[lorat] 完成：{len(seq_names)} 支／{total_frames} 幀／{wall/60:.1f} 分 → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
