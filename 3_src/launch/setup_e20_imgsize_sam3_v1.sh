#!/usr/bin/env bash
# ============================================================================
# setup_e20_imgsize_sam3_v1.sh — E20：SAM3 輸入解析度 ↑（image_size 1008 → 1568）
#
# 假設（v3「改輸入解析度」類，三戰三勝 vs 事後補丁六戰全敗）：
#   SAM3 是 image_size=1008 / backbone_stride=14 → 72×72 patch 網格。
#   原圖約 409×216 非等比 resize 到 1008² → nir-bee2 的 5.5px 目標只佔約 1×1.8 個 patch。
#   E19 的 crop-zoom（面積中位 4.2x ＝ 線性 2.05x）把它推到 2×3.8 patch，換得 LB +0.0175。
#   ⇒ image_size 1008→1568 是同一機制的**全域版**：線性 1.56x，但覆蓋 75/75 支而非 21 支。
#
# 與 E19 的關係＝相乘不是相加：crop 後的小圖再用 1568² 跑，小目標總放大 3.2x 線性。
#
# 產出兩份（一輪跑完，各自對照一個已知 LB 錨點，皆為單變因）：
#   A) e20_full.csv  = 純 image_size↑ 全量 75 支     → 對照 v006/E15  0.67383
#   B) e20_crop.csv  = A + crop-zoom（窗口與 E19 完全相同）→ 對照 v008/E19 0.69128
#   B 與 v008 的唯一差異就是 image_size ⇒ 其 LB 差即為本實驗的效果量。
#
# 性質：非學習性參數、零學習成分、不認序列名 → 不吃 D037 縮水、Ranking B 合法（D033 合規）。
# 全量替換 → 無 D038 不對稱風險（不是部分替換）。
#
# 紀律：D036（單 tar）、D018（用完必 API terminate）、D016（產出漸進回傳）、
#       D014（官方 rclone）、CLAUDE.md（隔離 venv、版本號檔名、pkill 單獨一條 ssh）。
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
PY=~/sam3env/bin/python
OUT=~/e20; mkdir -p "$OUT"
INSTANCE_NAME="${INSTANCE_NAME:-hsot-e20}"
# 候選必須是 backbone_stride=14 的倍數；閘門一會由大到小試，OOM 就退一級
IMG_CANDIDATES="${IMG_CANDIDATES:-1568 1400 1176}"

# --- 保險自毀（開頭就掛：D018，Lambda 關機仍計費）----------------------------
nohup bash -c "
  sleep 14400
  key=\$(cat ~/.lambda_key | tr -d '\n'); [ -z \"\$key\" ] && exit 0
  id=\$(curl -s -u \"\$key:\" https://cloud.lambda.ai/api/v1/instances | python3 -c \"
import json,sys
d=json.load(sys.stdin).get('data',[])
m=[i['id'] for i in d if i.get('name')=='$INSTANCE_NAME']
print(m[0] if m else '')\")
  [ -n \"\$id\" ] && curl -s -u \"\$key:\" -X POST \
    https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' -d \"{\\\"instance_ids\\\":[\\\"\$id\\\"]}\"
" > ~/insurance_destruct.log 2>&1 &
echo "⏰ 4 小時保險自毀已掛（成功結尾會改掛 15 分鐘快速自毀）"

echo "=== [0] rclone ==="
command -v rclone >/dev/null || curl -fsSL https://rclone.org/install.sh | sudo bash
rclone lsf "$GDRIVE/" >/dev/null || { echo "🚨 gDrive 失敗（rclone.conf 沒推上來？）"; exit 1; }

echo "=== [1] 背景拉 test 假色單 tar（D036）==="
(
  set -euo pipefail
  mkdir -p ~/test_fc
  rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" ~/t1test.tar
  tar -xf ~/t1test.tar -C ~/test_fc
  d=$(find ~/test_fc -mindepth 1 -maxdepth 1 -type d | wc -l)
  n=$(find ~/test_fc -name init_rect.txt | wc -l)
  [ "$d" -eq 75 ] && [ "$n" -eq 75 ] || { echo "🚨 序列 $d / init_rect $n（應各 75）"; exit 1; }
  echo "TEST_FC_READY"
) > ~/fc.log 2>&1 &
FC_PID=$!

echo "=== [2] sam3env + SAM3 權重（並行）==="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ~/sam3env ] || uv venv --python 3.12 ~/sam3env
VIRTUAL_ENV=~/sam3env uv pip install -q torch torchvision --torch-backend=auto
mkdir -p ~/ckpt
(
  set -euo pipefail
  [ -f ~/ckpt/sam3.pt ] || curl -fL -o ~/ckpt/sam3.pt "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"
  sz=$(stat -c%s ~/ckpt/sam3.pt); [ "$sz" -gt 3400000000 ] || { echo "🚨 權重 $sz 太小"; exit 1; }
  echo CKPT_READY
) > ~/ckpt.log 2>&1 &
CKPT_PID=$!

# SAM3 環境四坑（08-06 實測；setuptools 必須最後降級——81+ 移除 pkg_resources 而 BPE tokenizer 靠它）
VIRTUAL_ENV=~/sam3env uv pip install -q "git+https://github.com/facebookresearch/sam3.git" \
  "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
  pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless
VIRTUAL_ENV=~/sam3env uv pip install -q "setuptools<81"
$PY -c "import sam3, numpy; assert numpy.__version__.startswith('1.'); print('sam3 OK')" \
  2>&1 | grep -viE 'warning|deprecat|^ +import'

rclone copyto "$GDRIVE/3_src/track_t1.py" ~/track_t1.py
rclone copy "$GDRIVE/3_src/hsot/" ~/hsot/ --include "*.py"
rclone copyto "$GDRIVE/5_outputs/submissions/exp003_samurai_large.csv" ~/e02_test.csv
rclone copyto "$GDRIVE/5_outputs/submissions/sub_v006_e15sam3.csv" ~/e15_test.csv

wait $CKPT_PID; grep -q CKPT_READY ~/ckpt.log || { echo "🚨 權重失敗"; exit 1; }
wait $FC_PID;  grep -q TEST_FC_READY ~/fc.log || { echo "🚨 test 假色失敗"; exit 1; }
echo "✅ 環境 + 權重 + 資料就緒"

# --- [3] 閘門一：image_size 真的生效嗎 + 吃不吃得下 VRAM ---------------------
# 為何需要硬閘門：`predictor.image_size = N` 是設 attribute。若 backbone 的 position
# embedding 不支援插值，有兩種死法——(a) shape mismatch 直接炸（好，看得見）；
# (b) **靜默地沿用 1008 的網格**（壞，白跑一輪還以為測過了）。
# 故本閘門用 A/B 對照：同序列同 box，1008 vs 候選值各跑 3 幀，
# 斷言「影格張量尺寸」與「low-res mask 尺寸」兩者都必須改變。
echo "=== [3] 閘門一：image_size A/B 生效驗證 + OOM 探測 ==="
SEQ_PROBE=$(find ~/test_fc -mindepth 1 -maxdepth 1 -type d -name 'nir-*' -print -quit)
echo "probe 序列：$SEQ_PROBE"
$PY - "$SEQ_PROBE" $IMG_CANDIDATES <<'PROBE' | tee ~/probe_imgsize.log
import sys, json, torch, numpy as np
from pathlib import Path
from PIL import Image
from sam3.model_builder import build_sam3_video_model

seq_dir = Path(sys.argv[1]); candidates = [int(v) for v in sys.argv[2:]]
m = build_sam3_video_model(checkpoint_path="/home/ubuntu/ckpt/sam3.pt", device="cuda")
t = m.tracker; t.backbone = m.detector.backbone
stride = getattr(t, "backbone_stride", 14)
print(f"預設 image_size={t.image_size} | backbone_stride={stride}")

x, y, w, h = [float(v) for v in Path(seq_dir/"init_rect.txt").read_text().split()]
with Image.open(sorted(seq_dir.glob('*.jp*g'))[0]) as im:
    W, H = im.size
box = np.array([[x/W, y/H, (x+w)/W, (y+h)/H]], dtype=np.float32)

def run(size):
    """回傳 (影格張量尺寸, low-res mask 尺寸, 峰值 VRAM GB)；OOM 回 None。"""
    t.image_size = size
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            st = t.init_state(video_path=str(seq_dir), offload_video_to_cpu=True,
                              async_loading_frames=False)
            imgs = st.get("images")
            img_hw = tuple(imgs.shape[-2:]) if hasattr(imgs, "shape") else None
            t.add_new_points_or_box(inference_state=st, frame_idx=0, obj_id=0, box=box)
            low_hw = None
            for i, (fi, oid, low, vid, sc) in enumerate(t.propagate_in_video(
                    st, start_frame_idx=0, max_frame_num_to_track=3, reverse=False,
                    propagate_preflight=True, tqdm_disable=True)):
                low_hw = tuple(low.shape[-2:])
                if i >= 2: break
            t.reset_state(st)
        return img_hw, low_hw, torch.cuda.max_memory_allocated()/1e9
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

base = run(1008)
assert base, "❌ 連預設 1008 都 OOM，機器規格不對"
print(f"[1008 基準] 影格 {base[0]} | low-res mask {base[1]} | 峰值 VRAM {base[2]:.1f} GB")

chosen = None
for size in candidates:
    r = run(size)
    if r is None:
        print(f"[{size}] ❌ OOM，退下一級"); continue
    img_hw, low_hw, vram = r
    print(f"[{size}] 影格 {img_hw} | low-res mask {low_hw} | 峰值 VRAM {vram:.1f} GB")
    if img_hw == base[0]:
        print(f"  🚨 影格尺寸與 1008 相同 → image_size 未生效（靜默失敗），此值作廢"); continue
    if low_hw == base[1]:
        print(f"  🚨 low-res mask 尺寸未變 → backbone 仍用舊網格（靜默失敗），此值作廢"); continue
    if img_hw[-1] != size:
        print(f"  🚨 影格尺寸 {img_hw} 與請求的 {size} 不符，此值作廢"); continue
    chosen = size
    print(f"  ✅ 生效：patch 網格 {base[0][-1]//stride}² → {img_hw[-1]//stride}²，VRAM {vram:.1f} GB")
    break

assert chosen, "❌ 所有候選值皆未生效或 OOM —— SAM3 backbone 不支援可變解析度，E20 路線判死"
Path("/home/ubuntu/img_size.txt").write_text(str(chosen))
print(f"\n✅ 閘門一過：採用 image_size={chosen}")
PROBE

IMG_SIZE=$(cat ~/img_size.txt)
echo "採用 image_size=$IMG_SIZE"

# --- [4] 閘門二：冒煙 3 支（三模態各一）--------------------------------------
echo "=== [4] 閘門二：冒煙 3 支 @ ${IMG_SIZE}² ==="
SMOKE=~/smoke_e20.txt
for mod in vis nir rednir; do
  find ~/test_fc -mindepth 1 -maxdepth 1 -type d -name "${mod}-*" -print -quit \
    | xargs -r basename
done > "$SMOKE"
cat "$SMOKE"
$PY ~/track_t1.py --frames-root ~/test_fc --seq-list "$SMOKE" \
  --out-dir ~/out_smoke --backend sam3 --sam3-ckpt ~/ckpt/sam3.pt --sam3-image-size "$IMG_SIZE"
$PY -c "
import json; d=json.load(open('$HOME/out_smoke/diagnostics.json'))
assert d['_meta']['sam3_image_size']==$IMG_SIZE, '_meta 未記錄 image_size'
bad=[k for k,v in d.items() if k!='_meta' and 'error' in v]
assert not bad, f'❌ 冒煙失敗 {bad}'
fps=[v['fps'] for k,v in d.items() if k!='_meta' and 'fps' in v]
avg=sum(fps)/len(fps)
print(f'✅ 閘門二過：3/3 成功，FPS {fps} → 全 75 支(26,860 幀)預估 {26860/max(avg,0.1)/60:.0f} 分鐘')
"

# --- [5] 全量 75 支 @ IMG_SIZE（產出 A：對照 v006）---------------------------
echo "=== [5] 全量 75 支 @ ${IMG_SIZE}²（斷點續跑）==="
$PY ~/track_t1.py --frames-root ~/test_fc --out-dir "$OUT/full" \
  --backend sam3 --sam3-ckpt ~/ckpt/sam3.pt --sam3-image-size "$IMG_SIZE"
cp "$OUT/full/submission.csv" "$OUT/e20_full.csv"
rclone copy "$OUT" "$GDRIVE/5_outputs/e20_imgsize_20260806" --transfers 8   # D016 漸進回傳

# --- [6] crop prep：窗口與 E19 完全相同（E15 ∪ E02 軌跡，非本輪軌跡）--------
# 刻意沿用舊軌跡算窗：窗口只需涵蓋目標，用已驗證過的那組＝與 v008 少一個變因。
echo "=== [6] crop prep（窗口 = E15 ∪ E02，與 E19 逐位元相同）==="
PYTHONPATH=~ $PY -m hsot.crop_rerun prep --frames-root ~/test_fc \
  --base-csv ~/e15_test.csv --envelope-extra ~/e02_test.csv \
  --out-root ~/crop_fc --meta ~/crop_meta.json | tail -30
NSEL=$($PY -c "import json;print(len(json.load(open('$HOME/crop_meta.json'))))")
[ "$NSEL" -eq 21 ] || echo "⚠️ 選中 $NSEL 支（E19 當時為 21 支）——窗口規則應為確定性，請查明差異"
$PY -c "
import json,os
m=json.load(open(os.path.expanduser('~/crop_meta.json')))
open(os.path.expanduser('~/crop_seqs.txt'),'w').write('\n'.join(sorted(m))+'\n')"

echo "=== [7] 裁切序列 @ ${IMG_SIZE}²（crop 與 image_size 相乘）==="
$PY ~/track_t1.py --frames-root ~/crop_fc --seq-list ~/crop_seqs.txt \
  --out-dir "$OUT/crop" --backend sam3 --sam3-ckpt ~/ckpt/sam3.pt --sam3-image-size "$IMG_SIZE"

echo "=== [8] merge（base = 本輪全量，非 E15）→ 產出 B：對照 v008 ==="
PYTHONPATH=~ $PY -m hsot.crop_rerun merge --base-csv "$OUT/e20_full.csv" \
  --crop-csv "$OUT/crop/submission.csv" --meta ~/crop_meta.json --out "$OUT/e20_crop.csv"

# --- [9] 災難檢查（D040：無 GT，只驗管線完整性 + 與已知錨點的偏離量級）------
$PY - <<'CHK'
import os, pandas as pd, numpy as np
H = os.path.expanduser
ref  = pd.read_csv(H("~/e15_test.csv"))          # v006，已知 LB 0.67383
full = pd.read_csv(H("~/e20/e20_full.csv"))
crop = pd.read_csv(H("~/e20/e20_crop.csv"))
for name, df in (("e20_full", full), ("e20_crop", crop)):
    assert len(df) == len(ref) == 26860, f"{name} 列數 {len(df)}"
    assert (df.ID.values == ref.ID.values).all(), f"{name} ID 順序不符"
    assert df.isna().sum().sum() == 0, f"{name} 有 NaN"
    assert (df.iloc[:, 3] > 0).all() and (df.iloc[:, 4] > 0).all(), f"{name} 有非正的 w/h"
    ch = (df.iloc[:, 1:].values != ref.iloc[:, 1:].values).any(1)
    # 逐序列中心位移中位數：>50px 級的整體偏離＝疑似崩壞，非解析度微調該有的樣子
    d = df.copy()
    d["seq"] = d.ID.str.rsplit("_", n=1).str[0]
    r = ref.copy(); r["seq"] = r.ID.str.rsplit("_", n=1).str[0]
    cd = np.hypot((d.iloc[:, 1] + d.iloc[:, 3] / 2) - (r.iloc[:, 1] + r.iloc[:, 3] / 2),
                  (d.iloc[:, 2] + d.iloc[:, 4] / 2) - (r.iloc[:, 2] + r.iloc[:, 4] / 2))
    per = pd.DataFrame({"seq": d.seq, "cd": cd}).groupby("seq").cd.median().sort_values()
    print(f"[{name}] 變動 {ch.sum()} 幀 ({ch.mean():.1%})｜vs v006 中心位移中位 {np.median(cd):.1f}px")
    print(f"  偏離最大 5 支：{per.tail(5).round(1).to_dict()}")
CHK
rclone copy "$OUT" "$GDRIVE/5_outputs/e20_imgsize_20260806" --transfers 8
echo "E20_DONE image_size=$IMG_SIZE"

# --- 成功結尾：改掛 15 分鐘快速自毀（產出已在 gDrive，機器沒有留存價值）------
nohup bash -c "
  sleep 900
  key=\$(cat ~/.lambda_key | tr -d '\n'); [ -z \"\$key\" ] && exit 0
  id=\$(curl -s -u \"\$key:\" https://cloud.lambda.ai/api/v1/instances | python3 -c \"
import json,sys
d=json.load(sys.stdin).get('data',[])
m=[i['id'] for i in d if i.get('name')=='$INSTANCE_NAME']
print(m[0] if m else '')\")
  [ -n \"\$id\" ] && curl -s -u \"\$key:\" -X POST \
    https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' -d \"{\\\"instance_ids\\\":[\\\"\$id\\\"]}\"
" > ~/fast_destruct.log 2>&1 &
echo "⏰ 15 分鐘後自毀"
