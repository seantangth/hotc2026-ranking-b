#!/usr/bin/env bash
# ============================================================================
# setup_e20_imgsize_sam3_v2.sh — E20：SAM3 輸入解析度 ↑（image_size 1008 → 1568）
#
# v1 → v2 修正（v1 的閘門一自爆，但爆點在 probe 自身不在假設）：
#   1. probe 呼叫了 `tracker.reset_state()`——**SAM3 的 Sam3TrackerPredictor 沒有這個方法**
#      （track_t1.py 早有 `hasattr` 保護，v1 的 probe 漏抄）。改為有才呼叫。
#   2. v1 沿用 E19 的 `set -uo pipefail`（無 -e），閘門一死後**一路跑到底還印 E20_DONE
#      並掛上 15 分鐘自毀**——差點連同已就緒的環境與資料一起銷毀。v2 為每個閘門加顯式
#      硬停，且 E20_DONE 只在兩份產出都存在時才印。
#   3. probe 改用 10 幀的臨時序列：v1 對 690 幀序列做了 4 次 init_state，log 被 tqdm 洗版
#      且白等近一分鐘。
#   4. [1]/[2] 加冪等標記，重跑時跳過已完成的下載與解壓。
#
# 假設與產出同 v1：
#   SAM3 image_size=1008 / backbone_stride=14 → 72² patch。原圖約 409×216 時，
#   5.5px 目標只佔約 1 個 patch；E19 crop-zoom（線性 2.05x）換得 LB +0.0175。
#   image_size 1008→1568＝同機制全域版（線性 1.56x，覆蓋 75/75 支）。
#   A) e20_full.csv 純 image_size↑ 全量 → 對照 v006/E15 0.67383
#   B) e20_crop.csv ＝ A ＋ crop-zoom（窗口與 E19 逐位元相同）→ 對照 v008/E19 0.69128
#      B 與 v008 的唯一差異就是 image_size ⇒ 其 LB 差即效果量。
#
# 紀律：D036 / D018 / D016 / D014 / D033 / CLAUDE.md（隔離 venv、版本號檔名）。
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
PY=~/sam3env/bin/python
OUT=~/e20; mkdir -p "$OUT"
INSTANCE_NAME="${INSTANCE_NAME:-hsot-e20}"
# 候選必須「雙重整除」：整除 backbone_stride 14（patch 切得齊）**且**新網格整除
# window 尺寸 24（window partition 不需 padding）。1680→網格 120=24×5；1344→96=24×4。
# 1568 看似合理但網格 112 不是 24 的倍數，已排除。
IMG_CANDIDATES="${IMG_CANDIDATES:-1680 1344}"

die() { echo "🚨 $*"; exit 1; }

echo "=== [0] rclone ==="
command -v rclone >/dev/null || curl -fsSL https://rclone.org/install.sh | sudo bash
rclone lsf "$GDRIVE/" >/dev/null || die "gDrive 失敗（rclone.conf 沒推上來？）"

echo "=== [1] test 假色（冪等）==="
if [ ! -f ~/test_fc/.done ]; then
  mkdir -p ~/test_fc
  rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" ~/t1test.tar
  tar -xf ~/t1test.tar -C ~/test_fc
  touch ~/test_fc/.done
fi
d=$(find ~/test_fc -mindepth 1 -maxdepth 1 -type d | wc -l)
n=$(find ~/test_fc -name init_rect.txt | wc -l)
[ "$d" -eq 75 ] && [ "$n" -eq 75 ] || die "序列 $d / init_rect $n（應各 75）"
echo "資料就緒：75 序列"

echo "=== [2] 環境 + 權重（冪等）==="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ~/sam3env ] || uv venv --python 3.12 ~/sam3env
if [ ! -f ~/sam3env/.deps_done ]; then
  VIRTUAL_ENV=~/sam3env uv pip install -q torch torchvision --torch-backend=auto
  # SAM3 環境四坑（08-06 實測）；setuptools 必須最後降級——81+ 移除 pkg_resources，
  # 而 model_builder 靠它找 BPE tokenizer。
  VIRTUAL_ENV=~/sam3env uv pip install -q "git+https://github.com/facebookresearch/sam3.git" \
    "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
    pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless
  VIRTUAL_ENV=~/sam3env uv pip install -q "setuptools<81"
  touch ~/sam3env/.deps_done || die "deps_done 標記失敗（S-05：不可 fail-open，否則同機 resume 永不自我修復）"
fi
mkdir -p ~/ckpt
[ -f ~/ckpt/sam3.pt ] || curl -fL -o ~/ckpt/sam3.pt "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"
sz=$(stat -c%s ~/ckpt/sam3.pt); [ "$sz" -gt 3400000000 ] || die "權重 $sz 太小"
$PY -c "import sam3, numpy; assert numpy.__version__.startswith('1.'); print('sam3 OK')" \
  2>&1 | grep -viE 'warning|deprecat|^ +import'

rclone copyto "$GDRIVE/3_src/track_t1.py" ~/track_t1.py
rclone copy "$GDRIVE/3_src/hsot/" ~/hsot/ --include "*.py"
rclone copyto "$GDRIVE/5_outputs/submissions/exp003_samurai_large.csv" ~/e02_test.csv
rclone copyto "$GDRIVE/5_outputs/submissions/sub_v006_e15sam3.csv" ~/e15_test.csv
echo "✅ 環境 + 權重 + 資料 + 程式碼就緒"

# --- [3] 閘門一：image_size 真的生效嗎 + 吃不吃得下 VRAM ---------------------
# 兩種死法都要擋：(a) shape mismatch 直接炸（看得見）；
# (b) **靜默沿用 1008 的網格**（看不見，會白跑一輪還以為測過）。
# 故用 1008 vs 候選值的 A/B：斷言「影格張量尺寸」與「low-res mask 尺寸」雙雙改變。
echo "=== [3] 閘門一：image_size A/B 生效驗證 + OOM 探測 ==="
SRC=$(find ~/test_fc -mindepth 1 -maxdepth 1 -type d -name 'nir-*' -print -quit)
rm -rf ~/probe_seq && mkdir -p ~/probe_seq
# awk 取前 N 筆而非 head：CLAUDE.md 鐵律，head 提前關 pipe 會讓上游 SIGPIPE 靜默死亡
find "$SRC" -name '*.jp*g' | sort | awk 'NR<=10' | xargs -I{} cp {} ~/probe_seq/
cp "$SRC/init_rect.txt" ~/probe_seq/
echo "probe 序列：$SRC → 10 幀"

$PY - ~/probe_seq $IMG_CANDIDATES > ~/probe_imgsize.log 2>&1 <<'PROBE'
import sys, torch, numpy as np
sys.path.insert(0, "/home/ubuntu")      # 與正式跑用同一支實作，避免 probe 與推論分歧
from pathlib import Path
from PIL import Image
from sam3.model_builder import build_sam3_video_model
from track_t1 import _resize_sam3_input

seq_dir = Path(sys.argv[1]).expanduser(); candidates = [int(v) for v in sys.argv[2:]]
m = build_sam3_video_model(checkpoint_path="/home/ubuntu/ckpt/sam3.pt", device="cuda")
t = m.tracker; t.backbone = m.detector.backbone
stride = getattr(t, "backbone_stride", 14)
print(f"預設 image_size={t.image_size} | backbone_stride={stride}")

x, y, w, h = [float(v) for v in (seq_dir / "init_rect.txt").read_text().split()]
with Image.open(sorted(seq_dir.glob('*.jp*g'))[0]) as im:
    W, H = im.size
box = np.array([[x/W, y/H, (x+w)/W, (y+h)/H]], dtype=np.float32)

def run(size):
    """回傳 (影格尺寸, low-res mask 尺寸, 峰值 VRAM GB)；OOM 回 None。

    _resize_sam3_input 會一併重建 4 個 global attention 的 RoPE freqs_cis——
    只設 t.image_size 而不重建，forward 會撞 freqs_cis 的 shape assert（v2 實測）。
    """
    _resize_sam3_input(t, size, "cuda")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            st = t.init_state(video_path=str(seq_dir), offload_video_to_cpu=True,
                              async_loading_frames=False)
            imgs = st.get("images") if isinstance(st, dict) else getattr(st, "images", None)
            img_hw = tuple(imgs.shape[-2:]) if hasattr(imgs, "shape") else None
            t.add_new_points_or_box(inference_state=st, frame_idx=0, obj_id=0, box=box)
            low_hw = None
            for i, (fi, oid, low, vid, sc) in enumerate(t.propagate_in_video(
                    st, start_frame_idx=0, max_frame_num_to_track=3, reverse=False,
                    propagate_preflight=True, tqdm_disable=True)):
                low_hw = tuple(low.shape[-2:])
                if i >= 2:
                    break
            # SAM3 的 Sam3TrackerPredictor 沒有 reset_state（SAM2 才有）——有才呼叫
            if hasattr(t, "reset_state"):
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
    # 影格尺寸可能取不到（state 結構未知）；此時只認 low-res mask 這個硬證據
    if img_hw is not None and img_hw[-1] != size:
        print(f"  🚨 影格 {img_hw} 與請求的 {size} 不符 → 未生效，作廢"); continue
    if low_hw == base[1]:
        print(f"  🚨 low-res mask 尺寸未變（仍 {low_hw}）→ backbone 沿用舊網格，作廢"); continue
    chosen = size
    print(f"  ✅ 生效：low-res mask {base[1]} → {low_hw}，VRAM {vram:.1f} GB")
    break

assert chosen, "❌ 所有候選皆未生效或 OOM —— SAM3 backbone 不支援可變解析度，E20 判死"
Path("/home/ubuntu/img_size.txt").write_text(str(chosen))
print(f"\n✅ 閘門一過：採用 image_size={chosen}")
PROBE
grep -vE 'frame loading|it/s\]|UserWarning|FutureWarning|warnings.warn|pkg_resources|^\s*$' ~/probe_imgsize.log | tail -20

IMG_SIZE=$(cat ~/img_size.txt 2>/dev/null || true)
[ -n "$IMG_SIZE" ] || die "閘門一未產出 image_size（見 ~/probe_imgsize.log）——E20 路線判死，停止"
echo "採用 image_size=$IMG_SIZE"

# --- [4] 閘門二：冒煙 3 支（三模態各一）--------------------------------------
echo "=== [4] 閘門二：冒煙 3 支 @ ${IMG_SIZE}² ==="
SMOKE=~/smoke_e20.txt; : > "$SMOKE"
for mod in vis nir rednir; do
  find ~/test_fc -mindepth 1 -maxdepth 1 -type d -name "${mod}-*" -print -quit \
    | xargs -r -n1 basename >> "$SMOKE"
done
cat "$SMOKE"
rm -rf ~/out_smoke
$PY ~/track_t1.py --frames-root ~/test_fc --seq-list "$SMOKE" \
  --out-dir ~/out_smoke --backend sam3 --sam3-ckpt ~/ckpt/sam3.pt --sam3-image-size "$IMG_SIZE" \
  || die "冒煙執行失敗"
$PY -c "
import json; d=json.load(open('$HOME/out_smoke/diagnostics.json'))
assert d['_meta']['sam3_image_size']==$IMG_SIZE, '_meta 未記錄 image_size'
bad=[k for k,v in d.items() if k!='_meta' and 'error' in v]
assert not bad, f'❌ 冒煙失敗 {bad}'
fps=[v['fps'] for k,v in d.items() if k!='_meta' and 'fps' in v]
avg=sum(fps)/len(fps)
print(f'✅ 閘門二過：3/3 成功，FPS {fps} → 全 75 支(26,860 幀)預估 {26860/max(avg,0.1)/60:.0f} 分鐘')
" || die "冒煙驗收失敗"

# --- [5] 全量 75 支（產出 A）------------------------------------------------
echo "=== [5] 全量 75 支 @ ${IMG_SIZE}²（斷點續跑）==="
$PY ~/track_t1.py --frames-root ~/test_fc --out-dir "$OUT/full" \
  --backend sam3 --sam3-ckpt ~/ckpt/sam3.pt --sam3-image-size "$IMG_SIZE" || die "全量失敗"
cp "$OUT/full/submission.csv" "$OUT/e20_full.csv"
rclone copy "$OUT" "$GDRIVE/5_outputs/e20_imgsize_20260806" --transfers 8   # D016 漸進回傳

# --- [6] crop prep：窗口沿用 E15 ∪ E02（與 E19 逐位元相同，少一個變因）------
echo "=== [6] crop prep（窗口 = E15 ∪ E02）==="
if [ ! -f ~/crop_meta.json ]; then
  PYTHONPATH=~ $PY -m hsot.crop_rerun prep --frames-root ~/test_fc \
    --base-csv ~/e15_test.csv --envelope-extra ~/e02_test.csv \
    --out-root ~/crop_fc --meta ~/crop_meta.json | tail -5
fi
NSEL=$($PY -c "import json;print(len(json.load(open('$HOME/crop_meta.json'))))")
[ "$NSEL" -eq 21 ] || echo "⚠️ 選中 $NSEL 支（E19 為 21 支）——規則應確定性，請查明"
$PY -c "
import json,os
m=json.load(open(os.path.expanduser('~/crop_meta.json')))
open(os.path.expanduser('~/crop_seqs.txt'),'w').write('\n'.join(sorted(m))+'\n')"

echo "=== [7] 裁切序列 @ ${IMG_SIZE}²（crop 與 image_size 相乘）==="
$PY ~/track_t1.py --frames-root ~/crop_fc --seq-list ~/crop_seqs.txt \
  --out-dir "$OUT/crop" --backend sam3 --sam3-ckpt ~/ckpt/sam3.pt --sam3-image-size "$IMG_SIZE" \
  || die "裁切序列失敗"

echo "=== [8] merge（base ＝ 本輪全量）→ 產出 B ==="
PYTHONPATH=~ $PY -m hsot.crop_rerun merge --base-csv "$OUT/e20_full.csv" \
  --crop-csv "$OUT/crop/submission.csv" --meta ~/crop_meta.json --out "$OUT/e20_crop.csv" \
  || die "merge 失敗"

# --- [9] 災難檢查（D040：無 GT，只驗管線完整性 + 與已知錨點的偏離量級）------
$PY - <<'CHK' || die "災難檢查失敗"
import os, pandas as pd, numpy as np
H = os.path.expanduser
ref  = pd.read_csv(H("~/e15_test.csv"))          # v006，已知 LB 0.67383
full = pd.read_csv(H("~/e20/e20_full.csv"))
crop = pd.read_csv(H("~/e20/e20_crop.csv"))
for name, df in (("e20_full", full), ("e20_crop", crop)):
    assert len(df) == len(ref) == 26860, f"{name} 列數 {len(df)}"
    assert (df.ID.values == ref.ID.values).all(), f"{name} ID 順序不符"
    assert df.isna().sum().sum() == 0, f"{name} 有 NaN"
    assert (df.iloc[:, 3] > 0).all() and (df.iloc[:, 4] > 0).all(), f"{name} 有非正 w/h"
    ch = (df.iloc[:, 1:].values != ref.iloc[:, 1:].values).any(1)
    d = df.copy(); d["seq"] = d.ID.str.rsplit("_", n=1).str[0]
    r = ref.copy(); r["seq"] = r.ID.str.rsplit("_", n=1).str[0]
    cd = np.hypot((d.iloc[:, 1] + d.iloc[:, 3] / 2) - (r.iloc[:, 1] + r.iloc[:, 3] / 2),
                  (d.iloc[:, 2] + d.iloc[:, 4] / 2) - (r.iloc[:, 2] + r.iloc[:, 4] / 2))
    per = pd.DataFrame({"seq": d.seq, "cd": cd}).groupby("seq").cd.median().sort_values()
    print(f"[{name}] 變動 {ch.sum()} 幀 ({ch.mean():.1%})｜vs v006 中心位移中位 {np.median(cd):.1f}px")
    print(f"  偏離最大 5 支：{per.tail(5).round(1).to_dict()}")
CHK
rclone copy "$OUT" "$GDRIVE/5_outputs/e20_imgsize_20260806" --transfers 8
[ -s "$OUT/e20_full.csv" ] && [ -s "$OUT/e20_crop.csv" ] || die "產出不完整，不宣告成功"
echo "E20_DONE image_size=$IMG_SIZE"
echo "⚠️ 機器保留；確認 gDrive 產出後由本機主動 terminate（4 小時保險自毀仍在）"
