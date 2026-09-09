#!/usr/bin/env bash
# ============================================================================
# setup_e15_sam3_lambda_v1.sh — E15：SAM3 / SAM3.1 zero-shot 抽測（val_split_v1 65 序列）
#
# 為什麼是這條路線：E01→E02 唯一一次大跳（+0.047）就是「換更強的通用 checkpoint」，
# 零學習成分 → 對 D037「分佈外縮水定律」天然免疫、Ranking B 合法。
# 對照組＝E02 基準（val pooled 0.68985 / LB 0.66608），單變因＝只換 tracker 底座
# （mask→box、空 mask 沿用前框、首幀 init 全部共用，已由 test_track_t1_backends.py 證明）。
#
# 紀律：D036（單 tar 不逐檔 + 拉資料∥裝環境並行）、D018（用完必 API terminate）、
#       D016（產出漸進回傳）、D014（官方 rclone）、CLAUDE.md（完全隔離 venv、版本號檔名、
#       set -euo pipefail 下不用 ls|head、pkill 必須單獨一條 ssh）。
#
# 階段閘門：[探測] tracker 有無 box 介面 → [冒煙] 3 支 → [全量] 65 支 → 回傳 → 自毀保險
# 任一閘門失敗即停，機器保留供 debug（失敗不立刻自毀＝B3 教訓：別銷毀已拉好的資料）。
# ============================================================================
set -euo pipefail
trap 'echo "🚨 腳本死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DATA=~/t1_data
OUT=~/out_e15
CKPT=~/ckpt
PY=~/sam3env/bin/python
INSTANCE_NAME="${INSTANCE_NAME:-hsot-e15-sam3}"

# 版本選擇：sam3 = 官方 box-prompt 範例唯一背書（主路徑）
#           sam3.1 = VOS 7 benchmark 改進 6，但 box 介面未經官方範例驗證
SAM3_VERSION="${SAM3_VERSION:-sam3}"
if [ "$SAM3_VERSION" = "sam3.1" ]; then
  CKPT_FILE=sam3.1_multiplex.pt
  CKPT_URL="https://huggingface.co/AEmotionStudio/sam3.1/resolve/main/sam3.1_multiplex.pt"
else
  CKPT_FILE=sam3.pt
  CKPT_URL="https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"
fi
# 官方 gated repo 若已核准，帶 HF_TOKEN 走官方來源（Ranking B license 較乾淨）
if [ -n "${HF_TOKEN:-}" ]; then
  CKPT_URL="https://huggingface.co/facebook/${SAM3_VERSION}/resolve/main/${CKPT_FILE}"
fi
echo "=== E15：$SAM3_VERSION | ckpt=$CKPT_FILE | 官方權重=$([ -n "${HF_TOKEN:-}" ] && echo yes || echo no/鏡像) ==="

mkdir -p "$DATA" "$CKPT" "$OUT"

# --- [0/7] rclone（D014 官方新版；FULL 坑 5：全新 Lambda 映像不帶 rclone）---
if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash
fi
rclone version | head -1
rclone lsf "$GDRIVE/" >/dev/null || { echo "🚨 gDrive 存取失敗（rclone.conf 沒推上來？）"; exit 1; }

# --- [1/7] 拉資料（背景，與環境安裝並行——D036）-----------------------------
echo "=== [1/7] 背景拉 val 假色單 tar（D036：65 序列逐檔=40 分鐘 idle，單 tar<2 分鐘）==="
(
  set -euo pipefail
  if [ ! -f "$DATA/.tar_done" ]; then
    rclone copyto "$GDRIVE/1_data/packed/t1val_fc_65.tar" ~/t1val_fc_65.tar
    tar -xf ~/t1val_fc_65.tar -C "$DATA" --strip-components=0
    touch "$DATA/.tar_done"
  fi
  rclone copyto "$GDRIVE/1_data/val_split_v1.txt" ~/val_split_v1.txt
  rclone copyto "$GDRIVE/1_data/raw/2026training.csv" ~/2026training.csv
  rclone copyto "$GDRIVE/3_src/track_t1.py" ~/track_t1.py
  echo "DATA_READY"
) > ~/data_pull.log 2>&1 &
DATA_PID=$!

# --- [2/7] 隔離 venv（CLAUDE.md 鐵律：不用 --system-site-packages）----------
echo "=== [2/7] SAM3 環境：py3.12+/torch2.7+，完全隔離 venv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"
if [ ! -d ~/sam3env ]; then
  uv venv --python 3.12 ~/sam3env      # 映像自帶可能是 3.10，uv 自動下載 3.12
fi
VIRTUAL_ENV=~/sam3env uv pip install -q torch torchvision --torch-backend=auto
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print('torch', torch.__version__, torch.version.cuda)"

# --- [3/7] SAM3 套件 ∥ 權重下載 ---------------------------------------------
echo "=== [3/7] sam3 套件 + 權重（3.2 GiB）==="
(
  set -euo pipefail
  if [ ! -f "$CKPT/$CKPT_FILE" ]; then
    if [ -n "${HF_TOKEN:-}" ]; then
      curl -fL -H "Authorization: Bearer $HF_TOKEN" -o "$CKPT/$CKPT_FILE" "$CKPT_URL"
    else
      curl -fL -o "$CKPT/$CKPT_FILE" "$CKPT_URL"
    fi
  fi
  echo "CKPT_READY"
) > ~/ckpt_pull.log 2>&1 &
CKPT_PID=$!

VIRTUAL_ENV=~/sam3env uv pip install -q "git+https://github.com/facebookresearch/sam3.git" pandas pillow tqdm
$PY -c "import sam3; print('sam3 import OK')"

wait $CKPT_PID || { echo "🚨 權重下載失敗（見 ckpt_pull.log）"; exit 1; }
sz=$(stat -c%s "$CKPT/$CKPT_FILE")
[ "$sz" -gt 3000000000 ] || { echo "🚨 權重 ${sz}B 太小（應 ~3.2GiB）"; exit 1; }
echo "ckpt OK ($((sz/1000000)) MB)"

# --- [4/7] 閘門一：tracker 是否有 box prompt 介面 ---------------------------
# 官方只在 sam3（非 multiplex）示範 box prompt；multiplex 版未驗證。
# 先探測，省得跑一半才炸（E04 教訓：上機前 de-risk）。
echo "=== [4/7] 閘門一：探測 tracker box prompt 介面 ==="
$PY - "$SAM3_VERSION" "$CKPT/$CKPT_FILE" <<'PROBE'
import sys, torch
ver, ckpt = sys.argv[1], sys.argv[2]
from sam3.model_builder import build_sam3_video_model, build_sam3_multiplex_video_model
builder = build_sam3_multiplex_video_model if ver == "sam3.1" else build_sam3_video_model
m = builder(checkpoint_path=ckpt, device="cuda")
t = m.tracker
t.backbone = m.detector.backbone
need = ["init_state", "add_new_points_or_box", "propagate_in_video"]
missing = [n for n in need if not hasattr(t, n)]
assert not missing, f"❌ {ver} tracker 缺介面 {missing} —— 此版本只支援 text/session API，換 --sam3-version"
n = sum(p.numel() for p in m.parameters())
assert n > 500e6, f"❌ 參數量 {n/1e6:.0f}M 不像 SAM3(~848M)，權重可能載錯/鏡像不對"
print(f"✅ 閘門一過：{ver} | {n/1e6:.0f}M params | box prompt 介面齊備")
PROBE

wait $DATA_PID || { echo "🚨 資料拉取失敗（見 data_pull.log）"; exit 1; }
nseq=$(find "$DATA" -mindepth 1 -maxdepth 1 -type d | wc -l)
[ "$nseq" -ge 65 ] || { echo "🚨 只有 $nseq 序列，應 65"; exit 1; }
echo "資料就緒：$nseq 序列"

# --- [5/7] 閘門二：冒煙 3 支（三模態各一，含最難的 nir）---------------------
echo "=== [5/7] 閘門二：冒煙 3 支 ==="
SMOKE=~/smoke_e15.txt
grep -m1 '^vis-'    ~/val_split_v1.txt  > "$SMOKE"
grep -m1 '^nir-'    ~/val_split_v1.txt >> "$SMOKE"
grep -m1 '^rednir-' ~/val_split_v1.txt >> "$SMOKE"
cat "$SMOKE"
$PY ~/track_t1.py --frames-root "$DATA" --seq-list "$SMOKE" \
  --gt-csv ~/2026training.csv --out-dir ~/out_smoke \
  --backend sam3 --sam3-version "$SAM3_VERSION" --sam3-ckpt "$CKPT/$CKPT_FILE"
$PY -c "
import json; d=json.load(open('$HOME/out_smoke/diagnostics.json'))
bad=[k for k,v in d.items() if k!='_meta' and 'error' in v]
assert not bad, f'❌ 冒煙失敗序列 {bad}'
fps=[v['fps'] for k,v in d.items() if k!='_meta' and 'fps' in v]
print(f'✅ 閘門二過：3/3 成功，FPS {fps} → 全 65 序列（~40k 幀）預估 {40000/max(sum(fps)/len(fps),0.1)/60:.0f} 分鐘')
"

# --- [6/7] 全量 65 序列 ------------------------------------------------------
echo "=== [6/7] 全量 65 序列（斷點續跑：已完成序列自動略過）==="
$PY ~/track_t1.py --frames-root "$DATA" --seq-list ~/val_split_v1.txt \
  --gt-csv ~/2026training.csv --out-dir "$OUT" \
  --backend sam3 --sam3-version "$SAM3_VERSION" --sam3-ckpt "$CKPT/$CKPT_FILE"

# --- [7/7] 回傳 gDrive ------------------------------------------------------
echo "=== [7/7] 回傳 gDrive ==="
rclone copy "$OUT" "$GDRIVE/5_outputs/e15_sam3_20260806/$SAM3_VERSION" --transfers 8
echo "✅ E15 完成並回傳 5_outputs/e15_sam3_20260806/$SAM3_VERSION/"
echo "   下一步（本機）：python3.14 -m hsot.eval 對照 E02 基準 pooled 0.68985，看逐序列 delta 分佈（D033）"

# --- 自毀保險（D018：Lambda 關機仍計費，必須 API terminate）-----------------
nohup bash -c '
  sleep 5400
  key=$(cat ~/.lambda_key | tr -d "\n"); [ -z "$key" ] && exit 0
  id=$(curl -s -u "$key:" https://cloud.lambdalabs.com/api/v1/instances | python3 -c "
import json,sys,os
d=json.load(sys.stdin).get(\"data\",[])
m=[i[\"id\"] for i in d if i.get(\"name\")==os.environ.get(\"INSTANCE_NAME\",\"hsot-e15-sam3\")]
print(m[0] if m else \"\")")
  [ -n "$id" ] && curl -s -u "$key:" -X POST \
    https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
    -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$id\"]}"
' > ~/self_destruct.log 2>&1 &
echo "⏰ 90 分鐘自毀保險已啟動；驗收完請提前手動 terminate 省錢"
