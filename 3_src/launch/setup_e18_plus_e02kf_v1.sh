#!/usr/bin/env bash
# ============================================================================
# setup_e18_plus_e02kf_v1.sh — 一趟跑兩個實驗：
#   E18  = SAMURAI-on-SAM3（Kalman 運動先驗移植到 SAM3）
#   E02kf= E02 + 每序列重置 Kalman（量測跨序列污染上游 bug 的代價）
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
OUT=~/out_e18
CKPT=~/ckpt
PY=~/sam3env/bin/python
INSTANCE_NAME="${INSTANCE_NAME:-hsot-e18-samurai}"

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
  rclone copy "$GDRIVE/3_src/hsot/" ~/hsot/ --include "*.py"
  rclone copyto "$GDRIVE/3_src/prep/patch_sam3_samurai.py" ~/patch_sam3_samurai.py
  rclone copyto "$GDRIVE/5_outputs/e15_sam3_20260806/sam3/submission.csv" ~/e15_baseline.csv
  echo "DATA_READY"
) > ~/data_pull.log 2>&1 &
DATA_PID=$!

# --- [1.5] t1env（E02 用，背景並行；與 sam3env 完全隔離）--------------------
(
  set -euo pipefail
  [ -d ~/samurai ] || git clone --depth 1 https://github.com/yangchris11/samurai.git ~/samurai
  [ -d ~/t1env ] || python3 -m venv --system-site-packages ~/t1env
  ~/t1env/bin/pip install -q -e ~/samurai/sam2 loguru tqdm pandas
  mkdir -p ~/ckpt
  [ -f ~/ckpt/sam2.1_hiera_large.pt ] || \
    rclone copyto "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" ~/ckpt/sam2.1_hiera_large.pt
  echo T1ENV_READY
) > ~/t1env_setup.log 2>&1 &
T1ENV_PID=$!

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

# SAM3 的套件 metadata 未列全依賴，且 sam3/__init__.py 會連帶 import sam3.train
# （所以 pyproject 的 dev extra 也變成必裝）。以下為 08-06 實測補齊清單，Ranking B 重現必備：
#   einops/psutil/scipy/av      —— 未宣告但 module-level import，缺一即 ModuleNotFoundError
#   pycocotools/numba/rapidjson —— pyproject 的 dev extra，被 sam3.train 鏈式拉入
#   ftfy==6.1.1 / regex         —— 官方宣告（CLIP tokenizer 用）
#   numpy<2                     —— 官方 pyproject 明寫，與 T2 那次同一個 ABI 坑
VIRTUAL_ENV=~/sam3env uv pip install -q "git+https://github.com/facebookresearch/sam3.git" \
  "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
  pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless
# setuptools 必須「最後」降級：sam3 的 build-system 需要 >=61 會把它拉到最新，
# 但 81 起 deprecate、82+ 直接移除 pkg_resources，而 model_builder.py 靠它找 BPE tokenizer。
VIRTUAL_ENV=~/sam3env uv pip install -q "setuptools<81"
$PY -c "
import sam3, numpy, setuptools
assert numpy.__version__.startswith('1.'), f'numpy {numpy.__version__} 必須 <2（SAM3 宣告）'
assert int(setuptools.__version__.split('.')[0]) < 81, f'setuptools {setuptools.__version__} 已移除 pkg_resources'
print(f'sam3 import OK | numpy {numpy.__version__} | setuptools {setuptools.__version__}')
" 2>&1 | grep -viE 'userwarning|deprecat|^ +import'

wait $CKPT_PID || { echo "🚨 權重下載失敗（見 ckpt_pull.log）"; exit 1; }
sz=$(stat -c%s "$CKPT/$CKPT_FILE")
[ "$sz" -gt 3000000000 ] || { echo "🚨 權重 ${sz}B 太小（應 ~3.2GiB）"; exit 1; }
echo "ckpt OK ($((sz/1000000)) MB)"

# --- [3.5/7] E18：clone samurai 取 kalman_filter.py + 套 SAMURAI Kalman patch ---
echo "=== [3.5/7] 套用 SAMURAI Kalman patch ==="
[ -d ~/samurai ] || git clone --depth 1 https://github.com/yangchris11/samurai.git ~/samurai
$PY ~/patch_sam3_samurai.py --samurai-dir ~/samurai
$PY -c "
from sam3.model.sam3_tracker_base import enable_samurai, reset_kalman
print('PATCH_OK')
" 2>&1 | grep -q PATCH_OK || { echo "🚨 patch 未生效，停工"; exit 1; }
echo "✅ patch 生效：enable_samurai / reset_kalman 可 import"

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

# --- [5/7] 閘門二：直打病灶（E15 崩潰的 3 支 + 1 支對照）-------------------
# E15 診斷：這 3 支是 identity switch（CLE 中位 107px vs E02 的 1.6-1.8px）。
# Kalman 若有效，這裡就該看到；沒效就不必燒全量 46 分鐘。
echo "=== [5/7] 閘門二：病灶 3 支 + 對照 1 支 ==="
SMOKE=~/smoke_e18.txt
printf 'vis-droneshow2\nrednir-droneshow2\nrednir-drone2\nvis-backpack4\n' > "$SMOKE"
$PY ~/track_t1.py --frames-root "$DATA" --seq-list "$SMOKE" \
  --gt-csv ~/2026training.csv --out-dir ~/out_smoke \
  --backend sam3 --sam3-version "$SAM3_VERSION" --sam3-ckpt "$CKPT/$CKPT_FILE" --sam3-samurai
PYTHONPATH=~ $PY -m hsot.compare_runs ~/e15_baseline.csv ~/out_smoke/submission.csv \
  ~/2026training.csv --seqs "$SMOKE" --names E15 E18 | tail -25
set +e
PYTHONPATH=~ $PY - <<'GATE'
from hsot.eval import evaluate
import os
H = os.path.expanduser
r = evaluate(H('~/out_smoke/submission.csv'), H('~/2026training.csv'),
             ['vis-droneshow2','rednir-droneshow2','rednir-drone2'])
E15 = {'vis-droneshow2':0.1009,'rednir-droneshow2':0.1349,'rednir-drone2':0.3415}
E02 = {'vis-droneshow2':0.5923,'rednir-droneshow2':0.6053,'rednir-drone2':0.6758}
got = {k: v['auc'] for k, v in r['per_seq'].items()}
print('\n病灶 3 支：')
for k in E15:
    print(f"  {k:22s} E15 {E15[k]:.4f} → E18 {got.get(k,0):.4f}  (E02 {E02[k]:.4f})  Δ{got.get(k,0)-E15[k]:+.4f}")
mean_gain = sum(got.get(k,0)-E15[k] for k in E15)/len(E15)
print(f"  平均改善 {mean_gain:+.4f}")
import sys
if mean_gain > 0.10:
    print("✅ 閘門二過：Kalman 有效攔截 identity switch → 進 E18 全量")
else:
    print(f"❌ 閘門二 FAIL：病灶平均只改善 {mean_gain:+.4f}（需 >0.10）→ 跳過 E18 全量，仍跑 E02kf")
    sys.exit(3)
GATE
GATE_PASS=$?
set -e

# --- [6/7] E18 全量 65 序列（僅在閘門二通過時）------------------------------
if [ "${GATE_PASS:-3}" = "0" ]; then
  echo "=== [6/7] E18 全量 65 序列 ==="
  $PY ~/track_t1.py --frames-root "$DATA" --seq-list ~/val_split_v1.txt \
    --gt-csv ~/2026training.csv --out-dir "$OUT" \
    --backend sam3 --sam3-version "$SAM3_VERSION" --sam3-ckpt "$CKPT/$CKPT_FILE" --sam3-samurai
  PYTHONPATH=~ $PY -m hsot.compare_runs ~/e15_baseline.csv "$OUT/submission.csv" \
    ~/2026training.csv --seqs ~/val_split_v1.txt --names E15 E18 | tail -30
else
  echo "=== [6/7] 跳過 E18 全量（閘門二未過）==="
fi

# --- [6.5/7] E02kf：E02 + 每序列重置 Kalman（獨立實驗，不論 E18 結果都跑）----
# 上游 bug：SAMURAI 的 kf 狀態掛在 model 物件、只在 __init__ 設一次，reset_state() 不碰它
# → 單一 predictor 連跑 65 序列時前一支運動狀態污染下一支開頭。此輪量測其代價。
echo "=== [6.5/7] E02 + 每序列重置 Kalman ==="
wait $T1ENV_PID || { echo "🚨 t1env 建置失敗（見 t1env_setup.log）"; exit 1; }
grep -q T1ENV_READY ~/t1env_setup.log || { echo "🚨 t1env 未就緒"; exit 1; }
~/t1env/bin/python ~/track_t1.py --frames-root "$DATA" --seq-list ~/val_split_v1.txt \
  --gt-csv ~/2026training.csv --out-dir ~/out_e02kf \
  --backend samurai --samurai-dir ~/samurai --ckpt ~/ckpt/sam2.1_hiera_large.pt \
  --samurai-reset-kf
PYTHONPATH=~ $PY -m hsot.compare_runs ~/e15_baseline.csv ~/out_e02kf/submission.csv \
  ~/2026training.csv --seqs ~/val_split_v1.txt --names E15 E02kf | tail -12 || true

# --- [7/7] 回傳 gDrive ------------------------------------------------------
echo "=== [7/7] 回傳 gDrive ==="
[ -d "$OUT" ] && rclone copy "$OUT" "$GDRIVE/5_outputs/e18_samurai_sam3_20260806" --transfers 8
rclone copy ~/out_smoke "$GDRIVE/5_outputs/e18_samurai_sam3_20260806/smoke" --transfers 8
rclone copy ~/out_e02kf "$GDRIVE/5_outputs/e02_kfreset_20260806" --transfers 8
echo "✅ E15 完成並回傳 5_outputs/e18_samurai_sam3_20260806/"
echo "   下一步（本機）：python3.14 -m hsot.eval 對照 E02 基準 pooled 0.68985，看逐序列 delta 分佈（D033）"

# --- 自毀保險（D018：Lambda 關機仍計費，必須 API terminate）-----------------
nohup bash -c '
  sleep 10800
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
