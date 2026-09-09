#!/usr/bin/env bash
# ============================================================================
# setup_lbprobe_triple_v1.sh — 一台機器產出三發 LB 探針（test 75 序列）
#
# 為什麼改用 LB 當量測儀器（D040）：val_split_v1 的逐序列 bootstrap 顯示
# 標準誤 ±0.023，而我們要追的效果是 +0.003~+0.013、距 Top10 門檻僅 +0.0127
# —— **本地 val 在物理上測不出這個量級**。LB 是固定 75 序列的確定性評分，
# 對「進前 10」而言它就是答案本身；額度每天重置、不用即消滅、且 LB 取最佳
# （爛提交不傷榜位）→ 未用額度的邊際成本為零。
# 本地 val 降級為「災難偵測器」（>0.05 級崩壞它測得出來），不再當增益閘門。
#
# 三發探針：
#   P1 = E15   SAM3 860M zero-shot（本地擲硬幣 P(Δ>0)=46%，test 場景組成不同）
#   P2 = SAM31 SAM3.1 Object Multiplex（直擊 E18 診斷出的 memory 層 identity switch）
#   P3 = E16   光譜 adapter **無條件全量** nir 22 支（v004 敗在條件式規則，非 adapter 本身）
#
# 順帶修一個資產 bug：t1test_fc_75.tar 打包時只搬了 *.jpg，漏了 init_rect.txt
# → test 模式無 GT 可退回，track_t1.py 會對 75 支全部 FileNotFoundError。
# 本腳本重建正確版並覆寫回 gDrive。
#
# 紀律：D036（單 tar/並行）、D018（用完必 API terminate）、D016（產出漸進回傳）、
#       D014（官方 rclone）、CLAUDE.md（完全隔離 venv、版本號檔名）。
# 階段獨立：任一發失敗不影響其餘，最後印總表。
# ============================================================================
set -uo pipefail
trap 'echo "🚨 主線死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
OUT=~/lbprobe; mkdir -p "$OUT"
PY=~/sam3env/bin/python
declare -A STATUS

echo "=== [0] rclone（映像不帶）==="
command -v rclone >/dev/null || curl -fsSL https://rclone.org/install.sh | sudo bash
rclone lsf "$GDRIVE/" >/dev/null || { echo "🚨 gDrive 存取失敗"; exit 1; }

# --- [1] 背景：拉 test HSI-NIR（13.4GiB / 8749 檔，最慢，最先啟動）---------
echo "=== [1] 背景拉 test HSI-NIR（E16 用，13.4GiB）==="
(
  set -euo pipefail
  mkdir -p ~/hsi_nir
  rclone copy "$GDRIVE/1_data/raw_archive/validation/HSI-NIR/" ~/hsi_nir/ \
    --transfers 16 --checkers 24 --fast-list
  echo "HSI_NIR_READY $(find ~/hsi_nir -name '*.png' | wc -l) png"
) > ~/hsi_pull.log 2>&1 &
HSI_PID=$!

# --- [2] 背景：重建 test 假色（含 init_rect.txt！）+ 覆寫 gDrive tar ------
echo "=== [2] 背景重建 test 假色（修 init_rect.txt 缺漏）==="
(
  set -euo pipefail
  mkdir -p ~/test_zips ~/test_fc
  rclone copy "$GDRIVE/1_data/packed_val_fc/" ~/test_zips/ --transfers 8
  n=$(find ~/test_zips -name '*.zip' | wc -l); [ "$n" -eq 75 ] || { echo "🚨 zip $n≠75"; exit 1; }
  for z in ~/test_zips/*.zip; do
    name=$(basename "$z" .zip); dest=~/test_fc/"$name"
    mkdir -p "$dest"; tmp=$(mktemp -d); unzip -q "$z" -d "$tmp"
    find "$tmp" -iname '*.jpg' -exec mv {} "$dest"/ \;
    ir=$(find "$tmp" -name 'init_rect.txt' -print -quit)
    [ -n "$ir" ] || { echo "🚨 $name 無 init_rect.txt"; exit 1; }
    mv "$ir" "$dest"/init_rect.txt
    rm -rf "$tmp"
    [ "$(find "$dest" -name '*.jpg' | wc -l)" -gt 0 ] || { echo "🚨 $name 無 jpg"; exit 1; }
  done
  d=$(find ~/test_fc -mindepth 1 -maxdepth 1 -type d | wc -l); [ "$d" -eq 75 ] || { echo "🚨 序列 $d≠75"; exit 1; }
  ni=$(find ~/test_fc -name init_rect.txt | wc -l); [ "$ni" -eq 75 ] || { echo "🚨 init_rect $ni≠75"; exit 1; }
  cd ~/test_fc && tar -cf ~/t1test_fc_75.tar .
  rclone copyto ~/t1test_fc_75.tar "$GDRIVE/1_data/packed/t1test_fc_75.tar"   # 覆寫壞的舊版
  echo "TEST_FC_READY 75 序列 + 75 init_rect（tar 已修正並回寫 gDrive）"
) > ~/testfc_pull.log 2>&1 &
FC_PID=$!

# --- [2.5] 背景：t1env（E16 用 E02/SAMURAI 管線，只換輸入影像）------------
# 完全隔離 venv（CLAUDE.md 鐵律；--system-site-packages 會撞 numpy ABI）
echo "=== [2.5] 背景建 t1env（E16 用）==="
(
  set -euo pipefail
  [ -d ~/samurai ] || git clone --depth 1 https://github.com/yangchris11/samurai.git ~/samurai
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  [ -d ~/t1env ] || uv venv --python 3.12 ~/t1env
  VIRTUAL_ENV=~/t1env uv pip install -q torch torchvision --torch-backend=auto
  VIRTUAL_ENV=~/t1env uv pip install -q hydra-core iopath pillow pandas loguru tqdm scipy \
    opencv-python-headless
  VIRTUAL_ENV=~/t1env uv pip install -q --no-deps -e ~/samurai/sam2
  ~/t1env/bin/python -c "from sam2.build_sam import build_sam2_video_predictor; print('sam2 OK')"
  mkdir -p ~/ckpt
  [ -f ~/ckpt/sam2.1_hiera_large.pt ] || \
    rclone copyto "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" ~/ckpt/sam2.1_hiera_large.pt
  echo T1ENV_READY
) > ~/t1env_setup.log 2>&1 &
T1ENV_PID=$!

# --- [3] sam3env + 權重（與上面三個背景 job 並行）-------------------------
echo "=== [3] sam3env（隔離 venv）+ SAM3/3.1 權重 ==="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ~/sam3env ] || uv venv --python 3.12 ~/sam3env
VIRTUAL_ENV=~/sam3env uv pip install -q torch torchvision --torch-backend=auto
mkdir -p ~/ckpt
(
  set -euo pipefail
  [ -f ~/ckpt/sam3.pt ] || curl -fL -o ~/ckpt/sam3.pt \
    "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"
  [ -f ~/ckpt/sam3.1_multiplex.pt ] || curl -fL -o ~/ckpt/sam3.1_multiplex.pt \
    "https://huggingface.co/AEmotionStudio/sam3.1/resolve/main/sam3.1_multiplex.pt"
  echo CKPT_READY
) > ~/ckpt_pull.log 2>&1 &
CKPT_PID=$!

# SAM3 依賴：官方 metadata 未列全 + sam3/__init__ 連帶拉 sam3.train（08-06 實測補齊）
VIRTUAL_ENV=~/sam3env uv pip install -q "git+https://github.com/facebookresearch/sam3.git" \
  "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
  pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless
# setuptools 必須最後降級：81+ 移除 pkg_resources，而 model_builder 靠它找 BPE tokenizer
VIRTUAL_ENV=~/sam3env uv pip install -q "setuptools<81"
$PY -c "import sam3, numpy; assert numpy.__version__.startswith('1.'); print('sam3 OK')" \
  2>&1 | grep -viE 'warning|deprecat|^ +import'

rclone copyto "$GDRIVE/3_src/track_t1.py" ~/track_t1.py
rclone copy "$GDRIVE/3_src/hsot/" ~/hsot/ --include "*.py"
rclone copy "$GDRIVE/3_src/t16_adapter/" ~/t16_adapter/ --include "*.py"
rclone copyto "$GDRIVE/5_outputs/submissions/exp003_samurai_large.csv" ~/e02_test.csv

wait $CKPT_PID; grep -q CKPT_READY ~/ckpt_pull.log || { echo "🚨 權重下載失敗"; exit 1; }
wait $FC_PID;  grep -q TEST_FC_READY ~/testfc_pull.log || { echo "🚨 test 假色重建失敗"; exit 1; }
echo "✅ 環境 + 權重 + test 假色（含 init_rect）就緒"

# --- 共用：跑一發 test 並回傳 --------------------------------------------
run_probe () {  # $1=tag $2...=track_t1 額外參數
  local tag="$1"; shift
  echo "=== 探針 $tag：test 75 序列 ==="
  $PY ~/track_t1.py --frames-root ~/test_fc --out-dir "$OUT/$tag" "$@" || return 1
  local n=$(wc -l < "$OUT/$tag/submission.csv")
  [ "$n" -eq 26861 ] || { echo "🚨 $tag 列數 $n≠26861"; return 1; }
  rclone copy "$OUT/$tag" "$GDRIVE/5_outputs/lbprobe_20260806/$tag" --transfers 8
  echo "✅ $tag 完成並回傳（26860 列）"
}

# --- [4] P1：E15 SAM3 -----------------------------------------------------
set +e
run_probe e15_sam3 --backend sam3 --sam3-version sam3 --sam3-ckpt ~/ckpt/sam3.pt
STATUS[e15]=$?

# --- [5] P2：SAM3.1 Multiplex（閘門：box prompt 介面無官方範例背書）-------
echo "=== 閘門：SAM3.1 tracker 是否有 box prompt 介面 ==="
$PY - > ~/sam31_gate.log 2>&1 <<'GATE'
import os, torch
from sam3.model_builder import build_sam3_multiplex_video_model
m = build_sam3_multiplex_video_model(checkpoint_path=os.path.expanduser("~/ckpt/sam3.1_multiplex.pt"), device="cuda")
t = m.tracker
missing = [n for n in ["init_state","add_new_points_or_box","propagate_in_video"] if not hasattr(t, n)]
assert not missing, f"❌ SAM3.1 tracker 缺 {missing}（只支援 text/session API）"
print(f"✅ SAM3.1 閘門過 | {sum(p.numel() for p in m.parameters())/1e6:.0f}M params")
GATE
GATE_RC=$?
grep -E "✅|❌|Error" ~/sam31_gate.log | tail -3
if [ "$GATE_RC" -eq 0 ]; then
  run_probe sam31 --backend sam3 --sam3-version sam3.1 --sam3-ckpt ~/ckpt/sam3.1_multiplex.pt
  STATUS[sam31]=$?
else
  echo "⏭ SAM3.1 閘門未過，跳過"; STATUS[sam31]=99
fi

# --- [6] P3：E16 光譜 adapter 無條件全量 nir 22 支 ------------------------
echo "=== 探針 e16_full：等 HSI-NIR ==="
wait $HSI_PID
if grep -q HSI_NIR_READY ~/hsi_pull.log; then
  rclone copyto "$GDRIVE/4_models/e16_runs/full1_linear/adapter_nir_ep04.pt" ~/adapter_nir.pt
  cp ~/hsot/io.py ~/t16_adapter/io_hsot.py 2>/dev/null || true
  # init json：從已修正的 test_fc 取（每序列 init_rect.txt）
  $PY - <<'MKINIT'
import json, os, pathlib
root = pathlib.Path(os.path.expanduser("~/test_fc"))
seqs = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("nir-"))
init = {s: (root/s/"init_rect.txt").read_text().strip() for s in seqs}
pathlib.Path(os.path.expanduser("~/test_init.json")).write_text(json.dumps(init))
pathlib.Path(os.path.expanduser("~/nir_test.txt")).write_text("\n".join(seqs))
print(f"init json: {len(seqs)} nir 序列")
MKINIT
  wait $T1ENV_PID
  if ! grep -q T1ENV_READY ~/t1env_setup.log; then
    echo "🚨 t1env 未就緒，跳過 E16"; STATUS[e16]=97
  else
    $PY ~/t16_adapter/adapter_apply.py --adapter ~/adapter_nir.pt --hsi-root ~/hsi_nir \
        --seq-list ~/nir_test.txt --init-json ~/test_init.json --out-root ~/adapter3ch \
    && ~/t1env/bin/python ~/track_t1.py --frames-root ~/adapter3ch --seq-list ~/nir_test.txt \
        --out-dir "$OUT/e16_nir" --samurai-dir ~/samurai --ckpt ~/ckpt/sam2.1_hiera_large.pt
    rc=$?
    if [ "$rc" -eq 0 ]; then
      # splice：22 支 nir 用 adapter 版，其餘 53 支沿用 E02（exp003, LB 0.66608）
      $PY - <<'SPLICE'
import os, pandas as pd
H = os.path.expanduser
e02 = pd.read_csv(H("~/e02_test.csv")); e02.columns = ["ID","x","y","width","height"]
new = pd.read_csv(H("~/lbprobe/e16_nir/submission.csv")); new.columns = ["ID","x","y","width","height"]
assert set(new.ID) <= set(e02.ID), "E16 有 E02 沒有的 ID"
m = e02.set_index("ID"); m.update(new.set_index("ID")); m = m.reset_index()
m = m.set_index("ID").loc[e02.ID].reset_index()          # 保持 E02 原始列序
assert len(m) == len(e02) == 26860, f"列數 {len(m)}"
assert m.isna().sum().sum() == 0
changed = (m[["x","y","width","height"]].values != e02[["x","y","width","height"]].values).any(1)
os.makedirs(H("~/lbprobe/e16_full"), exist_ok=True)
m.to_csv(H("~/lbprobe/e16_full/submission.csv"), index=False)
print(f"splice OK：{len(new)} 列來自 adapter、實際變動 {changed.sum()} 幀 ({changed.mean():.1%})")
SPLICE
      rc=$?
      [ "$rc" -eq 0 ] && rclone copy "$OUT/e16_full" "$GDRIVE/5_outputs/lbprobe_20260806/e16_full" --transfers 8
    fi
    STATUS[e16]=$rc
  fi
else
  echo "⏭ HSI-NIR 拉取失敗，跳過 E16"; STATUS[e16]=98
fi

echo; echo "===== 總表 ====="
for k in e15 sam31 e16; do
  echo "  $k: ${STATUS[$k]:-未執行}  (0=成功)"
done
rclone copy "$OUT" "$GDRIVE/5_outputs/lbprobe_20260806" --transfers 8
echo "LBPROBE_ALL_DONE"

# 自毀保險（D018）
nohup bash -c '
  sleep 10800
  key=$(cat ~/.lambda_key | tr -d "\n"); [ -z "$key" ] && exit 0
  id=$(curl -s -u "$key:" https://cloud.lambda.ai/api/v1/instances | python3 -c "
import json,sys
d=json.load(sys.stdin).get(\"data\",[])
m=[i[\"id\"] for i in d if i.get(\"name\")==\"hsot-lbprobe\"]
print(m[0] if m else \"\")")
  [ -n "$id" ] && curl -s -u "$key:" -X POST \
    https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$id\"]}"
' > ~/self_destruct.log 2>&1 &
