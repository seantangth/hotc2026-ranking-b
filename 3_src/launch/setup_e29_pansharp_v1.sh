#!/usr/bin/env bash
# ============================================================================
# setup_e29_pansharp_v1.sh — E29：crop 窗內 pan-sharpen（顏色 ＋ 真實解析度）
#
# 【假說】D060（E26）判準未達結案，但機制解讀是關鍵：mosaic 腿
#   vis-S_jump2 0.1688 → 0.6747（+0.5059、跟丟率 74.4%→0%）＝ 解析度治好了跟丟；
#   兩支 droneshow −0.4872／−0.6229 ＝ **mosaic 被轉成灰階、顏色丟光**（D058 獨立佐證：
#   跟丟當下假色的 3 通道已足以分辨真目標）。D060 自己的結論是「顏色 vs 解析度的取捨」
#   ——但那是實作造成的，不是物理必然。**pan-sharpening 同時給兩者。**
#
# 【為什麼這條夠格衝 0.71，而不是保險】失分地圖（D057）：跟丟（IoU<0.1）占 12.7% 幀、
#   上限 +0.0932，是唯一大到能吃下 0.69292 → 0.715 所需 +0.022 的桶；而 S_jump2 的 +0.51
#   正是「跟丟被治好」的樣子。
#
# 【四腿，共用 E26 的同一組 crop 窗 ⇒ 唯一變因＝窗內像素合成方式】
#   fc                     現行 v008 行為（錨點，pooled 須重現 E26 的 0.4769）
#   pan_raw                pan-sharpen、不拉對比（隔離「對比拉伸」這個因素）
#   pan_raw_stretch        ＋亮度 p1–p99 拉伸
#   pan_equalized_stretch  ＋sub-position 均衡（抹平 4×4 固定紋樣）＋拉伸   ← 主候選
#
# 【事前判準】寫死在 hsot/e29_verdict.py 檔頭（主判準沿用 E26 ＋ 機制讀數分開記錄）。
#   ⚠️ 特別注意該檔的「機制成立、待調參」條款——droneshow 兩支已在 0.68–0.69，
#   本假說對它們的預測是「不會崩」而非「更好」，別把微跌讀成整條線判死。
#
# 【成本結構的關鍵】prep **已在本機做完並驗證**（08-10 零 GPU：合成 1,514 幀 ×3 變體僅 54 秒、
#   目視與逐幀統計皆已通過）⇒ 本機器只需「拉一個既有 tar → 合成 1 分鐘 → 跑 4 腿推論」。
#   不必回源拉 HSI zip。E26 的純推論是 2 腿 5 分鐘 ⇒ 本次 4 腿約 10–15 分鐘。
#
# 【紀律】D036 單 tar｜D018 用完必 API terminate｜D016 每腿完成即回傳｜bash 檔名帶版本號
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/e29_pansharp_20260811"
PY3=~/sam3env/bin/python
E26=~/e26_frames; E29=~/e29_frames; OUT=~/e29_out
mkdir -p "$OUT"
INSTANCE_NAME="${INSTANCE_NAME:-hsot-e29}"
STAMP=~/e29_timing.txt; : > "$STAMP"
die() { echo "🚨 $*"; exit 1; }
mark() { echo "$1 $(date +%s)" >> "$STAMP"; echo "⏱  $1 @ $(date +%H:%M:%S)"; }

mark E29_START

# --- 保險自毀（D018；⚠️ 延長死線時 cloud-init 的 /root/rearm_selfkill.sh 要一起延）------
nohup bash -c "
  sleep 7200
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
echo "⏰ 2 小時保險自毀已掛（instance name = $INSTANCE_NAME）"

command -v rclone >/dev/null || curl -fsSL https://rclone.org/install.sh | sudo bash
rclone lsf "$GDRIVE/" >/dev/null || die "gDrive 失敗（rclone.conf 沒推上來？）"

# --- [1] 背景：E26 影格單 tar（541MB，兩腿都在裡面）＋ SAM3 權重 -------------
echo "=== [1] 背景拉 e26_frames.tar（D036 單 tar）==="
(
  set -euo pipefail
  mkdir -p "$E26"
  rclone copyto "$GDRIVE/1_data/packed/e26_frames.tar" ~/e26.tar
  tar -xf ~/e26.tar -C "$E26"
  [ -s "$E26/prep_meta.json" ] || { echo "🚨 prep_meta.json 缺失"; exit 1; }
  n=$(find "$E26/fc" -mindepth 1 -maxdepth 1 -type d | wc -l)
  [ "$n" -eq 3 ] || { echo "🚨 fc 腿 $n 支（應 3）"; exit 1; }
  echo "E26_READY"
) > ~/e26.log 2>&1 &
E26_PID=$!

echo "=== [1b] 背景：SAM3 權重（3.45GB）==="
(
  set -euo pipefail
  mkdir -p ~/ckpt
  [ -f ~/ckpt/sam3.pt ] || curl -fL -o ~/ckpt/sam3.pt "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"
  sz=$(stat -c%s ~/ckpt/sam3.pt); [ "$sz" -gt 3400000000 ] || { echo "🚨 權重 $sz 太小"; exit 1; }
  echo CKPT3_READY
) > ~/ckpt3.log 2>&1 &
CKPT3_PID=$!

# --- [2] sam3env（釘 SHA，D059 ⑤；環境四坑見 PROVENANCE，setuptools 必須最後降級）----
echo "=== [2] sam3env ==="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
[ -d ~/sam3env ] || uv venv --python 3.12 ~/sam3env
if [ ! -f ~/sam3env/.deps_done ]; then
  VIRTUAL_ENV=~/sam3env uv pip install -q torch torchvision --torch-backend=auto
  VIRTUAL_ENV=~/sam3env uv pip install -q "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
    "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
    pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless
  VIRTUAL_ENV=~/sam3env uv pip install -q "setuptools<81"
  touch ~/sam3env/.deps_done || die "deps_done 標記失敗（S-05：不可 fail-open）"
fi
$PY3 -c "
import sam3, numpy, pkg_resources
assert numpy.__version__.startswith('1.'), f'numpy {numpy.__version__} 應 <2'
print(f'sam3env: numpy {numpy.__version__} | sam3 OK')
" 2>&1 | grep -viE 'warning|deprecat|^ +import' || die "sam3env 驗證失敗"

echo "=== [2b] 程式碼與 GT ==="
rclone copyto "$GDRIVE/3_src/track_t1.py" ~/track_t1.py
rclone copy "$GDRIVE/3_src/hsot/" ~/hsot/ --include "*.py"
rclone copyto "$GDRIVE/3_src/prep/pansharp_v1.py" ~/pansharp_v1.py
rclone copyto "$GDRIVE/1_data/raw/2026training.csv" ~/2026training.csv
[ -s ~/track_t1.py ] && [ -s ~/pansharp_v1.py ] || die "程式碼缺失"

wait $E26_PID;   grep -q E26_READY ~/e26.log   || die "e26_frames 失敗"
wait $CKPT3_PID; grep -q CKPT3_READY ~/ckpt3.log || die "SAM3 權重失敗"
mark ENV_READY

# --- [3] 合成三個 pan 變體（本機已驗證，約 1 分鐘）---------------------------
echo "=== [3] pan-sharpen 合成 ==="
mark SYNTH_START
$PY3 ~/pansharp_v1.py --e26-root "$E26" --out-root "$E29" \
  --variants raw raw_stretch equalized_stretch --preview-dir "$OUT/preview" \
  || die "pan-sharpen 合成失敗"
cp "$E26/prep_meta.json" "$OUT/prep_meta.json"
mark SYNTH_DONE
rclone copy "$OUT/preview" "$DEST/preview" --transfers 8   # 目視樣張先回傳

# --- [4] 四腿推論（每腿完成即回傳，D016）------------------------------------
run_leg () {                       # $1=腿名  $2=frames-root
  echo "=== 腿 $1 ==="
  mark "LEG_$1_START"
  $PY3 ~/track_t1.py --frames-root "$2" --out-dir "$OUT/out_$1" \
    --backend sam3 --sam3-version sam3 --sam3-ckpt ~/ckpt/sam3.pt \
    || die "腿 $1 追蹤失敗"
  mark "LEG_$1_DONE"
  rclone copy "$OUT" "$DEST" --transfers 8 --exclude "preview/**"
}
run_leg fc                    "$E26/fc"
run_leg pan_raw               "$E29/pan_raw"
run_leg pan_raw_stretch       "$E29/pan_raw_stretch"
run_leg pan_equalized_stretch "$E29/pan_equalized_stretch"

# --- [5] 判決（判準寫死在 e29_verdict.py 檔頭）------------------------------
echo "=== [5] 判決 ==="
PYTHONPATH=~ $PY3 -m hsot.e29_verdict --out-root "$OUT" --gt-csv ~/2026training.csv \
  | tee "$OUT/verdict.txt" || die "判決腳本失敗"

echo "=== [6] 時間帳 ==="
$PY3 - <<'TIME'
import os
H = os.path.expanduser
t = {k: int(v) for k, v in (l.split() for l in open(H("~/e29_timing.txt")).read().split("\n") if l.strip())}
def span(a, b, label):
    if a in t and b in t:
        s = t[b] - t[a]; print(f"  {label:34s} {s//60:3d} 分 {s%60:02d} 秒")
span("E29_START", "ENV_READY", "環境+權重+資料（並行）")
span("SYNTH_START", "SYNTH_DONE", "pan-sharpen 合成（CPU）")
for leg in ("fc", "pan_raw", "pan_raw_stretch", "pan_equalized_stretch"):
    span(f"LEG_{leg}_START", f"LEG_{leg}_DONE", f"腿 {leg}")
span("E29_START", "LEG_pan_equalized_stretch_DONE", "★ 總計")
TIME

rclone copy "$OUT" "$DEST" --transfers 8
rclone copyto "$STAMP" "$DEST/e29_timing.txt"
echo "E29_DONE"
echo "⚠️ 用完務必 API terminate（D018）；2 小時保險自毀仍在"
