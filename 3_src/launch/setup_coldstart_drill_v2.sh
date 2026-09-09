#!/usr/bin/env bash
# ============================================================================
# setup_coldstart_drill_v2.sh — 冷啟動演練 #2：**陌生資料泛化演練**（S-16 / D052 桶1 第 f 項）
#
# 【與演練 #1 的根本差異——這才是 §6 第 6 條的原始規格】
#   演練 #1 是**重現測試**：四道閘門全部比對 `~/ref/` 的 Ranking A 參考 CSV、完整性硬編
#   `len(d)==26860`、資料入口硬編 `t1test_fc_75.tar`、序列數 ≠75 即 `exit 1`
#   ⇒ 它證明的是「同一批資料能重跑出同樣結果」，**不是**「換一批資料還能跑」。
#   9/7 拿到的是**全新 75 支無標註影片**，上面五個假設全部不成立。
#
#   ⇒ 本腳本：**零 `~/ref/`、零 Ranking A 參考檔、零硬編序列數／幀數**。
#     資料改用 val 65 支當「假想新資料」——**故意不是 75**，硬編假設會當場現形。
#
# 【最關鍵的一條：不得傳 --gt-csv】（08-10 實測發現）
#   官方 test sample 的 ID ＝ **每序列 1-based 連續**（`nir-bee2_1`…，75 支零例外）；
#   而 2026training.csv 的 ID ＝ **跨序列流水號且各模態範圍重疊**。兩套不同方案。
#   `track_t1.py:70-78` 依「有無 --gt-csv」分流（有 → 全域幀號／無 → 1-based）。
#   ⇒ 演練若傳 --gt-csv，走的是 training 分支，**根本沒測到 9/7 會走的那條路**。
#   本腳本全程不傳，init 框改由封包的 init_rect.txt 提供（read_init 的優先來源）。
#
# 【閘門的性質也變了】新資料上**沒有參考答案**，故閘門不可能是「比對」，只能是：
#   G1  formal validator：對封包 sample 做 exact-set 比對（track_t1.py --sample-csv，不合格 exit 2）
#   G2  自洽：序列數／幀數**由封包推導**與產出一致（不硬編任何數字）
#   G3  crop 選序列規則在陌生資料上跑得出結果（數量 ≥1、且窗來自自產軌跡）
#   G4  事後 AUC（扣留 GT 到此刻才第一次被讀）：pooled > 0.5 ＝ 災難檢查；**數字記錄不設通過門檻**
#       （對照用：val65 上 SAM3 無 crop 的既有 pooled ≈ 0.6932，僅供合理性判讀，非閘門）
#
# 【紀律】D036 單 tar｜D018 用完必 API terminate｜D016 每階段完成即回傳｜bash 檔名帶版本號
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
DEST="$GDRIVE/5_outputs/coldstart_drill2_20260811"
PY1=~/t1env/bin/python      # SAMURAI + SAM2.1-L
PY3=~/sam3env/bin/python    # SAM3 860M
SAMURAI=~/samurai
NEW=~/newdata_fc            # 「假想新資料」的影格根目錄
PKG=~/newdata_pkg           # 「假想新資料」的封包（sample / init_rect / 扣留 GT）
FRESH=~/fresh; OUT=~/drill2
mkdir -p "$FRESH" "$OUT"
INSTANCE_NAME="${INSTANCE_NAME:-hsot-drill2}"
STAMP=~/drill2_timing.txt; : > "$STAMP"
die() { echo "🚨 $*"; exit 1; }
mark() { echo "$1 $(date +%s)" >> "$STAMP"; echo "⏱  $1 @ $(date +%H:%M:%S)"; }

mark DRILL2_START

# --- 保險自毀（D018：Lambda 關機仍持續計費）----------------------------------
# ⚠️ 延長死線時**兩套計時器都要延**：cloud-init 的 /root/rearm_selfkill.sh 與這條是獨立的。
#    盤點用 `pgrep -af "sleep [0-9]+"`；殺舊的用 bracket 寫法且**單獨一條 ssh**。
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
echo "⏰ 4 小時保險自毀已掛（instance name = $INSTANCE_NAME）"

echo "=== [0] rclone ==="
command -v rclone >/dev/null || curl -fsSL https://rclone.org/install.sh | sudo bash
rclone lsf "$GDRIVE/" >/dev/null || die "gDrive 失敗（rclone.conf 沒推上來？）"

# --- [1] 「假想新資料」：單 tar 拉取 + 解開（D036）---------------------------
# ⚠️ 這裡**不驗證序列數等於某個常數**——序列數是待推導量，不是已知量。
echo "=== [1] 背景拉假想新資料（val 65 假色單 tar）==="
(
  set -euo pipefail
  mkdir -p "$NEW"
  rclone copyto "$GDRIVE/1_data/packed/t1val_fc_65.tar" ~/newdata.tar
  tar -xf ~/newdata.tar -C "$NEW"
  d=$(find "$NEW" -mindepth 1 -maxdepth 1 -type d | wc -l)
  [ "$d" -ge 1 ] || { echo "🚨 解開後 0 個序列"; exit 1; }
  echo "NEWDATA_READY seqs=$d"
) > ~/newdata.log 2>&1 &
NEW_PID=$!

# --- [2] 環境 + 權重（能並行的全部並行）--------------------------------------
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/ckpt

echo "=== [2a] 背景：SAM3 權重（3.45GB）==="
(
  set -euo pipefail
  [ -f ~/ckpt/sam3.pt ] || curl -fL -o ~/ckpt/sam3.pt "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"
  sz=$(stat -c%s ~/ckpt/sam3.pt); [ "$sz" -gt 3400000000 ] || { echo "🚨 權重 $sz 太小"; exit 1; }
  echo CKPT3_READY
) > ~/ckpt3.log 2>&1 &
CKPT3_PID=$!

echo "=== [2b] 背景：SAM2.1-L 權重（898MB，gDrive）==="
(
  set -euo pipefail
  [ -f ~/ckpt/sam2.1_hiera_large.pt ] || \
    rclone copy "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" ~/ckpt/
  sz=$(stat -c%s ~/ckpt/sam2.1_hiera_large.pt)
  [ "$sz" -gt 800000000 ] || { echo "🚨 權重 ${sz}B 太小，非 Large"; exit 1; }
  echo CKPT2_READY
) > ~/ckpt2.log 2>&1 &
CKPT2_PID=$!

# t1env 兩坑：①禁用 --system-site-packages（系統 apt pandas 為 numpy 1.21.5 編譯，
# pip 裝 sam2 會把 venv numpy 拉到 2.x → ABI 連環炸）②scipy 是 SAMURAI 的未宣告依賴
echo "=== [2c] t1env（完全隔離 venv）+ SAMURAI fork ==="
[ -d ~/t1env ] || uv venv --python 3.12 ~/t1env
[ -d "$SAMURAI" ] || git clone --depth 1 https://github.com/yangchris11/samurai.git "$SAMURAI"
if [ ! -f ~/t1env/.deps_done ]; then
  VIRTUAL_ENV=~/t1env uv pip install -q torch torchvision --torch-backend=auto
  VIRTUAL_ENV=~/t1env uv pip install -q -e "$SAMURAI/sam2" \
    scipy loguru tqdm pandas pillow opencv-python-headless
  touch ~/t1env/.deps_done || die "deps_done 標記失敗（S-05：不可 fail-open）"
fi
$PY1 -c "
import torch, numpy, pandas, scipy, sam2
assert torch.cuda.is_available(), 'CUDA 不可用'
print(f't1env: torch {torch.__version__} | numpy {numpy.__version__} | sam2 OK')
" || die "t1env 驗證失敗"

echo "=== [2d] sam3env（完全隔離 venv）+ SAM3（釘 SHA，D059 ⑤）==="
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
[ -d ~/sam3env ] || uv venv --python 3.12 ~/sam3env
if [ ! -f ~/sam3env/.deps_done ]; then
  VIRTUAL_ENV=~/sam3env uv pip install -q torch torchvision --torch-backend=auto
  VIRTUAL_ENV=~/sam3env uv pip install -q "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
    "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
    pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless
  VIRTUAL_ENV=~/sam3env uv pip install -q "setuptools<81"   # 81 起移除 pkg_resources
  touch ~/sam3env/.deps_done || die "deps_done 標記失敗（S-05：不可 fail-open）"
fi
$PY3 -c "
import sam3, numpy, pkg_resources
assert numpy.__version__.startswith('1.'), f'numpy {numpy.__version__} 應 <2'
print(f'sam3env: numpy {numpy.__version__} | sam3 OK')
" 2>&1 | grep -viE 'warning|deprecat|^ +import' || die "sam3env 驗證失敗"

echo "=== [2e] 程式碼（**不拉任何 Ranking A 參考 CSV**）==="
rclone copyto "$GDRIVE/3_src/track_t1.py" ~/track_t1.py
rclone copy "$GDRIVE/3_src/hsot/" ~/hsot/ --include "*.py"
rclone copyto "$GDRIVE/3_src/prep/make_pretend_new_data_v1.py" ~/make_pretend_new_data_v1.py
rclone copyto "$GDRIVE/1_data/raw/2026training.csv" ~/2026training.csv
[ -s ~/track_t1.py ] || die "track_t1.py 缺失"
[ -s ~/make_pretend_new_data_v1.py ] || die "造包腳本缺失"

wait $CKPT2_PID; grep -q CKPT2_READY ~/ckpt2.log || die "SAM2.1 權重失敗"
wait $CKPT3_PID; grep -q CKPT3_READY ~/ckpt3.log || die "SAM3 權重失敗"
wait $NEW_PID;   grep -q NEWDATA_READY ~/newdata.log || die "假想新資料失敗"
mark ENV_READY

# --- [2f] 造「假想新資料」封包 ------------------------------------------------
# 2026training.csv 在這裡**只用來取首幀框與扣留評分用 GT**，之後全程不再出現。
echo "=== [2f] 造封包（sample / init_rect / 扣留 GT）==="
$PY1 ~/make_pretend_new_data_v1.py --frames-root "$NEW" \
  --gt-csv ~/2026training.csv --out-dir "$PKG" || die "造包失敗"
mkdir -p ~/scoring && mv "$PKG/_withheld_gt.csv" ~/scoring/withheld_gt.csv
[ -s ~/scoring/withheld_gt.csv ] || die "扣留 GT 缺失"
rm -f ~/2026training.csv        # 生成階段不得有 GT 在手（消除誤用可能）

# 序列數與幀數：**推導出來的**，不是寫死的
N_SEQ=$($PY1 -c "import json;print(json.load(open('$PKG/MANIFEST.json'))['n_seqs'])")
N_FRM=$($PY1 -c "import json;print(json.load(open('$PKG/MANIFEST.json'))['n_frames'])")
echo "📦 假想新資料：$N_SEQ 序列 / $N_FRM 幀（推導值，非硬編）"
[ "$N_SEQ" -ge 1 ] || die "封包 0 序列"
rclone copy "$PKG" "$DEST/pkg" --transfers 8

# 建構閘門：走完 hydra 實例化整條路徑才算環境真的可用
(cd "$SAMURAI/sam2" && $PY1 -c "
from sam2.build_sam import build_sam2_video_predictor
p = build_sam2_video_predictor('configs/samurai/sam2.1_hiera_l.yaml',
                               '$HOME/ckpt/sam2.1_hiera_large.pt', device='cuda:0')
n = sum(x.numel() for x in p.parameters())
assert n > 150e6, f'參數量 {n/1e6:.0f}M —— 不是 Large'
print(f'✅ SAM2.1 predictor 建構成功 | {n/1e6:.0f}M params')
") || die "SAM2.1 predictor 建構失敗"

# ============================================================================
# 生成階段：三段全部自產，全程 **不傳 --gt-csv**（走 9/7 會走的 1-based 分支）
# ============================================================================

echo "=== [3] 層0：SAMURAI + SAM2.1-L 全 $N_SEQ 支（自產窗源 A）==="
mark L0_START
$PY1 ~/track_t1.py --frames-root "$NEW" --out-dir "$FRESH/e02" \
  --backend samurai --samurai-dir "$SAMURAI" --ckpt ~/ckpt/sam2.1_hiera_large.pt \
  --sample-csv "$PKG/sample_submission.csv" \
  || die "層0 失敗（若 exit 2 ＝ formal validator 判定與 sample 不符，正是要抓的泛化 bug）"
cp "$FRESH/e02/submission.csv" "$FRESH/e02_fresh.csv"
mark L0_DONE
rclone copy "$FRESH" "$DEST/fresh" --transfers 8      # D016：每階段完成即回傳

echo "=== [4] 層1：SAM3 860M 全 $N_SEQ 支（自產窗源 B + merge base）==="
mark L1_START
$PY3 ~/track_t1.py --frames-root "$NEW" --out-dir "$FRESH/e15" \
  --backend sam3 --sam3-version sam3 --sam3-ckpt ~/ckpt/sam3.pt \
  --sample-csv "$PKG/sample_submission.csv" \
  || die "層1 失敗（exit 2 ＝ formal validator 不符）"
cp "$FRESH/e15/submission.csv" "$FRESH/e15_fresh.csv"
mark L1_DONE
rclone copy "$FRESH" "$DEST/fresh" --transfers 8

echo "=== [5] 層2a：crop prep（窗＝自產兩軌跡聯集；門檻 0.55 ＝ D061 定案）==="
mark CROP_PREP_START
PYTHONPATH=~ $PY3 -m hsot.crop_rerun prep --frames-root "$NEW" \
  --base-csv "$FRESH/e15_fresh.csv" --envelope-extra "$FRESH/e02_fresh.csv" \
  --area-frac-max 0.55 \
  --out-root ~/crop_fc --meta ~/crop_meta.json | tail -30 || die "crop prep 失敗"
NSEL=$($PY3 -c "import json;print(len(json.load(open('$HOME/crop_meta.json'))))")
echo "選中 $NSEL 支（陌生資料上由規則產生，無參考集合可比對）"
[ "$NSEL" -ge 1 ] || die "選中 0 支——選序列規則在陌生資料上失效"
$PY3 -c "
import json,os
m=json.load(open(os.path.expanduser('~/crop_meta.json')))
open(os.path.expanduser('~/crop_seqs.txt'),'w').write('\n'.join(sorted(m))+'\n')"
mark CROP_PREP_DONE

echo "=== [6] 層2b：SAM3 追蹤裁切序列 + merge ==="
mark L2_START
# ⚠️ 這一段跑的是**子集**，故**不可**傳 --sample-csv（exact-set 會必然失敗）
$PY3 ~/track_t1.py --frames-root ~/crop_fc --seq-list ~/crop_seqs.txt \
  --out-dir "$FRESH/crop" --backend sam3 --sam3-version sam3 --sam3-ckpt ~/ckpt/sam3.pt \
  || die "層2 裁切追蹤失敗"
PYTHONPATH=~ $PY3 -m hsot.crop_rerun merge --base-csv "$FRESH/e15_fresh.csv" \
  --crop-csv "$FRESH/crop/submission.csv" --meta ~/crop_meta.json \
  --out "$OUT/final_drill2.csv" || die "merge 失敗"
mark L2_DONE
rclone copy "$OUT" "$DEST" --transfers 8

# ============================================================================
# 驗收：四道閘門（扣留 GT 到 G4 才第一次被讀）
# ============================================================================
echo "=== [7] 驗收 ==="
$PY3 - "$PKG" "$N_SEQ" "$N_FRM" "$NSEL" <<'CHK' || die "驗收失敗"
import json, os, sys
import pandas as pd
H = os.path.expanduser
pkg, n_seq, n_frm, nsel = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
fails = []
print("=" * 74)

final = pd.read_csv(H("~/drill2/final_drill2.csv"))
smp   = pd.read_csv(os.path.join(pkg, "sample_submission.csv"))

# G1 formal validator：對封包 sample 做 exact-set 比對
want, got = set(smp.ID), set(final.ID)
miss, extra = want - got, got - want
print(f"G1  exact-set vs 封包 sample：缺 {len(miss)}｜多 {len(extra)}｜重複 "
      f"{int(final.ID.duplicated().sum())}")
if miss or extra or final.ID.duplicated().any():
    fails.append(f"G1 exact-set 不符（缺{len(miss)}/多{len(extra)}）")
    print(f"    例：缺 {sorted(miss)[:3]}｜多 {sorted(extra)[:3]}")

# G2 自洽：序列數／幀數由封包推導，與產出一致（不硬編任何數字）
seqs = final.ID.str.rsplit("_", n=1).str[0]
print(f"G2  自洽：產出 {seqs.nunique()} 序列 / {len(final)} 幀 vs 封包 {n_seq} / {n_frm}")
if seqs.nunique() != n_seq or len(final) != n_frm:
    fails.append(f"G2 不自洽（{seqs.nunique()}/{len(final)} vs {n_seq}/{n_frm}）")
if final.isna().sum().sum(): fails.append("G2 有 NaN")
if not ((final.width > 0).all() and (final.height > 0).all()): fails.append("G2 有非正 w/h")

# G3 選序列規則在陌生資料上可運作（無參考集合，只驗規則跑得出結果）
print(f"G3  crop 選序列：{nsel} 支（陌生資料上由規則產生）")
if nsel < 1: fails.append("G3 選中 0 支")

# G4 事後 AUC —— 扣留 GT 到這一刻才第一次被讀
sys.path.insert(0, H("~"))
from hsot.eval import evaluate  # noqa: E402
res = evaluate(H("~/drill2/final_drill2.csv"), H("~/scoring/withheld_gt.csv"))
pooled = float(res["pooled"]["auc"])
print(f"G4  事後 pooled AUC ＝ {pooled:.5f}｜seq-mean {res['seq_mean_auc']:.5f}"
      f"（對照：val65 上 SAM3 無 crop 既有 ≈ 0.6932，僅供判讀、非閘門）")
if pooled <= 0.5:
    fails.append(f"G4 pooled {pooled:.4f} ≤ 0.5 ＝ 管線災難")
json.dump(res, open(H("~/drill2/drill2_eval.json"), "w"), indent=1, default=float)

print("=" * 74)
if fails:
    print("❌ 冷啟動演練 #2 未通過：")
    for f in fails: print(f"   - {f}")
    sys.exit(1)
print("✅✅ 冷啟動演練 #2 通過——陌生資料規格下管線可端到端運作")
CHK

echo "--- G4 補算（CLI，逐序列）---"
PYTHONPATH=~ $PY3 -m hsot.eval "$OUT/final_drill2.csv" ~/scoring/withheld_gt.csv --per-seq \
  | tee "$OUT/drill2_auc.txt" || echo "⚠️ CLI 評分失敗（不阻斷，G4 已在上方判過）"

echo "=== [8] 時間帳（Ranking B 三天窗口的預算證據，D042）==="
$PY3 - <<'TIME'
import os
H = os.path.expanduser
t = {k: int(v) for k, v in (l.split() for l in open(H("~/drill2_timing.txt")).read().split("\n") if l.strip())}
def span(a, b, label):
    if a in t and b in t:
        s = t[b] - t[a]
        print(f"  {label:36s} {s//60:3d} 分 {s%60:02d} 秒")
span("DRILL2_START", "ENV_READY",        "環境+權重+資料（可並行部分）")
span("L0_START",     "L0_DONE",          "層0 SAMURAI+SAM2.1-L")
span("L1_START",     "L1_DONE",          "層1 SAM3 860M")
span("CROP_PREP_START", "CROP_PREP_DONE","層2a crop prep（CPU）")
span("L2_START",     "L2_DONE",          "層2b SAM3 裁切追蹤 + merge")
span("DRILL2_START", "L2_DONE",          "★ 總計（腳本啟動 → 產出 CSV）")
TIME

rclone copy "$OUT" "$DEST" --transfers 8
rclone copyto ~/crop_meta.json "$DEST/crop_meta.json"
rclone copyto "$STAMP" "$DEST/drill2_timing.txt"
echo "COLDSTART_DRILL2_DONE"
echo "⚠️ 機器可接力跑 E-F Phase 1；4 小時保險自毀仍在，用完務必 API terminate（D018）"
