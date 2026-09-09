#!/usr/bin/env bash
# ============================================================================
# setup_coldstart_drill_v1.sh — 冷啟動演練 #1：乾淨機器 → 一鍵重現 v008（LB 0.69128）
#
# 【為什麼是 P0】D034 列為 P0 但演練次數至今 **0**；**D042 使其升級為期程必要條件**——
# Ranking B 是 9/7–9/10 的**三天硬窗口**，屆時要對**全新 75 支無標註影片**跑完整管線。
# 現行 v008 的複現鏈從未在單一乾淨機器上從頭跑到尾，而是三個 session、三台機器的產物拼接。
#
# 【與 setup_e19_cropzoom_sam3_v1.sh 的關鍵差異——這才是「冷啟動」的定義】
#   E19 腳本第 70–71 行 rclone 拉 `exp003_samurai_large.csv`（E02）與 `sub_v006_e15sam3.csv`
#   （E15）**當作 crop 窗源**。那是「已經有軌跡了才跑第二層」，**不是冷啟動**。
#   9/7 拿到全新資料時沒有任何現成軌跡 → 兩條窗源軌跡都必須當場產出。
#   ⇒ 本腳本 **完整三段自產**：SAMURAI/SAM2.1-L 全 75 支 → SAM3 全 75 支 → crop-zoom 重跑。
#   ⇒ 參考 CSV 一律放 `~/ref/`，**只用於最後的驗收比對，絕不進入生成路徑**（生成一律用 ~/fresh/）。
#
# 【驗收分級（不採硬性 bit-exact）】追蹤是遞迴的，bf16 autocast 下 CUDA 本身非確定性
#   （t1_rerun 實測：同組態重跑 65 序列 pooled 差 −0.00002，但逐幀並非全等）。
#   硬性逐位元會因 CUDA 噪聲誤報失敗。故閘門＝**逐階段中心位移中位數 + 選中序列集合相等**：
#     G1  e02_fresh  vs exp003（E02, LB 0.66608）        中位 < 1px
#     G2  e15_fresh  vs sub_v006（E15, LB 0.67383）      中位 < 1px
#     G3  crop 選中序列集合 == E19 的 21 支              集合完全相等（規則確定性的硬證明）
#     G4  final      vs sub_v008（LB 0.69128）           中位 < 1px + 差異幀量化記錄
#
# 【同時要量的東西】各階段 wall-clock —— 這是 Ranking B 三天窗口的預算證據（D042）。
#
# 【紀律】D036 單 tar｜D018 用完必 API terminate｜D016 產出漸進回傳｜
#        D040 無 GT 只驗完整性與偏離量級｜bash 檔名帶版本號。
# 成功不自動自毀：機器保留供後續滑動窗實驗接力（4 小時保險自毀仍在）。
# ============================================================================
set -uo pipefail
trap 'echo "🚨 死於第 $LINENO 行(exit=$?)"' ERR
export PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
PY1=~/t1env/bin/python      # SAMURAI + SAM2.1-L
PY3=~/sam3env/bin/python    # SAM3 860M
SAMURAI=~/samurai
FRESH=~/fresh; REF=~/ref; OUT=~/drill
mkdir -p "$FRESH" "$REF" "$OUT"
INSTANCE_NAME="${INSTANCE_NAME:-hsot-drill}"
STAMP=~/drill_timing.txt; : > "$STAMP"
die() { echo "🚨 $*"; exit 1; }
mark() { echo "$1 $(date +%s)" >> "$STAMP"; echo "⏱  $1 @ $(date +%H:%M:%S)"; }

mark DRILL_START

# --- 保險自毀（開頭就掛：D018，Lambda 關機仍持續計費）------------------------
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

# --- [1] 資料（單 tar，D036：逐檔 rclone = 40 分鐘 GPU idle）-----------------
echo "=== [1] 背景拉 test 假色單 tar ==="
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

# --- [2] 兩套環境 + 兩份權重（能並行的全部並行）------------------------------
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

# t1env：SAMURAI + SAM2.1-L（PROVENANCE 備案鏈環境兩坑）
#   坑1 禁用 --system-site-packages（系統 apt pandas 為 numpy 1.21.5 編譯，pip 裝 sam2 會把
#        venv numpy 拉到 2.x → numpy.dtype size changed C header ABI 連環炸）
#   坑2 scipy 是未宣告依賴（SAMURAI fork 的 Kalman 用 scipy.linalg，module-level import；
#        缺它時 `import sam2` 會成功，卻在 hydra 實例化時報看不出真因的 locating target 錯誤）
echo "=== [2c] t1env（完全隔離 venv）+ SAMURAI fork ==="
[ -d ~/t1env ] || uv venv --python 3.12 ~/t1env
[ -d "$SAMURAI" ] || git clone --depth 1 https://github.com/yangchris11/samurai.git "$SAMURAI"
if [ ! -f ~/t1env/.deps_done ]; then
  VIRTUAL_ENV=~/t1env uv pip install -q torch torchvision --torch-backend=auto
  VIRTUAL_ENV=~/t1env uv pip install -q -e "$SAMURAI/sam2" \
    scipy loguru tqdm pandas pillow opencv-python-headless
  touch ~/t1env/.deps_done || die "deps_done 標記失敗（S-05：不可 fail-open，否則同機 resume 永不自我修復）"
fi
$PY1 -c "
import torch, numpy, pandas, scipy, sam2
assert torch.cuda.is_available(), 'CUDA 不可用'
print(f't1env: torch {torch.__version__} | numpy {numpy.__version__} | scipy {scipy.__version__} | sam2 OK')
" || die "t1env 驗證失敗"

# sam3env：SAM3（PROVENANCE 環境四坑，setuptools 必須最後降級）
echo "=== [2d] sam3env（完全隔離 venv）+ SAM3 上游 ==="
[ -d ~/sam3env ] || uv venv --python 3.12 ~/sam3env
if [ ! -f ~/sam3env/.deps_done ]; then
  VIRTUAL_ENV=~/sam3env uv pip install -q torch torchvision --torch-backend=auto
  VIRTUAL_ENV=~/sam3env uv pip install -q "git+https://github.com/facebookresearch/sam3.git" \
    "numpy<2" ftfy==6.1.1 regex einops psutil scipy av \
    pycocotools numba python-rapidjson pandas pillow tqdm opencv-python-headless
  VIRTUAL_ENV=~/sam3env uv pip install -q "setuptools<81"   # 81 起移除 pkg_resources
  touch ~/sam3env/.deps_done || die "deps_done 標記失敗（S-05：不可 fail-open，否則同機 resume 永不自我修復）"
fi
$PY3 -c "
import sam3, numpy, pkg_resources
assert numpy.__version__.startswith('1.'), f'numpy {numpy.__version__} 應 <2'
print(f'sam3env: numpy {numpy.__version__} | sam3 OK | pkg_resources OK')
" 2>&1 | grep -viE 'warning|deprecat|^ +import' || die "sam3env 驗證失敗"

echo "=== [2e] 程式碼 ==="
rclone copyto "$GDRIVE/3_src/track_t1.py" ~/track_t1.py
rclone copy "$GDRIVE/3_src/hsot/" ~/hsot/ --include "*.py"
[ -s ~/track_t1.py ] || die "track_t1.py 缺失"

# 參考檔：**只供驗收比對**，隔離在 ~/ref/，生成路徑一律用 ~/fresh/
echo "=== [2f] 參考 CSV（僅供驗收，不參與生成）==="
rclone copyto "$GDRIVE/5_outputs/submissions/exp003_samurai_large.csv" "$REF/e02_ref.csv"
rclone copyto "$GDRIVE/5_outputs/submissions/sub_v006_e15sam3.csv"     "$REF/e15_ref.csv"
rclone copyto "$GDRIVE/5_outputs/submissions/sub_v008_e19cropzoom.csv" "$REF/v008_ref.csv"
for f in "$REF"/*.csv; do [ -s "$f" ] || die "參考檔 $f 缺失"; done

wait $CKPT2_PID; grep -q CKPT2_READY ~/ckpt2.log || die "SAM2.1 權重失敗"
wait $CKPT3_PID; grep -q CKPT3_READY ~/ckpt3.log || die "SAM3 權重失敗"
wait $FC_PID;    grep -q TEST_FC_READY ~/fc.log  || die "test 假色失敗"
mark ENV_READY
echo "✅ 兩套環境 + 兩份權重 + 資料 + 程式碼就緒"

# 建構閘門：走完 hydra 實例化整條路徑才算環境真的可用（比 import 嚴格得多）
(cd "$SAMURAI/sam2" && $PY1 -c "
from sam2.build_sam import build_sam2_video_predictor
p = build_sam2_video_predictor('configs/samurai/sam2.1_hiera_l.yaml',
                               '$HOME/ckpt/sam2.1_hiera_large.pt', device='cuda:0')
n = sum(x.numel() for x in p.parameters())
assert n > 150e6, f'參數量 {n/1e6:.0f}M —— 不是 Large（~224M）'
print(f'✅ SAM2.1 predictor 建構成功 | {n/1e6:.0f}M params')
") || die "SAM2.1 predictor 建構失敗"

# ============================================================================
# 生成階段：三段全部自產（這才是 9/7 那天要走的路）
# ============================================================================

# --- [3] 層0：SAMURAI + SAM2.1-L 全 75 支（窗源之一，= E02）------------------
# 序列迭代順序：track_t1.py 內部 `sorted(...)`＝確定性，與 E02 原跑一致。
# 為何重要：exp008 證明 SAMURAI 的 Kalman 狀態掛在 model 物件、reset_state() 不碰它
# → 單一 predictor 連跑多序列時前一支狀態會污染下一支開頭，**序列順序改變即結果改變**。
echo "=== [3] 層0：SAMURAI + SAM2.1-L 全 75 支（自產窗源 A）==="
mark L0_START
$PY1 ~/track_t1.py --frames-root ~/test_fc --out-dir "$FRESH/e02" \
  --backend samurai --samurai-dir "$SAMURAI" --ckpt ~/ckpt/sam2.1_hiera_large.pt \
  || die "層0 SAMURAI 追蹤失敗"
cp "$FRESH/e02/submission.csv" "$FRESH/e02_fresh.csv"
mark L0_DONE
rclone copy "$FRESH" "$GDRIVE/5_outputs/coldstart_drill_20260807/fresh" --transfers 8

# --- [4] 層1：SAM3 860M 全 75 支（窗源之二 + merge base，= E15/v006）---------
echo "=== [4] 層1：SAM3 860M 全 75 支（自產窗源 B）==="
mark L1_START
$PY3 ~/track_t1.py --frames-root ~/test_fc --out-dir "$FRESH/e15" \
  --backend sam3 --sam3-version sam3 --sam3-ckpt ~/ckpt/sam3.pt \
  || die "層1 SAM3 追蹤失敗"
cp "$FRESH/e15/submission.csv" "$FRESH/e15_fresh.csv"
mark L1_DONE
rclone copy "$FRESH" "$GDRIVE/5_outputs/coldstart_drill_20260807/fresh" --transfers 8

# --- [5] 層2a：crop prep（窗 = 自產 e15 ∪ 自產 e02）--------------------------
echo "=== [5] 層2a：crop prep（窗＝自產兩軌跡聯集）==="
mark CROP_PREP_START
PYTHONPATH=~ $PY3 -m hsot.crop_rerun prep --frames-root ~/test_fc \
  --base-csv "$FRESH/e15_fresh.csv" --envelope-extra "$FRESH/e02_fresh.csv" \
  --out-root ~/crop_fc --meta ~/crop_meta.json | tail -30 || die "crop prep 失敗"
NSEL=$($PY3 -c "import json;print(len(json.load(open('$HOME/crop_meta.json'))))")
echo "選中 $NSEL 支"
[ "$NSEL" -ge 5 ] || die "只選中 $NSEL 支，管線有異"
$PY3 -c "
import json,os
m=json.load(open(os.path.expanduser('~/crop_meta.json')))
open(os.path.expanduser('~/crop_seqs.txt'),'w').write('\n'.join(sorted(m))+'\n')"
mark CROP_PREP_DONE

# --- [6] 層2b：SAM3 追蹤裁切序列 + merge ------------------------------------
echo "=== [6] 層2b：SAM3 追蹤裁切序列 ==="
mark L2_START
$PY3 ~/track_t1.py --frames-root ~/crop_fc --seq-list ~/crop_seqs.txt \
  --out-dir "$FRESH/crop" --backend sam3 --sam3-version sam3 --sam3-ckpt ~/ckpt/sam3.pt \
  || die "層2 裁切追蹤失敗"
PYTHONPATH=~ $PY3 -m hsot.crop_rerun merge --base-csv "$FRESH/e15_fresh.csv" \
  --crop-csv "$FRESH/crop/submission.csv" --meta ~/crop_meta.json \
  --out "$OUT/v008_drill.csv" || die "merge 失敗"
mark L2_DONE
rclone copy "$OUT" "$GDRIVE/5_outputs/coldstart_drill_20260807" --transfers 8

# ============================================================================
# 驗收：四道閘門 + 時間帳（參考檔到這裡才第一次被讀）
# ============================================================================
echo "=== [7] 驗收 ==="
$PY3 - <<'CHK' || die "驗收失敗"
import json, os, sys
import numpy as np, pandas as pd
H = os.path.expanduser

def load(p):
    df = pd.read_csv(H(p))
    df["seq"] = df.ID.str.rsplit("_", n=1).str[0]
    return df

def center_delta(a, b, label):
    """逐序列中心位移中位數。對 CUDA 噪聲耐受，但權重/設定/前處理錯了會立刻爆表。"""
    m = a.merge(b, on="ID", suffixes=("", "_r"))
    assert len(m) == len(a) == len(b), f"{label}: ID 對不上 {len(m)}/{len(a)}/{len(b)}"
    cd = np.hypot((m.x + m.width/2) - (m.x_r + m.width_r/2),
                  (m.y + m.height/2) - (m.y_r + m.height_r/2))
    per = pd.DataFrame({"seq": m.seq, "cd": cd}).groupby("seq").cd.median().sort_values()
    ch = (m[["x","y","width","height"]].to_numpy()
          != m[["x_r","y_r","width_r","height_r"]].to_numpy()).any(1)
    return float(np.median(cd)), per, float(ch.mean())

fails = []
print("=" * 74)

# G1 層0 vs E02（LB 0.66608）
a, b = load("~/fresh/e02_fresh.csv"), load("~/ref/e02_ref.csv")
med, per, chr_ = center_delta(a, b, "G1")
print(f"G1  層0(SAMURAI+SAM2.1-L) vs E02 參考：中心位移中位 {med:.2f}px｜差異幀 {chr_:.1%}")
print(f"    偏離最大 3 支：{per.tail(3).round(2).to_dict()}")
if med >= 1.0: fails.append(f"G1 中位 {med:.2f}px ≥ 1px")

# G2 層1 vs E15（LB 0.67383）
a, b = load("~/fresh/e15_fresh.csv"), load("~/ref/e15_ref.csv")
med, per, chr_ = center_delta(a, b, "G2")
print(f"G2  層1(SAM3 860M) vs E15 參考：中心位移中位 {med:.2f}px｜差異幀 {chr_:.1%}")
print(f"    偏離最大 3 支：{per.tail(3).round(2).to_dict()}")
if med >= 1.0: fails.append(f"G2 中位 {med:.2f}px ≥ 1px")

# G3 選中序列集合 == E19 的 21 支（選序列規則確定性的硬證明）
meta = json.load(open(H("~/crop_meta.json")))
got = set(meta)
E19_21 = None
ref_meta = H("~/ref/e19_meta.json")
if os.path.exists(ref_meta):
    E19_21 = set(json.load(open(ref_meta)))
if E19_21 is None:
    # 參考窗口表不在手邊時，退而以「v008 相對 E15 有變動的序列集合」還原
    v8, e15 = load("~/ref/v008_ref.csv"), load("~/ref/e15_ref.csv")
    m = v8.merge(e15, on="ID", suffixes=("", "_r"))
    ch = (m[["x","y","width","height"]].to_numpy()
          != m[["x_r","y_r","width_r","height_r"]].to_numpy()).any(1)
    E19_21 = set(m.loc[ch, "seq"].unique())
print(f"G3  選中序列：本次 {len(got)} 支｜E19 參考 {len(E19_21)} 支")
if got != E19_21:
    only_new, only_ref = sorted(got - E19_21), sorted(E19_21 - got)
    print(f"    ⚠️ 差異——僅本次 {only_new}｜僅參考 {only_ref}")
    fails.append(f"G3 選中集合不等（+{len(only_new)}/-{len(only_ref)}）")
else:
    print(f"    ✅ 集合完全相等——選序列規則在冷啟動下確定性重現")

# G4 最終 vs v008（LB 0.69128）
a, b = load("~/drill/v008_drill.csv"), load("~/ref/v008_ref.csv")
med, per, chr_ = center_delta(a, b, "G4")
print(f"G4  最終產出 vs v008 參考：中心位移中位 {med:.2f}px｜差異幀 {chr_:.1%}")
print(f"    偏離最大 5 支：{per.tail(5).round(2).to_dict()}")
if med >= 1.0: fails.append(f"G4 中位 {med:.2f}px ≥ 1px")

# 完整性（D040 災難檢查）
d = pd.read_csv(H("~/drill/v008_drill.csv"))
assert len(d) == 26860, f"列數 {len(d)}"
assert d.isna().sum().sum() == 0, "有 NaN"
assert (d.width > 0).all() and (d.height > 0).all(), "有非正 w/h"
print(f"完整性 ✅ 26,860 列｜無 NaN｜無非正 w/h")

print("=" * 74)
if fails:
    print("❌ 冷啟動演練未完全通過：")
    for f in fails: print(f"   - {f}")
    sys.exit(1)
print("✅✅ 冷啟動演練 #1 全數通過——乾淨機器可一鍵重現 v008")
CHK

echo "=== [8] 時間帳（Ranking B 三天窗口的預算證據，D042）==="
$PY3 - <<'TIME'
import os
H = os.path.expanduser
rows = [l.split() for l in open(H("~/drill_timing.txt")).read().split("\n") if l.strip()]
t = {k: int(v) for k, v in rows}
def span(a, b, label):
    if a in t and b in t:
        s = t[b] - t[a]
        print(f"  {label:34s} {s//60:3d} 分 {s%60:02d} 秒")
span("DRILL_START", "ENV_READY",       "環境+權重+資料（可並行部分）")
span("L0_START",    "L0_DONE",         "層0 SAMURAI+SAM2.1-L 全 75 支")
span("L1_START",    "L1_DONE",         "層1 SAM3 860M 全 75 支")
span("CROP_PREP_START", "CROP_PREP_DONE", "層2a crop prep（CPU）")
span("L2_START",    "L2_DONE",         "層2b SAM3 裁切追蹤 + merge")
span("DRILL_START", "L2_DONE",         "★ 總計（開機腳本啟動 → 產出 CSV）")
TIME

rclone copy "$OUT" "$GDRIVE/5_outputs/coldstart_drill_20260807" --transfers 8
rclone copyto ~/crop_meta.json "$GDRIVE/5_outputs/coldstart_drill_20260807/crop_meta.json"
rclone copyto "$STAMP" "$GDRIVE/5_outputs/coldstart_drill_20260807/drill_timing.txt"
echo "COLDSTART_DRILL_DONE"
echo "⚠️ 機器保留供滑動窗實驗接力；4 小時保險自毀仍在，用完務必 API terminate（D018）"
