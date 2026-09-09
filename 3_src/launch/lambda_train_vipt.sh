#!/usr/bin/env bash
# ============================================================================
# Lambda A100 T2 訓練 launch — ViPT_HOT2023（D006 定案底座）
# 開機自動全流程 + 跑完自毀。紀律：D019（gDrive 唯一真相源、不租 filesystem、
# 開機拉資料、跑完 ckpt 回傳 + 自動關機避免持續計費）。
#
# 用法（Lambda 開 A100 40GB 實例後）：
#   1. 上傳 rclone.conf 到 ~/.config/rclone/rclone.conf（免瀏覽器認證，D021）
#      + 上傳 Lambda API key 到 ~/.lambda_key（自毀 terminate 用，見下方 cleanup）
#   2. NO_TERMINATE=1 bash lambda_train_vipt.sh DRY vis   # DRY 保留機器供檢查
#   3. 確認 DRY 綠燈後：bash lambda_train_vipt.sh FULL vis   # 跑完自動 terminate
#      （nir / rednir 各另跑一輪——無 "all" 模式，見下方 DATATYPE 說明）
#
# ⚠️ 前置：T2 訓練資料需已由 RunPod 資料準備腳本產好並上 gDrive
#   （見 3_src/prep/prep_t2_trainset.py：解 405 zip → X2Cube→.npy cube +
#    假色 jpg + groundtruth_rect.txt + split txt；套用 update 修正）。
# ============================================================================
set -euo pipefail
# Python 的 stdout 被導向管線/檔案時是 block-buffered（8KB）→ 訓練每 50 步才印一行，
# 要 ~2000 步（約 20 分鐘）才刷一次緩衝區，log 嚴重落後、無法即時監控長時間訓練。
export PYTHONUNBUFFERED=1

MODE="${1:-DRY}"            # DRY（少量 step 驗通）| FULL
DATATYPE="${2:-vis}"       # vis | nir | rednir
# ⚠️ 沒有 "all" 這種模式：vit_ce_prompt_all.py 的 forward 依 train_data_type 走三選一分支，
#    其餘值直接 raise ValueError()。三模態＝各訓一輪（共用凍結 backbone、各自 prompt），
#    最後把三組 prompt 權重合併回單一 ckpt（deep_all 架構本來就同時容納三組）。
case "$DATATYPE" in
  vis|nir|rednir) ;;
  *) echo "DATATYPE 必須是 vis|nir|rednir（收到：$DATATYPE）"; exit 1 ;;
esac
GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
WORK="/workspace/hsot"
REPO="$WORK/ViPT_HOT2023"
CKPT_OUT="$GDRIVE/4_models/t2_vipt"
RUN_TAG="$(cat /proc/sys/kernel/random/uuid | cut -c1-8)"   # 本次 run 隨機標記
# ⚠️ save_dir 依模態分開：ViPT 啟動時會自動「續訓」save_dir 底下既有的 ckpt——共用一個
#    目錄會讓 FULL 誤載 DRY（或別的模態）留下的 ckpt。同模態重跑仍可正常續訓（崩潰復原）。
SAVE_DIR="$REPO/output_${DATATYPE}"

# --- 0. 自毀保險：無論成功/失敗/中斷，收尾都回傳 log 並「terminate」（省錢鐵律）---
# ⚠️ Lambda 沒有「已停止」狀態：OS 內 `shutdown -h` 只是關機，實例仍持續計費到
#    API terminate 為止。必須呼叫 instance-operations/terminate（2026-08-04 實測空
#    instance_ids 為安全 no-op，端點形式已驗）。API key 由 launch 前上傳到 ~/.lambda_key。
self_terminate() {
  local key id
  key="$(cat ~/.lambda_key 2>/dev/null | tr -d '\n')"
  if [ -z "$key" ]; then
    echo "🚨 找不到 ~/.lambda_key，無法 terminate——請手動終止本實例（否則持續計費）"
    sudo shutdown -h now; return
  fi
  # 自我辨識：Lambda hostname = IP 以 dash 連接，與 API 的 hostname 欄位一致
  id="$(curl -s -m 20 -u "$key:" https://cloud.lambda.ai/api/v1/instances \
        | python3 -c "import sys,json,socket;h=socket.gethostname();print(next((i['id'] for i in json.load(sys.stdin)['data'] if i.get('hostname')==h),''))" 2>/dev/null)"
  if [ -z "$id" ]; then
    echo "🚨 無法比對自身 instance id——請手動終止本實例（否則持續計費）"
    sudo shutdown -h now; return
  fi
  echo "=== terminate instance $id ==="
  curl -s -m 30 -u "$key:" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
       -H 'Content-Type: application/json' -d "{\"instance_ids\":[\"$id\"]}"
  sleep 60 && sudo shutdown -h now   # terminate 沒生效時的兜底
}
cleanup() {
  local rc=$?                       # 必須第一行取——後面任何指令都會覆蓋 $?
  echo "=== 收尾：回傳 log/ckpt（exit=$rc）==="
  rclone copy "$SAVE_DIR" "$CKPT_OUT/run_${RUN_TAG}" --include "*.pth.tar" --include "*.log" --include "*.txt" 2>/dev/null || true
  rclone copy "$WORK/train_${DATATYPE}.log" "$CKPT_OUT/run_${RUN_TAG}/" 2>/dev/null || true
  if [ "${NO_TERMINATE:-0}" = "1" ]; then
    echo "NO_TERMINATE=1 → 保留實例（記得手動終止！）"; return
  fi
  # 失敗時給搶救寬限期：失敗多半發生在拉完資料/解壓之後（本次實測 ~50 分鐘的成本），
  # 立刻 terminate 會把那些資料一起銷毀，重跑得從頭再等一次。成功則直接終止不浪費。
  if [ "$rc" != "0" ]; then
    echo "🚨 失敗退出（rc=$rc）——保留實例 ${FAIL_GRACE:-900}s 供搶救（SSH 進來 pkill 本腳本即可留住）"
    sleep "${FAIL_GRACE:-900}"
  fi
  self_terminate
}
trap cleanup EXIT

mkdir -p "$WORK" && cd "$WORK"

# --- 1. 環境 A100 化（研究 agent 標的最大坑：ViPT 出廠 torch1.9.1+cu102 跑不動 Ampere）---
echo "=== [1/5] 環境 ==="
# Lambda A100 映像（2026-08 實測）：Python 3.10 + torch 2.7.0+cu 且 cuda 可用
# → **不降級到 ViPT 出廠的 torch 1.9.1+cu102**（那組跑不動 Ampere，且無 py3.10 輪子）。
# ⚠️ 全新映像**沒有** timm/easydict/yacs（第一台之所以有，是先前會話手動裝的——別把那台的
#    狀態當成映像預設，2026-08-05 開新機時就是這樣連環失敗的）。全部明確安裝。
# ⚠️ timm 必須釘 0.6.11：ViPT 用的 `timm.models.layers` 等 API 在新版已搬家。
# 唯一版本衝突：映像帶 numpy 2.2.6，而 torch 2.7 是對 numpy 1.x 編譯的
# → 必須釘 numpy<2，否則 import 即噴 _ARRAY_API not found。
# jpeg4py 在 lib/train/data/image_loader.py 是 **module-level import**（即使實際用 opencv_loader
# 也必須裝得起來），且它靠 ctypes 載 libturbojpeg → 需先裝系統 .so。
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libturbojpeg >/dev/null 2>&1 || true
# ⚠️ 全新 Lambda 映像也**沒有 rclone**（同上：第一台有是先前會話裝的）。裝官方新版而非
#    apt 版——apt 的 v1.53 用舊 shared client，對 Google Drive 會撞 rateLimitExceeded（D014）。
if ! command -v rclone >/dev/null; then
  curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null 2>&1 || true
fi
command -v rclone >/dev/null || { echo "🚨 rclone 安裝失敗——無法取得資料"; exit 1; }
rclone version | head -1
# pycocotools/tensorboardX/visdom：lib/train/dataset/__init__.py 一次 import 全部資料集
# （COCO 等），即使我們只用 HSI3D 也必須裝得起來。numba/trax/vot 只在 VOT 評測路徑，不裝。
pip install -q "numpy<2" "timm==0.6.11" easydict yacs pandas \
    opencv-python-headless pyyaml lmdb tqdm jpeg4py \
    pycocotools tensorboardX visdom
python -c "
import torch, timm, numpy, cv2, pandas, easydict, yacs
assert torch.cuda.is_available(), 'CUDA 不可用'
assert timm.__version__.startswith('0.6'), f'timm {timm.__version__} 太新，ViPT 需 0.6.x'
assert numpy.__version__.startswith('1.'), f'numpy {numpy.__version__} 必須 <2'
print('torch', torch.__version__, '| timm', timm.__version__, '| numpy', numpy.__version__, '| ✓ 環境檢查通過')"

# --- 2. 拉 repo + 權重 + raw zip + 我方 prep/loader/GT（gDrive 真相源）---
echo "=== [2/5] 拉 repo/權重/raw資料/腳本 ==="
[ -d "$REPO" ] || git clone --depth 1 https://github.com/laisimiao/ViPT_HOT2023.git "$REPO"
mkdir -p "$REPO/pretrained_models" "$REPO/final_model"
# ViPT_all.pth 含凍結 OSTrack backbone + 三模態 prompt；config MODEL.PRETRAIN_FILE 為空，
# 整個模型從此 init，不需另外 OSTrack 底座檔（deep_all.yaml TRAIN.PRETRAIN_FILE=./final_model/ViPT_all.pth.tar）
rclone copy "$GDRIVE/4_models/pretrained_t2/" "$REPO/final_model/" --include "ViPT_all.pth.tar" -P
# 我方腳本（prep + 改好的現場 X2Cube loader）+ 逐幀 GT + val_split
SRC="$WORK/src"; mkdir -p "$SRC"
rclone copyto "$GDRIVE/3_src/prep/prep_t2_trainset.py" "$SRC/prep_t2_trainset.py"
rclone copyto "$GDRIVE/3_src/t2_vipt/hsi3d.py" "$SRC/hsi3d.py"
rclone copyto "$GDRIVE/1_data/raw/2026training.csv" "$SRC/2026training.csv"
rclone copyto "$GDRIVE/1_data/val_split_v1.txt" "$SRC/val_split_v1.txt"
# raw train zip（DRY 只拉幾序列；FULL 全量 ~400GB）
RAW="$WORK/raw_archive"; mkdir -p "$RAW/training"
if [ "$MODE" = "DRY" ]; then
  # DRY 序列必須與 DATATYPE 同模態，且 train/val 各有一支（否則 split 空、訓練空轉）。
  # vis-car（54MB，不在 val_split_v1）+ vis-coin（56MB，在 val_split_v1）＝最小可跑組合。
  # ⚠️ 不可用 basketball1（D025 已知來源幀數短缺）或 ball（NIR，與 DATATYPE=vis 不符）。
  DRY_SEQS="vis-car,vis-coin"
  MOD_UP="$(echo "$DATATYPE" | tr 'a-z' 'A-Z')"; [ "$DATATYPE" = "rednir" ] && MOD_UP="RedNIR"
  for z in car coin; do
    for sub in "HSI-$MOD_UP" "HSI-$MOD_UP-FalseColor"; do
      rclone copy "$GDRIVE/1_data/raw_archive/training/$sub/$z.zip" "$RAW/training/$sub/"
    done
  done
  PREP_LIMIT="--only $DRY_SEQS"
else
  # 一輪只訓一個模態 → 只拉該模態（vis ~122GB / nir ~124GB / rednir ~63GB），
  # 不是全部 ~400GB。⚠️ gDrive 上 update/ 位於 raw_archive/training/ 之內，但 prep 的
  # pick_zip 找的是 <zips>/update/<sub>/<name>.zip → 必須拆成兩份拉，否則 75 支修正包不會套用。
  MOD_UP="$(echo "$DATATYPE" | tr 'a-z' 'A-Z')"; [ "$DATATYPE" = "rednir" ] && MOD_UP="RedNIR"
  # ⚠️ 個別檔案失敗不可中止整輪：gDrive 上有失效捷徑（實測 HSI-RedNIR/backpack3.zip
  #    "can't read dangling shortcut"），rclone 會回非零而 set -e 直接砍掉整個腳本。
  #    改為容忍 + 事後對帳；缺資料的序列由 prep 自動排除於 split（不會在訓練中途炸）。
  for sub in "HSI-$MOD_UP" "HSI-$MOD_UP-FalseColor"; do
    rclone copy "$GDRIVE/1_data/raw_archive/training/$sub/" "$RAW/training/$sub/" -P --stats 60s || \
      echo "⚠️ $sub 有檔案未取得（見上方 rclone ERROR），繼續"
    rclone copy "$GDRIVE/1_data/raw_archive/training/update/$sub/" "$RAW/update/$sub/" -P --stats 60s || true
    echo "$sub 本地 zip 數：$(ls "$RAW/training/$sub" 2>/dev/null | wc -l)"
  done
  PREP_ONLY="$(python3 -c "
import csv,sys
seqs=sorted({r[0].rsplit('_',1)[0] for r in csv.reader(open('$SRC/2026training.csv')) if r[0].startswith('$DATATYPE-')})
print(','.join(seqs))")"
  PREP_LIMIT="--only $PREP_ONLY"
fi

# --- 2.5. 資料準備：解 zip → HOT 格式 + split（不預轉 npy，訓練 loader 現場 X2Cube）---
echo "=== [2.5] prep ==="
DATA_LOCAL="$WORK/data"; mkdir -p "$DATA_LOCAL"
python "$SRC/prep_t2_trainset.py" --zips "$RAW" --gt "$SRC/2026training.csv" \
    --val-split "$SRC/val_split_v1.txt" --out "$DATA_LOCAL/training" $PREP_LIMIT

# --- 3. 覆蓋 loader（現場 X2Cube）+ 放 split + local paths ---
echo "=== [3/5] loader/split/local ==="
cd "$REPO"
cp "$SRC/hsi3d.py" lib/train/dataset/hsi3d.py                    # 現場 X2Cube，省 ~300GB npy
# py3.10 / torch2.x 相容補丁（ViPT 出廠是 py3.7+torch1.9）：
#   (a) torch._six 已移除 → 不補連 local.py 產生器都 import 失敗
#   (b) collections.Mapping/Sequence 在 py3.10 移到 collections.abc → dataloader worker 全掛
sed -i 's/^from torch\._six import string_classes$/string_classes = str  # compat: torch>=1.13 已移除 torch._six/' \
    lib/train/data/loader.py
sed -i 's/collections\.\(Mapping\|Sequence\|Iterable\)/collections.abc.\1/g' lib/train/data/loader.py
#   (d) sampler 的裸 `except: valid = False` 會把**任何**載入例外靜默吞掉並重抽序列
#       → 資料壞掉時訓練照樣「成功」跑完（vis 實測靜默跳過 23/172 支）。改成印出前幾次
#       例外，讓問題在 log 裡留下痕跡（不改控制流，避免偏離上游行為）。
python - <<'PY'
import pathlib, re
p = pathlib.Path("lib/train/data/sampler.py"); s = p.read_text()
patched = s.replace(
    "            except:\n                valid = False\n",
    "            except Exception as _e:  # compat: 原為裸 except，會靜默吞掉壞資料\n"
    "                import os as _os\n"
    "                _n = getattr(self, '_load_err_n', 0) + 1\n"
    "                self._load_err_n = _n\n"
    "                if _n <= 20 or _n % 1000 == 0:\n"
    "                    print(f'[sampler] 載入失敗 #{_n} (pid {_os.getpid()}): {type(_e).__name__}: {_e}', flush=True)\n"
    "                valid = False\n")
p.write_text(patched)
print(f"sampler 例外記錄補丁：置換 {s.count('            except:')} 處")
PY
#   (c) torch>=2.6 把 torch.load 的 weights_only 預設改成 True，而 ViPT 存的 ckpt 內含
#       `lib.train.admin.stats.AverageMeter` 等物件 → 續訓/載入既有 ckpt 必炸
#       （`ViPT_all.pth.tar` 例外，它只有純張量所以載得動，容易誤以為沒事）。
grep -rl 'torch\.load(' lib/train/trainers/*.py lib/models/vipt/*.py | while read -r f; do
  sed -i "s/torch\.load(\(.*\)map_location=\(['\"]cpu['\"]\))/torch.load(\1map_location=\2, weights_only=False)/g" "$f"
done
grep -c "weights_only=False" lib/train/trainers/base_trainer.py
cp "$DATA_LOCAL/training"/hsi_*_split.txt lib/train/data_specs/  # 我方 405 序列 split
python tracking/create_default_local_file.py --workspace_dir "$REPO" --data_dir "$DATA_LOCAL" --save_dir "$SAVE_DIR" || true
# hsi_dir 須指向 $DATA_LOCAL（其下 training/HSI-*-FalseColor/{name}/）。
# ⚠️ create_default_local_file.py 產的 local.py **沒有 hsi_dir 欄位**（只有 lasot/got10k/coco…）
#    → 必須自己補進 EnvironmentSettings.__init__，不能靠 sed 取代既有行。
python - <<PY
import pathlib, re
p = pathlib.Path("lib/train/admin/local.py"); s = p.read_text()
line = "        self.hsi_dir = '$DATA_LOCAL'\n"
s = re.sub(r"^\s*self\.hsi_dir\s*=.*\n", "", s, flags=re.M)          # 冪等：先移除舊值
s = s.replace("        self.workspace_dir", line + "        self.workspace_dir", 1)
p.write_text(s)
print("local.py hsi_dir =", "$DATA_LOCAL")
PY
grep -n "hsi_dir" lib/train/admin/local.py

# --- 3.5 起飛前資料檢查（fail fast，不讓壞資料被靜默跳過）---
# ⚠️ 存在理由：ViPT 的 sampler 用 `while not valid: try: ... except: valid=False` 把
#    所有載入例外吞掉並重抽別支序列 → 空目錄／結構錯的序列會被**靜默跳過**，訓練照樣
#    跑完 20 epoch、崩潰計數 0，事後才發現只用了 87% 的資料（2026-08-05 vis 實測 149/172）。
#    因此必須在訓練前逐序列實際開一張圖，壞的就當場中止。
echo "=== [3.5] 資料起飛前檢查 ==="
python - <<PY || exit 1
import sys, pathlib, cv2
root = pathlib.Path("$DATA_LOCAL/training")
bad, total = [], 0
for split in ("train", "val"):
    f = pathlib.Path("lib/train/data_specs") / f"hsi_${DATATYPE}_{split}_split.txt"
    for line in f.read_text().split():
        total += 1
        fc = root / line                                  # HSI-*-FalseColor/{name}
        hsi = root / line.replace("-FalseColor", "")
        jpgs = sorted(fc.glob("*.jpg")); pngs = sorted(hsi.glob("*.png"))
        if not jpgs or not pngs:
            bad.append(f"{line}: jpg={len(jpgs)} png={len(pngs)}"); continue
        # 真的讀一張，擋掉「檔案在但讀不了」（權限／截斷／格式）
        if cv2.imread(str(jpgs[0])) is None or cv2.imread(str(pngs[0]), -1) is None:
            bad.append(f"{line}: 首幀讀取失敗")
print(f"檢查 {total} 支序列，異常 {len(bad)}")
for b in bad[:20]:
    print("  ", b)
if bad:
    print("🚨 有序列無法載入——訓練會靜默跳過它們，先修再跑")
    sys.exit(1)
print("✓ 全部序列可正常載入")
PY

# --- 4. 訓練（DRY=少量 step 驗 pipeline；FULL=正式）---
echo "=== [4/5] 訓練 MODE=$MODE DATATYPE=$DATATYPE ==="
CFG="deep_all"
# ⚠️ DATATYPE 只存在於 yaml（TRAIN.PROMPT.DATATYPE），命令列參數傳不進去——必須改檔。
#    凍結靠 `train_data_type in n` 字串比對（base_functions.get_optimizer_scheduler）：
#    2026-08-04 實測 ViPT_all.pth.tar 參數名 → vis 110 張量/3.40M、rednir 110/3.20M，
#    但 **nir 匹配 220 張量/8.36M（"nir" 是 "rednir" 的子字串，連 RedNIR prompt 一起解凍）**。
#    DATATYPE="" 則匹配全部 prompt（11.76M）。跑 nir 輪時務必檢查此數字。
# ⚠️ yaml 是**就地修改**的 → DRY 跑過會把 EPOCH=1/SAMPLE_PER_EPOCH=320 永久寫進檔案，
#    之後 FULL 若只設「DRY 專屬」欄位就會沿用被汙染的值，**10 步跑完並判定成功**
#    （2026-08-05 實測踩到）。因此：先從 git 還原，再把兩種模式的值都明確寫死。
git checkout -- experiments/vipt/${CFG}.yaml 2>/dev/null || echo "（無 git，靠下方明確賦值）"
python - <<PY
import yaml, pathlib, sys
p = pathlib.Path("experiments/vipt/${CFG}.yaml")
c = yaml.safe_load(p.read_text())
dt, mode = "${DATATYPE}", "${MODE}"
c["TRAIN"]["PROMPT"]["DATATYPE"] = dt
c["DATA"]["TRAIN"]["DATASETS_NAME"] = [f"HSI_train_{dt}"]
c["DATA"]["VAL"]["DATASETS_NAME"] = [f"HSI_val_{dt}"]
if mode == "DRY":
    c["TRAIN"]["EPOCH"] = 1
    c["TRAIN"]["VAL_EPOCH_INTERVAL"] = 1           # DRY 也要驗到 val 路徑
    c["DATA"]["TRAIN"]["SAMPLE_PER_EPOCH"] = 320   # bs32 → 10 step
    c["DATA"]["VAL"]["SAMPLE_PER_EPOCH"] = 64
    c["TRAIN"]["NUM_WORKER"] = 4
else:                                              # FULL＝原作者 HOT2023 配方（不可依賴檔案現況）
    c["TRAIN"]["EPOCH"] = 20
    c["TRAIN"]["VAL_EPOCH_INTERVAL"] = 5
    c["DATA"]["TRAIN"]["SAMPLE_PER_EPOCH"] = 60000
    c["DATA"]["VAL"]["SAMPLE_PER_EPOCH"] = 10000
    c["TRAIN"]["NUM_WORKER"] = 8
    if c["TRAIN"]["EPOCH"] < 20 or c["DATA"]["TRAIN"]["SAMPLE_PER_EPOCH"] < 60000:
        sys.exit("🚨 FULL 的 epoch/sample 設定不合理")
p.write_text(yaml.safe_dump(c))
print(f"yaml patched: MODE={mode} DATATYPE={dt} EPOCH={c['TRAIN']['EPOCH']} "
      f"SAMPLE_PER_EPOCH={c['DATA']['TRAIN']['SAMPLE_PER_EPOCH']} NUM_WORKER={c['TRAIN']['NUM_WORKER']}")
PY
# ⚠️ 不走 tracking/train.py：它用 os.system 起子行程 **吞掉退出碼**，訓練炸了仍回傳 0
#    → 腳本會照常往下走並自毀關機（失敗被當成功）。直接呼叫 run_training.py 讓 set -e 生效。
python lib/train/run_training.py --script vipt --config "$CFG" --save_dir "$SAVE_DIR" 2>&1 | tee "$WORK/train_${DATATYPE}.log"

# --- 4.5 成功閘門（沒有這關，失敗會被當成功並自毀關機）---
# ViPT 的 BaseTrainer 用 fail_safe=True 捕捉所有例外、重試該 epoch，最後照樣印
# "Finished training!" 並以 exit 0 收場 → 退出碼完全不可信。改用兩個實質判準：
TLOG="$WORK/train_${DATATYPE}.log"
if grep -q "Training crashed at epoch" "$TLOG"; then
  echo "🚨 訓練崩潰（fail_safe 吞掉例外）——見 $TLOG"; exit 1
fi
if ! ls "$SAVE_DIR"/checkpoints/train/vipt/"$CFG"/*.pth.tar >/dev/null 2>&1; then
  echo "🚨 沒有產出任何 checkpoint——訓練實質未進行"; exit 1
fi
# FULL 必須跑到最後一個 epoch：擋掉「沿用殘留 DRY 設定 → 10 步跑完卻判定成功」這類
# 靜默降級（2026-08-05 實測踩到；光看「有沒有 ckpt」是擋不住的）。
if [ "$MODE" = "FULL" ]; then
  LAST_CKPT="$SAVE_DIR/checkpoints/train/vipt/$CFG/ViPTrack_ep0020.pth.tar"
  if [ ! -f "$LAST_CKPT" ]; then
    echo "🚨 缺最後一個 epoch 的 ckpt（$LAST_CKPT）——訓練未跑滿 20 epoch"
    ls -la "$SAVE_DIR/checkpoints/train/vipt/$CFG/" || true
    exit 1
  fi
fi
# 凍結量核對：vis/rednir 應 110 張量（≈3.4M / 3.2M），nir 因子字串問題為 220
echo "可訓 prompt 張量數：$(grep -A400 'Only training prompt' "$TLOG" | grep -c '^backbone\.')"

# --- 5. ckpt 即傳（trap 也會收尾，此處額外保險把最佳 ckpt 立刻上傳）---
echo "=== [5/5] ckpt 回傳 ==="
rclone copy "$SAVE_DIR" "$CKPT_OUT/run_${RUN_TAG}" --include "*.pth.tar" -P || true
echo "=== 完成，trap 將自動關機 ==="
