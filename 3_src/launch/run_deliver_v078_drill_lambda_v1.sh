#!/usr/bin/env bash
# 9/7 交付演練：rankB_deliver_v078 端到端冷啟動（乾淨機、單一入口、零人工干預）。
#
# 這支腳本【就是 9/7 當天要跑的東西】——只有兩處會換：
#   1. FRAMES 的來源（演練用 test75 打包檔；9/7 換成主辦方釋出的新 75 支）
#   2. SAMPLE 的來源（9/7 用官方新 sample_submission.csv）
#   3. raw→frames_root 的 ingestion（官方發佈不是打包好的 tar，目前無腳本；稽核 G15）
# 其餘一律不動。演練的意義就是把這三處以外的一切都先跑穿一次。
#
# ⚠️ SAMPLE 有兩個互不相容的角色，**不可共用同一個檔**（09-04 稽核 G11）：
#   SAMPLE          ＝本次要跑的 sample（9/7 是官方新序列），只傳給 run_ranking_b.py
#   SELFTEST_SAMPLE ＝歷史 test75 sample，只給 finalize_submission.py --selftest 用
# 兩者混用 ⇒ 9/7 的 selftest (c)/(d) 拿到新序列、與凍結參照鏈對不上，
# 舊碼在 write() 丟未捕捉的 KeyError、腳本當場死、機器 5 分鐘後自毀。
#
# 交付鏈（profile rankB_deliver_v078，08-29 定案）：
#   corr both → frozen splice K=6 → selector-v2 (τ=0.05, 固化權重) → qhead v056
#   → 第三腿死區救援。test75 上位元級 == sub_v078 == LB 0.71666。
#
# ⏰ 自毀：唯一計時器＝cloud-init /root/rearm_selfkill.sh（開機端武裝）。
#    本腳本【不】自掛第二顆 sleep（08-07 事故：兩套計時器只延了一套，實驗差 3 分鐘被砍）。
#    開工先 rearm，分鐘數**依 profile 而定**（見 SELFKILL_MIN）：
#      v078 ＝ 240 分（08-29 演練 #4 實測全程 3h22m ⇒ 緩衝 38 分）
#      v090 ＝ 400 分（多兩組 crop-SAM3 ≈ +80 分 ⇒ 全程約 4.7h ⇒ 緩衝約 1.7h）
#    ⚠️ 腳本註解原寫「推論實測 1.7h/75 支」——那是 08-16 演練 #3 的數字，
#      當時管線沒有 crop 兩腿，08-30 已作廢。240 分**不足以跑 v090**，會在推論中途被砍。
set -euo pipefail
trap 'echo "死於第 $LINENO 行" >&2' ERR

export PYTHONUNBUFFERED=1
export PATH="/home/ubuntu/.local/bin:$PATH"
export RCLONE_CONFIG=/home/ubuntu/rclone.conf

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
# 產物目的地。**預設維持 08-29 演練 #4 的資料夾以免破壞既有引用**；
# 每次新演練都應以 DRILL_DEST 指定新名字（否則會蓋掉 PROVENANCE 引用的舊產物）。
DRILL_DEST="${DRILL_DEST:-deliver_v078_drill_20260829}"
DEST="$GDRIVE/5_outputs/$DRILL_DEST"
# ⚠️ repo 根【不可】叫 hsot：套件本身是 3_src/hsot 且無 __init__.py（namespace
#    package），同名時 pytest 的 `import hsot` 會解析到 repo 根而非套件。
#    08-29 演練實測撞到（rc=2, "cannot import name crop_rerun from hsot
#    (unknown location)"）。9/7 沿用此路徑名。
REPO=/home/ubuntu/hsot_repo
SRC="$REPO/3_src"
WORK="/home/ubuntu/$DRILL_DEST"
FRAMES=/home/ubuntu/test_fc
CKPT=/home/ubuntu/ckpt
SAMURAI=/home/ubuntu/samurai
T1ENV=/home/ubuntu/t1env
SAM3ENV=/home/ubuntu/sam3env
PY1="$T1ENV/bin/python"
PY3="$SAM3ENV/bin/python"
SAMURAI_SHA=76ba195984892b0d1e3db5d9c9f90bb62175680a
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
SAM3_CKPT_SHA256=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
SAMPLE=/home/ubuntu/sample_submisson.csv           # 本次要跑的（9/7 換官方新檔）
SELFTEST_SAMPLE=/home/ubuntu/selftest_sample_test75.csv  # 永遠是歷史 test75，selftest 專用
# G24：1＝從 3_src/requirements-{sam3env,t1env}.txt 的完全解析鎖安裝（預設）。
# 設 0 可退回舊的鬆散清單——若 9/6 演練因鎖檔裝不起來，這是逃生開關。
PINNED_ENV="${PINNED_ENV:-1}"
SYNC_PID=""

die() { echo "FATAL: $*" >&2; exit 1; }

# 交付 profile。**09-01 起預設 ＝ rankB_deliver_v090**（D104）：完整 drill 四閘門全過、
# 端到端產出對 sub_v090 逐列 100% 相同 ⇒ 依事前授權定案。
# 要跑舊的單一窗基準鏈：`DELIVER_PROFILE=rankB_deliver_v078 bash <本腳本>`。
PROFILE="${DELIVER_PROFILE:-rankB_deliver_v090}"
case "$PROFILE" in
  rankB_deliver_v078) REF_SUB=sub_v078_thirdleg_deadzone.csv; SELFTEST_GATE="自測 (c) 通過"
                      SELFKILL_MIN=240 ;;
  rankB_deliver_v090) REF_SUB=sub_v090_cropwin_medoid.csv;    SELFTEST_GATE="自測 (d) 通過"
                      SELFKILL_MIN=400 ;;
  *) die "未知 DELIVER_PROFILE=$PROFILE（只接受 rankB_deliver_v078／rankB_deliver_v090）" ;;
esac
echo "交付 profile = $PROFILE（參照檔 $REF_SUB、閘門「$SELFTEST_GATE」、自毀 $SELFKILL_MIN 分）"

# HF_TOKEN：gated facebook/sam3 需要。本機存放於 ~/.cache/huggingface/token，
# 部署時用 lambda_deploy.sh 推到箱上 /home/ubuntu/.hf_token 再 export（chmod 600）。
# 若 token 失效，G16 的 gDrive 備援可讓本階段不依賴 HF（見下方下載區塊）。
: "${HF_TOKEN:?HF_TOKEN is required (gated facebook/sam3); export it before running}"
sudo test -x /root/rearm_selfkill.sh \
  || die "缺 /root/rearm_selfkill.sh：cloud-init selfkill backstop 未就位（本腳本不自掛計時器）"
sudo /root/rearm_selfkill.sh "$SELFKILL_MIN"
test -d "$SRC" || die "repo 不在 $REPO（開機流程須把 3_src 放這裡）"

# 🚨 GPU 健康閘門（08-31 事故換來）：抽到的 A10 有 64 個未修正 ECC 錯誤
# ＋ Xid 48／Xid 64「All reserved rows for bank are remapped」＝顯存壞且備用列已用盡。
# 症狀是 track_t1.py 跑到一半 SIGABRT（terminate called without an active exception），
# **極易被誤判成自己的程式或參數有問題**——08-31 就誤判了一次，燒掉 40 分鐘與 $0.75。
# ⚠️ Lambda 的 instance id 是「本次租用」的編號，壞卡無法列黑名單，只能每次開機現驗。
ECC=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d " ")
case "$ECC" in
  0|"[N/A]"|"N/A"|"") echo "$(date -Is) GPU_ECC_OK ecc=${ECC:-none}" ;;
  *) echo "FATAL: GPU 有 $ECC 個未修正 ECC 錯誤 ⇒ 壞卡，terminate 換一台（勿在此機除錯）" >&2
     exit 1 ;;
esac

# 🚨 09-01 事故：跑完後機器多活了快 2 小時，因為 finish 的最後一次 rclone 要傳
#   **37,976 個檔案／624 MB**——其中絕大多數是三組窗的 crop 幀（`offline_two_pass/frames_*`
#   與各 crop track 目錄下的中繼檔）。那些幀是**可從 meta ＋ full CSV 重新產生的**，
#   沒有溯源價值，卻讓每次 run 的尾巴多燒 1–2 小時 GPU 機時。
#   ⚠️ 排掉它們之後，`rclone check --one-way` 才會在幾分鐘內完成、finish 才會準時把自毀縮到 5 分。
SYNC_SKIP=(--exclude 'run/offline_two_pass/frames_*/**')

sync_progress() {
  while true; do
    rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 \
      --exclude '*.part' --exclude '*.part/**' "${SYNC_SKIP[@]}" || true
    sleep 60
  done
}

finish() {
  RUN_RC=$?
  trap - EXIT
  set +e
  [ -n "$SYNC_PID" ] && { kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null; }
  echo "$(date -Is) finish rc=$RUN_RC" | tee -a "$WORK/finish.txt"
  SYNC_RC=0
  if command -v rclone >/dev/null 2>&1 && [ -f "$RCLONE_CONFIG" ]; then
    rclone copy "$WORK" "$DEST" --transfers 8 --checkers 16 \
      --exclude '*.part' --exclude '*.part/**' "${SYNC_SKIP[@]}" || SYNC_RC=$?
    [ "$SYNC_RC" -eq 0 ] && { rclone check "$WORK" "$DEST" --one-way \
      --exclude '*.part' --exclude '*.part/**' "${SYNC_SKIP[@]}" || SYNC_RC=$?; }
  else
    SYNC_RC=127
  fi
  if [ "$SYNC_RC" -eq 0 ]; then
    sudo /root/rearm_selfkill.sh 5
  else
    echo "$(date -Is) REMOTE_VERIFY_FAILED rc=$SYNC_RC；selfkill 未縮短" | tee -a "$WORK/finish.txt"
    [ "$RUN_RC" -eq 0 ] && RUN_RC=3
  fi
  exit "$RUN_RC"
}
trap finish EXIT

mkdir -p "$WORK/logs" "$CKPT" "$FRAMES"
chmod 600 "$RCLONE_CONFIG"
echo "$(date -Is) SETUP_START" | tee "$WORK/timeline.txt"

command -v rclone >/dev/null 2>&1 || curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null
rclone lsf "$GDRIVE/1_data/packed" >/dev/null
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null

# ── 資料／權重（背景並行；D036：單一 tar，不逐檔 rclone）────────────────────
(
  set -euo pipefail
  rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" /home/ubuntu/test_fc_75.tar
  tar -xf /home/ubuntu/test_fc_75.tar -C "$FRAMES"
  test "$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 75
  rclone copyto "$GDRIVE/1_data/raw/sample_submisson.csv" "$SAMPLE"
  test -s "$SAMPLE"
  # selftest 專用：**永遠**是歷史 test75 sample，與 $SAMPLE 是否被換掉無關（G11）
  rclone copyto "$GDRIVE/1_data/raw/sample_submisson.csv" "$SELFTEST_SAMPLE"
  test -s "$SELFTEST_SAMPLE"
  echo DATA_READY
) > "$WORK/logs/data_setup.log" 2>&1 &
DATA_PID=$!

(
  set -euo pipefail
  rclone copyto "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" "$CKPT/sam2.1_hiera_large.pt"
  test "$(stat -c%s "$CKPT/sam2.1_hiera_large.pt")" -gt 800000000
  echo SAM21_READY
) > "$WORK/logs/sam21_download.log" 2>&1 &
SAM21_PID=$!

(
  set -euo pipefail
  # G16：gDrive 優先。gated HF 是單點故障（token 失效／access 被收回／Meta 改網址），
  # 而 9/7 只有三天窗口。兩條路都走 sha256 fail-closed，來源不影響正確性。
  if rclone copyto "$GDRIVE/4_models/pretrained/sam3.pt" "$CKPT/sam3.pt" 2>/dev/null \
     && echo "$SAM3_CKPT_SHA256  $CKPT/sam3.pt" | sha256sum -c - >/dev/null 2>&1; then
    echo "sam3.pt 取自 gDrive 備援（sha256 已驗）"
    echo "gDrive mirror: $GDRIVE/4_models/pretrained/sam3.pt (sha256 $SAM3_CKPT_SHA256; byte-identical to official gated https://huggingface.co/facebook/sam3/resolve/main/sam3.pt)" \
      > "$WORK/SAM3_WEIGHT_SOURCE.txt"
  else
    rm -f "$CKPT/sam3.pt"
  curl -fL -H "Authorization: Bearer $HF_TOKEN" \
    -o "$CKPT/sam3.pt" "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"
  echo "$SAM3_CKPT_SHA256  $CKPT/sam3.pt" | sha256sum -c -
  echo "official gated: https://huggingface.co/facebook/sam3/resolve/main/sam3.pt (sha256 $SAM3_CKPT_SHA256)" \
    > "$WORK/SAM3_WEIGHT_SOURCE.txt"
  # G16：把驗過雜湊的檔回填 gDrive，讓下次開機不必依賴 gated HF（token 失效／Meta 改權限就沒退路）。
  # 只在 gDrive 還沒有時上傳；sha256 已在上面驗過，不會把壞檔推上去。
  if ! rclone lsf "$GDRIVE/4_models/pretrained/sam3.pt" >/dev/null 2>&1; then
    rclone copyto "$CKPT/sam3.pt" "$GDRIVE/4_models/pretrained/sam3.pt" && \
      echo "gDrive backup created: 4_models/pretrained/sam3.pt" >> "$WORK/SAM3_WEIGHT_SOURCE.txt"
  fi
  fi
  echo SAM3_READY
) > "$WORK/logs/sam3_download.log" 2>&1 &
SAM3_PID=$!

# ── 兩個完全隔離的 venv（禁 --system-site-packages）───────────────────────
[ -x "$T1ENV/bin/python" ] || uv venv --python 3.12 "$T1ENV"
[ -d "$SAMURAI/.git" ] || git clone https://github.com/yangchris11/samurai.git "$SAMURAI"
git -C "$SAMURAI" fetch --depth 1 origin "$SAMURAI_SHA"
git -C "$SAMURAI" checkout --detach "$SAMURAI_SHA"
VIRTUAL_ENV="$T1ENV" uv pip install -q torch torchvision --torch-backend=auto
if [ "$PINNED_ENV" = 1 ]; then
  # G24：從 08-22 的完全解析鎖安裝（torch/nvidia-* 不在鎖裡，上一行已依 driver 裝好）
  VIRTUAL_ENV="$T1ENV" uv pip install -q -r "$SRC/requirements-t1env.txt"
  VIRTUAL_ENV="$T1ENV" uv pip install -q -e "$SAMURAI/sam2" pytest
else
  VIRTUAL_ENV="$T1ENV" uv pip install -q -e "$SAMURAI/sam2" \
    scipy loguru tqdm pandas pillow opencv-python-headless pytest
fi
"$PY1" -c "import torch,sam2,pandas; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

[ -x "$SAM3ENV/bin/python" ] || uv venv --python 3.12 "$SAM3ENV"
VIRTUAL_ENV="$SAM3ENV" uv pip install -q torch torchvision --torch-backend=auto
if [ "$PINNED_ENV" = 1 ]; then
  VIRTUAL_ENV="$SAM3ENV" uv pip install -q -r "$SRC/requirements-sam3env.txt"
  VIRTUAL_ENV="$SAM3ENV" uv pip install -q \
    "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}"
else
  VIRTUAL_ENV="$SAM3ENV" uv pip install -q \
    "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
    "numpy<2" ftfy==6.1.1 regex einops psutil scipy av pycocotools numba \
    python-rapidjson pandas pillow tqdm opencv-python-headless
fi
# D086：81+ 移除 pkg_resources，sam3/model_builder.py 在 module scope import 它。
# 鎖檔已是 80.10.2，但 sam3 的 git 安裝可能把它升上去 ⇒ 無論如何最後再壓一次。
VIRTUAL_ENV="$SAM3ENV" uv pip install -q "setuptools<81"
"$PY3" -c "import torch,sam3,numpy; assert torch.cuda.is_available(); assert numpy.__version__.startswith('1.'); print(torch.__version__, torch.cuda.get_device_name(0))"

for JOB_PID in "$DATA_PID" "$SAM21_PID" "$SAM3_PID"; do wait "$JOB_PID"; done
grep -q DATA_READY "$WORK/logs/data_setup.log"
grep -q SAM21_READY "$WORK/logs/sam21_download.log"
grep -q SAM3_READY "$WORK/logs/sam3_download.log"

sha256sum "$CKPT/sam2.1_hiera_large.pt" "$CKPT/sam3.pt" > "$WORK/checkpoint_sha256.txt"
git -C "$SAMURAI" rev-parse HEAD > "$WORK/samurai_commit.txt"
uv pip freeze --python "$PY1" > "$WORK/t1_freeze.txt"
uv pip freeze --python "$PY3" > "$WORK/sam3_freeze.txt"
nvidia-smi > "$WORK/nvidia_smi.txt" || true

sync_progress &
SYNC_PID=$!

# ── 階段 0：交付自測（D067(f)：判別工具必須先在已知答案的資料上自測）────────
# 在燒 1.7 小時 GPU 之前先證明 finalize 這一段仍位元級等於 v078（LB 0.71666）。
# 需要 5_outputs/submissions 的歷史檔 ＋ rankb_robust test75 第三腿 ⇒ 一併拉下來。
echo "$(date -Is) SELFTEST_START" | tee -a "$WORK/timeline.txt"
mkdir -p "$REPO"/5_outputs/submissions "$REPO"/1_data/raw
rclone copy "$GDRIVE/5_outputs/submissions" "$REPO"/5_outputs/submissions \
  --include 'sub_v023_*.csv' --include 'sub_v012_*.csv' --include 'sub_v049_*.csv' \
  --include 'sub_v056_*.csv' --include 'sub_v078_*.csv' --include 'sub_v090_*.csv' --transfers 8
rclone copy "$GDRIVE/5_outputs/rankb_robust_test75_20260822/run/full_sam3" \
  "$REPO"/5_outputs/rankb_robust_test75_20260822/run/full_sam3 \
  --include 'submission.csv' --transfers 4
# selftest (d)＝三窗共識鏈（D101）的位元級守護，需要 08-31 那次 run 的 A/B/C merged 輸出。
# 即使本次跑 v078 也拉——(d) 順帶守護 run_ensemble_medoid 這條共用工具鏈，約 6MB。
rclone copy "$GDRIVE/5_outputs/cropwindow_ensemble_20260831" \
  "$REPO"/5_outputs/cropwindow_ensemble_20260831 --max-depth 1 \
  --include 'A_main_merged.csv' --include 'B_main_merged.csv' --include 'C_main_merged.csv' \
  --include 'A_source_merged.csv' --include 'A_full_sam3.csv' --transfers 8
# G11：這裡餵給 selftest，必須用歷史 test75，**不可**用本次的 $SAMPLE
cp "$SELFTEST_SAMPLE" "$REPO"/1_data/raw/sample_submisson.csv
"$PY1" "$SRC/finalize_submission.py" --selftest 2>&1 | tee "$WORK/logs/selftest.log"
grep -q "$SELFTEST_GATE" "$WORK/logs/selftest.log" \
  || die "selftest 缺「$SELFTEST_GATE」——$PROFILE 的交付鏈未經位元級驗證，停工"
grep -q "自測 (c) 通過" "$WORK/logs/selftest.log" \
  || die "selftest (c) 未通過或被略過——單一窗基準鏈未經驗證，停工"
# (d) 只在真的要跑三窗共識時才是閘門。**不要無條件擋**：v078 根本不用 medoid 這條工具鏈，
# 若 gDrive 的 cropwindow 檔案缺一個就把 v078 的交付擋死，等於為順帶守護加了一條斷點。
if [ "$PROFILE" = rankB_deliver_v090 ]; then
  grep -q "自測 (d) 通過" "$WORK/logs/selftest.log" \
    || die "selftest (d) 未通過或被略過——三窗共識佈線未經驗證，停工"
fi
# pytest 在【只有 3_src 的交付機器】上的兩個必要條件（08-29 演練兩次實測換來，
# 本機測不出來——本機靠專案根的 pytest.ini `pythonpath = 3_src`，而那個檔從不隨 3_src 同步）：
#   1. PYTHONPATH=3_src —— 否則 `from hsot.io import ...` 在 collection 階段就 ModuleNotFoundError
#   2. 排除 test_e32b_pilot.py —— 它驗的是 08-24 已結案探針的產物一致性，需要
#      5_outputs/submissions/ 的歷史 CSV（sub_v067 等），交付機器上不存在也不該存在
# 兩者皆已在本機重現環境驗證：123 passed / 2 skipped / 0 failed。
(cd "$REPO" && PYTHONPATH="$SRC" "$PY1" -m pytest -q 3_src \
   --ignore=3_src/hsot/test_e32b_pilot.py) 2>&1 | tee "$WORK/logs/pytest.log"
echo "$(date -Is) SELFTEST_OK" | tee -a "$WORK/timeline.txt"

# ── 階段 1：dry-run（single entrypoint 的前置檢查必須全綠才准 --execute）─────
echo "$(date -Is) DRYRUN_START" | tee -a "$WORK/timeline.txt"
"$PY1" "$SRC/run_ranking_b.py" \
  --profile "$PROFILE" --allow-offline-two-pass \
  --frames-root "$FRAMES" --sample "$SAMPLE" \
  --work-dir "$WORK/run" --out "$WORK/final.csv" \
  --sam3-python "$PY3" --samurai-python "$PY1" \
  --sam3-ckpt "$CKPT/sam3.pt" \
  --samurai-dir "$SAMURAI" --samurai-ckpt "$CKPT/sam2.1_hiera_large.pt" \
  --sam3-source-revision "$SAM3_SHA" --samurai-source-revision "$SAMURAI_SHA" \
  2>&1 | tee "$WORK/logs/dryrun.log"
grep -q "BLOCK=0" "$WORK/logs/dryrun.log" || die "dry-run 有 BLOCK，停工（見 logs/dryrun.log）"

# ── 階段 2：正式執行（9/7 當天就是這一段）────────────────────────────────
echo "$(date -Is) INFERENCE_START" | tee -a "$WORK/timeline.txt"
"$PY1" "$SRC/run_ranking_b.py" \
  --profile "$PROFILE" --allow-offline-two-pass \
  --frames-root "$FRAMES" --sample "$SAMPLE" \
  --work-dir "$WORK/run" --out "$WORK/final.csv" \
  --sam3-python "$PY3" --samurai-python "$PY1" \
  --sam3-ckpt "$CKPT/sam3.pt" \
  --samurai-dir "$SAMURAI" --samurai-ckpt "$CKPT/sam2.1_hiera_large.pt" \
  --sam3-source-revision "$SAM3_SHA" --samurai-source-revision "$SAMURAI_SHA" \
  --execute 2>&1 | tee "$WORK/logs/execute.log"
test -s "$WORK/final.csv"
echo "$(date -Is) INFERENCE_DONE" | tee -a "$WORK/timeline.txt"

# ── 階段 3：演練專屬驗收（9/7 沒有 GT，這段只在演練跑）────────────────────
# 演練資料＝現行 test75 ⇒ 產物應與 $REF_SUB 高度一致。
# ⚠️ 09-01 更新：**實測是逐列 100% 相同**（26860/26860，平均 IoU 1.00000）。
# 舊註解寫「D074 實測約 13% 序列會發散、位元級不可得」——那是 08-16 演練 #3 的觀察，
# 已由 D097/D104 取代：這條鏈跨機器、跨日期、跨獨立安裝的套件集合都是位元級確定的。
# ⇒ **本段的合格標準因此可以拉高**：若逐列相同率明顯低於 100%，那是異常訊號，不是預期方差。
# （唯一的已知非位元級差異是行尾：本管線寫 LF，finalize_submission.write 寫 CRLF。）
"$PY1" - "$WORK/final.csv" "$REPO/5_outputs/submissions/$REF_SUB" \
  > "$WORK/drill_compare.txt" 2>&1 <<'PY' || true
import csv, sys
def load(p):
    with open(p, newline="") as fh:
        return {r["ID"]: tuple(float(r[c]) for c in ("x","y","width","height"))
                for r in csv.DictReader(fh)}
got, ref = load(sys.argv[1]), load(sys.argv[2])
common = set(got) & set(ref)
same = sum(1 for k in common if got[k] == ref[k])
def iou(a, b):
    ax2, ay2, bx2, by2 = a[0]+a[2], a[1]+a[3], b[0]+b[2], b[1]+b[3]
    iw = max(0.0, min(ax2,bx2)-max(a[0],b[0])); ih = max(0.0, min(ay2,by2)-max(a[1],b[1]))
    inter = iw*ih; union = a[2]*a[3] + b[2]*b[3] - inter
    return inter/union if union > 0 else 0.0
ious = [iou(got[k], ref[k]) for k in common]
seqs = {}
for k in common:
    seqs.setdefault(k.rsplit("_",1)[0], []).append(iou(got[k], ref[k]))
bad = sorted(((sum(v)/len(v), s) for s, v in seqs.items()))[:5]
print(f"ID 覆蓋: got={len(got)} ref={len(ref)} common={len(common)}")
print(f"逐列完全相同: {same}/{len(common)} ({100*same/max(len(common),1):.1f}%)")
print(f"對參照檔（{sys.argv[2].rsplit(chr(47),1)[-1]}）的平均 IoU: {sum(ious)/max(len(ious),1):.5f}")
print("最發散的 5 支序列（09-01 實測全部 1.0000；低於 1.0 即為異常訊號，非預期方差）:")
for m, s in bad:
    print(f"  {s}: {m:.4f}")
PY
cat "$WORK/drill_compare.txt"
echo "$(date -Is) DRILL_DONE" | tee -a "$WORK/timeline.txt"
