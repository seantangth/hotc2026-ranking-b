#!/usr/bin/env bash
# 09-08 交付驗證：rankB_deliver_v090 在【主辦方 09-07 釋出的官方 75 支】上端到端跑一次。
#
# 由 run_deliver_v078_drill_lambda_v1.sh 派生（該腳本已演練 6 次）。**只改三處**：
#   1. 資料＝官方原始佈局 tar（HSI-<模態>-Falsecolor/<seq>/）→ 上箱後跑 ingestion
#   2. SAMPLE＝ingestion 的 --write-sample 產物（官方未附 sample_submission.csv）
#   3. 階段 3 驗收＝對官方 GT 算 Success AUC（不再是對 test75 參照檔的位元級比對——
#      本次是全新序列，沒有參照檔；GT 只用於印分數，不回饋進管線任何決策）
# 其餘（環境、selftest (c)/(d)、pytest、dry-run 閘門、自毀、同步）一律不動。
#
# 🎯 本次的目的是【證明交付包在官方資料上跑得起來】，不是衝分數：
#    REPO 必須是 dist/ 解開的交付包，不是 repo clone（09-06 只驗到 dry-run 為止）。
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
DRILL_DEST="${DRILL_DEST:-deliver_official75_20260908}"
DEST="$GDRIVE/5_outputs/$DRILL_DEST"
# ⚠️ repo 根【不可】叫 hsot：套件本身是 3_src/hsot 且無 __init__.py（namespace
#    package），同名時 pytest 的 `import hsot` 會解析到 repo 根而非套件。
#    08-29 演練實測撞到（rc=2, "cannot import name crop_rerun from hsot
#    (unknown location)"）。9/7 沿用此路徑名。
REPO=/home/ubuntu/hsot_repo
SRC="$REPO/3_src"
WORK="/home/ubuntu/$DRILL_DEST"
FRAMES=/home/ubuntu/frames_official75
RAW_OFFICIAL=/home/ubuntu/raw_official          # 官方原始佈局解壓處（HSI-*-Falsecolor/）
OFFICIAL_TAR=rankB_sample75_fc_20260907.tar
OFFICIAL_DIR=rankB_sample75_fc_20260907          # tar 內的頂層目錄名
GT_CSV=/home/ubuntu/rankB_sample75_gt_20260907.csv
CKPT=/home/ubuntu/ckpt
SAMURAI=/home/ubuntu/samurai
T1ENV=/home/ubuntu/t1env
SAM3ENV=/home/ubuntu/sam3env
PY1="$T1ENV/bin/python"
PY3="$SAM3ENV/bin/python"
SAMURAI_SHA=76ba195984892b0d1e3db5d9c9f90bb62175680a
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
SAM3_CKPT_SHA256=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
# 官方 09-07 的 ranking 測試序列【沒有附 sample_submission.csv】⇒ 由 ingestion 產生
# （prep 的 --write-sample，格式與 Ranking A 官方檔同型，經 _sample_contract 驗過）。
SAMPLE="$FRAMES/sample_submission.csv"             # 本次要跑的（ingestion 產物）
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
                      # 09-06 封板 drill 於 test75（26,860 幀）實測 4h40m；本次 30,338 幀
                      # ＝1.13× ⇒ 推論約 5h15m，加環境/selftest/同步約 6h ⇒ 480 分留 2h 緩衝。
                      SELFKILL_MIN=480 ;;
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
  # 官方原始佈局（D036：單一 tar，不逐檔 rclone——Drive 逐檔限速，30k 檔要 1 小時以上）
  rclone copyto "$GDRIVE/1_data/packed/$OFFICIAL_TAR" "/home/ubuntu/$OFFICIAL_TAR"
  mkdir -p "$RAW_OFFICIAL"
  tar -xf "/home/ubuntu/$OFFICIAL_TAR" -C "$RAW_OFFICIAL"
  # 解壓後是 <OFFICIAL_DIR>/HSI-{NIR,RedNIR,VIS}[-Falsecolor]/<seq>/…（不是 75 個頂層目錄，
  # 舊腳本那條「頂層恰 75 目錄」的檢查在這裡不成立；契約改由 ingestion 驗，見階段 0.5）
  test -d "$RAW_OFFICIAL/$OFFICIAL_DIR"
  # GT：只給階段 3 的分數用，**不進管線**（管線讀的是 $FRAMES，裡面沒有 GT）
  rclone copyto "$GDRIVE/1_data/gt/rankB_sample75_gt_20260907.csv" "$GT_CSV"
  test -s "$GT_CSV"
  # selftest 專用：**永遠**是歷史 test75 sample，與本次 $SAMPLE 無關（G11）
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

# ── 階段 0a：官方原始佈局 → frames_root（09-07 新增；6 次演練從未跑過這一步）──
# 官方把模態放在**上層資料夾名**（HSI-NIR-Falsecolor/blackball2/），序列目錄本身沒有
# nir- 前綴；而 finalize 的 quality_head_v1.modality() 對無前綴名一律回 VIS
# ⇒ 不經這支腳本直接把 --frames-root 指向官方資料夾，RedNIR 品質頭永不觸發、
#   兩顆頭的模態 one-hot 全錯，而且**沒有任何 BLOCK**（09-07 實測；已補 preflight 檢查）。
echo "$(date -Is) INGEST_START" | tee -a "$WORK/timeline.txt"
"$PY1" "$SRC/prep/prep_rankingb_frames_v1.py" \
  --source "$RAW_OFFICIAL/$OFFICIAL_DIR" \
  --frames-out "$FRAMES" \
  --write-sample "$SAMPLE" 2>&1 | tee "$WORK/logs/ingest.log"
grep -q "ingestion 契約全過" "$WORK/logs/ingest.log" \
  || die "ingestion 未全過——見 logs/ingest.log，停工（不要硬跑：座標慣例錯會燒掉整輪）"
test "$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 75 \
  || die "frames_root 不是 75 支"
test -s "$SAMPLE"
cp "$FRAMES/INGEST_MANIFEST.json" "$WORK/INGEST_MANIFEST.json"
echo "$(date -Is) INGEST_OK" | tee -a "$WORK/timeline.txt"

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

# ── 階段 3：對官方 GT 算分（本次是全新序列，沒有位元級參照檔可比）──────────
# ⚠️ GT 只用來【印分數】，不回饋進管線任何決策——官方明文這 75 支僅供運行驗證、不評分，
#    正式 Ranking B 在私有集上跑。這裡量的是「管線在官方新資料上的行為是否合理」，
#    以及 D072 懸案（座標校正 +1px 該不該留）的第一個有 GT 讀數。
echo "$(date -Is) SCORE_START" | tee -a "$WORK/timeline.txt"
PYTHONPATH="$SRC" "$PY1" "$SRC/hsot/eval.py" "$WORK/final.csv" "$GT_CSV" --per-seq \
  > "$WORK/score_main.txt" 2>&1 || echo "評分失敗（不影響交付產物）" >> "$WORK/score_main.txt"
cat "$WORK/score_main.txt" | head -8

# D072：三個 --corr 變體（CPU、各約 1 分鐘）。用 finalize 的**同一份中間產物**重跑，
# 唯一變因＝校正模式；不做首幀還原，故三者可互比但不等於最終檔（首幀在三者都是 exact init）。
VAR_DIR="$WORK/corr_variants"
mkdir -p "$VAR_DIR"
for CORR in both top-only none; do
  "$PY1" "$SRC/finalize_submission.py" \
    --main "$WORK/run/offline_two_pass/main_merged.csv" \
    --source "$WORK/run/offline_two_pass/source_merged.csv" \
    --sample "$SAMPLE" --out "$VAR_DIR/corr_$CORR.csv" \
    --corr "$CORR" --K 6 --qhead v056 \
    --qhead-weights "$SRC/hsot/qhead_weights_v056.npz" \
    --selector v2 --selector-weights "$SRC/hsot/selector_weights_v2.npz" \
    --third-leg "$WORK/run/full_sam3/submission.csv" \
    > "$VAR_DIR/finalize_$CORR.log" 2>&1 \
    && PYTHONPATH="$SRC" "$PY1" "$SRC/hsot/eval.py" "$VAR_DIR/corr_$CORR.csv" "$GT_CSV" \
       > "$VAR_DIR/score_$CORR.txt" 2>&1 \
    || echo "corr=$CORR 變體失敗" > "$VAR_DIR/score_$CORR.txt"
  echo "--- corr=$CORR ---" >> "$WORK/score_corr_variants.txt"
  head -3 "$VAR_DIR/score_$CORR.txt" >> "$WORK/score_corr_variants.txt"
done
cat "$WORK/score_corr_variants.txt"
echo "$(date -Is) DRILL_DONE" | tee -a "$WORK/timeline.txt"
