#!/usr/bin/env bash
# ============================================================================
# setup_q100_crop_pilot_lambda_v1.sh — crop JPEG q95 vs q100/4:4:4 paired pilot
#
# 用法（一次只選一個 mode；兩 mode 不共用 tracker cache）：
#   bash setup_q100_crop_pilot_lambda_v1.sh                  # rankA-historical
#   bash setup_q100_crop_pilot_lambda_v1.sh rankB-robust
#
# 實驗契約：
#   1. 只 fresh 重跑 crop legs；full SAM3/SAMURAI 使用既有 frozen CSV。
#   2. legacy-q95 與 q100-444 同機、共用 venv/權重/code/full pair。
#   3. 兩 profile 各自從原圖裁切，但 window geometry 必須逐欄相等。
#   4. VAL 先跑 2 profiles × 2 backends，以官方 Success AUC paired verdict。
#   5. GO 只是災難閘；通過才在 TEST 同機 fresh 跑 q95＋q100 paired，不代表已證明加分。
#   6. 不含 Kaggle submission，只輸出候選 CSV。
#
# mode 明確分流：
#   rankA-historical（預設／本輪主要）
#     E15 + 歷史 E02；SAM3 不加顯式 --sam3-eval；SAMURAI legacy cross-seq KF；
#     primary = corr both + K6 + qhead none。
#     v056 qhead 另輸出 *DIAGNOSTIC*，永不參與 GO。
#   rankB-robust
#     E15 + E02-kfreset（VAL）／08-22 robust full pair（TEST）；
#     SAM3 eval；SAMURAI per-sequence reset；corr top-only + K6 + qhead none。
#
# Lambda：不 launch、不另掛計時器。啟動前要求 cloud-init selfkill 已存在；
# 每階段同步 gDrive，EXIT trap 做 rclone check，驗證成功才把 backstop 縮成 5 分鐘。
# ============================================================================
set -euo pipefail
umask 077

export PYTHONUNBUFFERED=1
export PATH="/home/ubuntu/.local/bin:$PATH"
export RCLONE_CONFIG="${RCLONE_CONFIG:-/home/ubuntu/rclone.conf}"
KEEP_INSTANCE="${KEEP_INSTANCE:-0}"
case "$KEEP_INSTANCE" in 0|1) ;; *) echo "KEEP_INSTANCE 只能是 0 或 1" >&2; exit 2 ;; esac

MODE="${1:-rankA-historical}"
case "$MODE" in
  rankA-historical|rankB-robust) ;;
  *) echo "用法：$0 [rankA-historical|rankB-robust]" >&2; exit 2 ;;
esac

GDRIVE="gdrive:WHISPERS_2026_HyperSOT"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
case "$RUN_ID" in *[!A-Za-z0-9._-]*) echo "RUN_ID 含非法字元：$RUN_ID" >&2; exit 2 ;; esac
DEST="$GDRIVE/5_outputs/q100_crop_pilot_20260823/$MODE/$RUN_ID"
MODE_TAG="${MODE//-/_}"
WORK="${WORK:-/home/ubuntu/q100_crop_pilot_v1_${MODE_TAG}_${RUN_ID}}"
case "$WORK" in /*) ;; *) echo "WORK 必須是絕對路徑：$WORK" >&2; exit 2 ;; esac
RESULTS="$WORK/results"
SCRATCH="$WORK/scratch"
CODE="$WORK/code"
CKPT="$WORK/ckpt"
SAMURAI="$WORK/vendor/samurai"
T1ENV="$WORK/env/t1"
SAM3ENV="$WORK/env/sam3"
PY1="$T1ENV/bin/python"
PY3="$SAM3ENV/bin/python"

SAMURAI_SHA="76ba195984892b0d1e3db5d9c9f90bb62175680a"
SAM3_SHA="96914d2425f90a64f45ca977c2b5165418099543"
SAM21_WEIGHT_SHA="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
SAM3_WEIGHT_SHA="9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
QHEAD_WEIGHT_SHA="522bd4d102be6e93f0c05bdc362bc7b9480dae4c541a6d4d4126bc40fce7075e"
TRACK_T1_SHA="a13883137744507617d39ecb4bcf1e2d57d1176868a049315f0c3129eadda172"
FINALIZE_SHA="303469bde214f8699b349c0ac34eb58b3d0c2fc456b69c77d241aa6beef2600d"
CROP_RERUN_SHA="e35f235c37046e8d0867e57b366daaa8954fc44376c246f364e0010857c22d3e"
EVAL_SHA="3ac486c0b119020aa0433be493b5a8a711a123ee853b94698f06f2b2ae9ecfaf"
QUALITY_HEAD_SHA="b54d6e498664b2ffe0107835bc4ebf2214c025c06d2d2029ba6f2676f5816bd6"
VAL_MAIN_SHA="cc0d710ef8bfccc96fd8aa27ff8d461c72c6277fd72019dbc83b52a17b2c320c"
VAL_LIST_SHA="79558db33fc8c8dcdc73bc84e08b3b29563cbf8975dbbf35e3cfcdaea879b254"
VAL_GT_SHA="b4be4d3fd5f4004ab97e751850ded4198acf58bd25ed9cae8b5148b228ed3e2b"
TEST_SAMPLE_SHA="3aef22070aca0c83fb9866df806f7b3e8c66abc3dcddda3c89e27e2fdb35a866"
VAL_TAR_SHA="46043866ec27fc82667eb49a0a3f989e1c79e4fdfee048a85c443efd3b70fe9f"
TEST_TAR_SHA="b11dbfe146c07e386e8f7dbe008311bbd5d1eacdf075ab16751a91470877598c"

# 事前寫死的 DRY／GO 判準。
DRY_MIN_SELECTED_SEQS=1
DRY_MAX_INCOMPLETE_RUNS=0
GO_MIN_ALL_POOLED_DELTA=-0.020
GO_MIN_SELECTED_POOLED_DELTA=-0.050
GO_MIN_LEGACY_POOLED_AUC=0.600
GO_STABLE_AUC_THRESHOLD=0.80
GO_STABLE_BAD_DELTA=-0.050
GO_MAX_STABLE_BAD_FRACTION=0.20
GO_MAX_UNEXPECTED_CHANGED_SEQS=0

VAL_FRAMES_REMOTE="$GDRIVE/1_data/packed/t1val_fc_65.tar"
VAL_GT_REMOTE="$GDRIVE/1_data/raw/2026training.csv"
VAL_LIST_REMOTE="$GDRIVE/1_data/val_split_v1.txt"
VAL_MAIN_REMOTE="$GDRIVE/5_outputs/e15_sam3_20260806/submission_val65.csv"
TEST_FRAMES_REMOTE="$GDRIVE/1_data/packed/t1test_fc_75.tar"
TEST_SAMPLE_REMOTE="$GDRIVE/1_data/raw/sample_submisson.csv"

if [ "$MODE" = rankA-historical ]; then
  VAL_SOURCE_REMOTE="$GDRIVE/5_outputs/t1_rerun_20260805/submission.csv"
  TEST_MAIN_REMOTE="$GDRIVE/5_outputs/submissions/sub_v006_e15sam3.csv"
  TEST_SOURCE_REMOTE="$GDRIVE/5_outputs/submissions/exp003_samurai_large.csv"
  VAL_SOURCE_SHA="d1fc0418f54ce06ea2643da591856c294c1fa858d607f9b1499ff1dc517577fd"
  TEST_MAIN_SHA="85587a55733d920a1a8687cce19c4fde6b77745c75e06f42096f936a89489636"
  TEST_SOURCE_SHA="9a622bdf8318a55c33ef21850d6256a9991913ad972d8d1e89dfd370b42bdbfc"
  CORR_MODE=both
  MAKE_QHEAD_DIAGNOSTIC=1
  SAM3_MODE_LABEL=legacy-no-explicit-eval-flag
  SAMURAI_MODE_LABEL=legacy-cross-seq-kf
  SAM3_MODE_FLAG=""
  SAMURAI_MODE_FLAG=--samurai-legacy-cross-seq-kf
else
  VAL_SOURCE_REMOTE="$GDRIVE/5_outputs/e02_kfreset_20260806/submission_val65.csv"
  TEST_MAIN_REMOTE="$GDRIVE/5_outputs/rankb_robust_test75_20260822/run/full_sam3/submission.csv"
  TEST_SOURCE_REMOTE="$GDRIVE/5_outputs/rankb_robust_test75_20260822/run/full_samurai/submission.csv"
  VAL_SOURCE_SHA="f1feebf62d978a4cfda9afe3ec7730809fce9cdbb080e84723c210464102d8b7"
  TEST_MAIN_SHA="85587a55733d920a1a8687cce19c4fde6b77745c75e06f42096f936a89489636"
  TEST_SOURCE_SHA="b40b261711725850a23a9b40329d672bb88e4a3551e668c3c36541b066bfec86"
  CORR_MODE=top-only
  MAKE_QHEAD_DIAGNOSTIC=0
  SAM3_MODE_LABEL=eval
  SAMURAI_MODE_LABEL=per-sequence-reset
  SAM3_MODE_FLAG=--sam3-eval
  SAMURAI_MODE_FLAG=--samurai-reset-kf
fi

# 純 CPU/static 驗收：不建 WORK、不碰 gDrive/backstop/GPU。供部署前兩 mode 各跑一次。
if [ "${PILOT_VALIDATE_ONLY:-0}" = 1 ]; then
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  SRC_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
  bash -n "${BASH_SOURCE[0]}"
  grep -q -- '--jpeg-profile "\$PROFILE"' "${BASH_SOURCE[0]}"
  grep -q -- '--jpeg-profile q100-444' "${BASH_SOURCE[0]}"
  FORBIDDEN_ENDPOINT='/instance-operations/'"launch"
  if grep -q "$FORBIDDEN_ENDPOINT" "${BASH_SOURCE[0]}"; then
    echo "🚨 validate-only: 腳本含 launch endpoint" >&2
    exit 2
  fi
  PYTHONPATH="$SRC_ROOT" python3 - <<'PY'
from hsot.crop_rerun import _jpeg_encoding
assert _jpeg_encoding("legacy-q95")[1] == {"quality": 95}
assert _jpeg_encoding("q100-444")[1] == {"quality": 100, "subsampling": 0}
PY
  if [ "$MODE" = rankA-historical ]; then
    [ "$CORR_MODE" = both ] && [ "$MAKE_QHEAD_DIAGNOSTIC" -eq 1 ]
    [ "$SAM3_MODE_FLAG" = "" ]
    [ "$SAMURAI_MODE_FLAG" = "--samurai-legacy-cross-seq-kf" ]
  else
    [ "$CORR_MODE" = top-only ] && [ "$MAKE_QHEAD_DIAGNOSTIC" -eq 0 ]
    [ "$SAM3_MODE_FLAG" = "--sam3-eval" ]
    [ "$SAMURAI_MODE_FLAG" = "--samurai-reset-kf" ]
  fi
  echo "VALIDATE_ONLY_PASS mode=$MODE corr=$CORR_MODE sam3=$SAM3_MODE_LABEL samurai=$SAMURAI_MODE_LABEL keep=$KEEP_INSTANCE"
  exit 0
fi

mkdir -p "$RESULTS" "$SCRATCH" "$CODE/hsot" "$CKPT" "$WORK/vendor" "$WORK/env"
if [ -e "$WORK/.q100_pilot_started" ]; then
  echo "🚨 WORK 已有 run marker，拒絕混入舊 cache：$WORK" >&2
  echo "請指定全新的 WORK=/home/ubuntu/..." >&2
  exit 2
fi
touch "$WORK/.q100_pilot_started"

TIMELINE="$RESULTS/timeline.txt"
STATUS="$RESULTS/status.txt"
: > "$TIMELINE"
echo STARTING > "$STATUS"

mark() {
  printf '%s %s\n' "$1" "$(date +%s)" >> "$TIMELINE"
  printf '⏱  %s @ %s\n' "$1" "$(date -Is)"
}

hash_results() {
  local tmp="$RESULTS/.artifact_sha256.$$.part"
  if ! (
    cd "$RESULTS"
    find . -type f \
      ! -name artifact_sha256.txt ! -name pilot.log \
      ! -name status.txt ! -name finish.txt ! -name '*.part' -print0 \
      | LC_ALL=C sort -z | xargs -0 -r sha256sum > "$tmp"
  ); then
    rm -f "$tmp"
    return 1
  fi
  mv "$tmp" "$RESULTS/artifact_sha256.txt" || return 1
  (cd "$RESULTS" && sha256sum -c artifact_sha256.txt >/dev/null) || return 1
}

sync_stage() {
  local stage="$1"
  printf '%s\n' "$stage" > "$STATUS"
  hash_results
  rclone copy "$RESULTS" "$DEST/results" --transfers 8 --checkers 16 \
    --exclude pilot.log --exclude status.txt
  rclone check "$RESULTS" "$DEST/results" --one-way \
    --exclude pilot.log --exclude status.txt
  rclone copyto "$STATUS" "$DEST/results/status.txt"
  printf '☁️  stage synced+verified: %s\n' "$stage"
}

finish() {
  local run_rc=$?
  trap - EXIT
  trap - ERR
  set +eu
  local pid active_pids
  active_pids=$(jobs -pr)
  for pid in $active_pids; do kill "$pid" 2>/dev/null; done
  for pid in $active_pids; do wait "$pid" 2>/dev/null; done
  printf '%s finish rc=%s mode=%s\n' "$(date -Is)" "$run_rc" "$MODE" \
    > "$RESULTS/finish.txt"
  if [ "$KEEP_INSTANCE" -eq 1 ]; then
    printf '%s KEEP_INSTANCE=1; 保留既有 cloud-init deadline，由共同 orchestrator 控制\n' \
      "$(date -Is)" >> "$RESULTS/finish.txt"
  fi
  local sync_rc=0
  hash_results || sync_rc=125
  if [ "$sync_rc" -eq 0 ] && command -v rclone >/dev/null 2>&1 && [ -f "$RCLONE_CONFIG" ]; then
    rclone copy "$RESULTS" "$DEST/results" --transfers 8 --checkers 16 \
      --exclude pilot.log --exclude status.txt >/dev/null 2>&1 || sync_rc=$?
    if [ "$sync_rc" -eq 0 ]; then
      rclone check "$RESULTS" "$DEST/results" --one-way \
        --exclude pilot.log --exclude status.txt >/dev/null 2>&1 || sync_rc=$?
    fi
    if [ "$sync_rc" -eq 0 ]; then
      rclone copyto "$STATUS" "$DEST/results/status.txt" >/dev/null 2>&1 || sync_rc=$?
    fi
    if [ "$sync_rc" -eq 0 ] && [ -f "$RESULTS/pilot.log" ]; then
      rclone copyto "$RESULTS/pilot.log" "$DEST/results/pilot.log" \
        >/dev/null 2>&1 || sync_rc=$?
    fi
  else
    sync_rc=127
  fi
  if [ "$sync_rc" -ne 0 ]; then
    printf '%s REMOTE_VERIFY_FAILED rc=%s; backstop 未縮短\n' \
      "$(date -Is)" "$sync_rc" >> "$RESULTS/finish.txt"
    if [ "$run_rc" -eq 0 ]; then run_rc=3; fi
  elif [ "$KEEP_INSTANCE" -eq 0 ]; then
    local rearm_ok=0 attempt
    for attempt in 1 2; do
      sudo /root/rearm_selfkill.sh 5 >/dev/null 2>&1
      sleep 1
      if sudo pgrep -af '[s]elfkill_core.sh' | grep -q 'selfkill_core.sh 300'; then
        rearm_ok=1; break
      fi
    done
    if [ "$rearm_ok" -ne 1 ]; then
      sudo /root/rearm_selfkill.sh 15 >/dev/null 2>&1
      sleep 1
      local fallback_timer=FAILED
      if sudo pgrep -af '[s]elfkill_core.sh' | grep -q 'selfkill_core.sh 900'; then
        fallback_timer=ARMED_15M
      fi
      printf '%s rearm_selfkill 5m 驗證失敗；fallback=%s\n' \
        "$(date -Is)" "$fallback_timer" >> "$RESULTS/finish.txt"
      echo FAILED_REARM > "$STATUS"
      rclone copyto "$RESULTS/finish.txt" "$DEST/results/finish.txt" >/dev/null 2>&1
      rclone copyto "$STATUS" "$DEST/results/status.txt" >/dev/null 2>&1
      run_rc=4
    fi
  else
    : # 共同 orchestrator 接管；不改 cloud-init deadline。
  fi
  exit "$run_rc"
}

die() {
  printf '🚨 %s\n' "$*" >&2
  echo FAILED > "$STATUS"
  exit 1
}

assert_sha() {
  local file="$1" expected="$2" label="$3" got
  got=$(sha256sum "$file"); got=${got%% *}
  [ "$got" = "$expected" ] || die "$label SHA 不符：got=$got expected=$expected"
}

trap finish EXIT
trap 'rc=$?; echo FAILED > "$STATUS"; echo "🚨 死於第 $LINENO 行 exit=$rc" >&2; exit "$rc"' ERR
exec >> "$RESULTS/pilot.log" 2>&1

mark PILOT_START
echo "=== q100 crop pilot v1 | mode=$MODE | $(date -Is) ==="
echo "WORK=$WORK"
echo "DEST=$DEST"
printf '%s\n' \
  "mode=$MODE" \
  "val_main=$VAL_MAIN_REMOTE" \
  "val_source=$VAL_SOURCE_REMOTE" \
  "test_main=$TEST_MAIN_REMOTE" \
  "test_source=$TEST_SOURCE_REMOTE" \
  "run_id=$RUN_ID" \
  > "$RESULTS/input_sources.txt"

# 外部 cloud-init backstop 是硬前置；本腳本不自行 launch／掛第二套 timer。
sudo test -x /root/rearm_selfkill.sh \
  || die "缺 /root/rearm_selfkill.sh：cloud-init backstop 未就位"
sudo pgrep -af '[s]elfkill_core.sh' > "$RESULTS/backstop_preflight.txt" \
  || die "找不到 selfkill_core.sh process：拒絕無護欄執行"
test -f "$RCLONE_CONFIG" || die "缺 rclone config：$RCLONE_CONFIG"
chmod 600 "$RCLONE_CONFIG"
if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null
fi
rclone lsf "$GDRIVE/1_data/packed" >/dev/null || die "rclone 讀不到專案 gDrive"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi

python3 - "$RESULTS/protocol.json" "$MODE" "$CORR_MODE" "$KEEP_INSTANCE" "$RUN_ID" \
  "$SAM3_MODE_LABEL" "$SAMURAI_MODE_LABEL" \
  "$DRY_MIN_SELECTED_SEQS" "$DRY_MAX_INCOMPLETE_RUNS" \
  "$GO_MIN_ALL_POOLED_DELTA" "$GO_MIN_SELECTED_POOLED_DELTA" \
  "$GO_MIN_LEGACY_POOLED_AUC" \
  "$GO_STABLE_AUC_THRESHOLD" "$GO_STABLE_BAD_DELTA" \
  "$GO_MAX_STABLE_BAD_FRACTION" "$GO_MAX_UNEXPECTED_CHANGED_SEQS" <<'PY'
import json, sys
(out, mode, corr, keep_instance, run_id, sam3_mode, samurai_mode, dry_min, dry_bad,
 go_all, go_sel, min_legacy, stable_auc, stable_bad, max_bad_frac, max_unexpected) = sys.argv[1:]
doc = {
    "schema": 1, "experiment": "q100_crop_pilot_v1", "mode": mode, "run_id": run_id,
    "paired_variable_only": "crop JPEG encoding: legacy-q95 vs q100-444",
    "full_pair_reused": True, "fresh_crop_inference": True,
    "same_machine_and_environments": True,
    "window_geometry_must_match_exactly": True,
    "production_postprocess": {"corr": corr, "K": 6, "qhead": "none",
                               "preserve_first_frame": True},
    "tracker_modes": {"sam3": sam3_mode, "samurai": samurai_mode},
    "qhead_v056": ("diagnostic-only; never used for GO"
                    if mode == "rankA-historical" else "not generated"),
    "dry_gate": {"min_selected_sequences": int(dry_min),
                 "max_incomplete_tracker_runs": int(dry_bad),
                 "profiles": ["legacy-q95", "q100-444"],
                 "backends": ["sam3", "samurai"]},
    "go_disaster_gate": {
        "min_all_val_pooled_delta": float(go_all),
        "min_crop_selected_pooled_delta": float(go_sel),
        "min_legacy_control_pooled_auc": float(min_legacy),
        "stable_auc_threshold": float(stable_auc),
        "stable_bad_delta": float(stable_bad),
        "max_stable_bad_fraction": float(max_bad_frac),
        "max_unexpected_changed_sequences": int(max_unexpected)},
    "go_note": "No-catastrophe only; passing is not evidence of positive gain.",
    "keep_instance_for_orchestrator": bool(int(keep_instance)),
    "kaggle_submission": False,
}
open(out, "w").write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
PY
cp "$0" "$RESULTS/$(basename "$0")"

# 背景下載 VAL、frozen full pair、權重與本輪程式碼。
mkdir -p "$SCRATCH/val_extract" "$SCRATCH/inputs" "$RESULTS/setup"
(
  set -euo pipefail
  rclone copyto "$VAL_FRAMES_REMOTE" "$SCRATCH/inputs/t1val_fc_65.tar"
  echo "$VAL_TAR_SHA  $SCRATCH/inputs/t1val_fc_65.tar" | sha256sum -c -
  tar -xf "$SCRATCH/inputs/t1val_fc_65.tar" -C "$SCRATCH/val_extract"
  echo VAL_FRAMES_READY
) > "$RESULTS/setup/val_frames.log" 2>&1 &
VAL_DATA_PID=$!
(
  set -euo pipefail
  rclone copyto "$VAL_GT_REMOTE" "$SCRATCH/inputs/2026training.csv"
  rclone copyto "$VAL_LIST_REMOTE" "$SCRATCH/inputs/val_split_v1.txt"
  rclone copyto "$VAL_MAIN_REMOTE" "$SCRATCH/inputs/val_full_main.csv"
  rclone copyto "$VAL_SOURCE_REMOTE" "$SCRATCH/inputs/val_full_source.csv"
  echo VAL_INPUTS_READY
) > "$RESULTS/setup/val_inputs.log" 2>&1 &
VAL_INPUT_PID=$!
(
  set -euo pipefail
  curl -fL --retry 3 --retry-delay 5 -o "$CKPT/sam3.pt" \
    "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt"
  rclone copyto "$GDRIVE/4_models/pretrained/sam2.1_hiera_large.pt" \
    "$CKPT/sam2.1_hiera_large.pt"
  echo WEIGHTS_READY
) > "$RESULTS/setup/weights.log" 2>&1 &
WEIGHT_PID=$!
(
  set -euo pipefail
  DEPLOYED_CODE="${DEPLOYED_CODE:-/home/ubuntu/hsot}"
  if [ -s "$DEPLOYED_CODE/3_src/track_t1.py" ]; then
    DEPLOYED_CODE="$DEPLOYED_CODE/3_src"
  fi
  if [ -s "$DEPLOYED_CODE/track_t1.py" ] && \
     [ -s "$DEPLOYED_CODE/finalize_submission.py" ] && \
     [ -s "$DEPLOYED_CODE/hsot/crop_rerun.py" ] && \
     [ -s "$DEPLOYED_CODE/hsot/eval.py" ] && \
     [ -s "$DEPLOYED_CODE/hsot/quality_head_v1.py" ]; then
    cp "$DEPLOYED_CODE/track_t1.py" "$CODE/track_t1.py"
    cp "$DEPLOYED_CODE/finalize_submission.py" "$CODE/finalize_submission.py"
    cp "$DEPLOYED_CODE/hsot/"*.py "$CODE/hsot/"
    if [ -s "$DEPLOYED_CODE/hsot/qhead_weights_v056.npz" ]; then
      cp "$DEPLOYED_CODE/hsot/qhead_weights_v056.npz" "$CODE/hsot/"
    fi
    echo "CODE_SOURCE=lambda_deploy:$DEPLOYED_CODE"
  else
    rclone copyto "$GDRIVE/3_src/track_t1.py" "$CODE/track_t1.py"
    rclone copyto "$GDRIVE/3_src/finalize_submission.py" "$CODE/finalize_submission.py"
    rclone copy "$GDRIVE/3_src/hsot/" "$CODE/hsot/" --include '*.py'
    rclone copyto "$GDRIVE/3_src/hsot/qhead_weights_v056.npz" \
      "$CODE/hsot/qhead_weights_v056.npz" 2>/dev/null || true
    echo "CODE_SOURCE=gdrive-staging"
  fi
  test -s "$CODE/hsot/crop_rerun.py"
  grep -q -- '--jpeg-profile' "$CODE/hsot/crop_rerun.py"
  grep -q 'q100-444' "$CODE/hsot/crop_rerun.py"
  echo CODE_READY
) > "$RESULTS/setup/code.log" 2>&1 &
CODE_PID=$!

# Fresh isolated envs: paired legs share them; no old venv leakage.
uv venv --python 3.12 "$T1ENV"
git clone https://github.com/yangchris11/samurai.git "$SAMURAI"
git -C "$SAMURAI" fetch --depth 1 origin "$SAMURAI_SHA"
git -C "$SAMURAI" checkout --detach "$SAMURAI_SHA"
VIRTUAL_ENV="$T1ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$T1ENV" uv pip install -q -e "$SAMURAI/sam2" \
  scipy loguru tqdm pandas pillow opencv-python-headless

uv venv --python 3.12 "$SAM3ENV"
VIRTUAL_ENV="$SAM3ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$SAM3ENV" uv pip install -q \
  "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}" \
  "numpy<2" ftfy==6.1.1 regex einops psutil scipy av pycocotools numba \
  python-rapidjson pandas pillow tqdm opencv-python-headless
VIRTUAL_ENV="$SAM3ENV" uv pip install -q "setuptools<81"

wait "$VAL_DATA_PID" || die "VAL frames 下載／解包失敗"
wait "$VAL_INPUT_PID" || die "VAL frozen inputs 下載失敗"
wait "$WEIGHT_PID" || die "權重下載失敗"
wait "$CODE_PID" || die "code 下載或 q100 CLI 驗證失敗"
grep -q VAL_FRAMES_READY "$RESULTS/setup/val_frames.log" || die "VAL frames 未 ready"
grep -q VAL_INPUTS_READY "$RESULTS/setup/val_inputs.log" || die "VAL inputs 未 ready"
grep -q WEIGHTS_READY "$RESULTS/setup/weights.log" || die "權重未 ready"
grep -q CODE_READY "$RESULTS/setup/code.log" || die "code 未 ready"

actual=$(sha256sum "$CKPT/sam3.pt"); actual=${actual%% *}
[ "$actual" = "$SAM3_WEIGHT_SHA" ] || die "SAM3 weight SHA 不符：$actual"
actual=$(sha256sum "$CKPT/sam2.1_hiera_large.pt"); actual=${actual%% *}
[ "$actual" = "$SAM21_WEIGHT_SHA" ] || die "SAM2.1 weight SHA 不符：$actual"
QHEAD_AVAILABLE=0
assert_sha "$CODE/track_t1.py" "$TRACK_T1_SHA" "track_t1.py"
assert_sha "$CODE/finalize_submission.py" "$FINALIZE_SHA" "finalize_submission.py"
assert_sha "$CODE/hsot/crop_rerun.py" "$CROP_RERUN_SHA" "crop_rerun.py"
assert_sha "$CODE/hsot/eval.py" "$EVAL_SHA" "eval.py"
assert_sha "$CODE/hsot/quality_head_v1.py" "$QUALITY_HEAD_SHA" "quality_head_v1.py"
if [ -s "$CODE/hsot/qhead_weights_v056.npz" ]; then
  actual=$(sha256sum "$CODE/hsot/qhead_weights_v056.npz"); actual=${actual%% *}
  if [ "$actual" = "$QHEAD_WEIGHT_SHA" ]; then
    QHEAD_AVAILABLE=1
  fi
fi
if [ "$MAKE_QHEAD_DIAGNOSTIC" -eq 1 ] && [ "$QHEAD_AVAILABLE" -eq 0 ]; then
  MAKE_QHEAD_DIAGNOSTIC=0
  echo "SKIPPED: qhead_weights_v056.npz 缺失或 SHA 不符；primary 繼續" \
    > "$RESULTS/setup/qhead_diagnostic_status.txt"
else
  echo "AVAILABLE=$QHEAD_AVAILABLE" > "$RESULTS/setup/qhead_diagnostic_status.txt"
fi
"$PY3" - "$RESULTS/protocol.json" "$QHEAD_AVAILABLE" "$MAKE_QHEAD_DIAGNOSTIC" <<'PY'
import json, sys
p, available, enabled = sys.argv[1:]
d = json.load(open(p))
d["qhead_runtime"] = {"weight_available_and_hash_valid": bool(int(available)),
                      "diagnostic_enabled": bool(int(enabled)),
                      "primary_dependency": False}
open(p, "w").write(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
PY
assert_sha "$SCRATCH/inputs/val_full_main.csv" "$VAL_MAIN_SHA" "VAL full main"
assert_sha "$SCRATCH/inputs/val_full_source.csv" "$VAL_SOURCE_SHA" "VAL full source"
assert_sha "$SCRATCH/inputs/val_split_v1.txt" "$VAL_LIST_SHA" "VAL split"
assert_sha "$SCRATCH/inputs/2026training.csv" "$VAL_GT_SHA" "VAL GT"

"$PY1" -c "import torch,sam2,pandas; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
"$PY3" -c "import torch,sam3,numpy,PIL; assert torch.cuda.is_available(); assert numpy.__version__.startswith('1.'); print(torch.__version__, torch.cuda.get_device_name(0), PIL.__version__)"

VAL_SENTINEL=$(find "$SCRATCH/val_extract" -type d -name vis-ant -print -quit)
[ -n "$VAL_SENTINEL" ] || die "VAL 解包後找不到 vis-ant"
VAL_ROOT=$(dirname "$VAL_SENTINEL")
[ "$(find "$VAL_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 65 ] \
  || die "VAL root 不是 65 支：$VAL_ROOT"
export PYTHONPATH="$CODE"

uv pip freeze --python "$PY1" > "$RESULTS/setup/t1_freeze.txt"
uv pip freeze --python "$PY3" > "$RESULTS/setup/sam3_freeze.txt"
git -C "$SAMURAI" rev-parse HEAD > "$RESULTS/setup/samurai_commit.txt"
sha256sum "$CKPT/sam3.pt" "$CKPT/sam2.1_hiera_large.pt" \
  > "$RESULTS/setup/checkpoint_sha256.txt"
sha256sum "$CODE/track_t1.py" "$CODE/finalize_submission.py" \
  "$CODE/hsot/crop_rerun.py" "$CODE/hsot/eval.py" \
  "$CODE/hsot/quality_head_v1.py" > "$RESULTS/setup/source_sha256.txt"
if [ "$QHEAD_AVAILABLE" -eq 1 ]; then
  sha256sum "$CODE/hsot/qhead_weights_v056.npz" >> "$RESULTS/setup/source_sha256.txt"
fi
sha256sum "$SCRATCH/inputs/val_full_main.csv" "$SCRATCH/inputs/val_full_source.csv" \
  "$SCRATCH/inputs/val_split_v1.txt" "$SCRATCH/inputs/2026training.csv" \
  "$SCRATCH/inputs/t1val_fc_65.tar" \
  > "$RESULTS/setup/val_input_sha256.txt"

# Validate frozen full pair and create a strict VAL finalizer contract.
"$PY3" - "$SCRATCH/inputs/val_full_main.csv" "$SCRATCH/inputs/val_full_source.csv" \
  "$SCRATCH/inputs/val_split_v1.txt" "$RESULTS/val_contract.csv" \
  "$RESULTS/setup/val_full_pair_validation.json" <<'PY'
import json, sys
import numpy as np
import pandas as pd
main_p, src_p, list_p, contract_p, report_p = sys.argv[1:]
a, b = pd.read_csv(main_p), pd.read_csv(src_p)
assert list(a.columns) == ["ID", "x", "y", "width", "height"]
assert list(b.columns) == list(a.columns)
assert len(a) == len(b) == 40141
assert a.ID.tolist() == b.ID.tolist()
assert not a.ID.duplicated().any() and not b.ID.duplicated().any()
seq = a.ID.str.rsplit("_", n=1).str[0]
want = [s for s in open(list_p).read().split() if s]
assert seq.nunique() == len(want) == 65 and set(seq) == set(want)
for label, d in (("main", a), ("source", b)):
    x = d[["x", "y", "width", "height"]].to_numpy(float)
    assert np.isfinite(x).all(), f"{label}: NaN/Inf"
    assert (x[:, :2] >= 0).all(), f"{label}: negative x/y"
    assert (x[:, 2:] > 0).all(), f"{label}: non-positive w/h"
pd.DataFrame({"ID": a.ID, "x": 0, "y": 0, "width": 0, "height": 0}).to_csv(
    contract_p, index=False)
json.dump({"rows": len(a), "sequences": int(seq.nunique()),
           "exact_pair_ids_and_order": True}, open(report_p, "w"), indent=2)
PY
mark ENV_VAL_INPUT_READY
sync_stage ENV_VAL_INPUT_READY

rebase_1based() {
  "$PY3" - "$1" "$2" <<'PY'
import csv, sys
from collections import defaultdict
src, dst = sys.argv[1:]
rows = list(csv.DictReader(open(src)))
by, order = defaultdict(list), []
for r in rows:
    s, f = r["ID"].rsplit("_", 1)
    if s not in by: order.append(s)
    by[s].append((int(f), r))
out = []
for s in order:
    for i, (_, r) in enumerate(sorted(by[s], key=lambda z: z[0]), 1):
        out.append({"ID": f"{s}_{i}", "x": r["x"], "y": r["y"],
                    "width": r["width"], "height": r["height"]})
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ID", "x", "y", "width", "height"])
    w.writeheader(); w.writerows(out)
PY
}

map_1based_to_global() {
  "$PY3" - "$1" "$2" "$3" <<'PY'
import csv, sys
from collections import defaultdict
global_p, one_p, out_p = sys.argv[1:]
def grouped(path):
    rows = list(csv.DictReader(open(path)))
    by, order = defaultdict(list), []
    for r in rows:
        s, f = r["ID"].rsplit("_", 1)
        if s not in by: order.append(s)
        by[s].append((int(f), r))
    return order, {s: [r for _, r in sorted(v)] for s, v in by.items()}
go, gb = grouped(global_p); oo, ob = grouped(one_p)
assert go == oo
out = []
for s in go:
    assert len(gb[s]) == len(ob[s])
    for g, o in zip(gb[s], ob[s]):
        out.append({"ID": g["ID"], "x": o["x"], "y": o["y"],
                    "width": o["width"], "height": o["height"]})
with open(out_p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ID", "x", "y", "width", "height"])
    w.writeheader(); w.writerows(out)
PY
}

make_crop_contract() {
  "$PY3" - "$1" "$2" "$3" <<'PY'
import csv, json, sys
from pathlib import Path
frames, meta_p, out_p = map(Path, sys.argv[1:])
meta = json.loads(meta_p.read_text())
rows = []
for seq in sorted(meta):
    jpgs = sorted((frames / seq).glob("*.jpg"))
    assert jpgs, f"{seq}: no jpg"
    rows.extend((f"{seq}_{i}", 0, 0, 0, 0) for i in range(1, len(jpgs) + 1))
with out_p.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ID", "x", "y", "width", "height"])
    w.writerows(rows)
print(len(meta), len(rows))
PY
}

tree_hash_manifest() {
  local root="$1" out="$2"
  (
    cd "$root"
    find . -type f -name '*.jpg' -print0 | LC_ALL=C sort -z \
      | xargs -0 -r sha256sum > "$out"
  )
}

run_crop_tracker() {
  local frames="$1" list="$2" contract="$3" backend="$4" out="$5"
  mkdir -p "$out"
  if [ "$backend" = sam3 ]; then
    local sam3_cmd=("$PY3" "$CODE/track_t1.py" --frames-root "$frames"
      --seq-list "$list" --sample-csv "$contract" --out-dir "$out" --backend sam3
      --sam3-version sam3 --sam3-ckpt "$CKPT/sam3.pt" --source-revision "$SAM3_SHA")
    [ -n "$SAM3_MODE_FLAG" ] && sam3_cmd+=("$SAM3_MODE_FLAG")
    "${sam3_cmd[@]}"
  else
    local samurai_cmd=("$PY1" "$CODE/track_t1.py" --frames-root "$frames"
      --seq-list "$list" --sample-csv "$contract" --out-dir "$out" --backend samurai
      --samurai-dir "$SAMURAI" --ckpt "$CKPT/sam2.1_hiera_large.pt"
      --source-revision "$SAMURAI_SHA")
    [ -n "$SAMURAI_MODE_FLAG" ] && samurai_cmd+=("$SAMURAI_MODE_FLAG")
    "${samurai_cmd[@]}"
  fi
}

validate_tracker_output() {
  "$PY3" - "$1" "$2" "$3" "$4" "$MODE" <<'PY'
import json, sys
import pandas as pd
contract_p, out_dir, report_p, backend, mode = sys.argv[1:]
want = pd.read_csv(contract_p)
got = pd.read_csv(out_dir + "/submission.csv")
diag = json.load(open(out_dir + "/diagnostics.json"))
assert got.ID.tolist() == want.ID.tolist()
assert not got.ID.duplicated().any()
seqs = sorted(set(i.rsplit("_", 1)[0] for i in want.ID))
bad = {}
for s in seqs:
    d = diag.get(s, {})
    if d.get("status") != "complete" or "error" in d or d.get("fallback"):
        bad[s] = d
assert not bad, bad
meta = diag.get("_meta", {})
assert meta.get("validation") == "pass", meta
if backend == "sam3":
    assert bool(meta.get("sam3_eval")) == (mode == "rankB-robust"), meta
else:
    expect_reset = mode == "rankB-robust"
    assert bool(meta.get("samurai_reset_kf")) == expect_reset, meta
    assert bool(meta.get("samurai_legacy_cross_seq_kf")) == (not expect_reset), meta
json.dump({"rows": len(got), "sequences": len(seqs), "bad": 0,
           "formal_validation": "pass", "backend": backend, "mode": mode,
           "tracker_mode_asserted": True}, open(report_p, "w"), indent=2)
PY
}

rebase_1based "$SCRATCH/inputs/val_full_main.csv" "$SCRATCH/inputs/val_main_1b.csv"
rebase_1based "$SCRATCH/inputs/val_full_source.csv" "$SCRATCH/inputs/val_source_1b.csv"

# Prepare both encodings independently from original frames, then require identical geometry.
for PROFILE in legacy-q95 q100-444; do
  PDIR="$RESULTS/val/$PROFILE"
  CROP="$SCRATCH/val_crop/$PROFILE"
  mkdir -p "$PDIR" "$CROP"
  "$PY3" -m hsot.crop_rerun prep --frames-root "$VAL_ROOT" \
    --base-csv "$SCRATCH/inputs/val_main_1b.csv" \
    --envelope-extra "$SCRATCH/inputs/val_source_1b.csv" \
    --area-frac-max 0.55 --jpeg-profile "$PROFILE" \
    --out-root "$CROP" --meta "$PDIR/meta.json"
  "$PY3" - "$PDIR/meta.json" "$PDIR/crop_seqs.txt" "$PROFILE" \
    "$DRY_MIN_SELECTED_SEQS" <<'PY'
import json, sys
meta_p, list_p, profile, n_min = sys.argv[1:]
m = json.load(open(meta_p))
assert len(m) >= int(n_min), (len(m), n_min)
for name, w in m.items():
    enc = w.get("crop_image_encoding", {})
    assert enc.get("profile") == profile, (name, enc)
open(list_p, "w").write("".join(f"{s}\n" for s in sorted(m)))
PY
  make_crop_contract "$CROP" "$PDIR/meta.json" "$PDIR/crop_contract.csv" \
    > "$PDIR/crop_contract_summary.txt"
  tree_hash_manifest "$CROP" "$PDIR/crop_images_sha256.txt"
done

"$PY3" - "$RESULTS/val/legacy-q95/meta.json" "$RESULTS/val/q100-444/meta.json" \
  "$SCRATCH/val_crop/q100-444" "$RESULTS/val/window_geometry_verdict.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
from PIL import Image, JpegImagePlugin
legacy_p, q100_p, qroot, out_p = sys.argv[1:]
a, b = json.load(open(legacy_p)), json.load(open(q100_p))
assert set(a) == set(b), (set(a) - set(b), set(b) - set(a))
keys = ("x1", "y1", "w", "h", "orig", "seq", "frames", "zoom")
diff, geometry = [], {}
for name in sorted(a):
    ga = {k: a[name][k] for k in keys}
    gb = {k: b[name][k] for k in keys}
    if ga != gb: diff.append({"name": name, "legacy": ga, "q100": gb})
    geometry[name] = ga
assert not diff, diff[:3]
assert all(w["crop_image_encoding"]["profile"] == "legacy-q95" for w in a.values())
assert all(w["crop_image_encoding"]["profile"] == "q100-444" for w in b.values())
n_jpg = 0
for name in sorted(b):
    vals = []
    for p in sorted((Path(qroot) / name).glob("*.jpg")):
        with Image.open(p) as im: vals.append(JpegImagePlugin.get_sampling(im))
    assert vals and set(vals) == {0}, (name, sorted(set(vals)))
    n_jpg += len(vals)
blob = json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()
json.dump({"pass": True, "selected_sequences": len(a),
           "geometry_fields": list(keys),
           "geometry_sha256": hashlib.sha256(blob).hexdigest(),
           "geometry_exact_equal": True,
           "q100_all_sampling_zero_444": True,
           "q100_jpegs_audited": n_jpg}, open(out_p, "w"), indent=2)
PY
cmp "$RESULTS/val/legacy-q95/crop_seqs.txt" "$RESULTS/val/q100-444/crop_seqs.txt"
cmp "$RESULTS/val/legacy-q95/crop_contract.csv" "$RESULTS/val/q100-444/crop_contract.csv"
mark VAL_CROP_PREP_GEOMETRY_PASS
sync_stage VAL_CROP_PREP_GEOMETRY_PASS

# DRY: one deterministically selected crop sequence, all four profile/backend legs.
IFS= read -r DRY_SEQ < "$RESULTS/val/legacy-q95/crop_seqs.txt"
[ -n "$DRY_SEQ" ] || die "DRY sequence 為空"
printf '%s\n' "$DRY_SEQ" > "$RESULTS/val/dry_seq.txt"
"$PY3" - "$RESULTS/val/legacy-q95/crop_contract.csv" "$DRY_SEQ" \
  "$RESULTS/val/dry_contract.csv" <<'PY'
import sys
import pandas as pd
src, seq, out = sys.argv[1:]
d = pd.read_csv(src)
d = d[d.ID.str.rsplit("_", n=1).str[0] == seq]
assert len(d) > 0
d.to_csv(out, index=False)
PY
for PROFILE in legacy-q95 q100-444; do
  for BACKEND in sam3 samurai; do
    OUT="$RESULTS/val/$PROFILE/dry_$BACKEND"
    run_crop_tracker "$SCRATCH/val_crop/$PROFILE" "$RESULTS/val/dry_seq.txt" \
      "$RESULTS/val/dry_contract.csv" "$BACKEND" "$OUT"
    validate_tracker_output "$RESULTS/val/dry_contract.csv" "$OUT" \
      "$OUT/validation_report.json" "$BACKEND"
  done
done
"$PY3" - "$RESULTS/val/dry_gate_verdict.json" "$DRY_MAX_INCOMPLETE_RUNS" \
  "$RESULTS/val/legacy-q95/dry_sam3" "$RESULTS/val/legacy-q95/dry_samurai" \
  "$RESULTS/val/q100-444/dry_sam3" "$RESULTS/val/q100-444/dry_samurai" <<'PY'
import json, sys
out, max_bad, *dirs = sys.argv[1:]
bad, runs = [], []
for d in dirs:
    v = json.load(open(d + "/validation_report.json"))
    runs.append({"path": d, **v})
    if v["bad"]: bad.append(d)
doc = {"pass": len(bad) <= int(max_bad), "runs": runs,
       "incomplete_runs": bad, "threshold_max_incomplete_runs": int(max_bad)}
json.dump(doc, open(out, "w"), indent=2)
assert doc["pass"], doc
PY
mark VAL_DRY_PASS
sync_stage VAL_DRY_PASS

finalize_val_profile() {
  local profile="$1" pdir="$RESULTS/val/$1"
  "$PY3" -m hsot.crop_rerun merge \
    --base-csv "$SCRATCH/inputs/val_main_1b.csv" \
    --crop-csv "$pdir/crop_sam3/submission.csv" \
    --meta "$pdir/meta.json" --out "$pdir/main_merged_1b.csv"
  "$PY1" -m hsot.crop_rerun merge \
    --base-csv "$SCRATCH/inputs/val_source_1b.csv" \
    --crop-csv "$pdir/crop_samurai/submission.csv" \
    --meta "$pdir/meta.json" --out "$pdir/source_merged_1b.csv"
  map_1based_to_global "$SCRATCH/inputs/val_full_main.csv" \
    "$pdir/main_merged_1b.csv" "$pdir/main_merged_global.csv"
  map_1based_to_global "$SCRATCH/inputs/val_full_source.csv" \
    "$pdir/source_merged_1b.csv" "$pdir/source_merged_global.csv"
  "$PY1" "$CODE/finalize_submission.py" \
    --main "$pdir/main_merged_global.csv" --source "$pdir/source_merged_global.csv" \
    --sample "$RESULTS/val_contract.csv" --out "$pdir/final_primary.csv" \
    --corr "$CORR_MODE" --K 6 --qhead none
  if [ "$MAKE_QHEAD_DIAGNOSTIC" -eq 1 ]; then
    "$PY1" "$CODE/finalize_submission.py" \
      --main "$pdir/main_merged_global.csv" --source "$pdir/source_merged_global.csv" \
      --sample "$RESULTS/val_contract.csv" \
      --out "$pdir/final_qhead_v056_DIAGNOSTIC.csv" \
      --corr "$CORR_MODE" --K 6 --qhead v056 \
      --qhead-weights "$CODE/hsot/qhead_weights_v056.npz"
  fi
}

for PROFILE in legacy-q95 q100-444; do
  PDIR="$RESULTS/val/$PROFILE"
  run_crop_tracker "$SCRATCH/val_crop/$PROFILE" "$PDIR/crop_seqs.txt" \
    "$PDIR/crop_contract.csv" sam3 "$PDIR/crop_sam3"
  validate_tracker_output "$PDIR/crop_contract.csv" "$PDIR/crop_sam3" \
    "$PDIR/crop_sam3/validation_report.json" sam3
  mark "VAL_${PROFILE}_SAM3_DONE"
  sync_stage "VAL_${PROFILE}_SAM3_DONE"
  run_crop_tracker "$SCRATCH/val_crop/$PROFILE" "$PDIR/crop_seqs.txt" \
    "$PDIR/crop_contract.csv" samurai "$PDIR/crop_samurai"
  validate_tracker_output "$PDIR/crop_contract.csv" "$PDIR/crop_samurai" \
    "$PDIR/crop_samurai/validation_report.json" samurai
  finalize_val_profile "$PROFILE"
  mark "VAL_${PROFILE}_FINAL_DONE"
  sync_stage "VAL_${PROFILE}_FINAL_DONE"
done

# Official GT paired verdict. qhead diagnostic is reported but never gated.
LEGACY_DIAG=NONE; Q100_DIAG=NONE
if [ "$MAKE_QHEAD_DIAGNOSTIC" -eq 1 ]; then
  LEGACY_DIAG="$RESULTS/val/legacy-q95/final_qhead_v056_DIAGNOSTIC.csv"
  Q100_DIAG="$RESULTS/val/q100-444/final_qhead_v056_DIAGNOSTIC.csv"
fi
"$PY3" - "$RESULTS/val/legacy-q95/final_primary.csv" \
  "$RESULTS/val/q100-444/final_primary.csv" "$SCRATCH/inputs/2026training.csv" \
  "$SCRATCH/inputs/val_split_v1.txt" "$RESULTS/val/legacy-q95/crop_seqs.txt" \
  "$RESULTS/val/paired_verdict.json" "$RESULTS/val/paired_verdict.txt" \
  "$GO_MIN_ALL_POOLED_DELTA" "$GO_MIN_SELECTED_POOLED_DELTA" \
  "$GO_MIN_LEGACY_POOLED_AUC" \
  "$GO_STABLE_AUC_THRESHOLD" "$GO_STABLE_BAD_DELTA" \
  "$GO_MAX_STABLE_BAD_FRACTION" "$GO_MAX_UNEXPECTED_CHANGED_SEQS" \
  "$LEGACY_DIAG" "$Q100_DIAG" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from hsot.eval import evaluate
(legacy_p, q100_p, gt_p, list_p, selected_p, json_p, text_p, min_all, min_sel,
 min_legacy, stable_auc_t, stable_bad_d, max_bad_frac, max_unexpected,
 legacy_diag, q100_diag) = sys.argv[1:]
seqs = Path(list_p).read_text().split()
selected = Path(selected_p).read_text().split()
L, Q = evaluate(legacy_p, gt_p, seqs), evaluate(q100_p, gt_p, seqs)
Ls, Qs = evaluate(legacy_p, gt_p, selected), evaluate(q100_p, gt_p, selected)
d = {s: Q["per_seq"][s]["auc"] - L["per_seq"][s]["auc"] for s in seqs}
stable = [s for s in selected if L["per_seq"][s]["auc"] >= float(stable_auc_t)]
stable_bad = [s for s in stable if d[s] < float(stable_bad_d)]
stable_bad_frac = len(stable_bad) / max(len(stable), 1)
a, b = pd.read_csv(legacy_p), pd.read_csv(q100_p)
assert a.ID.tolist() == b.ID.tolist()
changed = (a[["x", "y", "width", "height"]].to_numpy() !=
           b[["x", "y", "width", "height"]].to_numpy()).any(axis=1)
changed_seqs = set(a.loc[changed, "ID"].str.rsplit("_", n=1).str[0])
unexpected = sorted(changed_seqs - set(selected))
delta_all = Q["pooled"]["auc"] - L["pooled"]["auc"]
delta_sel = Qs["pooled"]["auc"] - Ls["pooled"]["auc"]
reasons = []
if L["pooled"]["auc"] < float(min_legacy):
    reasons.append(f"legacy absolute AUC {L['pooled']['auc']:.5f} < {float(min_legacy):.5f}")
if delta_all < float(min_all):
    reasons.append(f"all pooled delta {delta_all:+.5f} < {float(min_all):+.5f}")
if delta_sel < float(min_sel):
    reasons.append(f"selected pooled delta {delta_sel:+.5f} < {float(min_sel):+.5f}")
if stable_bad_frac > float(max_bad_frac):
    reasons.append(f"stable bad fraction {stable_bad_frac:.3f} > {float(max_bad_frac):.3f}")
if len(unexpected) > int(max_unexpected):
    reasons.append(f"unexpected changed sequences {len(unexpected)} > {int(max_unexpected)}")
# Paired cluster bootstrap: report only, not a gate.
rng = np.random.default_rng(42)
dv = np.array([d[s] for s in seqs])
w = np.array([L["per_seq"][s]["n"] for s in seqs], float)
idx = rng.integers(0, len(seqs), size=(10000, len(seqs)))
boot = (dv[idx] * w[idx]).sum(1) / w[idx].sum(1)
doc = {
    "official_metric": "Success AUC at thresholds 0.02..1.00",
    "primary_only_used_for_gate": True,
    "legacy": {"pooled_auc": L["pooled"]["auc"],
               "selected_pooled_auc": Ls["pooled"]["auc"]},
    "q100_444": {"pooled_auc": Q["pooled"]["auc"],
                 "selected_pooled_auc": Qs["pooled"]["auc"]},
    "paired_delta": {"all_pooled": delta_all, "selected_pooled": delta_sel,
                     "seq_median": float(np.median(list(d.values()))),
                     "seq_mean": float(np.mean(list(d.values()))),
                     "changed_rows": int(changed.sum()),
                     "changed_sequences": sorted(changed_seqs),
                     "unexpected_changed_sequences": unexpected,
                     "bootstrap95_all_pooled": [float(x) for x in np.percentile(boot, [2.5, 97.5])]},
    "stable_selected_group": {"auc_threshold": float(stable_auc_t), "n": len(stable),
                     "bad_delta_threshold": float(stable_bad_d),
                     "bad_sequences": stable_bad, "bad_fraction": stable_bad_frac},
    "per_sequence_delta": d, "go": not reasons, "gate_reasons": reasons,
    "thresholds": {"min_all_pooled_delta": float(min_all),
                   "min_selected_pooled_delta": float(min_sel),
                   "min_legacy_control_pooled_auc": float(min_legacy),
                   "max_stable_bad_fraction": float(max_bad_frac),
                   "max_unexpected_changed_sequences": int(max_unexpected)},
}
if legacy_diag != "NONE":
    LD = evaluate(legacy_diag, gt_p, seqs)
    QD = evaluate(q100_diag, gt_p, seqs)
    doc["qhead_v056_DIAGNOSTIC_not_gated"] = {
        "legacy_auc": LD["pooled"]["auc"], "q100_auc": QD["pooled"]["auc"],
        "delta": QD["pooled"]["auc"] - LD["pooled"]["auc"],
        "warning": "q100 changes exact A/B pair; v056 was not OOF-trained for it"}
Path(json_p).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
lines = [
    "Q100 crop paired verdict (official Success AUC)",
    f"legacy={L['pooled']['auc']:.5f} q100={Q['pooled']['auc']:.5f} delta={delta_all:+.5f}",
    f"selected legacy={Ls['pooled']['auc']:.5f} q100={Qs['pooled']['auc']:.5f} delta={delta_sel:+.5f}",
    f"changed rows={changed.sum()} sequences={len(changed_seqs)} unexpected={len(unexpected)}",
    f"stable bad={len(stable_bad)}/{len(stable)} ({stable_bad_frac:.1%})",
    "VERDICT=" + ("GO_TEST_FRESH_PAIRED" if not reasons else "NO_GO_CATASTROPHE")]
if reasons: lines.extend("- " + r for r in reasons)
Path(text_p).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
mark VAL_PAIRED_VERDICT_DONE
sync_stage VAL_PAIRED_VERDICT_DONE

GO=$("$PY3" -c "import json; print('1' if json.load(open('$RESULTS/val/paired_verdict.json'))['go'] else '0')")
if [ "$GO" != 1 ]; then
  echo NO_GO_VAL_CATASTROPHE > "$STATUS"
  mark PILOT_STOPPED_BEFORE_TEST
  sync_stage NO_GO_VAL_CATASTROPHE
  exit 0
fi

# TEST only after GO: fresh same-machine q95 AND q100 (D074 attribution guard).
mkdir -p "$SCRATCH/test_extract" "$RESULTS/test"
(
  set -euo pipefail
  rclone copyto "$TEST_FRAMES_REMOTE" "$SCRATCH/inputs/t1test_fc_75.tar"
  echo "$TEST_TAR_SHA  $SCRATCH/inputs/t1test_fc_75.tar" | sha256sum -c -
  tar -xf "$SCRATCH/inputs/t1test_fc_75.tar" -C "$SCRATCH/test_extract"
  echo TEST_FRAMES_READY
) > "$RESULTS/setup/test_frames.log" 2>&1 &
TEST_DATA_PID=$!
(
  set -euo pipefail
  rclone copyto "$TEST_SAMPLE_REMOTE" "$SCRATCH/inputs/test_sample.csv"
  rclone copyto "$TEST_MAIN_REMOTE" "$SCRATCH/inputs/test_full_main.csv"
  rclone copyto "$TEST_SOURCE_REMOTE" "$SCRATCH/inputs/test_full_source.csv"
  echo TEST_INPUTS_READY
) > "$RESULTS/setup/test_inputs.log" 2>&1 &
TEST_INPUT_PID=$!
wait "$TEST_DATA_PID" || die "TEST frames 下載／解包失敗"
wait "$TEST_INPUT_PID" || die "TEST frozen inputs 下載失敗"
grep -q TEST_FRAMES_READY "$RESULTS/setup/test_frames.log" || die "TEST frames 未 ready"
grep -q TEST_INPUTS_READY "$RESULTS/setup/test_inputs.log" || die "TEST inputs 未 ready"
assert_sha "$SCRATCH/inputs/test_full_main.csv" "$TEST_MAIN_SHA" "TEST full main"
assert_sha "$SCRATCH/inputs/test_full_source.csv" "$TEST_SOURCE_SHA" "TEST full source"
assert_sha "$SCRATCH/inputs/test_sample.csv" "$TEST_SAMPLE_SHA" "TEST sample"

TEST_SENTINEL=$(find "$SCRATCH/test_extract" -type d -name nir-bee2 -print -quit)
[ -n "$TEST_SENTINEL" ] || die "TEST 解包後找不到 nir-bee2"
TEST_ROOT=$(dirname "$TEST_SENTINEL")
[ "$(find "$TEST_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 75 ] \
  || die "TEST root 不是 75 支：$TEST_ROOT"
"$PY3" - "$SCRATCH/inputs/test_full_main.csv" "$SCRATCH/inputs/test_full_source.csv" \
  "$SCRATCH/inputs/test_sample.csv" "$RESULTS/setup/test_full_pair_validation.json" <<'PY'
import json, sys
import numpy as np
import pandas as pd
main_p, src_p, sample_p, out_p = sys.argv[1:]
a, b, s = pd.read_csv(main_p), pd.read_csv(src_p), pd.read_csv(sample_p)
assert len(a) == len(b) == len(s) == 26860
assert a.ID.tolist() == b.ID.tolist() == s.ID.tolist()
assert not a.ID.duplicated().any()
assert a.ID.str.rsplit("_", n=1).str[0].nunique() == 75
for label, d in (("main", a), ("source", b)):
    x = d[["x", "y", "width", "height"]].to_numpy(float)
    assert np.isfinite(x).all() and (x[:, :2] >= 0).all() and (x[:, 2:] > 0).all(), label
json.dump({"rows": len(a), "sequences": 75,
           "exact_main_source_sample_ids_and_order": True}, open(out_p, "w"), indent=2)
PY
sha256sum "$SCRATCH/inputs/test_full_main.csv" "$SCRATCH/inputs/test_full_source.csv" \
  "$SCRATCH/inputs/test_sample.csv" "$SCRATCH/inputs/t1test_fc_75.tar" \
  > "$RESULTS/setup/test_input_sha256.txt"

for PROFILE in legacy-q95 q100-444; do
  PDIR="$RESULTS/test/$PROFILE"
  CROP="$SCRATCH/test_crop/$PROFILE"
  mkdir -p "$PDIR" "$CROP"
  "$PY3" -m hsot.crop_rerun prep --frames-root "$TEST_ROOT" \
    --base-csv "$SCRATCH/inputs/test_full_main.csv" \
    --envelope-extra "$SCRATCH/inputs/test_full_source.csv" \
    --area-frac-max 0.55 --jpeg-profile "$PROFILE" \
    --out-root "$CROP" --meta "$PDIR/meta.json"
  "$PY3" - "$PDIR/meta.json" "$PDIR/crop_seqs.txt" "$PROFILE" <<'PY'
import json, sys
meta_p, list_p, profile = sys.argv[1:]
m = json.load(open(meta_p)); assert m
for name, w in m.items():
    enc = w.get("crop_image_encoding", {})
    assert enc.get("profile") == profile, (name, enc)
open(list_p, "w").write("".join(f"{s}\n" for s in sorted(m)))
PY
  make_crop_contract "$CROP" "$PDIR/meta.json" "$PDIR/crop_contract.csv" \
    > "$PDIR/crop_contract_summary.txt"
  tree_hash_manifest "$CROP" "$PDIR/crop_images_sha256.txt"
done

"$PY3" - "$RESULTS/test/legacy-q95/meta.json" "$RESULTS/test/q100-444/meta.json" \
  "$SCRATCH/test_crop/q100-444" "$RESULTS/test/window_geometry_verdict.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
from PIL import Image, JpegImagePlugin
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
qroot, out_p = Path(sys.argv[3]), sys.argv[4]
assert set(a) == set(b)
keys = ("x1", "y1", "w", "h", "orig", "seq", "frames", "zoom")
geometry = {}
for name in sorted(a):
    ga, gb = {k: a[name][k] for k in keys}, {k: b[name][k] for k in keys}
    assert ga == gb, (name, ga, gb)
    geometry[name] = ga
n = 0
for seq in sorted(b):
    for p in sorted((qroot / seq).glob("*.jpg")):
        with Image.open(p) as im: assert JpegImagePlugin.get_sampling(im) == 0, p
        n += 1
assert n > 0
blob = json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()
json.dump({"pass": True, "selected_sequences": len(a), "geometry_exact_equal": True,
           "geometry_sha256": hashlib.sha256(blob).hexdigest(),
           "q100_all_sampling_zero_444": True, "q100_jpegs_audited": n},
          open(out_p, "w"), indent=2)
PY
cmp "$RESULTS/test/legacy-q95/crop_seqs.txt" "$RESULTS/test/q100-444/crop_seqs.txt"
cmp "$RESULTS/test/legacy-q95/crop_contract.csv" "$RESULTS/test/q100-444/crop_contract.csv"
mark TEST_PAIRED_PREP_DONE
sync_stage TEST_PAIRED_PREP_DONE

finalize_test_profile() {
  local profile="$1" pdir="$RESULTS/test/$1"
  "$PY3" -m hsot.crop_rerun merge --base-csv "$SCRATCH/inputs/test_full_main.csv" \
    --crop-csv "$pdir/crop_sam3/submission.csv" --meta "$pdir/meta.json" \
    --out "$pdir/main_merged.csv"
  "$PY1" -m hsot.crop_rerun merge --base-csv "$SCRATCH/inputs/test_full_source.csv" \
    --crop-csv "$pdir/crop_samurai/submission.csv" --meta "$pdir/meta.json" \
    --out "$pdir/source_merged.csv"
  "$PY1" "$CODE/finalize_submission.py" \
    --main "$pdir/main_merged.csv" --source "$pdir/source_merged.csv" \
    --sample "$SCRATCH/inputs/test_sample.csv" --out "$pdir/final_primary.csv" \
    --corr "$CORR_MODE" --K 6 --qhead none
  if [ "$MAKE_QHEAD_DIAGNOSTIC" -eq 1 ]; then
    "$PY1" "$CODE/finalize_submission.py" \
      --main "$pdir/main_merged.csv" --source "$pdir/source_merged.csv" \
      --sample "$SCRATCH/inputs/test_sample.csv" \
      --out "$pdir/final_qhead_v056_DIAGNOSTIC.csv" \
      --corr "$CORR_MODE" --K 6 --qhead v056 \
      --qhead-weights "$CODE/hsot/qhead_weights_v056.npz"
  fi
}

for PROFILE in legacy-q95 q100-444; do
  PDIR="$RESULTS/test/$PROFILE"
  CROP="$SCRATCH/test_crop/$PROFILE"
  run_crop_tracker "$CROP" "$PDIR/crop_seqs.txt" "$PDIR/crop_contract.csv" \
    sam3 "$PDIR/crop_sam3"
  validate_tracker_output "$PDIR/crop_contract.csv" "$PDIR/crop_sam3" \
    "$PDIR/crop_sam3/validation_report.json" sam3
  mark "TEST_${PROFILE}_SAM3_DONE"
  sync_stage "TEST_${PROFILE}_SAM3_DONE"
  run_crop_tracker "$CROP" "$PDIR/crop_seqs.txt" "$PDIR/crop_contract.csv" \
    samurai "$PDIR/crop_samurai"
  validate_tracker_output "$PDIR/crop_contract.csv" "$PDIR/crop_samurai" \
    "$PDIR/crop_samurai/validation_report.json" samurai
  finalize_test_profile "$PROFILE"
  mark "TEST_${PROFILE}_FINAL_DONE"
  sync_stage "TEST_${PROFILE}_FINAL_DONE"
done

Q95_PRIMARY="$RESULTS/test/legacy-q95/final_primary.csv"
Q100_PRIMARY="$RESULTS/test/q100-444/final_primary.csv"
Q95_DIAG=NONE; Q100_DIAG=NONE
if [ "$MAKE_QHEAD_DIAGNOSTIC" -eq 1 ]; then
  Q95_DIAG="$RESULTS/test/legacy-q95/final_qhead_v056_DIAGNOSTIC.csv"
  Q100_DIAG="$RESULTS/test/q100-444/final_qhead_v056_DIAGNOSTIC.csv"
fi
"$PY3" - "$MODE" "$Q95_PRIMARY" "$Q100_PRIMARY" "$Q95_DIAG" "$Q100_DIAG" \
  "$SCRATCH/inputs/test_sample.csv" "$SCRATCH/inputs/test_full_main.csv" \
  "$RESULTS/test/legacy-q95/crop_seqs.txt" "$RESULTS/test/paired_candidate_manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
(mode, q95_p, q100_p, q95_diag_p, q100_diag_p, sample_p, raw_main_p,
 selected_p, out_p) = sys.argv[1:]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
q95, q100, sample = pd.read_csv(q95_p), pd.read_csv(q100_p), pd.read_csv(sample_p)
assert q95.ID.tolist() == q100.ID.tolist() == sample.ID.tolist()
assert len(q95) == 26860 and not q95.ID.duplicated().any()
for d in (q95, q100):
    box = d[["x", "y", "width", "height"]].to_numpy(float)
    assert np.isfinite(box).all() and (box[:, :2] >= 0).all() and (box[:, 2:] > 0).all()
seq = q95.ID.str.rsplit("_", n=1).str[0]
first_ids = q95.assign(_seq=seq).groupby("_seq", sort=False).ID.first().tolist()
raw = pd.read_csv(raw_main_p).set_index("ID")
for d in (q95, q100):
    assert np.array_equal(d.set_index("ID").loc[first_ids].iloc[:, :4].to_numpy(float),
                          raw.loc[first_ids].iloc[:, :4].to_numpy(float))
changed = (q95.iloc[:, 1:].to_numpy() != q100.iloc[:, 1:].to_numpy()).any(1)
changed_seqs = set(seq[changed])
selected = set(Path(selected_p).read_text().split())
unexpected = sorted(changed_seqs - selected)
assert not unexpected, unexpected
doc = {"mode": mode, "comparison": "fresh same-machine q100-444 vs legacy-q95",
       "D074_cross_environment_confounded": False,
       "primary_q95_control": {"path": q95_p, "sha256": sha(q95_p),
                               "eligible_for_manual_LB_review": True},
       "primary_q100_experimental": {"path": q100_p, "sha256": sha(q100_p),
                                     "eligible_for_manual_LB_review": True},
       "paired_primary_delta": {"changed_rows": int(changed.sum()),
                                "changed_sequences": sorted(changed_seqs),
                                "unexpected_changed_sequences": unexpected,
                                "first_frames_exact_raw_main_both": len(first_ids)},
       "submission_note": "If tested on LB, submit q95 and q100 as a paired two-run comparison.",
       "kaggle_submitted": False}
if q95_diag_p != "NONE":
    a, b = pd.read_csv(q95_diag_p), pd.read_csv(q100_diag_p)
    assert a.ID.tolist() == b.ID.tolist() == sample.ID.tolist()
    dc = (a.iloc[:, 1:].to_numpy() != b.iloc[:, 1:].to_numpy()).any(1)
    doc["qhead_v056_DIAGNOSTIC"] = {
        "q95_sha256": sha(q95_diag_p), "q100_sha256": sha(q100_diag_p),
        "changed_rows": int(dc.sum()), "eligible_for_manual_LB_review": False,
        "reason": "q100 changes exact A/B pair; v056 was not OOF-trained for it"}
json.dump(doc, open(out_p, "w"), indent=2, ensure_ascii=False)
PY
sha256sum "$Q95_PRIMARY" "$Q100_PRIMARY" > "$RESULTS/test/primary_candidates_sha256.txt"
if [ "$Q95_DIAG" != NONE ]; then
  sha256sum "$Q95_DIAG" "$Q100_DIAG" > "$RESULTS/test/qhead_DIAGNOSTIC_sha256.txt"
fi
mark TEST_PAIRED_CANDIDATES_DONE
echo COMPLETE_GO_PAIRED_CANDIDATES > "$STATUS"
sync_stage COMPLETE_GO_PAIRED_CANDIDATES
echo "✅ q100 crop pilot 完成：mode=$MODE run_id=$RUN_ID"
echo "fresh q95 control=$Q95_PRIMARY"
echo "fresh q100 experimental=$Q100_PRIMARY"
if [ "$Q95_DIAG" != NONE ]; then
  echo "qhead outputs = DIAGNOSTIC ONLY"
fi
echo "未執行 Kaggle submission；結果已同步並驗 hash。"
