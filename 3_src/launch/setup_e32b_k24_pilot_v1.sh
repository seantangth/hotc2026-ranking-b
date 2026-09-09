#!/usr/bin/env bash
# E32b K=24 production-pair exemplar pilot.
#
# This script operates an ALREADY-CREATED Lambda VM.  It does not create one.
# It runs only the offline [first,current] detector-box substitution proxy; it
# neither re-initializes tracker memory nor submits anything to Kaggle.
# Design/gates: E32B_K24_PRODUCTION_PILOT_DESIGN_20260823.md (written pre-GPU).
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"

export INSTANCE_ID="${INSTANCE_ID:?Pass only the Lambda INSTANCE_ID explicitly authorized for this run}"
export KEEP_INSTANCE="${KEEP_INSTANCE:-0}"
export NO_TERMINATE="${NO_TERMINATE:-0}"
export E32B_REUSE_SHARED="${E32B_REUSE_SHARED:-0}"
case "$KEEP_INSTANCE" in 0|1) ;; *) echo "KEEP_INSTANCE must be 0 or 1" >&2; exit 2 ;; esac
case "$NO_TERMINATE" in 0|1) ;; *) echo "NO_TERMINATE must be 0 or 1" >&2; exit 2 ;; esac
case "$E32B_REUSE_SHARED" in 0|1) ;; *) echo "E32B_REUSE_SHARED must be 0 or 1" >&2; exit 2 ;; esac
if test "$KEEP_INSTANCE" -eq 1 || test "$NO_TERMINATE" -eq 1; then
  export KEEP_INSTANCE_EFFECTIVE=1
else
  export KEEP_INSTANCE_EFFECTIVE=0
fi
GDRIVE="${GDRIVE_REMOTE:-gdrive:WHISPERS_2026_HyperSOT}"
export RCLONE_CONFIG="${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}"
export E32B_RUN_ID="${E32B_RUN_ID:-$INSTANCE_ID}"
case "$E32B_RUN_ID" in *[!A-Za-z0-9._-]*|'') echo "E32B_RUN_ID contains unsafe characters" >&2; exit 2 ;; esac
DEST="$GDRIVE/5_outputs/e32b_k24_pilot_20260823/runs/$E32B_RUN_ID"
SAM3_REVISION="96914d2425f90a64f45ca977c2b5165418099543"
SAM3_CHECKPOINT_SHA256="9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
E32B_DIR="$HOME/e32b_k24"
SRC_DIR="$E32B_DIR/src"
INPUT_DIR="$E32B_DIR/inputs"
OUT_DIR="$E32B_DIR/out"
WORK_DIR="$E32B_DIR/work"

if test "${E32B_VALIDATE_ONLY:-0}" -eq 1; then
  bash -n "${BASH_SOURCE[0]}"
  grep -q 'E32B_SHARED_PY3' "${BASH_SOURCE[0]}"
  grep -q 'E32B_SHARED_SAM3_CHECKPOINT' "${BASH_SOURCE[0]}"
  grep -q 'E32B_SHARED_TEST_FRAMES_ROOT' "${BASH_SOURCE[0]}"
  grep -q -- '--reuse-extracted-frames' "${BASH_SOURCE[0]}"
  grep -q -- '--expected-checkpoint-sha256' "${BASH_SOURCE[0]}"
  forbidden_endpoint='/instance-operations/'"launch"
  if grep -q "$forbidden_endpoint" "${BASH_SOURCE[0]}"; then
    echo "validate-only: launch endpoint is forbidden" >&2
    exit 2
  fi
  if test "$E32B_REUSE_SHARED" -eq 1 && test "$KEEP_INSTANCE" -ne 1; then
    echo "validate-only: shared reuse requires KEEP_INSTANCE=1" >&2
    exit 2
  fi
  echo "E32B_VALIDATE_ONLY_PASS reuse=$E32B_REUSE_SHARED keep=$KEEP_INSTANCE"
  exit 0
fi

if test "$E32B_REUSE_SHARED" -eq 1; then
  test "$KEEP_INSTANCE" -eq 1 \
    || { echo "shared reuse requires explicit KEEP_INSTANCE=1" >&2; exit 2; }
  : "${E32B_SHARED_PY3:?shared reuse requires E32B_SHARED_PY3}"
  : "${E32B_SHARED_SAM3_CHECKPOINT:?shared reuse requires E32B_SHARED_SAM3_CHECKPOINT}"
  : "${E32B_SHARED_TEST_FRAMES_ROOT:?shared reuse requires E32B_SHARED_TEST_FRAMES_ROOT}"
  for shared_path in \
    "$E32B_SHARED_PY3" "$E32B_SHARED_SAM3_CHECKPOINT" "$E32B_SHARED_TEST_FRAMES_ROOT"; do
    case "$shared_path" in /*) ;; *) echo "shared reuse paths must be absolute: $shared_path" >&2; exit 2 ;; esac
  done
  test -x "$E32B_SHARED_PY3" || { echo "shared Python is not executable" >&2; exit 2; }
  test -f "$E32B_SHARED_SAM3_CHECKPOINT" || { echo "shared SAM3 checkpoint missing" >&2; exit 2; }
  test -d "$E32B_SHARED_TEST_FRAMES_ROOT" || { echo "shared test frames root missing" >&2; exit 2; }
  PY3="$E32B_SHARED_PY3"
  CHECKPOINT="$E32B_SHARED_SAM3_CHECKPOINT"
  DATA_DIR="$E32B_SHARED_TEST_FRAMES_ROOT"
  VENV_DIR="$(dirname "$(dirname "$PY3")")"
else
  DATA_DIR="$E32B_DIR/testfc"
  VENV_DIR="$HOME/e32b_sam3env"
  PY3="$VENV_DIR/bin/python"
  CHECKPOINT="$E32B_DIR/ckpt/sam3.pt"
  mkdir -p "$DATA_DIR"
fi
mkdir -p "$SRC_DIR" "$INPUT_DIR" "$OUT_DIR" "$WORK_DIR"

failure_active=0

self_terminate() {
  local key
  key="$(tr -d '\n' < "$HOME/.lambda_key")"
  test -n "$key"
  curl -fsS -u "$key:" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' \
    -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}"
}

maybe_terminate() {
  if test "$KEEP_INSTANCE_EFFECTIVE" -eq 1; then
    echo "KEEP_INSTANCE/NO_TERMINATE opt-in active; lifecycle ownership remains with the shared-VM orchestrator."
    return 0
  fi
  self_terminate
}

sync_out() {
  local stage="$1"
  python3 - "$stage" "$OUT_DIR/stage.json" <<'PY'
import json, os, pathlib, sys, tempfile, time
stage, dst = sys.argv[1:]
p = pathlib.Path(dst)
p.parent.mkdir(parents=True, exist_ok=True)
doc = {
    "stage": stage,
    "unix_time": time.time(),
    "instance_id": os.environ.get("INSTANCE_ID", ""),
    "run_id": os.environ.get("E32B_RUN_ID", ""),
    "keep_instance_effective": os.environ.get("KEEP_INSTANCE_EFFECTIVE") == "1",
    "reuse_shared": os.environ.get("E32B_REUSE_SHARED") == "1",
}
fd, tmp = tempfile.mkstemp(prefix=".stage.", suffix=".part", dir=p.parent)
with os.fdopen(fd, "w") as f:
    json.dump(doc, f, indent=1)
    f.write("\n")
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp, p)
PY
  rclone copy "$OUT_DIR" "$DEST/out" --checksum --transfers 8
  rclone check "$OUT_DIR" "$DEST/out" --one-way
}

on_error() {
  local rc=$?
  local line="$1"
  trap - ERR
  if test "$failure_active" -eq 1; then
    exit "$rc"
  fi
  failure_active=1
  set +e
  python3 - "$rc" "$line" "$OUT_DIR/FAILED.json" <<'PY'
import json, os, pathlib, sys, tempfile, time
rc, line, dst = sys.argv[1:]
p = pathlib.Path(dst)
p.parent.mkdir(parents=True, exist_ok=True)
doc = {
    "status": "FAILED",
    "exit_code": int(rc),
    "line": int(line),
    "instance_id": os.environ.get("INSTANCE_ID", ""),
    "run_id": os.environ.get("E32B_RUN_ID", ""),
    "unix_time": time.time(),
    "fallbacks": 0,
    "keep_instance_effective": os.environ.get("KEEP_INSTANCE_EFFECTIVE") == "1",
    "reuse_shared": os.environ.get("E32B_REUSE_SHARED") == "1",
}
fd, tmp = tempfile.mkstemp(prefix=".failed.", suffix=".part", dir=p.parent)
with os.fdopen(fd, "w") as f:
    json.dump(doc, f, indent=1)
    f.write("\n")
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp, p)
PY
  if command -v rclone >/dev/null 2>&1; then
    rclone copy "$OUT_DIR" "$DEST/out" --checksum --transfers 8
  fi
  maybe_terminate
  exit "$rc"
}
trap 'on_error "$LINENO"' ERR

# Standalone retains the existing API-key termination contract.  Shared reuse
# explicitly transfers lifecycle ownership and never needs/reads that key.
if test "$E32B_REUSE_SHARED" -eq 0; then
  test -s "$HOME/.lambda_key"
fi

# Hard stop after four hours in standalone mode.  On an explicitly shared VM,
# both success/failure termination and this timer are disabled; the caller then
# owns final termination after all co-located jobs (for example q100) finish.
if test "$KEEP_INSTANCE_EFFECTIVE" -eq 0; then
  nohup bash -c "sleep 14400; key=\$(tr -d '\n' < '$HOME/.lambda_key'); \
    curl -fsS -u \"\$key:\" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H 'Content-Type: application/json' \
    -d '{\"instance_ids\":[\"$INSTANCE_ID\"]}'" \
    >"$OUT_DIR/insurance_terminate.log" 2>&1 &
else
  echo "Shared VM mode: no E32b termination timer installed."
fi

if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash
fi
test -s "$RCLONE_CONFIG"
rclone lsf "$GDRIVE/1_data" --max-depth 1 >/dev/null
sync_out "boot"

# Source and immutable input artifacts.  No alternative path is tried.
rclone copyto "$GDRIVE/3_src/hsot/e32b_common.py" "$SRC_DIR/e32b_common.py"
rclone copyto "$GDRIVE/3_src/hsot/e32b_preflight.py" "$SRC_DIR/e32b_preflight.py"
rclone copyto "$GDRIVE/3_src/hsot/e32b_verdict.py" "$SRC_DIR/e32b_verdict.py"
rclone copyto "$GDRIVE/3_src/peft/e32b_exemplar_probe_cuda.py" "$SRC_DIR/e32b_exemplar_probe_cuda.py"
rclone copyto "$GDRIVE/3_src/peft/e32b_frames.json" "$INPUT_DIR/e32b_frames.json"
rclone copyto "$GDRIVE/5_outputs/submissions/sub_v023_cropwiden55.csv" "$INPUT_DIR/a_v023.csv"
rclone copyto "$GDRIVE/5_outputs/submissions/sub_v012_e23b_sam21_ablation.csv" "$INPUT_DIR/b_v012.csv"
rclone copyto "$GDRIVE/5_outputs/submissions/sub_v067_v056_firstframe_init.csv" "$INPUT_DIR/base_v067.csv"
rclone copyto "$GDRIVE/1_data/raw/sample_submisson.csv" "$INPUT_DIR/sample.csv"

if test "$E32B_REUSE_SHARED" -eq 0; then
  # Standalone path is unchanged in intent: own tar, own venv, own weight.
  rclone copyto "$GDRIVE/1_data/packed/t1test_fc_75.tar" "$E32B_DIR/test_fc.tar"
  tar -xf "$E32B_DIR/test_fc.tar" -C "$DATA_DIR"

  python3 "$SRC_DIR/e32b_preflight.py" \
    --manifest "$INPUT_DIR/e32b_frames.json" \
    --a-csv "$INPUT_DIR/a_v023.csv" \
    --b-csv "$INPUT_DIR/b_v012.csv" \
    --sample "$INPUT_DIR/sample.csv" \
    --frames-archive "$E32B_DIR/test_fc.tar" \
    --frames-root "$DATA_DIR" --require-frames \
    --report "$OUT_DIR/preflight_inputs.json"
  sync_out "input_preflight_pass"

  # Isolated environment: never reuse the generic ~/sam3env marker.
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="$HOME/.local/bin:$PATH"
  if ! test -d "$VENV_DIR"; then
    uv venv --python 3.12 "$VENV_DIR"
  fi
  VIRTUAL_ENV="$VENV_DIR" uv pip install -q torch torchvision --torch-backend=auto
  VIRTUAL_ENV="$VENV_DIR" uv pip install -q \
    "git+https://github.com/facebookresearch/sam3.git@${SAM3_REVISION}" \
    "numpy<2" ftfy==6.1.1 regex einops psutil scipy av pycocotools numba \
    python-rapidjson pandas pillow tqdm opencv-python-headless "setuptools<81" huggingface_hub

  "$PY3" - "$E32B_DIR/ckpt" <<'PY'
import pathlib, sys
from huggingface_hub import hf_hub_download
dst = pathlib.Path(sys.argv[1])
dst.mkdir(parents=True, exist_ok=True)
print(hf_hub_download("1038lab/sam3", "sam3.pt", local_dir=dst))
PY

  "$PY3" "$SRC_DIR/e32b_preflight.py" \
    --manifest "$INPUT_DIR/e32b_frames.json" \
    --a-csv "$INPUT_DIR/a_v023.csv" \
    --b-csv "$INPUT_DIR/b_v012.csv" \
    --sample "$INPUT_DIR/sample.csv" \
    --frames-archive "$E32B_DIR/test_fc.tar" \
    --frames-root "$DATA_DIR" --require-frames \
    --checkpoint "$CHECKPOINT" \
    --expected-checkpoint-sha256 "$SAM3_CHECKPOINT_SHA256" \
    --expected-python "$PY3" --expected-sam3-revision "$SAM3_REVISION" --require-cuda \
    --report "$OUT_DIR/preflight_full.json"
  PREFLIGHT_STAGE="standalone_environment_and_checkpoint_pass"
else
  # Shared-q100 reuse: the three paths were explicitly supplied above.  This
  # branch performs no uv/pip install, no checkpoint download, and no tar copy
  # or extraction.  Exact content/frame/runtime checks precede every GPU call.
  "$PY3" "$SRC_DIR/e32b_preflight.py" \
    --manifest "$INPUT_DIR/e32b_frames.json" \
    --a-csv "$INPUT_DIR/a_v023.csv" \
    --b-csv "$INPUT_DIR/b_v012.csv" \
    --sample "$INPUT_DIR/sample.csv" \
    --reuse-extracted-frames \
    --frames-root "$DATA_DIR" --require-frames \
    --checkpoint "$CHECKPOINT" \
    --expected-checkpoint-sha256 "$SAM3_CHECKPOINT_SHA256" \
    --expected-python "$PY3" --expected-sam3-revision "$SAM3_REVISION" --require-cuda \
    --report "$OUT_DIR/preflight_full.json"
  cp "$OUT_DIR/preflight_full.json" "$OUT_DIR/preflight_inputs.json"
  PREFLIGHT_STAGE="shared_q100_reuse_preflight_pass"
fi

"$PY3" - "$OUT_DIR/preflight_full.json" "$OUT_DIR/environment.json" \
  "$E32B_REUSE_SHARED" "$DATA_DIR" "$CHECKPOINT" <<'PY'
import importlib.metadata as md, json, pathlib, sys
preflight_path, output_path, reuse, frames, checkpoint = sys.argv[1:]
preflight = json.loads(pathlib.Path(preflight_path).read_text())
assert preflight["status"] == "PASS"
packages = sorted(
    {f"{d.metadata.get('Name', '')}=={d.version}" for d in md.distributions()}
)
doc = {
    "reuse_shared": reuse == "1",
    "frames_root": frames,
    "checkpoint": checkpoint,
    "preflight_manifest_payload_sha256": preflight["manifest_payload_sha256"],
    "runtime": preflight["python_runtime"],
    "checkpoint_identity": preflight["checkpoint"],
    "packages": packages,
}
pathlib.Path(output_path).write_text(json.dumps(doc, indent=1) + "\n")
PY
sync_out "$PREFLIGHT_STAGE"

# One model process per sequence costs a few extra model loads but permits an
# exact, remotely verified checkpoint after every sequence.  Resume is allowed
# only when the per-sequence payload validates against this manifest and model.
"$PY3" - "$INPUT_DIR/e32b_frames.json" "$E32B_DIR/sequence_order.txt" <<'PY'
import json, pathlib, sys
manifest, output = map(pathlib.Path, sys.argv[1:])
doc = json.loads(manifest.read_text())
output.write_text("\n".join(doc["sequence_order"]) + "\n")
PY

while IFS= read -r seq; do
  test -n "$seq"
  "$PY3" "$SRC_DIR/e32b_exemplar_probe_cuda.py" \
    --manifest "$INPUT_DIR/e32b_frames.json" \
    --frames-root "$DATA_DIR" \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$OUT_DIR" \
    --work-dir "$WORK_DIR" \
    --sam3-revision "$SAM3_REVISION" \
    --seq "$seq" 2>&1 | tee "$OUT_DIR/${seq}_console.log"
  sync_out "probe_${seq}_pass"
done < "$E32B_DIR/sequence_order.txt"

"$PY3" "$SRC_DIR/e32b_exemplar_probe_cuda.py" \
  --manifest "$INPUT_DIR/e32b_frames.json" \
  --out-dir "$OUT_DIR" \
  --assemble-only
sync_out "probe_assembled_pass"

"$PY3" "$SRC_DIR/e32b_verdict.py" \
  --manifest "$INPUT_DIR/e32b_frames.json" \
  --probe-result "$OUT_DIR/probe_result.json" \
  --base "$INPUT_DIR/base_v067.csv" \
  --sample "$INPUT_DIR/sample.csv" \
  --candidate-out "$OUT_DIR/candidate_unscored_offline.csv" \
  --report "$OUT_DIR/verdict.json" 2>&1 | tee "$OUT_DIR/verdict_console.log"

"$PY3" - "$OUT_DIR/SUCCESS.json" <<'PY'
import json, os, pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status": "COMPLETE",
    "fallbacks": 0,
    "kaggle_submitted": False,
    "lambda_was_created_by_this_script": False,
    "run_id": os.environ.get("E32B_RUN_ID", ""),
    "keep_instance_effective": os.environ.get("KEEP_INSTANCE_EFFECTIVE") == "1",
    "reuse_shared": os.environ.get("E32B_REUSE_SHARED") == "1",
    "unix_time": time.time(),
}, indent=1) + "\n")
PY
sync_out "complete"
maybe_terminate
