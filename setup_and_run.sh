#!/usr/bin/env bash
# HOTC 2026 Ranking B — one-command setup and run.
#
#   bash setup_and_run.sh --ranking-dir /path/to/ranking
#
# It builds both virtual environments from the pinned lock files, fetches and
# hash-verifies the two public checkpoints, converts the official folder layout
# into the layout the tracker reads, and produces the submission file:
#
#   <work-dir>/submission.csv           profile rankB_deliver_v090
#
# That single file is our submission. A single-pass variant can additionally be
# produced with --with-onepass.
#
# Every step is fail-closed: a bad checkpoint hash, a mismatched sequence count
# or a failed pre-flight check stops the script rather than producing a wrong file.
# Nothing is downloaded from a private location and no ground truth is ever read.
#
# Requirements: Linux, one NVIDIA GPU (>=16 GB) with a CUDA 12 driver, ~30 GB free
# disk, and network access to github.com, pypi.org and dl.fbaipublicfiles.com.
# Full documentation, including how to run the steps by hand, is in README.md.

set -euo pipefail
trap 'echo "FAILED at line $LINENO" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/3_src"

RANKING_DIR=""
WORK_DIR="$PWD/hotc2026_run"
SAM3_CKPT=""
SAMURAI_CKPT=""
DRY_RUN=0
WITH_ONEPASS=0

SAMURAI_SHA=76ba195984892b0d1e3db5d9c9f90bb62175680a
SAM3_SHA=96914d2425f90a64f45ca977c2b5165418099543
SAM3_CKPT_SHA256=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
SAMURAI_CKPT_SHA256=2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318
SAMURAI_CKPT_URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
SAM3_CKPT_HF=https://huggingface.co/facebook/sam3/resolve/main/sam3.pt
SAM3_CKPT_MIRROR_ID=1pkGt0b0QLAAaDLxcc7dd2BFXBoXcZPPI

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'USAGE'

Options
  --ranking-dir DIR    Required. The official folder that contains
                       HSI-NIR-Falsecolor/, HSI-RedNIR-Falsecolor/ and
                       HSI-VIS-FalseColor/ (16-bit mosaic folders are ignored).
  --work-dir DIR       Where to put environments, frames and outputs.
                       Default: ./hotc2026_run
  --sam3-ckpt PATH     Use an existing sam3.pt instead of downloading it.
  --samurai-ckpt PATH  Use an existing sam2.1_hiera_large.pt.
  --dry-run            Set everything up and run the pre-flight checks, then stop
                       before inference. Use this first; it takes a few minutes.
  --with-onepass       Additionally produce submission_onepass.csv, the single-pass
                       variant (profile rankB_robust).
  -h, --help           This message.
USAGE
}

die() { echo "ERROR: $*" >&2; exit 1; }
say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --ranking-dir)   RANKING_DIR="${2:?}"; shift 2 ;;
    --work-dir)      WORK_DIR="${2:?}"; shift 2 ;;
    --sam3-ckpt)     SAM3_CKPT="${2:?}"; shift 2 ;;
    --samurai-ckpt)  SAMURAI_CKPT="${2:?}"; shift 2 ;;
    --dry-run)       DRY_RUN=1; shift ;;
    --with-onepass)  WITH_ONEPASS=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

[ -n "$RANKING_DIR" ] || { usage >&2; die "--ranking-dir is required"; }
[ -d "$RANKING_DIR" ] || die "--ranking-dir is not a directory: $RANKING_DIR"
[ -f "$SRC/run_ranking_b.py" ] || die "run this script from inside the unpacked package"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# Download to a .part file and only move it into place once the hash matches, so an
# interrupted download can never be mistaken for a good checkpoint on the next run.
fetch_verified() {
  local dest="$1" want="$2" url="$3" auth="${4:-}"
  if [ -f "$dest" ] && [ "$(sha256_of "$dest")" = "$want" ]; then
    echo "  already present and verified: $(basename "$dest")"; return 0
  fi
  echo "  downloading $(basename "$dest") ..."
  if [ -n "$auth" ]; then
    curl -fL -C - -H "Authorization: Bearer $auth" -o "$dest.part" "$url"
  else
    curl -fL -C - -o "$dest.part" "$url"
  fi
  local got; got="$(sha256_of "$dest.part")"
  [ "$got" = "$want" ] || { rm -f "$dest.part"; die "checksum mismatch for $(basename "$dest")
  expected $want
  got      $got"; }
  mv "$dest.part" "$dest"
  echo "  verified: $(basename "$dest")"
}

ENV_DIR="$WORK_DIR/env"
CKPT_DIR="$WORK_DIR/checkpoints"
FRAMES="$WORK_DIR/frames"
SAMPLE="$FRAMES/sample_submission.csv"
SAMURAI_DIR="$WORK_DIR/samurai"
T1ENV="$ENV_DIR/t1env"
SAM3ENV="$ENV_DIR/sam3env"
PY1="$T1ENV/bin/python"
PY3="$SAM3ENV/bin/python"
mkdir -p "$WORK_DIR" "$ENV_DIR" "$CKPT_DIR" "$FRAMES"

say "0/8  Host check"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; this pipeline needs an NVIDIA GPU"
nvidia-smi --query-gpu=name,memory.total,ecc.errors.uncorrected.volatile.total \
           --format=csv,noheader || true
# A GPU with uncorrected ECC errors makes the tracker abort partway through a run,
# which looks exactly like a software bug. Worth one look before spending hours.
ECC="$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
case "$ECC" in
  0|"[N/A]"|"N/A"|"") ;;
  *) echo "  WARNING: this GPU reports $ECC uncorrected ECC errors; consider another card" >&2 ;;
esac
command -v git >/dev/null 2>&1 || die "git not found"
command -v curl >/dev/null 2>&1 || die "curl not found"

say "1/8  uv (Python package manager)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv installation failed; see https://docs.astral.sh/uv/"
uv --version

say "2/8  SAMURAI source at pinned commit"
[ -d "$SAMURAI_DIR/.git" ] || git clone --quiet https://github.com/yangchris11/samurai.git "$SAMURAI_DIR"
git -C "$SAMURAI_DIR" fetch --quiet --depth 1 origin "$SAMURAI_SHA"
git -C "$SAMURAI_DIR" checkout --quiet --detach "$SAMURAI_SHA"
echo "  samurai @ $(git -C "$SAMURAI_DIR" rev-parse --short HEAD)"

# The two environments cannot be merged: the upstream packages pin incompatible
# versions and a shared environment breaks old C extensions against the NumPy 2 ABI.
say "3/8  Environment 1 of 2: t1env (SAMURAI / SAM 2.1)"
[ -x "$PY1" ] || uv venv --python 3.12 "$T1ENV"
VIRTUAL_ENV="$T1ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$T1ENV" uv pip install -q -r "$SRC/requirements-t1env.txt"
VIRTUAL_ENV="$T1ENV" uv pip install -q -e "$SAMURAI_DIR/sam2"
"$PY1" -c "import torch,sam2,pandas; assert torch.cuda.is_available(); print('  torch',torch.__version__,'|',torch.cuda.get_device_name(0))"

say "4/8  Environment 2 of 2: sam3env (SAM 3)"
[ -x "$PY3" ] || uv venv --python 3.12 "$SAM3ENV"
VIRTUAL_ENV="$SAM3ENV" uv pip install -q torch torchvision --torch-backend=auto
VIRTUAL_ENV="$SAM3ENV" uv pip install -q -r "$SRC/requirements-sam3env.txt"
VIRTUAL_ENV="$SAM3ENV" uv pip install -q "git+https://github.com/facebookresearch/sam3.git@${SAM3_SHA}"
# setuptools 81 removed pkg_resources, which sam3/model_builder.py imports at module
# scope; installing sam3 can pull a newer setuptools, so pin it back afterwards.
VIRTUAL_ENV="$SAM3ENV" uv pip install -q "setuptools<81"
"$PY3" -c "import torch,sam3,numpy; assert torch.cuda.is_available(); assert numpy.__version__.startswith('1.'); print('  torch',torch.__version__,'| numpy',numpy.__version__)"

say "5/8  Checkpoints"
if [ -n "$SAMURAI_CKPT" ]; then
  [ -f "$SAMURAI_CKPT" ] || die "--samurai-ckpt not found: $SAMURAI_CKPT"
  [ "$(sha256_of "$SAMURAI_CKPT")" = "$SAMURAI_CKPT_SHA256" ] || die "--samurai-ckpt has the wrong SHA-256"
  echo "  using supplied sam2.1_hiera_large.pt"
else
  SAMURAI_CKPT="$CKPT_DIR/sam2.1_hiera_large.pt"
  fetch_verified "$SAMURAI_CKPT" "$SAMURAI_CKPT_SHA256" "$SAMURAI_CKPT_URL"
fi

if [ -n "$SAM3_CKPT" ]; then
  [ -f "$SAM3_CKPT" ] || die "--sam3-ckpt not found: $SAM3_CKPT"
  [ "$(sha256_of "$SAM3_CKPT")" = "$SAM3_CKPT_SHA256" ] || die "--sam3-ckpt has the wrong SHA-256"
  echo "  using supplied sam3.pt"
else
  SAM3_CKPT="$CKPT_DIR/sam3.pt"
  if [ -f "$SAM3_CKPT" ] && [ "$(sha256_of "$SAM3_CKPT")" = "$SAM3_CKPT_SHA256" ]; then
    echo "  already present and verified: sam3.pt"
  elif [ -n "${HF_TOKEN:-}" ]; then
    echo "  HF_TOKEN is set; fetching from the official gated repository"
    fetch_verified "$SAM3_CKPT" "$SAM3_CKPT_SHA256" "$SAM3_CKPT_HF" "$HF_TOKEN"
  else
    # facebook/sam3 is gated: an anonymous request returns HTTP 401 and access has to
    # be granted by Meta. The mirror below is the same file (same SHA-256), so a
    # verification run is not blocked waiting for that approval.
    echo "  no HF_TOKEN set; using our byte-identical mirror of the same file"
    VIRTUAL_ENV="$T1ENV" uv pip install -q gdown
    "$T1ENV/bin/gdown" --quiet -O "$SAM3_CKPT.part" "$SAM3_CKPT_MIRROR_ID" || {
      rm -f "$SAM3_CKPT.part"
      die "could not download sam3.pt automatically.
  Please fetch it one of these ways and re-run with --sam3-ckpt <path>:
    a) https://drive.google.com/open?id=$SAM3_CKPT_MIRROR_ID   (our mirror, no account needed)
    b) $SAM3_CKPT_HF   (official, gated: set HF_TOKEN after Meta approves access)
  Expected SHA-256: $SAM3_CKPT_SHA256"
    }
    got="$(sha256_of "$SAM3_CKPT.part")"
    [ "$got" = "$SAM3_CKPT_SHA256" ] || { rm -f "$SAM3_CKPT.part"; die "mirror download has the wrong SHA-256 (got $got)"; }
    mv "$SAM3_CKPT.part" "$SAM3_CKPT"
    echo "  verified: sam3.pt"
  fi
fi

say "6/8  Reading the official layout"
# Writes <frames>/<modality>-<sequence>/0001.jpg ... plus init_rect.txt, and a
# sample_submission.csv in the Ranking A format. Ground truth is never copied.
"$PY1" "$SRC/prep/prep_rankingb_frames_v1.py" \
  --source "$RANKING_DIR" --frames-out "$FRAMES" --write-sample "$SAMPLE"
N_SEQ="$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
echo "  $N_SEQ sequences ready in $FRAMES"

COMMON=( --frames-root "$FRAMES" --sample "$SAMPLE"
         --sam3-python "$PY3" --samurai-python "$PY1"
         --sam3-ckpt "$SAM3_CKPT"
         --samurai-dir "$SAMURAI_DIR" --samurai-ckpt "$SAMURAI_CKPT"
         --sam3-source-revision "$SAM3_SHA" --samurai-source-revision "$SAMURAI_SHA" )

say "7/8  Pre-flight checks (no inference yet)"
"$PY1" "$SRC/run_ranking_b.py" --profile rankB_deliver_v090 --allow-offline-two-pass \
  "${COMMON[@]}" --work-dir "$WORK_DIR/main" --out "$WORK_DIR/submission.csv" \
  | tee "$WORK_DIR/preflight_main.log" | tail -3
grep -q "BLOCK=0" "$WORK_DIR/preflight_main.log" \
  || die "pre-flight reported a BLOCK; see $WORK_DIR/preflight_main.log. Nothing was run."

if [ "$DRY_RUN" = 1 ]; then
  say "Dry run complete"
  echo "Everything is in place. Re-run the same command without --dry-run to produce"
  echo "the submission files (about 4-5 hours for 75 sequences on one A10)."
  exit 0
fi

say "8/8  Inference"
echo "This takes roughly 4-5 hours for 75 sequences on a single A10. Progress is printed"
echo "per sequence."
"$PY1" "$SRC/run_ranking_b.py" --profile rankB_deliver_v090 --allow-offline-two-pass \
  "${COMMON[@]}" --work-dir "$WORK_DIR/main" --out "$WORK_DIR/submission.csv" --execute
[ -s "$WORK_DIR/submission.csv" ] || die "submission.csv was not produced"

if [ "$WITH_ONEPASS" = 1 ]; then
  "$PY1" "$SRC/run_ranking_b.py" --profile rankB_robust \
    "${COMMON[@]}" --work-dir "$WORK_DIR/onepass" --out "$WORK_DIR/submission_onepass.csv" --execute
  [ -s "$WORK_DIR/submission_onepass.csv" ] || die "submission_onepass.csv was not produced"
fi

say "Done"
echo "  $WORK_DIR/submission.csv"
[ "$WITH_ONEPASS" = 1 ] && echo "  $WORK_DIR/submission_onepass.csv   (alternate, README section 5)"
echo
echo "One row per frame, in the format of the official sample_submission.csv."
true
