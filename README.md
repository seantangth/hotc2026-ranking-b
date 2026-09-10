# Ranking B submission — HOTC 2026 (team `seantangth`)

This repository is our Stage II submission. It contains everything needed to reproduce our
Ranking B results from the official data: source code, configuration, pinned dependency locks,
the two learned weight files used in post-processing, and the licences of the third-party
models it builds on.

## Quick start

On a Linux machine with one NVIDIA GPU (16 GB or more) and a CUDA 12 driver:

```bash
git clone https://github.com/seantangth/hotc2026-ranking-b.git
cd hotc2026-ranking-b
bash setup_and_run.sh --ranking-dir /path/to/ranking --dry-run   # setup + checks only
bash setup_and_run.sh --ranking-dir /path/to/ranking             # 4-5 h for 75 sequences
```

`--ranking-dir` is the official folder holding `HSI-NIR-Falsecolor/`,
`HSI-RedNIR-Falsecolor/` and `HSI-VIS-FalseColor/`. The script builds both virtual
environments, fetches and hash-verifies the two public checkpoints, converts the official
folder layout, runs the pre-flight checks and then writes **`submission.csv`** — one row per
frame, in the format of the official `sample_submission.csv`. That single file is our submission.

A single-pass variant, `submission_onepass.csv`, can also be produced with `--with-onepass`.

Run `--dry-run` first: it does the whole setup and every fail-closed check, then stops before
inference, so any problem surfaces immediately rather than hours in. It took **2 min 35 s** on a
cloud A10 (dominated by ~4.3 GB of checkpoint downloads and two `torch` installs, so allow longer
on a slower link). Every step is also documented as a standalone command below, in case you would
rather drive the stages yourself.

> **Verified end to end.** 2026-09-08: run on a clean cloud GPU machine over the 75 sample ranking
> sequences released on 7 September, starting from the official folder layout — the main chain
> finished in 4 h 04 min on one NVIDIA A10 with no manual intervention.
> 2026-09-09: the Quick start above was re-run verbatim from a fresh `git clone` of this
> repository on a new machine, through to `BLOCK=0`, in 2 min 35 s.

---

## 0. Package contents

| File | What it is |
|---|---|
| `setup_and_run.sh` | One command: environment, checkpoints, ingestion, checks, inference. |
| `3_src/` | All source code. Single entry point is `3_src/run_ranking_b.py`. |
| `3_src/configs/ranking_profiles.json` | Every setting of every profile. Nothing is hard-coded outside this file. |
| `3_src/requirements-rankingb.txt` | Dependency notes, weight URLs and SHA-256 hashes. |
| `3_src/env_locks/` | Fully resolved package locks from the machines the results were produced on. |
| `SAM_LICENSE.txt` | Meta's SAM License, redistributed with the `sam3.pt` checkpoint as that licence requires (§3). |

---

## 1. Which command produces which file

Everything runs through the **same entry point**, `3_src/run_ranking_b.py`; what changes is
`--profile` (and one explicit flag, see the note). `setup_and_run.sh` wraps these calls.

### `submission.csv` — our submission

```bash
python 3_src/run_ranking_b.py \
    --profile rankB_deliver_v090 \
    --allow-offline-two-pass \
    --frames-root  <FRAMES_ROOT> \
    --sample       <sample_submission.csv> \
    --sam3-python    <SAM3_VENV>/bin/python \
    --samurai-python <T1_VENV>/bin/python \
    --sam3-ckpt      <PATH>/sam3.pt \
    --samurai-dir    <PATH>/samurai \
    --samurai-ckpt   <PATH>/sam2.1_hiera_large.pt \
    --out            submission.csv \
    --execute
```

> ⚠️ **`--allow-offline-two-pass` is required for this profile.** Without it the run
> stops at pre-flight with a `BLOCK` and produces no output.

### `submission_onepass.csv` — single-pass variant (optional)

```bash
python 3_src/run_ranking_b.py \
    --profile rankB_robust \
    --frames-root <FRAMES_ROOT> \
    --sample      <sample_submission.csv> \
    ...same environment flags as above... \
    --out         submission_onepass.csv \
    --execute
```

Do **not** pass `--allow-offline-two-pass` with this profile; pre-flight refuses the
combination with a `BLOCK`.

### Dry run first (recommended)

Omit `--execute` to get the full command plan plus fail-closed pre-flight checks without
running anything. The entry point itself uses only the Python standard library, so a dry run
works even before the two model environments exist:

```bash
python 3_src/run_ranking_b.py --profile rankB_deliver_v090 --allow-offline-two-pass
# → prints every step, then "DRY-RUN ONLY（BLOCK=n）"
```

`BLOCK=0` means the environment is complete. Add `--execute` only then.

---

## 2. Environment

**Two isolated virtual environments are required** — one per tracker leg. They cannot be
merged: the two upstream packages pull incompatible pins, and a shared env silently breaks
old C extensions against the NumPy 2.x ABI.

| venv | Contents |
|---|---|
| `sam3env` | `torch`, `torchvision`, SAM 3 (pinned commit below), `einops`, `pycocotools`, `psutil`, `hydra-core`, `iopath`, `timm`, `tqdm`, `pillow`, `numpy`, `pandas` |
| `t1env` | `torch`, `torchvision`, SAMURAI + its bundled `sam2` package, `scipy`, `loguru`, `opencv-python-headless` |

```bash
# both venvs — install torch first so the CUDA build matches the host driver
uv pip install torch torchvision --torch-backend=auto

# sam3env
uv pip install "setuptools<81"        # see warning below
uv pip install git+https://github.com/facebookresearch/sam3.git@96914d2425f90a64f45ca977c2b5165418099543
uv pip install einops pycocotools psutil hydra-core iopath timm tqdm pillow numpy pandas

# t1env
git clone https://github.com/yangchris11/samurai.git && \
  git -C samurai checkout 76ba195984892b0d1e3db5d9c9f90bb62175680a
uv pip install -e samurai/sam2
uv pip install scipy loguru opencv-python-headless
```

> 🚨 **`setuptools<81` is a hard requirement, not a preference.** setuptools 81 removed
> `pkg_resources`, which `sam3/model_builder.py` imports at module scope. On any clean machine
> that resolves a current setuptools, `import sam3` dies immediately with `ModuleNotFoundError`.

> 🚨 The published `sam3` package does **not declare** `einops`, `pycocotools` or `psutil`,
> yet `import sam3` needs all three. Installing the `[train]` extras does not help.

Instead of the two `--sam3-python` / `--samurai-python` flags you may export the
environment variables **`SAM3_PYTHON`** and **`SAMURAI_PYTHON`**.

Fully resolved locks from the machines that produced the reported results are in
`3_src/env_locks/`. Observed torch on those runs: `2.13.0+cu129`.

---

## 3. Model weights

Both checkpoints are supplied with this submission, together with their SHA-256 hashes and the
licence text the SAM 3 checkpoint is distributed under (the SAM License requires a copy of the
agreement to accompany any redistribution; it is `SAM_LICENSE.txt` in this package and next to
the file in the folder below).

**Download folder** (Google Drive, anyone with the link):
`https://drive.google.com/open?id=1H91SQacOIZke2ZeWxGF_GzcxMmgkJP4N`

| File | Size | SHA-256 | Direct link |
|---|---|---|---|
| `sam3.pt` | 3,450,062,241 B | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` | `https://drive.google.com/open?id=1pkGt0b0QLAAaDLxcc7dd2BFXBoXcZPPI` |
| `sam2.1_hiera_large.pt` | 898,083,611 B | `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` | `https://drive.google.com/open?id=1Kakd297Ws9HKCf9u_hn_w62I5OytPZhV` |
| `SAM_LICENSE.txt` | — | — | `https://drive.google.com/open?id=1bgj12fqHxwXZ49wD3qi_hC8Ovh4VVaa6` |

`SHA256SUMS` in the same folder lets you verify with `sha256sum -c SHA256SUMS`. The pipeline also
checks the SAM 3 hash itself and refuses to start on a mismatch.

The files are byte-identical to the public releases: `sam3.pt` to
`https://huggingface.co/facebook/sam3/resolve/main/sam3.pt` (a gated repository — access is
granted by Meta on request, typically within a day) and `sam2.1_hiera_large.pt` to the SAM 2.1
release used by SAMURAI. Either source may be used interchangeably; the hashes above are the
identity, not the download location.

---

## 4. Input layout

`--frames-root` must point at a directory with one sub-directory per sequence:

```
<FRAMES_ROOT>/
  <sequence-name>/
      0001.jpg  0002.jpg  ...      # false-colour frames
      init_rect.txt                # first-frame box: x,y,w,h
```

**Sequence names must carry the modality prefix** `nir-`, `rednir-` or `vis-` (e.g.
`nir-blackball2`). The official distribution puts the modality in the *parent* folder name
instead (`HSI-NIR-Falsecolor/blackball2/`), so do not point `--frames-root` at the official
folders directly — run the ingestion script, which derives the prefix from the folder name
and writes the layout above:

```bash
# <RANKING_DIR> is the official folder that contains HSI-NIR-Falsecolor/,
# HSI-RedNIR-Falsecolor/ and HSI-VIS-FalseColor/ (the 16-bit HSI-* mosaic folders are
# skipped automatically; only the false-colour JPEG frames are consumed).
python 3_src/prep/prep_rankingb_frames_v1.py \
    --source <RANKING_DIR> --frames-out <FRAMES_ROOT> \
    --write-sample <FRAMES_ROOT>/sample_submission.csv
```

It reads `init_rect.txt` (falling back to the first line of `groundtruth_rect.txt`), never copies
ground truth into `<FRAMES_ROOT>`, refuses 16-bit mosaic PNGs and `(x1,y1,x2,y2)` boxes, and
writes `INGEST_MANIFEST.json`. Add `--dry-run` to check without writing.

`--sample` is the official `sample_submission.csv`. It is treated as a **contract**: the set of
IDs, their order within each sequence, and the per-sequence frame counts must all match the
frames on disk, or pre-flight fails closed. Row order inside the file is significant — it is
the order predictions are emitted in — and is checked to be strictly ascending per sequence.
If no official sample file exists for the ranking set, use the one produced by
`--write-sample` above: same format (`ID,x,y,width,height`, `ID = <sequence>_<1-based frame>`),
sequences in sorted order.

**Expected runtime**: **4–5 hours** for 75 sequences on a single NVIDIA A10, for the whole
main-profile chain (four tracker legs plus three crop-window passes). Measured end-to-end:
4 h 40 min for 26,860 frames and 4 h 04 min for 30,338 frames, on two different A10 cards —
individual cards vary more than frame count does, so treat this as a range. The one-pass
alternate is roughly half. An A100 is roughly twice as fast.
(four tracker legs plus three crop-window passes). An A100 is roughly twice as fast.

---

## 5. Method

The pipeline is zero-shot at the tracker level — no model is trained on the competition
training set. It combines two public trackers and then arbitrates between them per frame:

1. **Two tracker legs, both zero-shot**: SAM 3 (primary) and SAM 2.1 driven by SAMURAI's
   motion model (secondary). Each is initialised from the first-frame ground-truth box only.
2. **Target-aware crop**: a first pass tracks the full frame; the envelope of that trajectory
   defines a crop window, and a second pass re-tracks inside the crop at higher effective
   resolution. Three crop windows are computed from the run's own first pass and combined per
   frame by a median-of-three consensus.
3. **Coordinate-convention correction** of +1 px on the top and left edges.
4. **Cross-backbone frozen-run rescue** (`K=6`): where the primary leg's box has been frozen
   for ≥6 consecutive frames and the secondary leg is still moving, the secondary box is taken.
5. **Per-frame selector** (ridge regression on a 25-dimensional feature vector — 22 geometric
   terms plus 3 modality one-hots — with threshold `τ=0.05`) choosing between the two legs.
6. **RedNIR quality head** and a **third-leg dead-zone rescue** for frames where both legs froze.

Run end-to-end by exactly these commands on the Ranking A validation set (fresh A10,
2026-09-06), `submission.csv` (profile `rankB_deliver_v090`) scored **0.71096**. The
single-pass variant `submission_onepass.csv` (profile `rankB_robust`: no crop, no selector,
no quality head) scored **0.68148**.

---

## 6. Third-party components and licences

| Component | Version / commit | Licence |
|---|---|---|
| SAM 3 (`facebookresearch/sam3`) | `96914d2425f90a64f45ca977c2b5165418099543` | SAM License |
| `sam3.pt` weights | see §3 | SAM License (a copy is included as `SAM_LICENSE.txt`) |
| SAMURAI (`yangchris11/samurai`) | `76ba195984892b0d1e3db5d9c9f90bb62175680a` | Apache-2.0 |
| SAM 2.1 / `sam2` package + weights | bundled with SAMURAI | Apache-2.0 |
| PyTorch, torchvision, timm, hydra-core, einops, iopath, pycocotools, psutil, scipy, loguru, OpenCV, NumPy, pandas, Pillow, tqdm | see `3_src/env_locks/` | BSD-3-Clause / Apache-2.0 / MIT (per package) |

Our own code in `3_src/` is released under the licence in `LICENSE`.


---

## 7. Notes for whoever runs this

- Pre-flight messages, profile descriptions and in-code comments are in **Traditional Chinese**.
  The command-line interface, file names and this document are in English. If any diagnostic is
  unclear, please contact us rather than guessing.
- Every stage is **fail-closed**: a missing input, a hash mismatch, a sample-contract violation
  or a sequence-count mismatch stops the run rather than producing a partial file.
- `python 3_src/finalize_submission.py --selftest` reproduces four historical submissions
  bit-for-bit from archived intermediates. It is a regression check on the post-processing chain
  and needs files that are not part of this package; it is not required to reproduce a submission.
- `python -m pytest 3_src` runs the unit test suite (no GPU, no data required).

Contact: via the Kaggle competition page or the address used for our participation confirmation.
