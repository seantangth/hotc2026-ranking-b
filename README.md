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
bash setup_and_run.sh --ranking-dir /path/to/ranking --dry-run   # a few minutes
bash setup_and_run.sh --ranking-dir /path/to/ranking             # 4-5 h for 75 sequences
```

`--ranking-dir` is the official folder holding `HSI-NIR-Falsecolor/`,
`HSI-RedNIR-Falsecolor/` and `HSI-VIS-FalseColor/`. The script builds both virtual
environments, fetches and hash-verifies the two public checkpoints, converts the official
folder layout, runs the pre-flight checks and then produces:

| Output | Profile | What it is |
|---|---|---|
| `submission_main.csv` | `rankB_deliver_v090` | **Primary submission.** |
| `submission_onepass.csv` | `rankB_robust` | Strictly causal alternate, see §5. |

Run `--dry-run` first: it does the whole setup and every fail-closed check, then stops before
inference, so a problem surfaces in minutes rather than hours. Every step is also documented
as a standalone command below, in case you would rather drive the stages yourself.

> Verified on 2026-09-08: unpacked on a clean cloud GPU machine, run end-to-end on the 75
> sample ranking sequences released on 7 September, starting from the official folder layout.
> The main chain finished in 4 h 04 min on one NVIDIA A10 with no manual intervention.

---

## 0. Package contents

| File | What it is |
|---|---|
| `submission_main.csv` | **Primary submission.** Produced by profile `rankB_deliver_v090`. |
| `submission_onepass.csv` | Strictly causal one-pass alternate. Produced by profile `rankB_robust`. See §5. |
| `3_src/` | All source code. Single entry point is `3_src/run_ranking_b.py`. |
| `3_src/configs/ranking_profiles.json` | Every setting of every profile. Nothing is hard-coded outside this file. |
| `3_src/requirements-rankingb.txt` | Dependency notes, weight URLs and SHA-256 hashes. |
| `3_src/env_locks/` | Fully resolved package locks from the machines the results were produced on. |
| `setup_and_run.sh` | One-command setup and run (see Quick start above). |
| `SAM_LICENSE.txt` | Meta's SAM License, redistributed with the `sam3.pt` checkpoint as that licence requires (§3). |

---

## 1. Which command produces which file

Both files come from the **same entry point**; they differ only by `--profile`
(and by one explicit flag, see the note).

### `submission_main.csv` — primary

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
    --out            submission_main.csv \
    --execute
```

> ⚠️ **`--allow-offline-two-pass` is mandatory for this profile.** Without it the run
> stops at pre-flight with a `BLOCK` and produces no output. This is deliberate: the flag is
> how the pipeline forces the two-pass structure (§5) to be an explicit, auditable choice
> rather than a silent default.

### `submission_onepass.csv` — strictly causal alternate

```bash
python 3_src/run_ranking_b.py \
    --profile rankB_robust \
    --frames-root <FRAMES_ROOT> \
    --sample      <sample_submission.csv> \
    ...same environment flags as above... \
    --out         submission_onepass.csv \
    --execute
```

This profile is `profile_intent=disabled`, so it does **not** take `--allow-offline-two-pass`
and never enters the two-pass path.

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

## 5. Method, and a disclosure about the two-pass structure

The pipeline is zero-shot at the tracker level — no model is trained on the competition
training set. It combines two public trackers and then arbitrates between them per frame:

1. **Two tracker legs, both zero-shot**: SAM 3 (primary) and SAM 2.1 driven by SAMURAI's
   motion model (secondary). Each is initialised from the first-frame ground-truth box only.
2. **Coordinate-convention correction** of +1 px on the top and left edges.
3. **Cross-backbone frozen-run rescue** (`K=6`): where the primary leg's box has been frozen
   for ≥6 consecutive frames and the secondary leg is still moving, the secondary box is taken.
4. **Per-frame selector** (ridge regression on a 25-dimensional feature vector — 22 geometric
   terms plus 3 modality one-hots, see §5's modality disclosure — with threshold `τ=0.05`)
   choosing between the two legs.
5. **RedNIR quality head** and a **third-leg dead-zone rescue** for frames where both legs froze.

### Disclosure: `rankB_deliver_v090` is a two-pass pipeline

We want to be explicit about this rather than leave it to be discovered in the code.

The primary submission uses a **target-aware crop**: a first pass tracks the full frame, the
envelope of that trajectory is used to define a crop window, and a second pass re-tracks inside
the crop at higher effective resolution. **The crop window for a sequence is therefore derived
from predictions on later frames of that same sequence.** No ground truth beyond the first frame
is used at any point, and no information crosses between sequences — but the second pass is not
causal within a sequence, so it is not a literal one-pass run in the strictest reading of OPE.

We could not find a ruling on this in the rules or the forum, and our question to the organizers
went unanswered, so we are supplying **both** interpretations and leaving the choice to you:

- `submission_main.csv` — two-pass, as described.
- `submission_onepass.csv` — `rankB_robust`, strictly causal, single pass, no crop.

If the two-pass structure is not acceptable, please score `submission_onepass.csv`.

For reference, both files were produced end-to-end by exactly these commands on the public
Ranking A validation set and scored there as follows: `submission_main.csv` (profile
`rankB_deliver_v090`) **0.71096**; `submission_onepass.csv` (profile `rankB_robust`) **0.68148**
(fresh A10 run, 2026-09-06). The two-pass crop and its consensus account for the difference.

Two further points of transparency about how constants were chosen: the frozen-run length
`K=6` and the selector threshold `τ=0.05` were selected from dose curves measured on the
Ranking A leaderboard, and the +1 px correction's **top** edge was derived from the training-set
annotation convention while its **left** edge was measured on the Ranking A leaderboard only.
The three crop windows combined by the median-of-three consensus are fixed in advance and are
computed from each run's own first pass — they are not selected per sequence or per dataset.

### Disclosure: the main file reads the modality prefix of the sequence ID

Protocol 3 requires the same model hyper-parameters for all sequences, so we should be precise
about the one place where our post-processing is not literally identical across sequences.

`submission_main.csv` reads the **modality prefix** of the sequence folder name — `nir-`,
`rednir-` or `vis-` — in exactly two places, both in `3_src/finalize_submission.py` via
`hsot/quality_head_v1.py::modality`:

1. The RedNIR quality head is **gated to `rednir-` sequences** (`finalize_submission.py:221`).
2. The modality enters both learned heads as **three one-hot features** of the 25-dimensional
   feature vector; the other 22 terms are geometric (`quality_head_v1.py:86-115`).

Our reading is that this is a dataset-level attribute rather than per-sequence tuning: the three
modalities are distinct sensors with 16/25/15 bands, the official distribution encodes the label
in the folder name, and the same fixed weights are applied to every sequence of a given modality.
No component reads the identity of an *individual* sequence, and there is no per-sequence
constant or lookup table anywhere in the pipeline.

If you nevertheless consider the modality gate to be outside Protocol 3, two fallbacks are
already in this package and need no code change:

- `submission_onepass.csv` (`rankB_robust`) uses **no** quality head and **no** selector, so it
  reads no modality at all in post-processing.
- `3_src/configs/ranking_profiles_lean.json` defines `rankB_deliver_v090_lean`: identical to the
  main profile with the selector and quality head disabled, keeping the crop consensus. Run it
  with `--config 3_src/configs/ranking_profiles_lean.json --profile rankB_deliver_v090_lean`.

The tracking stage itself (`track_t1.py`) never branches on modality; the prefix appears there
only in log counts and in a diagnostic field.

---

## 6. Third-party components and licences

| Component | Version / commit | Licence |
|---|---|---|
| SAM 3 (`facebookresearch/sam3`) | `96914d2425f90a64f45ca977c2b5165418099543` | **SAM License** (Meta custom; not OSI-approved; contains use restrictions) |
| `sam3.pt` weights | see §3 | SAM License (redistributed with a copy of the licence, `SAM_LICENSE.txt`, as its §1.b.i requires) |
| SAMURAI (`yangchris11/samurai`) | `76ba195984892b0d1e3db5d9c9f90bb62175680a` | Apache-2.0 |
| SAM 2.1 / `sam2` package + weights | bundled with SAMURAI | Apache-2.0 |
| PyTorch, torchvision, timm, hydra-core, einops, iopath, pycocotools, psutil, scipy, loguru, OpenCV, NumPy, pandas, Pillow, tqdm | see `3_src/env_locks/` | BSD-3-Clause / Apache-2.0 / MIT (per package) |

Our own code in `3_src/` is released under the licence in `LICENSE`.

The pipeline can also be run **without SAM 3** — `track_t1.py --backend samurai` uses only the
Apache-2.0 legs. That configuration scores lower, but it exists and is a single flag away should
the SAM License be a problem for the competition's open-source requirement.

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
