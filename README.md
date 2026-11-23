# SAGD: Selection-Aware Gaze Distillation for Aerial Object Detection

Official PyTorch implementation of the paper *"SAGD: Selection-Aware Gaze Distillation for Efficient Object Detection in Remote Sensing Imagery"* (paper link coming soon).

> SAGD injects human gaze supervision from the \*\*AerialGaze-1.0 \*\*eye-tracking dataset into the token-selection supervision of a sparse detection transformer. The inference path stays **architecturally unchanged — no added parameters, no added FLOPs**: all gaze information is consumed by the training loss.

The codebase is built on top of [Salience-DETR](https://github.com/xiuqhou/Salience-DETR) (CVPR 2024).

## Method

SAGD consists of two loss-side components:

1. **Mixed saliency target.** The box-derived salience mask `M` (center-symmetric geometric prior of the baseline) is blended with a gaze density map `D` (fixations Gaussian-splatted onto each feature level):
   ```
   T = (1 − λ) · M + λ · D          λ = 0.5 (default)
   ```
   Images without gaze coverage fall back to the pure box target automatically (per-image gating), so the training set is never restricted to gaze-covered tiles.
2. **Selection-aware weighting.** At deployment, the encoder keeps the **top-k tokens** ranked by foreground score under a retention ratio `r`. Training therefore concentrates the weighted focal loss around that truncation point: tokens are ranked by the mixed target `T`, and an exponential kernel `w = exp(−|rank − k| / τ)` with `τ = 0.1·k` puts a bell-shaped weight peak at `k`. Weights are row-mean normalized, so only the *allocation* of supervision changes — not the loss magnitude. Per batch, `r` is sampled from a deployment pool `{0.25, 0.35, 0.5}`.

Because everything happens in the criterion, the model structure, parameter count, and inference FLOPs are identical to the Salience-DETR baseline.

### Experiment switch: `GAZE_DIR`

One environment variable switches the distillation target (see configs):

| Experiment                                                 | `GAZE_DIR`            | Config file                        |
| ---------------------------------------------------------- | --------------------- | ---------------------------------- |
| Baseline (box-only supervision)                            | unset                 | `configs/train_config.py`          |
| Pseudo-gaze distillation (statistical transfer from boxes) | `data/pseudo_density` | `configs/train_config.py`          |
| Real-gaze distillation (AerialGaze-1.0)                    | `data/gaze_density`   | `configs/train_config.py`          |
| **SAGD** (real gaze + selection-aware weighting)           | `data/gaze_density`   | `configs/train_config_selaware.py` |

ConvNeXt-T variants use `configs/train_config_convnext_t_base.py` / `configs/train_config_convnext_t_selaware.py`.

## Repository layout

```
SAGD-main/
├── main.py                  # training entry point (accelerate-based)
├── test.py                  # evaluation entry point (COCO protocol)
├── inference.py             # visualization demo on a folder of images
├── models/
│   ├── backbones/           # ResNet-50 / ConvNeXt-T / Swin / ViT / FocalNet
│   ├── bricks/
│   │   ├── salience_transformer.py   # hierarchical salience filtering encoder/decoder
│   │   ├── set_criterion.py          # Hungarian-matched set losses
│   │   └── ops/cuda/                 # multi-scale deformable attention CUDA op (JIT-built)
│   ├── detectors/
│   │   └── salience_detr.py          # ★ mixed target + selection-aware boundary weighting
│   ├── matcher/             # Hungarian matching
│   ├── necks/               # channel mapper + RepVGG neck
├── datasets/coco.py         # DOTA-slice COCO loader + gaze density map loading/augmentation
├── transforms/              # data augmentation (gaze maps are transformed in sync with images)
├── util/                    # training/eval engine, COCO eval, logging, visualization
├── optimizer/param_dict.py  # backbone/backbone-projection lr grouping
├── tools/
│   ├── convert_dota15_to_coco.py  # per-tile DOTA labelTxt -> COCO json
│   ├── merge_gaze_stimuli.py      # gaze stimulus tiles -> train COCO entries
│   ├── build_gaze_density.py      # AerialGaze-1.0 eye-tracking JSONs -> gaze density maps
│   └── build_pseudo_gaze.py       # box statistics from real gaze -> pseudo-gaze maps
└── configs/                 # model configs + training-session configs
```

> **`data/` is not part of this repository.** Datasets, eye-tracking recordings
> and derived gaze density maps are several hundred MB, so they are excluded
> from version control (see [`.gitignore`](.gitignore)). Everything under
> `data/` is produced locally with the steps in §2.

## Getting started

### 1. Installation

Requirements: PyTorch (CUDA build), CUDA toolkit (nvcc) and a C++ compiler for the JIT build of the deformable-attention op, plus:

```bash
pip install -r requirements.txt
```

The multi-scale deformable attention CUDA extension (`models/bricks/ops/cuda/`) is compiled automatically on first use via `torch.utils.cpp_extension`. You can verify the build with:

```bash
python -c "from models.bricks.ms_deform_attn import MultiScaleDeformableAttention; print('CUDA-EXT-BUILD-OK')"
```

### 2. Data preparation

> **Tile grid & naming — read this first.** All tiles (detection slices *and* gaze
> stimuli) are 1024×1024 patches on the official DOTA split grid (gap 200,
> stride 824) and share **one** naming convention:
> `<source-image>__<xstart>___<ystart>` (e.g. `P0000__0___4120.jpg`). If your
> splitter emits a different convention (e.g. mmrotate's
> `<source>__1024__<x>___<y>`), strip the size token once so every stage uses the
> same names. **Every pipeline stage matches data by tile stem** — detection
> annotations, gaze density maps and stimulus images must share the same naming;
> a mismatch silently drops data.

#### 2.1 Detection data — DOTA-1.5, 1024×1024 slices, COCO format

Option A (recommended): download the prepared dataset (link coming soon):

```
data/DOTA1.5_sliced/
├── annotations/instances_train2017.json     # train tiles + gaze stimulus tiles
├── annotations/instances_val2017.json
├── train/images/*.jpg  ├── train/labelTxt/*.txt
└── val/images/*.jpg
```

Option B: build it yourself from raw DOTA-1.5 with the official splitting code:

```bash
# 1) official split, patch 1024 / gap 200 (mmrotate example; DOTA_devkit works too)
python mmrotate/tools/data/dota/split/img_split.py \
    --base-dir <DOTA1.5-root> --patch-size 1024 --gap 200 \
    --nproc 8 --save-ext jpg

# 2) per-tile DOTA labelTxt -> COCO (repeat for the val split with
#    --file-prefix val/images)
python tools/convert_dota15_to_coco.py \
    --img-dir data/DOTA1.5_sliced/train/images \
    --label-dir data/DOTA1.5_sliced/train/labelTxt \
    --out data/DOTA1.5_sliced/annotations/instances_train2017.json
```

`convert_dota15_to_coco.py` reproduces the released annotations exactly
(per-tile DOTA txt → unclipped polygon bounding boxes, 16 classes,
`category_id` 1–16 with index 0 reserved following the DETR/COCO convention).

#### 2.2 Gaze data — AerialGaze-1.0 (not publicly available)

> **The eye-tracking data is not included in this repository and is not publicly
> released.** Raw recordings and derived density maps are several hundred MB, so a
> fresh clone contains no `data/` directory. Each experiment below states which
> generated directory it expects; how to rebuild that directory from your own
> gaze recordings is described here.

AerialGaze-1.0 was collected for internal research use only and is **not
distributed**. Because the recordings were acquired without a formal
ethics-review approval, we cannot redistribute either the raw eye-tracking
records or the derived density maps. This repository documents the expected
layout and the scripts that turn gaze records into density maps, so the pipeline
can be reproduced with user-collected gaze data. For reference, the internal
layout is:

```
AerialGaze-1.0/                       # internal layout only; not distributed
├── README.md
├── trials.csv                       # one row per trial: participant/task/class/stem/path
├── stats.json                       # aggregate counts (trials, fixations, sessions)
├── P001/
│   ├── search/<class>/<stem>.json       # 1,800 target-search trials
│   └── free_viewing/<stem>.json         # 200 free-viewing trials
├── P002/
└── P003/
```

Each JSON carries a `meta` block (`participant` / `task` / `class` / `image`),
the raw 30 Hz gaze samples (`gaze`) and I-DT fixations (`fixations`: `x`, `y`,
`dur` in seconds). Gaze coordinates are **integer pixels 0–1023**, expressed in
the 1024×1024 stimulus-tile pixel space. Stimulus tiles use the same naming as
the detection slices (`<source>__<xstart>___<ystart>`), so gaze data, stimulus
images and detection annotations match by stem with no renaming step.

**How the data was obtained.** AerialGaze-1.0 was recorded by our group with a
Tobii screen-based eye tracker (30 Hz sampling). The acquisition protocol was:

1. **Stimuli** — 1,800 tiles of 1024×1024 px on the official DOTA-1.5 split
   grid (gap 200, stride 824): 200 tiles for each of the 9 target classes used
   in this work (bridge, court, ground track field, harborship, plane,
   roundabout, storage tank, swimming pool, vehicle).
2. **Task** — target search: the participant searched the tile for instances of
   the cued class. A free-viewing task was also recorded on 200 tiles per
   participant; the maps used by SAGD come from the *search* task, hence the
   `search/` task directory in `data/gaze_density/` (`gaze_task="search"` in the
   loader).
3. **Participants** — 3 participants (P001, P002, P003); calibration was redone
   for every participant and session.
4. **Fixation extraction** — raw 30 Hz samples are converted offline with the
   I-DT (dispersion-threshold identification) algorithm; each fixation keeps
   its pixel position and its duration in seconds.
5. **Alignment** — gaze is mapped onto the stimulus tile's own 1024×1024 pixel
   grid, so coordinates are directly comparable with DOTA annotations without
   any rescaling.
6. **De-identification** — only anonymous participant codes are stored; no
   demographic, face, or stimulus data is distributed.

No formal ethics-review approval was obtained for public redistribution of these
recordings, so neither the raw 30 Hz samples nor the derived density maps are
released. The data were used solely for the internal experiments reported in the
paper.

The training signal is **not** the fixation list itself but a Gaussian-splatted
density map built from these JSONs with `tools/build_gaze_density.py` (§2.4).
For our internal recordings this yields 1,800 maps (9 classes × 200).

#### 2.3 Rebuild the stimulus tiles and merge them into the detection train set

The repository ships no images and no gaze records. If you have your own gaze
trials aligned to DOTA tile stems, list them in a `trials.csv` with the same
columns (`participant/task/class/stem/path`) and reconstruct the corresponding
stimuli from your DOTA-1.5 split. Save the snippet below as `make_stimuli.py` and
run it once (it accepts both `<stem>.jpg` and the mmrotate name
`<src>__1024__<x>___<y>.jpg`):

```python
import csv, re, shutil
from pathlib import Path

root  = Path("AerialGaze-1.0")
split = Path("data/DOTA1.5_sliced/train")          # images/ + labelTxt/
rows  = [r for r in csv.DictReader(open(root / "trials.csv", encoding="utf-8-sig"))
         if r["task"] == "target_search"]

missing = 0
for r in rows:
    stem = r["stem"]
    m = re.match(r"(.+)__(\d+)___(\d+)$", stem)
    names = [stem] + ([f"{m.group(1)}__1024__{m.group(2)}___{m.group(3)}"] if m else [])
    img = next((split / "images" / f"{n}.jpg" for n in names
                if (split / "images" / f"{n}.jpg").exists()), None)
    lbl = next((split / "labelTxt" / f"{n}.txt" for n in names
                if (split / "labelTxt" / f"{n}.txt").exists()), None)
    if img is None or lbl is None:
        missing += 1
        continue
    for sub, src, ext in (("images", img, ".jpg"), ("annfiles", lbl, ".txt")):
        d = root / sub / r["class"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d / f"{stem}{ext}")
print("missing:", missing)
```

If tiles are absent from the standard train split, re-run the official split
(§2.1 Option B) over the full DOTA-1.5; the stems encode the source image and
top-left corner on the same 1024 / gap-200 grid.

Then merge the reconstructed tiles into the detection COCO json (idempotent):

```bash
python tools/merge_gaze_stimuli.py --gaze-root AerialGaze-1.0 \
    --coco data/DOTA1.5_sliced/annotations/instances_train2017.json \
    --stim-dir data/DOTA1.5_sliced/stimuli
```

#### 2.4 Build gaze density maps from the eye-tracking JSONs

This is the step that creates the `data/gaze_density/` directory consumed by
`GAZE_DIR`. The outputs are **not** committed to git, so run it once locally
after §2.3:

```bash
python tools/build_gaze_density.py --src AerialGaze-1.0 \
    --out data/gaze_density \
    --coord-space pixel \
    --meta-task 1 \
    --images AerialGaze-1.0/images \
    --coco-ann data/DOTA1.5_sliced/annotations/instances_train2017.json \
    --classes bridge court "ground track field" harborship plane roundabout \
              "storage tank" "swimming pool" vehicle
```

`--meta-task 1` keeps only target-search trials; without it the free-viewing
JSONs are rendered too and can collide with search maps on shared tile stems.

The script is layout-agnostic (grouped by each JSON's `meta` fields), applies the
training-time fusion protocol (σ = 16 px splats, duration weighting, per-subject
max-normalization before summation), supports both pixel and normalized
coordinates (`--coord-space pixel|norm01|auto`), and validates output stems
against both the stimulus images and the detection annotations — tiles that match
no annotated image are listed under `orphan_stems` in `index.json` and would be
silently skipped by training, so this report should be empty before training.

Each `.npz` stores a float16 `density` array (1024×1024): the sum of per-fixation
Gaussian splats (σ = 16 px). For multi-subject tiles, each subject's map is
max-normalized before summation ("one person, one vote"). The loader max-normalizes
each map on load, matches tiles by filename stem, and co-transforms density maps
with images (flip/crop/scale-jitter) before downsampling to 128×128.

#### 2.5 Build pseudo-gaze density maps (statistical-transfer arm)

The pseudo-gaze arm replaces human gaze with a box-derived surrogate, so **every**
annotated training tile receives a distillation target (arm 2 in the switch
table). `tools/build_pseudo_gaze.py` fits the surrogate on tiles that have both
annotations and a real density map (§2.4), then transfers it to the whole train
split:

1. **Within-box gaze distribution** — for each GT box, the real density is
   normalized inside the box and pooled into a K×K relative-coordinate grid;
   grids are aggregated by `class × area-bin`, weighted by each box's gaze mass.
2. **Box gaze share** — the fraction of the tile's total gaze mass falling in
   each box, used to split a per-tile point budget across boxes.

Transfer matches every box by class and area bin, falling back
`class × bin → class → bin → global` so classes without real gaze still get a
target. Sampled points are splatted with the same σ = 16 px kernel as §2.4 and
written in the identical `.npz` layout:

```bash
python tools/build_pseudo_gaze.py \
    --coco data/DOTA1.5_sliced/annotations/instances_train2017.json \
    --gaze-dir data/gaze_density \
    --out data/pseudo_density \
    --points 32 --grid 8 --area-bins 3 --seed 42
```

Area bins are quantiles of √(w·h) over all annotated boxes. The outputs are
`<out>/search/<class>/<stem>.npz` (float16, 1024×1024, max-normalized on load),
so switching to this arm only requires `GAZE_DIR`:

```bash
export GAZE_DIR=data/pseudo_density       # Windows: set GAZE_DIR=data/pseudo_density
python main.py --config-file configs/train_config.py --seed 42
```

`index.json` records the fitted parameters, per-`(class, area-bin)` support counts
and how many boxes used each fallback level. Only the real density maps are
needed for fitting — raw eye-tracking JSONs are not read by this script.

### 3. Training

Run everything from the repository root. Single GPU (batch size 2), 12 epochs, AdamW (lr 1e-4), step decay at epoch 10:

```bash
# Baseline (no gaze)
python main.py --config-file configs/train_config.py --seed 42

# Real-gaze distillation
export GAZE_DIR=data/gaze_density            # Windows: set GAZE_DIR=data/gaze_density
python main.py --config-file configs/train_config.py --seed 42

# SAGD (selection-aware weighting on top of gaze distillation)
export GAZE_DIR=data/gaze_density
python main.py --config-file configs/train_config_selaware.py --seed 42

# ConvNeXt-T backbone: use configs/train_config_convnext_t_base.py
# and configs/train_config_convnext_t_selaware.py instead.
```

To resume an interrupted run, point `RESUME_FROM` at the run directory:

```bash
export RESUME_FROM=checkpoints/salience_detr_resnet50_dota15_selaware/train/<timestamp>
python main.py --config-file configs/train_config_selaware.py
```

By default checkpoints are written to `checkpoints/<model-config-name>/train/<timestamp>/`. Sensitivity studies (λ, τ, retention-ratio pool) only require one-line changes to the corresponding config.

### 4. Evaluation

```bash
python test.py --coco-path data/DOTA1.5_sliced --subset val \
    --model-config configs/salience_detr/salience_detr_resnet50_dota15_selaware.py \
    --checkpoint <path/to/best_ap.pth>
```

Metrics follow the COCO bounding-box protocol. Token-budget evaluation at deployment ratios {1.0, 0.75, 0.5, 0.35, 0.25} is done by adjusting `level_filter_ratio` / `layer_filter_ratio` in the model config.

For qualitative results on a folder of images:

```bash
python inference.py --image-dir <folder> --model-config <model config> \
    --checkpoint <ckpt> --show-dir <output folder>
```

## Results

Trained-model checkpoints and the full result tables will be released upon paper acceptance. Please refer to the paper for detailed comparisons (baselines, distillation-target ablations, budget sweeps, and backbone generality).

## License

This project is released under the [Apache 2.0 license](LICENSE). The framework is derived from [Salience-DETR](https://github.com/xiuqhou/Salience-DETR) (Apache 2.0).

## Acknowledgements

- [Salience-DETR](https://github.com/xiuqhou/Salience-DETR) (CVPR 2024) — base detection framework
- [DOTA](https://captain-whu.github.io/DOTA/) — aerial image dataset
- All participants who contributed eye-tracking data to AerialGaze-1.0

## Citation

If this repository helps your research, please cite the paper (BibTeX will be added upon publication).
