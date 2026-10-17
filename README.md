# SPINS

**This repository currently provides an approximate implementation. Paper-specific implementations, configurations, preprocessing, and checkpoints will be updated soon.** It is a runnable research scaffold, not a claim of exact reproduction of the paper. No trained checkpoints or raw dataset files are distributed here.

## Abstract

Existing human novel view synthesis (NVS) methods typically rely on textured 3D scans, dense pose-aligned multiview images, or parametric body priors such as SMPL, making their training data costly to acquire and difficult to scale beyond controlled capture settings.
We present **SPINS**, a framework for single-image human NVS that learns feed-forward human 3D Gaussian Splatting (3DGS) from sparse 2D view supervision.
Given a single input image, a human-specific multiview diffusion module first synthesizes a small set of anatomically plausible canonical views.
An object-centric 3DGS reconstruction model, initialized from large-scale pretraining, is then adapted with only sparse 2D human views to convert the synthesized views into a 3DGS representation.
We introduce a sparsely supervised perturb-and-reconstruct objective: Gaussians predicted from the synthesized views are rendered at offset viewpoints, and the reconstruction obtained from those renderings must reproduce the original views.
Geometric errors that are invisible from the supervised views become visible once the asset is rendered from elsewhere, and the round trip converts them into reconstruction error that the model is driven to correct.
The objective, therefore, enforces stable geometry and appearance across viewpoints with nothing beyond the sparse 2D views as supervision.
Trained once on four views per subject of a single dataset, SPINS remains competitive with densely supervised methods across three benchmarks despite using no dense multiview, textured-mesh, or SMPL supervision, and is markedly more resilient to partial occlusions of the input view.
These results show that single-image human 3DGS can be learned without dense multiview capture, textured meshes, or explicit parametric body priors.

## Current implementation

The abstract describes the full framework. The current approximate release covers the four-view reconstruction/adaptation stage; the human-specific diffusion module and paper-specific evaluation are not included yet.

The current code adapts [LGM](https://github.com/3DTopia/LGM) for four-view cycle-consistency training. Four observed images produce Gaussians; rendered images from another camera set are detached and fed back into LGM; the returned reconstruction is supervised at the original four cameras. The loss is RGB MSE + mask MSE + VGG LPIPS. Ground-truth images at intermediate viewpoints are not used in the training loss.

## Qualitative results

<p align="center">
  <img src="assets/results/comparison_0515.gif" alt="Baseline and OURS synchronized novel-view comparison" width="548" />
</p>

<p align="center">
  <img src="assets/results/comparison_0516.gif" alt="Baseline and OURS synchronized novel-view comparison" width="548" />
</p>

<p align="center">
  <img src="assets/results/comparison_0518.gif" alt="Baseline and OURS synchronized novel-view comparison" width="548" />
</p>

<p align="center">
  <img src="assets/results/comparison_0513.gif" alt="Baseline and OURS synchronized novel-view comparison" width="548" />
</p>

<p align="center">
  <img src="assets/results/comparison_0521.gif" alt="Baseline and OURS synchronized novel-view comparison" width="548" />
</p>

<p align="center">
  <img src="assets/results/comparison_0512.gif" alt="Baseline and OURS synchronized novel-view comparison" width="548" />
</p>

## Install

Use Linux, Python 3.10, an NVIDIA GPU with BF16 support, a CUDA-compatible driver, CUDA toolkit 12.1 (`nvcc`), and a C++ compiler for the Gaussian rasterizer. The training code was exercised with PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, xformers 0.0.29.post1, and an RTX 6000 Ada. Batch size 1 used approximately 6.5 GiB of allocated GPU memory; allow additional device memory and host RAM for initialization/checkpointing. This code is single-GPU.

```bash
git clone https://github.com/arid29/SPINS.git
cd SPINS
conda create -n spins python=3.10 -y
conda activate spins
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-deps xformers==0.0.29.post1
python -m pip install -r requirements.txt
bash scripts/install_rasterizer.sh
```

The install script pins the [ashawkey rasterizer](https://github.com/ashawkey/diff-gaussian-rasterization) at `8829d14f814fccdaf840b7b0f3021a616583c0a1`, including its submodules. This fork supplies depth and alpha outputs; the unmodified upstream rasterizer is not a drop-in replacement. Set `CUDA_HOME` to your CUDA toolkit if the build cannot locate it. Install PyTorch before compiling the extension.

For mesh preprocessing, also install:

```bash
python -m pip install -r requirements-render.txt
```

Headless preprocessing uses EGL. The NVIDIA EGL/OpenGL driver libraries must be available on the rendering node. Installing PyOpenGL alone does not supply those system libraries. You can choose another available backend with `PYOPENGL_PLATFORM`; EGL is the tested path.

On Ubuntu/Debian, the rendering runtime also needs `libegl1`, `libgl1`, and `libglu1-mesa` (install these with your system package manager or include them in your container). A `Library "GLU" not found` error means the system GLU library is missing, not a missing Python package.

## Obtain THuman2.0

Request the textured scans through the [official THuman dataset repository](https://github.com/ytrock/THuman2.0-Dataset#download-instructions). Follow its current agreement and request-form instructions to obtain access. The repository also describes the newer THuman2.1 release: check that the provided subject IDs match the subset used below. Obtain optional SMPL-X fittings through the links maintained by the data provider.

Keep the original OBJ, MTL, and texture files together. Dataset use and redistribution are governed by the provider's terms; this repository does not mirror source scans or preprocessed training data.

Arrange the authorized scans as:

```text
/path/to/THuman2.0/
  0000/
    0000.obj
    material0.mtl
    material0.jpeg
  0001/
    0001.obj
    material0.mtl
    material0.jpeg
  ...
  0525/
    0525.obj
    material0.mtl
    material0.jpeg
```

The training entry points explicitly use **IDs 0000–0499 for training (500 subjects)** and **0500–0525 for held-out inference/evaluation (26 subjects)**. This is the split configured in this implementation, not a claim about an official dataset split. If your download is missing any of these IDs, resolve the dataset version/subset first; do not silently replace or duplicate subjects.

## Preprocess the scans

The included [preprocessor](scripts/preprocess_thuman.py) is an approximate, portable renderer. It preserves textured meshes, normalizes their height to 1, centers their vertical extent, and renders 12 views at 30° intervals on a horizontal radius-1.5 orbit. It uses square perspective images, 49.1° vertical field of view, near/far planes 0.5/2.5, and depth-derived foreground masks. Its lighting and rasterization are not asserted to match the paper's preprocessing.

First render and validate a single subject:

```bash
python scripts/preprocess_thuman.py \
  --mesh-root /path/to/THuman2.0 \
  --output THumanDataset/thuman2_rendered_lgm \
  --start 0 --end 0

python scripts/validate_data.py \
  --data-root THumanDataset/thuman2_rendered_lgm --start 0 --end 0
```

Inspect the RGB images and masks, then render the rest:

```bash
python scripts/preprocess_thuman.py \
  --mesh-root /path/to/THuman2.0 \
  --output THumanDataset/thuman2_rendered_lgm \
  --start 1 --end 525

python scripts/validate_data.py --data-root THumanDataset/thuman2_rendered_lgm
```

Default rendering resolution is 1024×1024. Training resizes images to 256×256. `--resolution 256` is useful for checking the rendering setup. `--skip-existing` skips complete subjects; partial output is rejected so it cannot silently enter training.

Optionally pass `--smplx-root /path/to/Smpl-X`, containing `0000/smplx_param.pkl`, etc. The helper uses the Y component of `global_orient` as an approximate initial azimuth offset. This is not a full pose canonicalization. With no fittings supplied, initial azimuth is zero. Use only trusted local fitting files, since the optional loader reads pickle files.

The output layout is:

```text
THumanDataset/thuman2_rendered_lgm/
  img/0000/00.jpg ... 11.jpg
  mask/0000/00.png ... 11.png
  parm/0000/00_extrinsic.npy ... 11_extrinsic.npy
  parm/0000/00_intrinsic.npy ... 11_intrinsic.npy
  parm/0000/preprocess.json
  ... the same layout for 0001 through 0525
```

Extrinsics are OpenCV/COLMAP **world-to-camera** matrices (3×4 or homogeneous 4×4). Masks have nonzero foreground. The loader composites RGB onto white and converts poses to its normalized OpenGL/rasterizer conventions. It uses the configured 49.1° FOV rather than reading per-image intrinsics, so render with the matching camera settings. Inputs are views **00,03,06,09** (90° apart).

If you already have rendered data, run the validator before training. It checks required files, image/mask shapes, foreground, valid rotations, intrinsics when available, and the ordered circular orbit. Arbitrary camera trajectories are not supported by the progressive-offset helper.

## Prepare initialization and perceptual weights

Paper-specific checkpoints will be released later. For this approximate implementation, download the public LGM initialization and prepare the VGG/LPIPS files:

```bash
python scripts/prepare_weights.py --download-public
```

This obtains LGM's public `model_fp16_fixrot.safetensors`, torchvision's ImageNet VGG16 features, and the VGG v0.1 calibration weights from the installed LPIPS package. Downloads require internet access and sufficient disk space. The VGG download temporarily includes the full network.

Alternatively, use an existing compatible initialization:

```bash
python scripts/prepare_weights.py --lgm-checkpoint /path/to/model.safetensors
```

The resulting local files are `pretrained/model.safetensors`, `pretrained/vgg16_features.pth`, and `pretrained/lpips_vgg.pth`. Existing files are kept. Checkpoints are ignored by Git. Public initialization and this approximate preprocessing are not guaranteed to reproduce results obtained with a different initialization or rendering pipeline.

## Train

Fixed-offset baseline (30 epochs):

```bash
bash scripts/run.sh fixed --workspace workspace_fixed --epochs 30 \
  --data-root THumanDataset/thuman2_rendered_lgm
```

Progressive schedule (60 epochs):

```bash
bash scripts/run.sh progressive --workspace workspace_progressive --epochs 60 \
  --epochs-per-stage 3 --data-root THumanDataset/thuman2_rendered_lgm
```

Both start from `pretrained/model.safetensors`; override with `--checkpoint`. They use AdamW, learning rate `4e-6`, batch size 1, BF16, 500 optimizer steps per epoch, four real input images per subject, and RGB MSE + mask MSE + LPIPS (weight 1). The intermediate render is detached. Only the return to the four original views contributes to the training loss.

Progressive offsets are sampled uniformly from signed multiples of 5° up to the current cap: epochs 1–3 use ±5°; epochs 4–6 use ±5°/±10°; the cap reaches 90° at epoch 52. The geometry helper rotates both camera centers and orientations on the orbit, and derives view/projection/ray inputs consistently. Evaluation always uses the same 12 stored cameras and a fixed 30° cycle probe, independent of training stage.

Each epoch saves evaluation image grids and per-subject metrics. Files include:

- `config.json`: split, arguments, source hashes, and schedule.
- `train.jsonl`, `epochs.jsonl`, `progress.json`: training and evaluation progress.
- `model.safetensors`: latest completed epoch.
- `best.safetensors`, `best.json`: best checkpoint by the selected metric.
- `training_state.pt`: completed-epoch model, optimizer, and RNG state.
- `summary.json`: written after successful completion.

**All-view PSNR is the default checkpoint-selection metric.** Use `--selection-metric direct_psnr_unseen` to select by unseen views instead. All-view PSNR includes all 12 views, including the four inputs. Unseen-view PSNR uses the other eight. Each is computed on white-composited full images at 256×256: average RGB squared error over the selected views/pixels, convert to PSNR per subject, then average PSNR over the 26 evaluation subjects. These are not foreground-only scores. Because this same set selects checkpoints, treat it as validation rather than an untouched final test set.

Resume from a saved completed epoch with the same arguments plus `--resume`. If a run was interrupted during evaluation, move its incomplete `epoch_NNN` directory aside before resuming. Keep the code, split, learning rate, metric and schedule unchanged; the resume guard checks their configuration. A fresh workspace plus `--checkpoint path/to/model.safetensors` is a weights-only restart with a fresh optimizer and schedule.

## Inference/evaluation

```bash
bash scripts/run.sh evaluate \
  --checkpoint workspace_progressive/best.safetensors \
  --data-root THumanDataset/thuman2_rendered_lgm \
  --workspace workspace_eval
```

This performs inference on subjects 0500–0525, saves RGB comparisons and cycle grids, and writes per-subject and aggregate metrics. It requires four calibrated input views; this entry point is not a single-image generation tool. Direct grids show GT above predictions for views 00–11. Cycle grids show the original four views, the four intermediate renders, then the returned four views.

## Acknowledgments and licensing

Built on [LGM](https://github.com/3DTopia/LGM), the [modified Gaussian rasterizer](https://github.com/ashawkey/diff-gaussian-rasterization), [DINOv2 attention code](https://github.com/facebookresearch/dinov2), [LPIPS](https://github.com/richzhang/PerceptualSimilarity), [pyrender](https://github.com/mmatl/pyrender), and [trimesh](https://github.com/mikedh/trimesh). Please cite the original projects and the THuman/Function4D dataset when using them. See [THIRD_PARTY.md](THIRD_PARTY.md), [LICENSE](LICENSE), and the retained notices in `licenses/`. Dataset and external dependency licenses remain separate from this repository's code license.
