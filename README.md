# Hierarchical Prompt Injection

This repository contains the public implementation of Hierarchical Prompt Injection (HPI) for domain-adaptive semantic segmentation. The code keeps the HPI training and evaluation paths for CLIP and EVA-CLIP backbones, with paper-facing names for the main modules:

- `HPI`: Hierarchical Prompt Injection
- `SHP`: semantic/hierarchical prompt utilities
- `SAC`: semantic attention contribution
- `SCC`: spatial confidence calibration

## Repository Layout

```text
HPI/
  configs/                         # Public CLIP/EVA experiment configs
  mmseg/                           # Project-local MMSegmentation code
  models/                          # HPI segmentors, backbones, adapters, prompts
  Talk2DINO/
    configs/                       # Talk2DINO projection configs
    src/                           # Talk2DINO source code
  tools/
    convert_datasets/              # Dataset conversion/statistics helpers
    convert_hpi_checkpoint_names.py
    profile_hpi.py
    profile_hpi_gate_values.py
    report_hpi_module_cost.py
  train.py
  test.py
  speed.py
```

The following local/runtime directories are intentionally not included in the public repository:

```text
data/
pretrained/
checkpoints/
work_dirs/
Talk2DINO/weights/
```

Create them locally before training or evaluation.

## Environment

The reference environment used for cleanup and parity checks was Python 3.10 with CUDA 11.8 wheels:

```bash
conda create -n hpi python=3.10 -y
conda activate hpi
pip install -r requirements.txt
```

`requirements-py310.txt` records the same dependencies grouped by installation stage. The code can import without `mamba-ssm`; when `mamba-ssm` is installed, the optional ContextDecoder sequence mixer uses the exact selective-scan path needed for bitwise parity with converted legacy checkpoints.

## Pre-trained VFM & VLM Models

Please download the pre-trained VFM and VLM models and save them in the `./pretrained` folder.

| Model | Type | Link |
| --- | --- | --- |
| DINOv2 | `dinov2_vitl14_pretrain.pth` | [download link](https://drive.google.com/file/d/1Rrl0RfU51eU8orbNVWHtNr1L3k5xhnld/view?usp=sharing) |
| CLIP | `ViT-L-14-336px.pt` | [download link](https://drive.google.com/file/d/1s00ofvxn0NCVVgnycXd2wUx4Gs53O6mj/view?usp=sharing) |
| EVA02-CLIP | `EVA02_CLIP_L_336_psz14_s6B.pt` | [download link](https://drive.google.com/file/d/1mQJ1zc_YLt7qAbaAET4-2EGNtIp2I6eB/view?usp=sharing) |

## HPI Weights

Please download the Talk2DINO projection weight and save it in `./Talk2DINO/weights`.

| Model | Type | Link |
| --- | --- | --- |
| Talk2DINO | `vitl_mlp_infonce.pth` | [download link](https://drive.google.com/drive/folders/1DxtNRlQlSFkI6r3hzO0rvpQKc4VDb3XP?usp=sharing) |

## Checkpoints

Please download HPI model checkpoints and save them in the `./checkpoints` folder.

| Model | Pretrained | Setting | Config | Link |
| --- | --- | --- | --- | --- |
| `hpi-clip-vit-l-g2c` | CLIP | GTA -> Cityscapes | [config](configs/hpi_clip_vit-l_1e-4_20k-g2c-512.py) | [download link](https://drive.google.com/drive/folders/1DxtNRlQlSFkI6r3hzO0rvpQKc4VDb3XP?usp=sharing) |
| `hpi-eva02-clip-vit-l-g2c` | EVA02-CLIP | GTA -> Cityscapes | [config](configs/hpi_eva_vit-l_1e-4_20k-g2c-512.py) | [download link](https://drive.google.com/drive/folders/1DxtNRlQlSFkI6r3hzO0rvpQKc4VDb3XP?usp=sharing) |
| `hpi-eva02-clip-vit-l-c2m` | EVA02-CLIP | Cityscapes -> Mapillary | [config](configs/hpi_eva_vit-l_1e-4_20k-c2m-512.py) | [download link](https://drive.google.com/drive/folders/1DxtNRlQlSFkI6r3hzO0rvpQKc4VDb3XP?usp=sharing) |

The expected local asset layout is:

```text
pretrained/
  ViT-L-14-336px.pt
  EVA02_CLIP_L_336_psz14_s6B.pt
  dinov2_vitl14_pretrain.pth

Talk2DINO/
  weights/
    vitl_mlp_infonce.pth

checkpoints/
  <converted HPI checkpoints>.pth
```

You may either place the files at these paths or use symbolic links.

## Data Layout

The public configs use relative paths under `data/`.

```text
data/
  gta/
    images/
    labels/
    sample_class_stats.json
    sample_class_stats_dict.json
    samples_with_class.json

  cityscapes/
    leftImg8bit/
      train/
      val/
    gtFine/
      train/
      val/
    sample_class_stats.json
    sample_class_stats_dict.json
    samples_with_class.json

  mapillary/
    validation/
      images/
      labels/

  bdd100k/
    images/
      10k/
        val/
    labels/
      sem_seg/
        colormaps/
          val/
```

The `sample_class_stats*.json` and `samples_with_class.json` files are required by rare class sampling for the source domain. Generate them with the helpers in `tools/convert_datasets/`, for example:

```bash
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

## Public Configs

```text
configs/hpi_clip_vit-l_1e-4_20k-g2c-512.py      # GTA -> Cityscapes, CLIP-L
configs/hpi_eva_vit-l_1e-4_20k-g2c-512.py       # GTA -> Cityscapes, EVA-CLIP-L
configs/hpi_eva_vit-l_1e-4_20k-c2m-512.py       # Cityscapes -> Mapillary, EVA-CLIP-L
```

All public configs use:

```python
hpi_layers = [23]
hpi_layers_dino = [23]
```

## Training

Single-GPU examples:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  configs/hpi_clip_vit-l_1e-4_20k-g2c-512.py

CUDA_VISIBLE_DEVICES=0 python train.py \
  configs/hpi_eva_vit-l_1e-4_20k-g2c-512.py
```

Set `work_dir` in the config or pass standard MMCV runner options as needed.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  configs/hpi_eva_vit-l_1e-4_20k-g2c-512.py \
  checkpoints/hpi_eva_vit-l_1e-4_20k-g2c-512_best.pth \
  --eval mIoU
```

For a quick smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  configs/hpi_eva_vit-l_1e-4_20k-g2c-512.py \
  checkpoints/hpi_eva_vit-l_1e-4_20k-g2c-512_best.pth \
  --eval mIoU \
  --max-samples 10
```

## Checkpoint Conversion

Older internal checkpoints used historical HPI names such as `sap_*` and `msgate_*`. Convert them before loading with the public configs:

```bash
python tools/convert_hpi_checkpoint_names.py \
  path/to/old_checkpoint.pth \
  checkpoints/hpi_checkpoint_public_names.pth
```

The converter only renames model parameter keys. It does not guarantee optimizer-state compatibility for resuming old training jobs.

## Notes

- Large assets, generated checkpoints, logs, caches, and datasets are ignored by `.gitignore`.
- The cleaned HPI code was checked against the legacy implementation with CUDA tensor parity on the full `encode_decode` path after checkpoint conversion.
- Exact numerical reproducibility during training still depends on the CUDA stack, data order, random seeds, and distributed training settings.
