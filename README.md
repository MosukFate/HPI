# Hierarchical Prompt Injection

Official implementation of Hierarchical Prompt Injection (HPI) for domain-adaptive semantic segmentation.

## Installation

The code was tested with Python 3.10, PyTorch 2.0.1, CUDA 11.8, and MMCV 1.7.2.

```bash
conda create -n hpi python=3.10 -y
conda activate hpi
pip install -r requirements.txt
```

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

The expected weight structure is:

```text
pretrained/
  ViT-L-14-336px.pt
  EVA02_CLIP_L_336_psz14_s6B.pt
  dinov2_vitl14_pretrain.pth

Talk2DINO/
  weights/
    vitl_mlp_infonce.pth
```

## Checkpoints

Please download HPI checkpoints and save them in the `./checkpoints` folder.

| Model | Backbone | Setting | Link |
| --- | --- | --- | --- |
| HPI-CLIP | CLIP | GTA -> Cityscapes | [download link](https://drive.google.com/drive/folders/1DxtNRlQlSFkI6r3hzO0rvpQKc4VDb3XP?usp=sharing) |
| HPI-EVA | EVA02-CLIP | GTA -> Cityscapes | [download link](https://drive.google.com/drive/folders/1DxtNRlQlSFkI6r3hzO0rvpQKc4VDb3XP?usp=sharing) |
| HPI-EVA | EVA02-CLIP | Cityscapes -> Mapillary | [download link](https://drive.google.com/drive/folders/1DxtNRlQlSFkI6r3hzO0rvpQKc4VDb3XP?usp=sharing) |

## Data Preparation

Please arrange datasets as follows:

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

The rare-class sampling files can be generated with:

```bash
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

## Training

```bash
python train.py configs/hpi_clip_vit-l_1e-4_20k-g2c-512.py
python train.py configs/hpi_eva_vit-l_1e-4_20k-g2c-512.py
python train.py configs/hpi_eva_vit-l_1e-4_20k-c2m-512.py
```

## Testing

```bash
python test.py configs/hpi_clip_vit-l_1e-4_20k-g2c-512.py checkpoints/hpi_clip_vit-l_1e-4_20k-g2c-512_best.pth --eval mIoU
python test.py configs/hpi_eva_vit-l_1e-4_20k-g2c-512.py checkpoints/hpi_eva_vit-l_1e-4_20k-g2c-512_best.pth --eval mIoU
python test.py configs/hpi_eva_vit-l_1e-4_20k-c2m-512.py checkpoints/hpi_eva_vit-l_1e-4_20k-c2m-512_best.pth --eval mIoU
```
