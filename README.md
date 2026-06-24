# [ECCV 2026] Hierarchical Prompt Injector for Domain Generalization Segmentation

Official implementation of the ECCV 2026 paper "Hierarchical Prompt Injector for Domain Generalization Segmentation".

XIN KUN LIN, Ruoyu Guo, Jiaqi Guo, Maurice Pagnucco, Yang Song

University of New South Wales

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

Please download the [Talk2DINO](https://github.com/lorebianchi98/Talk2DINO) projection weight and save it in `./Talk2DINO/weights`.

| Model | Type | Link |
| --- | --- | --- |
| Talk2DINO | `vitl_mlp_infonce.pth` | [download link](https://drive.google.com/file/d/1I8o_br-__iApEE1oNSTcEzi6XAE893U6/view?usp=drive_link) |

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

| Model | Pretrained | Trained on | Link |
| --- | --- | --- | --- |
| HPI-CLIP-G2C | CLIP | GTA -> Cityscapes | [download link](https://drive.google.com/drive/folders/1bJMWy1izDSOnSVlXfeFnRtnZIL6duXeM?usp=drive_link) |
| HPI-EVA-G2C | EVA02-CLIP | GTA -> Cityscapes | [download link](https://drive.google.com/drive/folders/1RrBWqW198O2mKsyN-_Nnxj8xGpteV4Y8?usp=drive_link) |
| HPI-CLIP-C2M | CLIP | Cityscapes -> Mapillary | [download link](https://drive.google.com/drive/folders/16reC5UDlsIX9PPopv7Uxw_esJzkRui6c?usp=drive_link) |
| HPI-EVA-C2M | EVA02-CLIP | Cityscapes -> Mapillary | [download link](https://drive.google.com/drive/folders/1TiPQDVBPpsfAPhnRdtQu9DmufoRqXCSS?usp=drive_link) |

## Datasets

Please prepare the datasets and edit the data roots in the config files according to your environment:

```python
src_dataset_dict = dict(..., data_root='[YOUR_DATA_FOLDER_ROOT]', ...)
tgt_dataset_dict = dict(..., data_root='[YOUR_DATA_FOLDER_ROOT]', ...)
```

The expected dataset layout is:

```text
HPI/
  data/
    gta/
      images/
      labels/

    cityscapes/
      leftImg8bit/
        train/
        val/
      gtFine/
        train/
        val/

    bdd100k/
      images/
        10k/
          train/
          val/
      labels/
        sem_seg/
          masks/
            train/
            val/

    mapillary/
      training/
        images/
      cityscapes_trainIdLabel/
        train/
          label/
      half/
        val_img/
        val_label/
```

The release configs use the dataset directories above; auxiliary class-statistics JSON files are not required in the README setup.

## Training

```bash
python train.py configs/[TRAIN_CONFIG]
```

## Evaluation

```bash
python test.py configs/[TEST_CONFIG] checkpoints/[MODEL] --eval mIoU
```

## Citation

If you find our code helpful, please cite our paper:

```bibtex
@inproceedings{lin2026hierarchical,
  title     = {Hierarchical Prompt Injector for Domain Generalization Segmentation},
  author    = {Lin, Xinkun and Guo, Ruoyu and Guo, Jiaqi and Pagnucco, Maurice and Song, Yang},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgements

This project is based on the following open-source projects. We thank the authors for sharing their code.

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [TLDR](https://github.com/ssssshwan/TLDR)
- [Talk2DINO](https://github.com/lorebianchi98/Talk2DINO): `Talk2DINO/src` and `Talk2DINO/configs` are adapted from the official repository.
