# Filename: convert_bdd100k_to_trainid_color.py
import argparse
import os
import os.path as osp
import mmcv
import numpy as np
from PIL import Image

# Cityscapes color palette to TrainId mapping
COLOR2TRAINID = {
    (128,  64, 128): 0,   # road
    (244,  35, 232): 1,   # sidewalk
    ( 70,  70,  70): 2,   # building
    (102, 102, 156): 3,   # wall
    (190, 153, 153): 4,   # fence
    (153, 153, 153): 5,   # pole
    (250, 170,  30): 6,   # traffic light
    (220, 220,   0): 7,   # traffic sign
    (107, 142,  35): 8,   # vegetation
    (152, 251, 152): 9,   # terrain
    ( 70, 130, 180): 10,  # sky
    (220,  20,  60): 11,  # person
    (255,   0,   0): 12,  # rider
    (  0,   0, 142): 13,  # car
    (  0,   0,  70): 14,  # truck
    (  0,  60, 100): 15,  # bus
    (  0,  80, 100): 16,  # train
    (  0,   0, 230): 17,  # motorcycle
    (119,  11,  32): 18,  # bicycle
    # Other colors map to ignore label 255
}

def convert_color_png_to_trainid(src_path, dst_path):
    """Map a color segmentation label to single-channel TrainId and save it."""
    with Image.open(src_path) as im:
        im = im.convert('RGB')
        img = np.asarray(im, dtype=np.uint8)
    h, w, _ = img.shape
    out = np.full((h, w), 255, dtype=np.uint8)
    for color, tid in COLOR2TRAINID.items():
        mask = np.all(img == color, axis=-1)
        out[mask] = tid
    mmcv.mkdir_or_exist(osp.dirname(dst_path))
    Image.fromarray(out, mode='L').save(dst_path)
    return dst_path

def process_task(task):
    """Wrapper for multiprocessing: task is (src, dst) tuple"""
    src, dst = task
    return convert_color_png_to_trainid(src, dst)


def delete_old_labels(out_gt_dir):
    """Delete old *_labelTrainIds.png files."""
    cnt = 0
    for root, _, files in os.walk(out_gt_dir):
        for f in files:
            if f.endswith('_labelTrainIds.png'):
                os.remove(osp.join(root, f))
                cnt += 1
    print(f"Deleted {cnt} old TrainIds files in {out_gt_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch remap BDD100K color labels to Cityscapes TrainIds, GTA-style.'
    )
    parser.add_argument('root', help='BDD100K dataset root directory')
    parser.add_argument('--gt-dir', default='labels', help='Label subdirectory relative to root')
    parser.add_argument('-o', '--out-dir', help='Output root directory; defaults to writing under the input root')
    parser.add_argument('--nproc', type=int, default=4, help='Number of parallel processes')
    return parser.parse_args()


def main():
    args = parse_args()
    in_gt = osp.join(args.root, args.gt_dir)
    out_root = args.out_dir or args.root
    out_gt = osp.join(out_root, args.gt_dir)

    # Delete old files
    delete_old_labels(out_gt)

    # Collect all .png labels
    src_paths = []
    for root, _, files in os.walk(in_gt):
        for f in files:
            if f.endswith('.png') and not f.endswith('_labelTrainIds.png'):
                src_paths.append(osp.join(root, f))
    if not src_paths:
        print(f"Warning: no PNG files found in in_gt: {in_gt}")
        return
    print(f"Found {len(src_paths)} PNG files to convert in {in_gt}")

    # Build task list
    tasks = []
    for src in src_paths:
        rel = osp.relpath(src, in_gt)
        dst = osp.join(out_gt, osp.dirname(rel), osp.basename(src).replace('.png', '_labelTrainIds.png'))
        tasks.append((src, dst))

    # Batch conversion
    if args.nproc > 1:
        results = mmcv.track_parallel_progress(process_task, tasks, args.nproc)
    else:
        results = mmcv.track_progress(process_task, tasks)

    # Print statistics
    for dst in results:
        print(f"[saved] {dst}")
    print('Conversion completed.')

if __name__ == '__main__':
    main()
