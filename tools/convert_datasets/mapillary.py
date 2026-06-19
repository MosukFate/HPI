# Filename: convert_mapillary_to_trainid.py
import argparse
import json
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

def convert_to_train_id(file_path):
    """Remap a single Mapillary TrainId annotation to Cityscapes TrainIds."""
    pil_label = Image.open(file_path)
    label = np.asarray(pil_label, dtype=np.uint8)
    # Mapillary TrainId to Cityscapes TrainId
    id_to_trainid = {
        13:  0, 24:  0, 41:  0,  # Road
         2:  1, 15:  1,          # Sidewalk
        17:  2,                  # Building
         6:  3,                  # Wall
         3:  4,                  # Fence
        45:  5, 47:  5,          # Pole
        48:  6,                  # Traffic light
        50:  7,                  # Traffic sign
        30:  8,                  # Vegetation
        29:  9,                  # Terrain
        27: 10,                  # Sky
        19: 11,                  # Person
        20: 12, 21: 12, 22: 12,  # Rider
        55: 13,                  # Car
        61: 14,                  # Truck
        54: 15,                  # Bus
        58: 16,                  # Train
        57: 17,                  # Motorcycle
        52: 18                   # Bicycle
    }
    # Initialize as ignore label 255
    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    stats = {}
    for ori_id, train_id in id_to_trainid.items():
        mask = (label == ori_id)
        label_copy[mask] = train_id
        cnt = int(mask.sum())
        if cnt > 0:
            stats[train_id] = cnt
    new_file = file_path.replace('.png', '_labelTrainIds.png')
    Image.fromarray(label_copy, mode='L').save(new_file)
    stats['file'] = new_file
    return stats

def parse_args():
    parser = argparse.ArgumentParser(description='Convert Mapillary annotations to Cityscapes TrainIds')
    parser.add_argument('mapillary_path', help='Mapillary root path')
    parser.add_argument('--gt-dir', default='labels', help='Single-channel TrainId label directory')
    parser.add_argument('-o', '--out-dir', help='Output path; defaults to overwriting under the input directory')
    parser.add_argument('--nproc', type=int, default=4, help='Number of parallel processes')
    return parser.parse_args()

def save_class_stats(out_dir, stats_list):
    with open(osp.join(out_dir, 'mapillary_sample_class_stats.json'), 'w') as f:
        json.dump(stats_list, f, indent=2)
    stats_dict = {s.pop('file'): s for s in stats_list}
    with open(osp.join(out_dir, 'mapillary_sample_class_stats_dict.json'), 'w') as f:
        json.dump(stats_dict, f, indent=2)

def main():
    args = parse_args()
    in_root = args.mapillary_path
    out_root = args.out_dir or in_root
    mmcv.mkdir_or_exist(out_root)
    gt_dir = osp.join(in_root, args.gt_dir)
    pngs = sorted([
        osp.join(gt_dir, f)
        for f in mmcv.scandir(gt_dir, suffix='.png', recursive=True)
    ])
    if args.nproc > 1:
        stats = mmcv.track_parallel_progress(convert_to_train_id, pngs, args.nproc)
    else:
        stats = mmcv.track_progress(convert_to_train_id, pngs)
    save_class_stats(out_root, stats)

if __name__ == '__main__':
    main()
