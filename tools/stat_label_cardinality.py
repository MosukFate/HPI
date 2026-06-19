#!/usr/bin/env python3
import argparse
import copy
import os
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PRESETS = {
    "cityscapes": {
        "mask_dir": "data/cityscapes/gtFine/train",
        "suffix": "_labelTrainIds.png",
        "recursive": True,
        "ignore": [255, -1],
        "stats_candidates": [
            "data/cityscapes/sample_class_stats_dict.json",
            "data/cityscapes/sample_class_stats.json",
        ],
    },
    "gta": {
        "mask_dir": "data/gta/labels",
        "suffix": "_labelTrainIds.png",
        "recursive": True,
        "ignore": [255, -1],
        "stats_candidates": [
            "data/gta/sample_class_stats_dict.json",
            "data/gta/sample_class_stats.json",
        ],
    },
    "bdd": {
        "mask_dir": "data/bdd100k/labels/sem_seg/masks/train",
        "suffix": ".png",
        "recursive": True,
        "ignore": [255, -1],
        "num_workers": 8,
    },
    "mapillary": {
        "mask_dir": "data/mapillary/validation/labels",
        "suffix": "_labelTrainIds.png",
        "recursive": True,
        "ignore": [255, -1],
        "stats_candidates": [
            "data/mapillary/validation/sample_class_stats_dict.json",
            "data/mapillary/validation/sample_class_stats.json",
        ],
    },
}

DOMAIN_DATASETS = {
    "gta": {
        "type": "GTADataset",
        "data_root": "data/gta",
        "resize": (1280, 720),
        "sources": [
            dict(img_dir="images", ann_dir="labels", split="default"),
        ],
    },
    "cityscapes": {
        "type": "CityscapesDataset",
        "data_root": "data/cityscapes",
        "resize": (1024, 512),
        "sources": [
            dict(img_dir="leftImg8bit/train", ann_dir="gtFine/train", split="train"),
            dict(img_dir="leftImg8bit/val", ann_dir="gtFine/val", split="val"),
        ],
    },
    "bdd": {
        "type": "BDD100kDataset",
        "data_root": "data/bdd100k",
        "resize": (1280, 720),
        "sources": [
            dict(img_dir="images/10k/train", ann_dir="labels/sem_seg/masks/train", split="train"),
            dict(img_dir="images/10k/val", ann_dir="labels/sem_seg/masks/val", split="val"),
            dict(img_dir="images/10k/val", ann_dir="labels/sem_seg/colormaps/val", split="val_colormap"),
        ],
    },
    "mapillary": {
        "type": "MapillaryDataset",
        "data_root": "data/mapillary",
        "resize": (1024, 512),
        "sources": [
            dict(img_dir="training/images", ann_dir="cityscapes_trainIdLabel/train/label", split="train"),
            dict(img_dir="validation/images", ann_dir="validation/labels", split="validation"),
            dict(img_dir="half/val_img", ann_dir="half/val_label", split="half_val"),
        ],
    },
}

FIXED_SUFFIX_DATASET_TYPES = {
    "GTADataset",
    "BDD100kDataset",
    "MapillaryDataset",
    "CityscapesDataset",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count how many semantic classes appear in each label map."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Use a built-in dataset preset.",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default=None,
        help="Directory containing segmentation masks.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Only include files ending with this suffix.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan mask_dir.",
    )
    parser.add_argument(
        "--ignore",
        type=int,
        nargs="*",
        default=None,
        help="Label ids to ignore when counting unique classes. Default: 255 -1",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Show the top-k most frequent class-count values.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save full stats as JSON.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N masks. Use 0 to disable.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker threads for PNG scanning. Default uses preset or 1.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional mmseg config. When set, count classes after dataset pipeline instead of raw masks.",
    )
    parser.add_argument(
        "--dataset-var",
        type=str,
        default=None,
        help="Optional top-level dataset variable name in config mode, e.g. train_gta.",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which cfg.data split to use in --config mode.",
    )
    parser.add_argument(
        "--seg-key",
        type=str,
        default=None,
        help="Segmentation key to count in --config mode. Default auto-detects gt_semantic_seg/source_gt_semantic_seg/target_gt_semantic_seg.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="How many dataset samples to draw in --config mode. 0 means len(dataset).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for --config mode sampling and pipeline randomness.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        nargs="+",
        choices=sorted(DOMAIN_DATASETS.keys()),
        default=None,
        help="Count classes after the built-in 512 training pipeline for one or more domains.",
    )
    parser.add_argument(
        "--all-domains",
        action="store_true",
        help="Shortcut for --domain gta cityscapes bdd mapillary.",
    )
    return parser.parse_args()


def resolve_args(args):
    cfg = {}
    if args.preset is not None:
        cfg.update(PRESETS[args.preset])

    if args.mask_dir is not None:
        cfg["mask_dir"] = args.mask_dir
    if args.suffix is not None:
        cfg["suffix"] = args.suffix
    if args.recursive:
        cfg["recursive"] = True
    if args.ignore is not None:
        cfg["ignore"] = args.ignore
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers

    cfg.setdefault("recursive", True)
    cfg.setdefault("ignore", [255, -1])
    cfg.setdefault("suffix", ".png")
    cfg.setdefault("num_workers", 1)
    cfg.setdefault("stats_candidates", [])

    if "mask_dir" not in cfg:
        raise ValueError("Please provide --preset or --mask-dir.")
    return cfg


def collect_files(mask_dir: Path, suffix: str, recursive: bool):
    pattern = f"*{suffix}"
    if recursive:
        files = sorted(p for p in mask_dir.rglob(pattern) if p.is_file())
    else:
        files = sorted(p for p in mask_dir.glob(pattern) if p.is_file())
    return files


def count_classes(mask_path: Path, ignore_values):
    arr = np.array(Image.open(mask_path))
    unique = np.unique(arr)
    valid = [int(x) for x in unique.tolist() if int(x) not in ignore_values]
    return {
        "file": str(mask_path),
        "num_classes": len(valid),
        "classes": valid,
    }


def load_records_from_cache(stats_path: Path, ignore_values):
    with stats_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    records = []
    if isinstance(obj, dict):
        iterator = obj.items()
        for file_path, class_stats in iterator:
            valid = sorted(
                int(k) for k in class_stats.keys()
                if int(k) not in ignore_values
            )
            records.append({
                "file": str(file_path),
                "num_classes": len(valid),
                "classes": valid,
            })
    elif isinstance(obj, list):
        for item in obj:
            valid = sorted(
                int(k) for k in item.keys()
                if k != "file" and int(k) not in ignore_values
            )
            records.append({
                "file": str(item.get("file", "")),
                "num_classes": len(valid),
                "classes": valid,
            })
    else:
        raise TypeError(f"Unsupported cached stats type: {type(obj).__name__}")

    return records


def collect_records_serial(files, ignore_values, progress_every):
    records = []
    total = len(files)
    for idx, path in enumerate(files, start=1):
        records.append(count_classes(path, ignore_values))
        if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == total):
            print(f"[progress] {idx}/{total}: {path}", file=sys.stderr, flush=True)
    return records


def collect_records_threaded(files, ignore_values, progress_every, num_workers):
    records = []
    total = len(files)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(count_classes, path, ignore_values): path for path in files}
        for idx, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            records.append(future.result())
            if progress_every > 0 and (idx == 1 or idx % progress_every == 0 or idx == total):
                print(f"[progress] {idx}/{total}: {path}", file=sys.stderr, flush=True)
    return records


def collect_records(files, ignore_values, progress_every, num_workers):
    if num_workers <= 1:
        return collect_records_serial(files, ignore_values, progress_every)
    return collect_records_threaded(files, ignore_values, progress_every, num_workers)


def summarize(records, topk):
    counts = np.array([r["num_classes"] for r in records], dtype=np.int64)
    hist = {}
    for c in counts.tolist():
        hist[c] = hist.get(c, 0) + 1
    hist_sorted = sorted(hist.items(), key=lambda x: (-x[1], x[0]))

    summary = {
        "num_images": int(len(records)),
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "p10": float(np.percentile(counts, 10)),
        "p25": float(np.percentile(counts, 25)),
        "p75": float(np.percentile(counts, 75)),
        "p90": float(np.percentile(counts, 90)),
        "hist_topk": hist_sorted[:topk],
    }
    return summary


def unwrap_data(value):
    return value.data if hasattr(value, "data") else value


def tensor_to_numpy(value):
    value = unwrap_data(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def extract_filename(sample):
    meta_keys = [
        "img_metas",
        "source_img_metas",
        "target_img_metas",
    ]
    for key in meta_keys:
        if key not in sample:
            continue
        meta = unwrap_data(sample[key])
        if isinstance(meta, dict):
            return meta.get("filename", "")
    return ""


def infer_seg_key(sample, requested_key=None):
    if requested_key is not None:
        if requested_key not in sample:
            raise KeyError(
                f"Requested seg key `{requested_key}` not found. "
                f"Available keys: {sorted(sample.keys())}"
            )
        return requested_key

    candidates = [
        "gt_semantic_seg",
        "source_gt_semantic_seg",
        "target_gt_semantic_seg",
    ]
    for key in candidates:
        if key in sample:
            return key

    raise KeyError(
        "Could not auto-detect a segmentation key in dataset sample. "
        f"Available keys: {sorted(sample.keys())}"
    )


def count_classes_in_array(arr, ignore_values):
    unique = np.unique(arr)
    return sorted(int(x) for x in unique.tolist() if int(x) not in ignore_values)


def count_classes_from_sample(sample, ignore_values, seg_key):
    arr = tensor_to_numpy(sample[seg_key])
    if arr.ndim >= 3:
        arr = arr.squeeze()
    classes = count_classes_in_array(arr, ignore_values)
    return {
        "file": extract_filename(sample),
        "num_classes": len(classes),
        "classes": classes,
    }


def sanitize_dataset_cfg(dataset_cfg):
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_type = dataset_cfg.get("type")
    if dataset_type in FIXED_SUFFIX_DATASET_TYPES:
        dataset_cfg.pop("img_suffix", None)
        dataset_cfg.pop("seg_map_suffix", None)
    data_prefix = dataset_cfg.pop("data_prefix", None)
    if isinstance(data_prefix, dict):
        if "img_path" in data_prefix and "img_dir" not in dataset_cfg:
            dataset_cfg["img_dir"] = data_prefix["img_path"]
        if "seg_map_path" in data_prefix and "ann_dir" not in dataset_cfg:
            dataset_cfg["ann_dir"] = data_prefix["seg_map_path"]
    pipeline = dataset_cfg.get("pipeline")
    if isinstance(pipeline, list):
        for step in pipeline:
            if not isinstance(step, dict):
                continue
            if step.get("type") == "Resize" and "scale" in step and "img_scale" not in step:
                step["img_scale"] = step.pop("scale")
                step.setdefault("keep_ratio", False)
            if step.get("type") == "LoadAnnotations":
                step.pop("reduce_zero_label", None)
            if step.get("type") == "PackSegInputs":
                step["type"] = "Collect"
                step["keys"] = ["img", "gt_semantic_seg"]
    return dataset_cfg


def resolve_domain_source(domain_name):
    domain_cfg = DOMAIN_DATASETS[domain_name]
    data_root = Path(domain_cfg["data_root"])
    for source in domain_cfg["sources"]:
        img_dir = data_root / source["img_dir"]
        ann_dir = data_root / source["ann_dir"]
        if img_dir.is_dir() and ann_dir.is_dir():
            return source

    attempted = ", ".join(
        f"({source['img_dir']} | {source['ann_dir']})"
        for source in domain_cfg["sources"]
    )
    raise FileNotFoundError(
        f"No valid data directories found for domain `{domain_name}` under "
        f"{data_root}. Tried: {attempted}"
    )


def build_domain_dataset_cfg(domain_name):
    domain_cfg = DOMAIN_DATASETS[domain_name]
    source = resolve_domain_source(domain_name)
    dataset_cfg = dict(
        type=domain_cfg["type"],
        data_root=domain_cfg["data_root"],
        img_dir=source["img_dir"],
        ann_dir=source["ann_dir"],
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="Resize", img_scale=domain_cfg["resize"], keep_ratio=False),
            dict(type="RandomCrop", crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type="RandomFlip", prob=0.5),
        ],
    )
    return dataset_cfg, source["split"]


def try_build_domain_dataset(domain_name):
    from mmseg.datasets import build_dataset

    domain_cfg = DOMAIN_DATASETS[domain_name]
    data_root = Path(domain_cfg["data_root"])
    errors = []

    for source in domain_cfg["sources"]:
        img_dir = data_root / source["img_dir"]
        ann_dir = data_root / source["ann_dir"]
        if not (img_dir.is_dir() and ann_dir.is_dir()):
            errors.append(
                f"{source['split']}: missing dir(s) img={img_dir} ann={ann_dir}"
            )
            continue

        dataset_cfg = dict(
            type=domain_cfg["type"],
            data_root=domain_cfg["data_root"],
            img_dir=source["img_dir"],
            ann_dir=source["ann_dir"],
            pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="LoadAnnotations"),
                dict(type="Resize", img_scale=domain_cfg["resize"], keep_ratio=False),
                dict(type="RandomCrop", crop_size=(512, 512), cat_max_ratio=0.75),
                dict(type="RandomFlip", prob=0.5),
            ],
        )

        try:
            dataset = build_dataset(sanitize_dataset_cfg(dataset_cfg))
            return dataset, dataset_cfg, source["split"]
        except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
            errors.append(f"{source['split']}: {exc}")

    error_text = "\n".join(errors)
    raise RuntimeError(
        f"Could not build a usable dataset for domain `{domain_name}`.\n{error_text}"
    )


def collect_records_from_dataset(dataset,
                                 ignore_values,
                                 progress_every,
                                 num_samples=0,
                                 seg_key=None):
    dataset_len = len(dataset)
    if dataset_len == 0:
        raise ValueError("Dataset is empty.")

    sample_count = dataset_len if num_samples <= 0 else num_samples
    records = []
    resolved_seg_key = None

    for idx in range(sample_count):
        sample = dataset[idx % dataset_len]
        if resolved_seg_key is None:
            resolved_seg_key = infer_seg_key(sample, seg_key)
            print(
                f"[config-mode] using seg key: {resolved_seg_key}",
                file=sys.stderr,
                flush=True,
            )
        record = count_classes_from_sample(sample, ignore_values, resolved_seg_key)
        records.append(record)

        if progress_every > 0 and (
            idx == 0 or (idx + 1) % progress_every == 0 or (idx + 1) == sample_count
        ):
            msg_file = record["file"] if record["file"] else f"sample_{idx}"
            print(
                f"[progress] {idx + 1}/{sample_count}: {msg_file}",
                file=sys.stderr,
                flush=True,
            )

    return records, resolved_seg_key


def run_config_mode(args, ignore_values):
    try:
        from mmcv import Config
    except ImportError as exc:
        raise ImportError(
            "--config mode requires mmcv to be installed in the active environment."
        ) from exc

    try:
        from mmseg.datasets import build_dataset
    except ImportError as exc:
        raise ImportError(
            "--config mode requires mmseg dataset imports to succeed in the active environment."
        ) from exc

    cfg = Config.fromfile(args.config)

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    if args.dataset_var is not None:
        if args.dataset_var not in cfg:
            raise KeyError(
                f"Config {args.config} does not contain top-level dataset var "
                f"`{args.dataset_var}`"
            )
        dataset_cfg = cfg[args.dataset_var]
        dataset_name = args.dataset_var
    else:
        if "data" not in cfg or args.data_split not in cfg.data:
            raise KeyError(
                f"Config {args.config} does not contain cfg.data.{args.data_split}. "
                "If this is a dataset-only config, pass --dataset-var."
            )
        dataset_cfg = cfg.data[args.data_split]
        dataset_name = f"data.{args.data_split}"

    dataset = build_dataset(sanitize_dataset_cfg(dataset_cfg))
    records, resolved_seg_key = collect_records_from_dataset(
        dataset=dataset,
        ignore_values=ignore_values,
        progress_every=args.progress_every,
        num_samples=args.num_samples,
        seg_key=args.seg_key,
    )
    summary = summarize(records, args.topk)

    print(f"config: {args.config}")
    print(f"dataset_ref: {dataset_name}")
    print(f"dataset_type: {type(dataset).__name__}")
    print(f"seg_key: {resolved_seg_key}")
    print(f"seed: {args.seed}")
    print(f"num_samples: {summary['num_images']}")
    print(f"mean classes/input: {summary['mean']:.3f}")
    print(f"median classes/input: {summary['median']:.3f}")
    print(
        "min / p10 / p25 / p75 / p90 / max: "
        f"{summary['min']} / {summary['p10']:.1f} / {summary['p25']:.1f} / "
        f"{summary['p75']:.1f} / {summary['p90']:.1f} / {summary['max']}"
    )
    print("most frequent class-count values:")
    for count_value, freq in summary["hist_topk"]:
        print(f"  {count_value}: {freq}")

    if args.save_json is not None:
        payload = {
            "config": {
                "config_path": args.config,
                "dataset_ref": dataset_name,
                "seg_key": resolved_seg_key,
                "ignore": sorted(ignore_values),
                "seed": args.seed,
                "num_samples": args.num_samples if args.num_samples > 0 else len(dataset),
            },
            "summary": summary,
            "records": records,
        }
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"saved json: {out_path}")


def summarize_domain(domain_name, args, ignore_values):
    domain_args = argparse.Namespace(**vars(args))
    domain_args.config = None
    domain_args.dataset_var = domain_name
    domain_args.data_split = "train"
    domain_args.save_json = None

    try:
        from mmseg.datasets import build_dataset  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "--domain/--all-domains mode requires mmcv and repo-local mmseg imports "
            "to succeed in the active environment."
        ) from exc

    random.seed(domain_args.seed)
    np.random.seed(domain_args.seed)
    try:
        import torch
        torch.manual_seed(domain_args.seed)
    except ImportError:
        pass

    dataset, dataset_cfg, resolved_split = try_build_domain_dataset(domain_name)
    records, resolved_seg_key = collect_records_from_dataset(
        dataset=dataset,
        ignore_values=ignore_values,
        progress_every=domain_args.progress_every,
        num_samples=domain_args.num_samples,
        seg_key=domain_args.seg_key,
    )
    summary = summarize(records, domain_args.topk)
    return {
        "config_path": None,
        "dataset_var": domain_args.dataset_var,
        "dataset_type": type(dataset).__name__,
        "seg_key": resolved_seg_key,
        "resolved_split": resolved_split,
        "summary": summary,
        "records": records,
        "dataset_cfg": dataset_cfg,
    }


def run_domain_mode(args, ignore_values):
    domains = list(DOMAIN_DATASETS.keys()) if args.all_domains else args.domain
    if not domains:
        raise ValueError("run_domain_mode requires --domain or --all-domains.")

    payload = {
        "config": {
            "mode": "domain_presets",
            "domains": domains,
            "ignore": sorted(ignore_values),
            "seed": args.seed,
            "num_samples": args.num_samples,
        },
        "domains": {},
    }

    for domain_name in domains:
        print(f"\n=== domain: {domain_name} ===", flush=True)
        result = summarize_domain(domain_name, args, ignore_values)
        summary = result["summary"]

        print(f"config: {result['config_path']}")
        print(f"dataset_ref: {result['dataset_var']}")
        print(f"dataset_type: {result['dataset_type']}")
        print(f"resolved_split: {result['resolved_split']}")
        print(f"seg_key: {result['seg_key']}")
        print(f"seed: {args.seed}")
        print(f"num_samples: {summary['num_images']}")
        print(f"mean classes/input: {summary['mean']:.3f}")
        print(f"median classes/input: {summary['median']:.3f}")
        print(
            "min / p10 / p25 / p75 / p90 / max: "
            f"{summary['min']} / {summary['p10']:.1f} / {summary['p25']:.1f} / "
            f"{summary['p75']:.1f} / {summary['p90']:.1f} / {summary['max']}"
        )
        print("most frequent class-count values:")
        for count_value, freq in summary["hist_topk"]:
            print(f"  {count_value}: {freq}")

        payload["domains"][domain_name] = result

    if args.save_json is not None:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nsaved json: {out_path}")


def main():
    args = parse_args()
    if args.domain is not None or args.all_domains:
        ignore_values = set([255, -1] if args.ignore is None else args.ignore)
        run_domain_mode(args, ignore_values)
        return

    if args.config is not None:
        ignore_values = set([255, -1] if args.ignore is None else args.ignore)
        run_config_mode(args, ignore_values)
        return

    cfg = resolve_args(args)
    ignore_values = set(cfg["ignore"])

    for stats_path_str in cfg["stats_candidates"]:
        stats_path = Path(stats_path_str)
        if stats_path.exists():
            print(f"using cached stats: {stats_path}", file=sys.stderr, flush=True)
            records = load_records_from_cache(stats_path, ignore_values)
            summary = summarize(records, args.topk)

            print(f"stats_file: {stats_path}")
            print(f"num_images: {summary['num_images']}")
            print(f"mean classes/image: {summary['mean']:.3f}")
            print(f"median classes/image: {summary['median']:.3f}")
            print(
                "min / p10 / p25 / p75 / p90 / max: "
                f"{summary['min']} / {summary['p10']:.1f} / {summary['p25']:.1f} / "
                f"{summary['p75']:.1f} / {summary['p90']:.1f} / {summary['max']}"
            )
            print("most frequent class-count values:")
            for count_value, freq in summary["hist_topk"]:
                print(f"  {count_value}: {freq}")

            if args.save_json is not None:
                payload = {
                    "config": {
                        "stats_file": str(stats_path),
                        "ignore": cfg["ignore"],
                    },
                    "summary": summary,
                    "records": records,
                }
                out_path = Path(args.save_json)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print(f"saved json: {out_path}")
            return

    mask_dir = Path(cfg["mask_dir"])
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    files = collect_files(mask_dir, cfg["suffix"], cfg["recursive"])
    if not files:
        raise FileNotFoundError(
            f"No mask files found in {mask_dir} with suffix {cfg['suffix']}"
        )

    print(
        f"found {len(files)} masks in {mask_dir} (suffix={cfg['suffix']})",
        file=sys.stderr,
        flush=True,
    )
    num_workers = max(1, int(cfg["num_workers"]))
    if num_workers > 1:
        print(f"using {num_workers} worker threads", file=sys.stderr, flush=True)
    records = collect_records(files, ignore_values, args.progress_every, num_workers)
    summary = summarize(records, args.topk)

    print(f"mask_dir: {mask_dir}")
    print(f"suffix: {cfg['suffix']}")
    print(f"num_images: {summary['num_images']}")
    print(f"mean classes/image: {summary['mean']:.3f}")
    print(f"median classes/image: {summary['median']:.3f}")
    print(
        "min / p10 / p25 / p75 / p90 / max: "
        f"{summary['min']} / {summary['p10']:.1f} / {summary['p25']:.1f} / "
        f"{summary['p75']:.1f} / {summary['p90']:.1f} / {summary['max']}"
    )
    print("most frequent class-count values:")
    for count_value, freq in summary["hist_topk"]:
        print(f"  {count_value}: {freq}")

    if args.save_json is not None:
        payload = {
            "config": {
                "mask_dir": str(mask_dir),
                "suffix": cfg["suffix"],
                "recursive": cfg["recursive"],
                "ignore": cfg["ignore"],
                "num_workers": num_workers,
            },
            "summary": summary,
            "records": records,
        }
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"saved json: {out_path}")


if __name__ == "__main__":
    main()
