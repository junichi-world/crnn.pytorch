#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create LMDB dataset for CRNN training.

Supports a folder layout like:
    data/letter_images/
      b/*.png
      7/*.png
      ...
where the folder name is used as the label.

Outputs an LMDB compatible with `dataset.lmdbDataset` in this repository:
  - image-%09d
  - label-%09d
  - num-samples
"""

from __future__ import annotations

import argparse
import io
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import lmdb
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


Sample = Tuple[str, str]  # (image_path, label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CRNN LMDB dataset")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--folder",
        type=str,
        help="Root folder containing class subfolders (subfolder name = label)",
    )
    src.add_argument(
        "--list",
        dest="list_file",
        type=str,
        help="Text file with one sample per line: <image_path>\t<label>",
    )

    parser.add_argument("--out", type=str, help="Output LMDB directory (single dataset mode)")
    parser.add_argument("--trainOut", type=str, help="Output train LMDB directory (split mode)")
    parser.add_argument("--valOut", type=str, help="Output val LMDB directory (split mode)")
    parser.add_argument(
        "--valRatio",
        type=float,
        default=0.0,
        help="Validation split ratio (0.0-1.0). Use with --trainOut/--valOut",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for split")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search image files under each label folder",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples before writing (single dataset mode) or before splitting",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip image validity checks",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=1 << 40,
        help="LMDB map_size in bytes (default: 1TB virtual map)",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def gather_samples_from_folder(root: Path, recursive: bool = False) -> List[Sample]:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root}")

    samples: List[Sample] = []
    label_dirs = [p for p in root.iterdir() if p.is_dir()]
    label_dirs.sort(key=lambda p: p.name)

    for label_dir in label_dirs:
        label = label_dir.name
        iterator: Iterable[Path]
        iterator = label_dir.rglob("*") if recursive else label_dir.iterdir()
        files = [p for p in iterator if is_image_file(p)]
        files.sort()
        for p in files:
            samples.append((str(p.resolve()), label))

    return samples


def gather_samples_from_list(list_file: Path) -> List[Sample]:
    if not list_file.exists() or not list_file.is_file():
        raise FileNotFoundError(f"List file not found: {list_file}")

    base_dir = list_file.parent
    samples: List[Sample] = []
    with list_file.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                path_str, label = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid line {line_no}: {raw.rstrip()}")
                path_str, label = parts
            p = Path(path_str)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            samples.append((str(p), label))
    return samples


def check_image_valid(img_bin: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(img_bin)) as im:
            im.verify()
        return True
    except Exception:
        return False


def write_cache(env: lmdb.Environment, cache: dict[bytes, bytes]) -> None:
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def create_lmdb_dataset(
    output_path: Path,
    samples: Sequence[Sample],
    check_valid: bool = True,
    map_size: int = 1 << 40,
) -> None:
    if not samples:
        raise ValueError(f"No samples to write: {output_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(output_path), map_size=map_size)

    cache: dict[bytes, bytes] = {}
    n_written = 0
    skipped = 0

    for idx, (image_path, label) in enumerate(samples, 1):
        try:
            with open(image_path, "rb") as f:
                image_bin = f.read()
        except OSError as exc:
            print(f"[skip] cannot read {image_path}: {exc}")
            skipped += 1
            continue

        if check_valid and not check_image_valid(image_bin):
            print(f"[skip] invalid image: {image_path}")
            skipped += 1
            continue

        n_written += 1
        image_key = f"image-{n_written:09d}".encode("ascii")
        label_key = f"label-{n_written:09d}".encode("ascii")
        cache[image_key] = image_bin
        cache[label_key] = label.encode("utf-8")

        if n_written % 1000 == 0:
            write_cache(env, cache)
            cache.clear()
            print(f"Written {n_written}/{len(samples)} (processed {idx})")

    cache[b"num-samples"] = str(n_written).encode("ascii")
    write_cache(env, cache)
    env.sync()
    env.close()

    print(f"Done: {output_path}")
    print(f"  written: {n_written}")
    print(f"  skipped: {skipped}")


def stratified_split(samples: Sequence[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")

    rng = random.Random(seed)
    grouped: dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample[1]].append(sample)

    train_samples: List[Sample] = []
    val_samples: List[Sample] = []

    for label in sorted(grouped.keys()):
        items = grouped[label][:]
        rng.shuffle(items)
        n = len(items)
        if n <= 1:
            train_samples.extend(items)
            continue

        n_val = int(round(n * val_ratio))
        n_val = max(1, n_val) if val_ratio > 0 else 0
        n_val = min(n - 1, n_val)

        val_samples.extend(items[:n_val])
        train_samples.extend(items[n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def summarize_labels(samples: Sequence[Sample], title: str) -> None:
    counts: dict[str, int] = defaultdict(int)
    for _, label in samples:
        counts[label] += 1
    print(f"{title}: {len(samples)} samples, {len(counts)} labels")
    for label in sorted(counts.keys()):
        print(f"  {label}\t{counts[label]}")


def main() -> None:
    args = parse_args()

    if args.folder:
        samples = gather_samples_from_folder(Path(args.folder), recursive=args.recursive)
    else:
        samples = gather_samples_from_list(Path(args.list_file))

    if not samples:
        raise SystemExit("No samples found. Check --folder/--list input.")

    if args.shuffle:
        random.Random(args.seed).shuffle(samples)

    split_mode = bool(args.trainOut or args.valOut or args.valRatio > 0)
    if split_mode:
        if not args.trainOut or not args.valOut:
            raise SystemExit("Split mode requires both --trainOut and --valOut")
        if not (0.0 < args.valRatio < 1.0):
            raise SystemExit("Split mode requires --valRatio in (0, 1)")

        train_samples, val_samples = stratified_split(samples, args.valRatio, args.seed)
        summarize_labels(train_samples, "Train")
        summarize_labels(val_samples, "Val")
        create_lmdb_dataset(Path(args.trainOut), train_samples, check_valid=not args.no_check, map_size=args.map_size)
        create_lmdb_dataset(Path(args.valOut), val_samples, check_valid=not args.no_check, map_size=args.map_size)
        return

    if not args.out:
        raise SystemExit("Single dataset mode requires --out")

    summarize_labels(samples, "Dataset")
    create_lmdb_dataset(Path(args.out), samples, check_valid=not args.no_check, map_size=args.map_size)


if __name__ == "__main__":
    main()
