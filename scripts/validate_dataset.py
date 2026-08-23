#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

EXPECTED = {
    "train": {"images": 1538, "instances": 18605},
    "val": {"images": 443, "instances": 5358},
    "test": {"images": 222, "instances": 2650},
}
EXPECTED_AUG_IMAGES = 7690
EXPECTED_AUG_INSTANCES = 93025
MAX_LONG_SIDE = 1920


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_image_cv(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image with OpenCV: {path}")
    return img


def parse_label(path: Path):
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise RuntimeError(f"Empty label file: {path}")

    areas, ratios = [], []
    count = 0

    for lineno, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise RuntimeError(f"{path}:{lineno}: expected 5 YOLO fields")

        try:
            cls, x, y, w, h = map(float, parts)
        except ValueError:
            raise RuntimeError(f"{path}:{lineno}: non-numeric YOLO value")

        if int(cls) != 0:
            raise RuntimeError(f"{path}:{lineno}: class must be 0")

        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise RuntimeError(f"{path}:{lineno}: non-finite value")

        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise RuntimeError(f"{path}:{lineno}: invalid normalized box")

        eps = 1e-6
        if x - w / 2 < -eps or x + w / 2 > 1 + eps:
            raise RuntimeError(f"{path}:{lineno}: x bounds outside image")
        if y - h / 2 < -eps or y + h / 2 > 1 + eps:
            raise RuntimeError(f"{path}:{lineno}: y bounds outside image")

        areas.append(w * h)
        ratios.append(w / h)
        count += 1

    return count, areas, ratios


def split_stats(root: Path, split: str):
    img_dir = root / "original" / split / "images"
    lbl_dir = root / "original" / split / "labels"

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})
    labels = sorted(p for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt")

    exp = EXPECTED[split]
    if len(images) != exp["images"] or len(labels) != exp["images"]:
        raise RuntimeError(
            f"{split}: expected {exp['images']} images/labels; "
            f"found {len(images)}/{len(labels)}"
        )

    if {p.stem for p in images} != {p.stem for p in labels}:
        raise RuntimeError(f"{split}: image-label correspondence mismatch")

    counts = []
    all_areas = []
    all_ratios = []
    brightness = []
    contrast = []
    sharpness = []
    per_image = []
    hashes = {}

    for i, image_path in enumerate(images, 1):
        label_path = lbl_dir / f"{image_path.stem}.txt"
        n, areas, ratios = parse_label(label_path)

        counts.append(n)
        all_areas.extend(areas)
        all_ratios.extend(ratios)

        with Image.open(image_path) as im:
            im.load()
            if max(im.size) > MAX_LONG_SIDE:
                raise RuntimeError(
                    f"Released image exceeds {MAX_LONG_SIDE}px long side: "
                    f"{image_path} {im.size}"
                )

        image = read_image_cv(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        b = float(np.mean(gray))
        c = float(np.std(gray))
        s = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        brightness.append(b)
        contrast.append(c)
        sharpness.append(s)

        h, w = gray.shape
        per_image.append({
            "split": split,
            "image_name": image_path.name,
            "width_px": w,
            "height_px": h,
            "instances": n,
            "brightness": b,
            "contrast": c,
            "sharpness": s,
        })

        digest = sha256_file(image_path)
        if digest in hashes:
            raise RuntimeError(
                f"Exact duplicate inside {split}: {hashes[digest]} and {image_path.name}"
            )
        hashes[digest] = image_path.name

        if i % 250 == 0 or i == len(images):
            print(f"{split}: {i}/{len(images)}")

    if sum(counts) != exp["instances"]:
        raise RuntimeError(
            f"{split}: expected {exp['instances']} instances, found {sum(counts)}"
        )

    summary = {
        "split": split,
        "images": len(images),
        "instances": int(sum(counts)),
        "mean_instances_per_image": float(np.mean(counts)),
        "median_normalized_bbox_area": float(np.median(all_areas)),
        "median_bbox_aspect_ratio": float(np.median(all_ratios)),
        "median_brightness": float(np.median(brightness)),
        "median_contrast": float(np.median(contrast)),
        "median_sharpness": float(np.median(sharpness)),
    }

    return summary, per_image, hashes


def check_augmented(root: Path):
    img_dir = root / "augmented" / "train" / "images"
    lbl_dir = root / "augmented" / "train" / "labels"

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})
    labels = sorted(p for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt")

    if len(images) != EXPECTED_AUG_IMAGES or len(labels) != EXPECTED_AUG_IMAGES:
        raise RuntimeError(
            f"augmented/train expected {EXPECTED_AUG_IMAGES} images/labels; "
            f"found {len(images)}/{len(labels)}"
        )

    if {p.stem for p in images} != {p.stem for p in labels}:
        raise RuntimeError("augmented/train image-label correspondence mismatch")

    instances = 0
    for p in labels:
        n, _, _ = parse_label(p)
        instances += n

    if instances != EXPECTED_AUG_INSTANCES:
        raise RuntimeError(
            f"augmented/train expected {EXPECTED_AUG_INSTANCES} instances; "
            f"found {instances}"
        )

    return len(images), instances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    root = args.dataset_root
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    per_image = []
    hashes_by_split = {}

    for split in ("train", "val", "test"):
        summary, rows, hashes = split_stats(root, split)
        summaries.append(summary)
        per_image.extend(rows)
        hashes_by_split[split] = hashes

    split_names = ["train", "val", "test"]
    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            common = set(hashes_by_split[a]) & set(hashes_by_split[b])
            if common:
                digest = next(iter(common))
                raise RuntimeError(
                    f"Exact cross-split duplicate: "
                    f"{a}/{hashes_by_split[a][digest]} and "
                    f"{b}/{hashes_by_split[b][digest]}"
                )

    aug_images, aug_instances = check_augmented(root)

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.DataFrame(per_image)

    summary_df.to_csv(args.output_dir / "table4_statistics_summary.csv", index=False)
    detail_df.to_csv(
        args.output_dir / "image_appearance_metrics_per_image.csv", index=False
    )

    print("\nTABLE 4 STATISTICS")
    print(summary_df.to_string(index=False))
    print("\nINTEGRITY CHECK: PASS")
    print(f"Original images     : {sum(EXPECTED[s]['images'] for s in EXPECTED)}")
    print(f"Original instances  : {sum(EXPECTED[s]['instances'] for s in EXPECTED)}")
    print("Exact cross-split duplicates: 0")
    print(f"Augmented images    : {aug_images}")
    print(f"Augmented instances : {aug_instances}")
    print(f"Long side > {MAX_LONG_SIDE}: 0")


if __name__ == "__main__":
    main()
