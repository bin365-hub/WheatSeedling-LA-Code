#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def read_boxes(path: Path):
    boxes = []
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return np.empty((0, 4), dtype=float)

    for line in text.splitlines():
        cls, x, y, w, h = map(float, line.split())
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2
        boxes.append((x1, y1, x2, y2))

    return np.asarray(boxes, dtype=float)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)

    out = np.zeros((len(a), len(b)), dtype=float)

    for i, aa in enumerate(a):
        ax1, ay1, ax2, ay2 = aa
        aarea = max(0, ax2 - ax1) * max(0, ay2 - ay1)

        for j, bb in enumerate(b):
            bx1, by1, bx2, by2 = bb
            barea = max(0, bx2 - bx1) * max(0, by2 - by1)

            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = aarea + barea - inter

            out[i, j] = 0.0 if union <= 0 else inter / union

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--comparison", type=Path, required=True)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    args = ap.parse_args()

    names = sorted(
        {p.name for p in args.reference.glob("*.txt")}
        & {p.name for p in args.comparison.glob("*.txt")}
    )

    if not names:
        raise RuntimeError("No matching TXT filenames found.")

    ref_total = 0
    comp_total = 0
    matched_total = 0
    matched_ious = []
    exact_count_images = 0

    for name in names:
        ref = read_boxes(args.reference / name)
        comp = read_boxes(args.comparison / name)

        ref_total += len(ref)
        comp_total += len(comp)
        exact_count_images += int(len(ref) == len(comp))

        M = iou_matrix(ref, comp)
        if M.size == 0:
            continue

        rr, cc = linear_sum_assignment(1.0 - M)
        for r, c in zip(rr, cc):
            if M[r, c] >= args.iou_threshold:
                matched_total += 1
                matched_ious.append(float(M[r, c]))

    precision = matched_total / comp_total if comp_total else 0.0
    recall = matched_total / ref_total if ref_total else 0.0
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )

    print(f"Images               : {len(names)}")
    print(f"Reference instances  : {ref_total}")
    print(f"Comparison instances : {comp_total}")
    print(f"Matched pairs        : {matched_total}")
    print(f"Precision            : {precision:.6f}")
    print(f"Recall               : {recall:.6f}")
    print(f"F1                   : {f1:.6f}")
    print(
        f"Mean matched IoU     : {np.mean(matched_ious):.6f}"
        if matched_ious
        else "Mean matched IoU     : NA"
    )
    print(
        f"Median matched IoU   : {np.median(matched_ious):.6f}"
        if matched_ious
        else "Median matched IoU   : NA"
    )
    print(f"Exact-count images   : {exact_count_images}/{len(names)}")


if __name__ == "__main__":
    main()
