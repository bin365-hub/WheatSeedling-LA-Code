#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

TRAINING = {
    "imgsz": 640,
    "batch": 16,
    "epochs": 300,
    "patience": 50,
    "optimizer": "AdamW",
    "lr0": 0.002,
    "momentum": 0.9,  # AdamW beta1 in the Ultralytics optimizer interface
    "weight_decay": 0.0005,
    "seed": 42,
    "deterministic": True,
    "amp": True,
    "workers": 2,
}


def r2_score(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def dataset_counts(root: Path, split: str):
    img_dir = root / "original" / split / "images"
    lbl_dir = root / "original" / split / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})
    gt = []
    for image in images:
        text = (lbl_dir / f"{image.stem}.txt").read_text(
            encoding="utf-8-sig"
        ).strip()
        gt.append(0 if not text else len(text.splitlines()))
    return images, np.asarray(gt, dtype=float)


def command_train(args):
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        imgsz=TRAINING["imgsz"],
        batch=TRAINING["batch"],
        epochs=TRAINING["epochs"],
        patience=TRAINING["patience"],
        optimizer=TRAINING["optimizer"],
        lr0=TRAINING["lr0"],
        momentum=TRAINING["momentum"],
        weight_decay=TRAINING["weight_decay"],
        seed=TRAINING["seed"],
        deterministic=TRAINING["deterministic"],
        amp=TRAINING["amp"],
        device=args.device,
        workers=TRAINING["workers"],
        project=args.project,
        name=args.name,
    )


def command_detection(args):
    model = YOLO(args.weights)
    metrics = model.val(
        data=str(args.data),
        split="test",
        imgsz=640,
        batch=16,
        device=args.device,
        plots=False,
        save_json=False,
    )
    result = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def prediction_confidences(model, images, device, iou):
    results = model.predict(
        source=[str(p) for p in images],
        imgsz=640,
        conf=0.001,
        iou=iou,
        device=device,
        verbose=False,
        stream=False,
    )
    out = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            out.append(np.asarray([], dtype=float))
        else:
            out.append(
                r.boxes.conf.detach().cpu().numpy().astype(float)
            )
    return out


def command_threshold(args):
    images, gt = dataset_counts(args.dataset_root, args.split)
    model = YOLO(args.weights)
    confs = prediction_confidences(model, images, args.device, args.iou)

    rows = []
    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        pred = np.asarray(
            [int(np.sum(c >= threshold)) for c in confs],
            dtype=float,
        )
        err = pred - gt
        rows.append({
            "threshold": float(threshold),
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "R2": r2_score(gt, pred),
            "mean_signed_error": float(np.mean(err)),
            "within_2_percent": float(np.mean(np.abs(err) <= 2) * 100),
            "predicted_total": int(np.sum(pred)),
            "ground_truth_total": int(np.sum(gt)),
        })

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    best = df.sort_values(
        ["MAE", "threshold"],
        ascending=[True, True],
    ).iloc[0]
    print("Best threshold")
    print(best.to_string())


def command_counting(args):
    images, gt = dataset_counts(args.dataset_root, args.split)
    model = YOLO(args.weights)

    results = model.predict(
        source=[str(p) for p in images],
        imgsz=640,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
        stream=False,
    )

    pred = np.asarray(
        [0 if r.boxes is None else len(r.boxes) for r in results],
        dtype=float,
    )
    err = pred - gt

    df = pd.DataFrame({
        "image_name": [p.name for p in images],
        "ground_truth": gt.astype(int),
        "prediction": pred.astype(int),
        "error": err.astype(int),
        "abs_error": np.abs(err).astype(int),
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    metrics = {
        "confidence_threshold": float(args.conf),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": r2_score(gt, pred),
        "mean_signed_error": float(np.mean(err)),
        "within_2_percent": float(np.mean(np.abs(err) <= 2) * 100),
        "predicted_total": int(np.sum(pred)),
        "ground_truth_total": int(np.sum(gt)),
        "n_images": int(len(images)),
    }

    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(
        description="WheatSeedling-LA baseline training and evaluation"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train")
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--project", default="runs/baselines")
    p.add_argument("--device", default="0")
    p.set_defaults(func=command_train)

    p = sub.add_parser("detection")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="0")
    p.set_defaults(func=command_detection)

    p = sub.add_parser("threshold")
    p.add_argument("--weights", required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--iou", type=float, default=0.7)
    p.set_defaults(func=command_threshold)

    p = sub.add_parser("counting")
    p.add_argument("--weights", required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--conf", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--iou", type=float, default=0.7)
    p.set_defaults(func=command_counting)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
