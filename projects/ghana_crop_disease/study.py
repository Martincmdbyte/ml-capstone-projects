import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import get_annotations, split_images
from detector import build_model, train_one_epoch
from evaluate import SCORE_CUTOFF, evaluate


def plot_recall(results, output):
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    x = np.arange(2)
    peak = max(frame[['localization_recall', 'class_correct_recall']].max().max()
               for frame in results.values())
    chart_top = min(1, max(0.1, 1.25 * peak))
    for i, (name, frame) in enumerate(results.items()):
        for axis, metric, title in zip(
            axes, ["localization_recall", "class_correct_recall"],
            ["Any class at IoU � 0.50", "Correct class at IoU � 0.50"]
        ):
            bars = axis.bar(x + (i - 0.5) * 0.34, frame[metric], width=0.34, label=name)
            labels = [f'{value:.1%}' if np.isfinite(value) else '' for value in frame[metric]]
            axis.bar_label(bars, labels=labels, padding=2, fontsize=8)
            axis.set_xticks(x, ["Small disease", "Other disease"])
            axis.set_ylim(0, chart_top)
            axis.set_title(title)
            axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
    axes[0].set_ylabel("Share of annotated boxes found")
    axes[1].legend(loc="upper right")
    figure.suptitle(f"Held-out crop photos � score threshold {SCORE_CUTOFF:g}")
    figure.tight_layout()
    figure.savefig(output / "recall_by_box_size.png", dpi=145)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a crop disease detector.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sample", type=int, default=0, help="Use fewer photos for a test run.")
    parser.add_argument("--resume", action="store_true", help="Continue from the saved epoch.")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error("Epochs, batch size, and threads must be positive.")
    if args.sample and args.sample < 20:
        parser.error("A sample run needs at least 20 photos.")

    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)

    records, classes = get_annotations(args.data, args.output / "image_cache", args.sample)
    train, test = split_images(records)
    print(f"{len(train):,} train photos, {len(test):,} test photos, {len(classes)} classes", flush=True)

    pd.DataFrame([
        {"split": "train", "images": len(train), "boxes": sum(len(row[1]) for row in train)},
        {"split": "test", "images": len(test), "boxes": sum(len(row[1]) for row in test)},
    ]).to_csv(args.output / "split.csv", index=False)

    model = build_model(len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    checkpoint = args.output / "detector.pt"
    loss_log = []
    if args.resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["classes"] != classes or saved["train_files"] != [row[0].name for row in train]:
            raise ValueError("Use the same data and sample size when resuming.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        loss_log = saved["loss"]
        if args.epochs < len(loss_log):
            parser.error("Epochs must be at least the number already trained.")

    for epoch in range(len(loss_log), args.epochs):
        loss = train_one_epoch(model, optimizer, train, args.batch_size, seed=42 + epoch)
        loss_log.append(loss)
        print(f"Epoch {epoch + 1:2d}/{args.epochs}: training loss {loss:.3f}", flush=True)
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "classes": classes, "loss": loss_log, "train_files": [row[0].name for row in train],
        }, temporary)
        temporary.replace(checkpoint)
    pd.DataFrame({"epoch": range(1, len(loss_log) + 1), "loss": loss_log}).to_csv(
        args.output / "training_loss.csv", index=False
    )

    model.eval()
    size_results = {}
    summary = []
    for tiled, slug, name in [(False, "whole", "Whole photo"), (True, "tiled", "Overlapping tiles")]:
        print(f"Evaluating {name.lower()} on held-out photos", flush=True)
        scores, sizes, confusion, image_hits = evaluate(model, test, classes, tiled)
        scores.to_csv(args.output / f"{slug}_ap50.csv", index=False)
        sizes.to_csv(args.output / f"{slug}_box_sizes.csv", index=False)
        confusion.to_csv(args.output / f"{slug}_confusion.csv", index_label="actual")
        pd.Series(image_hits, name="correct_small_boxes").to_csv(
            args.output / f"{slug}_image_hits.csv", index_label="image"
        )
        size_results[name] = sizes
        mean_ap = scores["AP50"].mean()
        summary.append((name, mean_ap))
        print(f"{name}: mAP50={mean_ap:.3f}", flush=True)
    pd.DataFrame(summary, columns=["view", "mAP50"]).to_csv(args.output / "summary.csv", index=False)
    plot_recall(size_results, args.output)


if __name__ == "__main__":
    main()
