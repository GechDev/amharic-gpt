#!/usr/bin/env python3
"""
Publication-Quality Training Curve Visualization for Amharic GPT.

Generates training loss, validation loss, learning rate schedule,
perplexity, and bits-per-character (BPC) curves from logged metrics.csv.
"""

import os
import sys
import csv
import argparse
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def load_metrics(metrics_csv_path: str) -> Dict[str, List[float]]:
    """Loads metrics columns from CSV into typed lists."""
    if not os.path.exists(metrics_csv_path):
        raise FileNotFoundError(f"Metrics CSV file not found: {metrics_csv_path}")

    data: Dict[str, List[float]] = {
        "step": [],
        "train_loss": [],
        "val_step": [],
        "val_loss": [],
        "train_val_gap": [],
        "val_ppl": [],
        "val_bpc": [],
        "lr_step": [],
        "lr": [],
        "grad_norm_step": [],
        "grad_norm": [],
    }

    with open(metrics_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s = int(row["step"])
            except (ValueError, KeyError):
                continue

            if row.get("train_loss"):
                data["step"].append(s)
                data["train_loss"].append(float(row["train_loss"]))

            if row.get("val_loss"):
                data["val_step"].append(s)
                data["val_loss"].append(float(row["val_loss"]))
                if row.get("train_val_gap"):
                    data["train_val_gap"].append(float(row["train_val_gap"]))
                if row.get("val_ppl"):
                    data["val_ppl"].append(float(row["val_ppl"]))
                if row.get("val_bpc"):
                    data["val_bpc"].append(float(row["val_bpc"]))

            if row.get("lr"):
                data["lr_step"].append(s)
                data["lr"].append(float(row["lr"]))

            if row.get("grad_norm"):
                data["grad_norm_step"].append(s)
                data["grad_norm"].append(float(row["grad_norm"]))

    return data


def plot_training_curves(
    metrics_path: str,
    output_png_path: str,
    title: str = "Amharic GPT Training Dynamics"
) -> str:
    """Plots multi-panel diagnostic curves from metrics CSV."""
    data = load_metrics(metrics_path)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    # 1. Top Panel: Training and Validation Loss + Gap
    ax1 = axes[0]
    ax1.plot(data["step"], data["train_loss"], label="Train Loss (nats)", color="#1f77b4", alpha=0.7, lw=1.5)
    if data["val_loss"]:
        ax1.plot(data["val_step"], data["val_loss"], "o-", label="Val Loss (nats)", color="#d62728", lw=2, markersize=5)
    ax1.set_ylabel("Cross-Entropy Loss (nats)", fontsize=11, fontweight="bold")
    ax1.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Middle Panel: Learning Rate Schedule & Gradient Norm
    ax2 = axes[1]
    color_lr = "#2ca02c"
    ax2.plot(data["lr_step"], data["lr"], label="Learning Rate", color=color_lr, lw=2)
    ax2.set_ylabel("Learning Rate", color=color_lr, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_lr)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Twin axis for Gradient Norm
    if data["grad_norm"]:
        ax2_norm = ax2.twinx()
        color_norm = "#7f7f7f"
        ax2_norm.plot(data["grad_norm_step"], data["grad_norm"], color=color_norm, alpha=0.4, lw=1, label="Grad Norm (pre-clip)")
        ax2_norm.set_ylabel("Pre-Clip Grad Norm", color=color_norm, fontsize=11)
        ax2_norm.tick_params(axis="y", labelcolor=color_norm)

    # 3. Bottom Panel: Perplexity and Bits-Per-Character (BPC)
    ax3 = axes[2]
    if data["val_ppl"]:
        color_ppl = "#9467bd"
        ax3.plot(data["val_step"], data["val_ppl"], "s-", color=color_ppl, lw=2, markersize=5, label="Val Perplexity")
        ax3.set_ylabel("Validation Perplexity", color=color_ppl, fontsize=11, fontweight="bold")
        ax3.tick_params(axis="y", labelcolor=color_ppl)
        ax3.grid(True, linestyle="--", alpha=0.6)

        if data["val_bpc"]:
            ax3_bpc = ax3.twinx()
            color_bpc = "#ff7f0e"
            ax3_bpc.plot(data["val_step"], data["val_bpc"], "^--", color=color_bpc, lw=2, markersize=5, label="Val BPC")
            ax3_bpc.set_ylabel("Bits Per Character (BPC)", color=color_bpc, fontsize=11, fontweight="bold")
            ax3_bpc.tick_params(axis="y", labelcolor=color_bpc)
    else:
        ax3.text(0.5, 0.5, "No Validation Data Logged", ha="center", va="center", transform=ax3.transAxes)

    ax3.set_xlabel("Optimization Step", fontsize=11, fontweight="bold")

    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)
    plt.savefig(output_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plotting] Saved training curves to -> {output_png_path}")
    return output_png_path


def main():
    parser = argparse.ArgumentParser(description="Plot Amharic GPT Training Curves")
    parser.add_argument("--metrics_file", type=str, default=None, help="Path to metrics.csv")
    parser.add_argument("--run_dir", type=str, default="outputs/small_run", help="Path to run output directory containing metrics.csv")
    parser.add_argument("--output", type=str, default=None, help="Destination PNG file path")
    parser.add_argument("--title", type=str, default="Amharic GPT Training Metrics", help="Title for the plot")
    args = parser.parse_args()

    metrics_csv = args.metrics_file or os.path.join(args.run_dir, "metrics.csv")
    output_png = args.output or os.path.join(args.run_dir, "training_curves.png")

    plot_training_curves(metrics_csv, output_png, title=args.title)


if __name__ == "__main__":
    main()
