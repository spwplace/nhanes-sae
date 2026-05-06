#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tweet_composites"
OUT.mkdir(parents=True, exist_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def trim_background(img, bg=(246, 243, 236), tol=18, pad=14):
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    diff = np.abs(arr - np.asarray(bg, dtype=np.int16)).max(axis=2)
    mask = diff > tol
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    left = max(0, xs.min() - pad)
    right = min(img.width, xs.max() + pad)
    top = max(0, ys.min() - pad)
    bottom = min(img.height, ys.max() + pad)
    return img.crop((left, top, right, bottom))


def load_img(path, trim=True):
    img = Image.open(path).convert("RGB")
    if trim:
        img = trim_background(img)
    return np.asarray(img)


def main_result():
    multiview = read_json(ROOT / "outputs/nhanes_multiview/summary.json")
    clean = read_json(ROOT / "outputs/nhanes_mining_clean/summary.json")
    quotient_lab = read_json(ROOT / "outputs/nhanes_quotient_lab_exam/summary.json")
    quotient_q = read_json(ROOT / "outputs/nhanes_quotient_questionnaire/summary.json")
    independent = read_json(ROOT / "outputs/nhanes_independent_blocks/summary.json")

    fig = plt.figure(figsize=(16, 10), facecolor="#f6f3ec")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], width_ratios=[1.15, 1.05, 0.9])

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(load_img(ROOT / "outputs/nhanes_multiview/umap_clusters.png"))
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1:])
    ax1.imshow(load_img(ROOT / "outputs/nhanes_multiview/cluster_archetypes.png"))
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, 1])
    labels = ["combined", "lab quotient", "questionnaire quotient", "multiview"]
    vals = [
        clean["cluster_silhouette"],
        quotient_lab["cluster_silhouette"],
        quotient_q["cluster_silhouette"],
        multiview["cluster_silhouette"],
    ]
    colors = ["#9aa0a6", "#4c78a8", "#f58518", "#54a24b"]
    ax2.barh(labels, vals, color=colors)
    ax2.set_xlim(0, max(vals) * 1.2)
    ax2.set_title("Cluster separation improves after block balancing", loc="left", fontsize=13, fontweight="bold")
    ax2.set_xlabel("silhouette score")
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    for i, v in enumerate(vals):
        ax2.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=10)

    ax3 = fig.add_subplot(gs[1, 2])
    blocks = sorted(independent["blocks"], key=lambda b: b["cluster_silhouette"], reverse=True)[:7]
    names = [b["name"].replace(" history", "").replace("Utilization/meds", "Util/meds") for b in blocks]
    scores = [b["cluster_silhouette"] for b in blocks]
    ax3.barh(names[::-1], scores[::-1], color="#b279a2")
    ax3.set_xlim(0, 1)
    ax3.set_title("Sharpest independent blocks", loc="left", fontsize=13, fontweight="bold")
    ax3.set_xlabel("silhouette")
    ax3.spines[["top", "right", "left"]].set_visible(False)
    ax3.tick_params(axis="y", length=0)
    for i, v in enumerate(scores[::-1]):
        ax3.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=9)

    fig.suptitle(
        "NHANES phenome SAE: quotienting out modality reveals sharper structure",
        x=0.03,
        y=0.99,
        ha="left",
        fontsize=24,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.025,
        "Public NHANES adults (n=23,969). Blocks: body size, BP, lipids/glucose, renal/liver, CBC, diabetes, cardio, smoking, mood/sleep, meds/utilization, activity/alcohol.",
        fontsize=10,
        color="#333333",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = OUT / "main_result_composite.png"
    plt.savefig(path, dpi=220)
    plt.close(fig)
    return path


def block_grid():
    independent = read_json(ROOT / "outputs/nhanes_independent_blocks/summary.json")
    blocks = sorted(independent["blocks"], key=lambda b: b["cluster_silhouette"], reverse=True)
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), facecolor="#f6f3ec")
    for ax, block in zip(axes.ravel(), blocks):
        img_path = ROOT / block["plots"]["umap_clusters"]
        ax.imshow(load_img(img_path))
        ax.axis("off")
        ax.set_title(
            f"{block['name']}  |  k={block['cluster_k']}  sil={block['cluster_silhouette']:.2f}",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
    for ax in axes.ravel()[len(blocks):]:
        ax.axis("off")
    fig.suptitle(
        "Independent phenotype-family maps: where structure is actually crisp",
        x=0.03,
        y=0.99,
        ha="left",
        fontsize=24,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.02,
        "Each panel is a separate SAE/PCA/UMAP run on one curated variable block. High-silhouette blocks are mostly discrete questionnaire/condition/medication spaces; lab/body blocks are more continuous.",
        fontsize=10,
        color="#333333",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = OUT / "independent_blocks_grid.png"
    plt.savefig(path, dpi=220)
    plt.close(fig)
    return path


if __name__ == "__main__":
    paths = [main_result(), block_grid()]
    for path in paths:
        print(path)
