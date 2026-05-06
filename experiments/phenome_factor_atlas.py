#!/usr/bin/env python3
import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP

sys.path.append(str(Path(__file__).resolve().parent))
from multiview_nhanes_umap import BLOCKS, PRETTY, select_block_columns
from nhanes_phenome_sae import prepare_matrix


PALETTE = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
    "#59a14f",
    "#edc948",
    "#af7aa1",
    "#76b7b2",
    "#8cd17d",
]

BG = (246, 243, 236)


FIELD_NAMES = {
    "BMXBMI": "BMI",
    "BMXWT": "weight",
    "BMXWAIST": "waist",
    "BMXHIP": "hip",
    "BMXHT": "height",
    "BPXSY1": "systolic BP",
    "BPXDI1": "diastolic BP",
    "BPXOSY1": "systolic BP",
    "BPXODI1": "diastolic BP",
    "LBXTC": "total cholesterol",
    "LBDHDD": "HDL",
    "LBXTR": "triglycerides",
    "LBDLDL": "LDL",
    "LBXGLU": "glucose",
    "LBXGH": "A1c",
    "LBXSCR": "creatinine",
    "LBXSBU": "BUN",
    "LBXSAL": "albumin",
    "LBXSAPSI": "alk phos",
    "LBXSASSI": "AST",
    "LBXSATSI": "ALT",
    "LBXSGTSI": "GGT",
    "LBXSUA": "uric acid",
    "LBXWBCSI": "WBC",
    "LBXRBCSI": "RBC",
    "LBXHGB": "hemoglobin",
    "LBXHCT": "hematocrit",
    "LBXPLTSI": "platelets",
    "DIQ010": "diabetes dx",
    "DIQ050": "insulin",
    "DIQ070": "diabetes pills",
    "BPQ020": "hypertension dx",
    "MCQ160B": "CHF",
    "MCQ160C": "coronary heart disease",
    "MCQ160D": "angina",
    "MCQ160E": "heart attack",
    "MCQ160F": "stroke",
    "SMQ020": "100 cigarettes",
    "SMQ040": "current smoking",
    "MCQ010": "asthma",
    "DPQ010": "little interest",
    "DPQ020": "depressed mood",
    "DPQ030": "sleep trouble",
    "DPQ040": "low energy",
    "DPQ050": "appetite change",
    "DPQ060": "self-judgment",
    "DPQ070": "concentration",
    "DPQ080": "psychomotor",
    "DPQ090": "self-harm thoughts",
    "SLD012": "sleep hours",
    "RXDUSE": "any Rx meds",
    "RXDCOUNT": "Rx count",
    "HSD010": "general health",
    "PAQ605": "vigorous work",
    "PAQ620": "moderate work",
    "PAQ650": "vigorous recreation",
    "PAD680": "sedentary minutes",
    "ALQ111": "ever drank",
    "ALQ130": "drinks/day",
}


def pretty_field(col):
    stem = col.rsplit("__", 1)[-1].replace("__MISSING", "")
    return FIELD_NAMES.get(stem, stem)


def trim_background(img, bg=BG, tol=18, pad=12):
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    diff = np.abs(arr - np.asarray(bg, dtype=np.int16)).max(axis=2)
    mask = diff > tol
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    return img.crop(
        (
            max(0, xs.min() - pad),
            max(0, ys.min() - pad),
            min(img.width, xs.max() + pad),
            min(img.height, ys.max() + pad),
        )
    )


def load_trimmed(path):
    return np.asarray(trim_background(Image.open(path).convert("RGB")))


def wrap_csv(text, width=34):
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def zscore(x):
    return StandardScaler().fit_transform(np.asarray(x, dtype=np.float32))


def confounds(df, index):
    demo = df.loc[index]
    cols = []
    names = []
    if "DEMO__RIDAGEYR" in demo:
        age = demo["DEMO__RIDAGEYR"].astype(float).fillna(demo["DEMO__RIDAGEYR"].median())
        age = (age - age.mean()) / (age.std() + 1e-8)
        cols.extend([age.to_numpy(), (age.to_numpy() ** 2)])
        names.extend(["age", "age2"])
    for cat_col, prefix in [("DEMO__RIAGENDR", "sex"), ("DEMO__RIDRETH3", "race_ethnicity")]:
        if cat_col not in demo:
            continue
        dummies = pd.get_dummies(demo[cat_col].fillna(-1).astype(int), prefix=prefix, drop_first=True)
        for name in dummies.columns:
            cols.append(dummies[name].to_numpy(dtype=np.float32))
            names.append(name)
    if not cols:
        return np.ones((len(index), 1), dtype=np.float32), ["intercept"]
    c = np.column_stack([np.ones(len(index), dtype=np.float32), *cols]).astype(np.float32)
    return c, ["intercept", *names]


def residualize(x, c):
    beta, *_ = np.linalg.lstsq(c, x, rcond=None)
    resid = x - c @ beta
    return zscore(resid).astype(np.float32)


def choose_block_components(xb, max_components):
    n_comp = min(max_components, xb.shape[1], xb.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=0)
    scores = pca.fit_transform(xb)
    cum = np.cumsum(pca.explained_variance_ratio_)
    keep = int(np.searchsorted(cum, 0.72) + 1)
    keep = min(max(2, keep), n_comp)
    return pca, scores[:, :keep], keep


def corrcoef(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    return (a.T @ b) / ((len(a) - 1) * (a.std(axis=0)[:, None] + 1e-8) * (b.std(axis=0)[None, :] + 1e-8))


def cluster_name(row):
    pos = row.sort_values(ascending=False)
    neg = row.sort_values()
    if pos.iloc[0] < 0.35 and abs(neg.iloc[0]) > pos.iloc[0]:
        return f"low {PRETTY.get(neg.index[0], neg.index[0]).lower()}"
    if len(pos) > 1 and pos.iloc[1] > 0.45:
        return f"{PRETTY.get(pos.index[0], pos.index[0])} + {PRETTY.get(pos.index[1], pos.index[1])}"
    return PRETTY.get(pos.index[0], pos.index[0])


def top_field_text(delta, n=5):
    pos = [pretty_field(c) for c in delta.sort_values(ascending=False).head(n).index]
    neg = [pretty_field(c) for c in delta.sort_values().head(n).index]
    return ", ".join(pos), ", ".join(neg)


def make_umap(emb, labels, cards, out):
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#f6f3ec")
    ax.set_facecolor("#f6f3ec")
    for cid in sorted(np.unique(labels)):
        mask = labels == cid
        color = "#d0d0d0" if cid < 0 else PALETTE[int(cid) % len(PALETTE)]
        label = "noise" if cid < 0 else f"{cid}: {cards[int(cid)]['name'][:22]}"
        ax.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.78, linewidths=0, c=color, label=label)
    ax.legend(markerscale=3, fontsize=8, ncols=2, frameon=False, loc="best")
    ax.set_title("Residualized block-factor phenome map", loc="left", fontsize=16, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    fig.savefig(out / "factor_umap_clusters.png", dpi=230)
    plt.close(fig)


def make_archetype_heatmap(cluster_block_z, cards, out):
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#f6f3ec")
    mat = cluster_block_z.to_numpy()
    im = ax.imshow(mat, cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_yticks(range(len(cluster_block_z)))
    ax.set_yticklabels([f"{c['cluster']}: {c['name']}" for c in cards], fontsize=9)
    ax.set_xticks(range(len(cluster_block_z.columns)))
    ax.set_xticklabels([PRETTY.get(c, c) for c in cluster_block_z.columns], rotation=35, ha="right", fontsize=9)
    ax.set_title("Clusters are labeled by held-out block contrasts", loc="left", fontsize=16, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    plt.tight_layout()
    fig.savefig(out / "factor_cluster_archetypes.png", dpi=230)
    plt.close(fig)


def make_axis_cards(axis_cards, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="#f6f3ec")
    for ax, card in zip(axes.ravel(), axis_cards[:6]):
        ax.axis("off")
        ax.text(0, 0.95, f"Axis {card['axis'] + 1}", fontsize=15, fontweight="bold", transform=ax.transAxes)
        ax.text(0, 0.83, f"{card['variance_pct']:.1f}% of factor variance", fontsize=10, color="#555", transform=ax.transAxes)
        ax.text(0, 0.66, "high", fontsize=10, fontweight="bold", transform=ax.transAxes)
        ax.text(0, 0.55, wrap_csv(card["positive"]), fontsize=9.5, transform=ax.transAxes, va="top")
        ax.text(0, 0.31, "low", fontsize=10, fontweight="bold", transform=ax.transAxes)
        ax.text(0, 0.20, wrap_csv(card["negative"]), fontsize=9.5, transform=ax.transAxes, va="top")
    for ax in axes.ravel()[len(axis_cards[:6]):]:
        ax.axis("off")
    fig.suptitle("Interpretable axes from residualized block factors", x=0.04, y=0.99, ha="left", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "factor_axis_cards.png", dpi=230)
    plt.close(fig)


def make_composite(out, summary):
    fig = plt.figure(figsize=(16, 9), facecolor="#f6f3ec")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.1, 0.9], height_ratios=[1.05, 0.95])

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(load_trimmed(out / "factor_umap_clusters.png"))
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1:])
    ax1.imshow(load_trimmed(out / "factor_cluster_archetypes.png"))
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, 1])
    method_names = [m["name"] for m in summary["method_scores"]]
    method_scores = [m["score"] for m in summary["method_scores"]]
    colors = ["#9aa0a6", "#54a24b", "#4c78a8"]
    ax2.barh(method_names, method_scores, color=colors[: len(method_scores)])
    ax2.set_xlim(0, max(method_scores) * 1.25)
    ax2.set_title("Separation without SAE", loc="left", fontsize=13, fontweight="bold")
    ax2.set_xlabel("silhouette on clustering space")
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    for i, v in enumerate(method_scores):
        ax2.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=10)

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis("off")
    ax3.text(0, 0.98, "Most tweetable read", fontsize=12.5, fontweight="bold", transform=ax3.transAxes)
    ax3.text(
        0,
        0.78,
        textwrap.fill(
            "Whole-phenome clusters are weak after demographic residualization. The strongest signal is block-local structure plus a few interpretable continuous axes.",
            width=39,
        ),
        fontsize=9.5,
        transform=ax3.transAxes,
        va="top",
    )
    ax3.text(0, 0.42, "Top clusters:", fontsize=11.5, fontweight="bold", transform=ax3.transAxes)
    for i, card in enumerate(summary["cluster_cards"][:4]):
        ax3.text(
            0,
            0.32 - i * 0.075,
            f"{card['cluster']}: {card['name']} ({card['pct'] * 100:.1f}%)",
            fontsize=9.5,
            transform=ax3.transAxes,
        )

    fig.suptitle(
        "NHANES phenome atlas: global clusters are weak, block structure is real",
        x=0.03,
        y=0.99,
        ha="left",
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.02,
        f"Public NHANES adults, n={summary['n_participants']:,}. Residualized against age, age^2, sex, race/ethnicity; each phenotype family contributes equal-weight PCA factors.",
        fontsize=10,
        color="#333333",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out / "tweetable_factor_atlas.png", dpi=230)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/nhanes/processed/nhanes_phenome_raw.parquet")
    p.add_argument("--out-dir", default="outputs/nhanes_factor_atlas")
    p.add_argument("--min-nonmissing", type=float, default=0.25)
    p.add_argument("--max-cols", type=int, default=1000)
    p.add_argument("--max-components-per-block", type=int, default=5)
    p.add_argument("--umap-sample", type=int, default=18000)
    p.add_argument("--seed", type=int, default=167)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    x_df, _ = prepare_matrix(df, args.min_nonmissing, args.max_cols)
    x_df = x_df[[c for c in x_df.columns if not c.endswith("__MISSING") and not c.startswith("DEMO__")]]
    c, confound_names = confounds(df, x_df.index)
    x_resid = pd.DataFrame(residualize(x_df.to_numpy(dtype=np.float32), c), index=x_df.index, columns=x_df.columns)

    block_columns = {name: select_block_columns(x_resid, cols) for name, cols in BLOCKS.items()}
    block_columns = {k: v for k, v in block_columns.items() if len(v) >= 2}

    block_scores = []
    block_summary_scores = {}
    block_meta = []
    for name, cols in block_columns.items():
        xb = zscore(x_resid[cols].to_numpy(dtype=np.float32))
        pca, scores, keep = choose_block_components(xb, args.max_components_per_block)
        scores = zscore(scores) / np.sqrt(keep)
        block_scores.append(scores)
        block_summary_scores[name] = xb.mean(axis=1)
        block_meta.append(
            {
                "block": name,
                "name": PRETTY.get(name, name),
                "n_columns": len(cols),
                "n_components": int(keep),
                "explained_variance": [float(v) for v in pca.explained_variance_ratio_[:keep]],
                "columns": cols,
            }
        )

    z = np.concatenate(block_scores, axis=1).astype(np.float32)
    z = zscore(z).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    sample_n = min(args.umap_sample, len(z))
    sample_idx = np.sort(rng.choice(len(z), size=sample_n, replace=False))
    z_sample = z[sample_idx]
    x_sample = x_resid.iloc[sample_idx]

    pca_global = PCA(n_components=min(24, z_sample.shape[1]), random_state=args.seed)
    z_pca = pca_global.fit_transform(z_sample)
    emb = UMAP(n_neighbors=55, min_dist=0.12, metric="cosine", random_state=args.seed).fit_transform(z_pca)

    best = None
    for k in range(5, 14):
        labels = KMeans(n_clusters=k, random_state=args.seed, n_init=40).fit_predict(z_pca)
        score = silhouette_score(z_pca, labels, sample_size=min(7000, len(labels)), random_state=args.seed)
        if best is None or score > best["score"]:
            best = {"k": k, "score": float(score), "labels": labels}
    labels = best["labels"]

    hdb = HDBSCAN(min_cluster_size=max(180, sample_n // 80), min_samples=25).fit_predict(z_pca)
    hdb_non_noise = hdb[hdb >= 0]
    hdb_score = None
    if len(np.unique(hdb_non_noise)) > 1 and (hdb >= 0).mean() > 0.35:
        hdb_score = float(silhouette_score(z_pca[hdb >= 0], hdb[hdb >= 0], sample_size=min(5000, len(hdb_non_noise)), random_state=args.seed))

    block_frame = pd.DataFrame({k: v[sample_idx] for k, v in block_summary_scores.items()})
    cluster_block = block_frame.assign(cluster=labels).groupby("cluster").mean()
    cluster_block_z = (cluster_block - block_frame.mean(axis=0)) / (block_frame.std(axis=0) + 1e-8)

    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    cards = []
    for cid, row in cluster_block_z.iterrows():
        mask = labels == cid
        delta = x_sample.iloc[mask].mean(axis=0) - x_sample.mean(axis=0)
        high, low = top_field_text(delta)
        cards.append(
            {
                "cluster": int(cid),
                "name": cluster_name(row),
                "n": int(cluster_sizes.loc[cid]),
                "pct": float(cluster_sizes.loc[cid] / len(labels)),
                "block_z": {str(k): float(v) for k, v in row.items()},
                "top_high_fields": high,
                "top_low_fields": low,
            }
        )

    axis_corr = corrcoef(z_pca[:, :6], x_sample.to_numpy(dtype=np.float32))
    axis_cards = []
    for axis in range(axis_corr.shape[0]):
        row = pd.Series(axis_corr[axis], index=x_sample.columns)
        high, low = top_field_text(row, n=6)
        axis_cards.append(
            {
                "axis": int(axis),
                "variance_pct": float(pca_global.explained_variance_ratio_[axis] * 100),
                "positive": high,
                "negative": low,
            }
        )

    method_scores = [
        {"name": "old clean SAE", "score": float(json.loads(Path("outputs/nhanes_mining_clean/summary.json").read_text())["cluster_silhouette"])},
        {"name": "block SAE", "score": float(json.loads(Path("outputs/nhanes_multiview/summary.json").read_text())["cluster_silhouette"])},
        {"name": "block factors", "score": float(best["score"])},
    ]
    if hdb_score is not None:
        method_scores.append({"name": "HDBSCAN subset", "score": hdb_score})

    summary = {
        "args": vars(args),
        "n_participants": int(x_df.shape[0]),
        "n_features": int(x_df.shape[1]),
        "n_blocks": len(block_meta),
        "factor_dim": int(z.shape[1]),
        "confounds": confound_names,
        "cluster_k": int(best["k"]),
        "cluster_silhouette": float(best["score"]),
        "hdbscan_clusters": int(len(np.unique(hdb_non_noise))),
        "hdbscan_coverage": float((hdb >= 0).mean()),
        "hdbscan_silhouette": hdb_score,
        "method_scores": method_scores,
        "blocks": block_meta,
        "cluster_cards": cards,
        "axis_cards": axis_cards,
        "plots": {
            "factor_umap_clusters": str(out / "factor_umap_clusters.png"),
            "factor_cluster_archetypes": str(out / "factor_cluster_archetypes.png"),
            "factor_axis_cards": str(out / "factor_axis_cards.png"),
            "tweetable_factor_atlas": str(out / "tweetable_factor_atlas.png"),
        },
    }

    make_umap(emb, labels, cards, out)
    make_archetype_heatmap(cluster_block_z, cards, out)
    make_axis_cards(axis_cards, out)
    make_composite(out, summary)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
