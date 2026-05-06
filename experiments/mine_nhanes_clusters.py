#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from nhanes_phenome_sae import ANCHORS, prepare_matrix, train_sparse_autoencoder


PRETTY_ANCHORS = {
    "adiposity_body_size": "Adiposity",
    "blood_pressure": "Blood pressure",
    "lipids_glucose": "Lipids/glucose",
    "renal_liver_biochem": "Renal/liver",
    "blood_counts": "Blood counts",
    "diabetes": "Diabetes",
    "cardio_history": "Cardio history",
    "respiratory_smoking": "Resp/smoking",
    "mental_health_sleep": "Mood/sleep",
    "general_health_utilization": "Health/utilization",
}


def corrcoef(a, b):
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    return (a.T @ b) / ((len(a) - 1) * (a.std(axis=0)[:, None] + 1e-8) * (b.std(axis=0)[None, :] + 1e-8))


def anchor_frame(x_df):
    scores = {}
    for name, cols in ANCHORS.items():
        present = [c for c in cols if c in x_df.columns]
        if present:
            scores[name] = x_df[present].mean(axis=1)
    return pd.DataFrame(scores, index=x_df.index)


def top_terms(row, positive=True, n=4):
    ordered = row.sort_values(ascending=not positive)
    return [f"{idx.replace('_', ' ')} {val:+.2f}" for idx, val in ordered.head(n).items()]


def cluster_name(row):
    pos = row.sort_values(ascending=False)
    neg = row.sort_values()
    lead = PRETTY_ANCHORS.get(pos.index[0], pos.index[0])
    if pos.iloc[0] < 0.25 and abs(neg.iloc[0]) > pos.iloc[0]:
        return f"Low {PRETTY_ANCHORS.get(neg.index[0], neg.index[0]).lower()}"
    second = PRETTY_ANCHORS.get(pos.index[1], pos.index[1])
    if pos.iloc[1] > 0.25:
        return f"{lead} + {second}"
    return lead


def make_umap_panel(emb, color, title, path, cmap="viridis", categorical=False):
    plt.figure(figsize=(8, 7), facecolor="#f6f3ec")
    ax = plt.gca()
    ax.set_facecolor("#f6f3ec")
    if categorical:
        vals = np.asarray(color)
        for val in np.unique(vals):
            mask = vals == val
            ax.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.75, label=str(val), linewidths=0)
        ax.legend(markerscale=3, fontsize=8, ncols=2, frameon=False, loc="best")
    else:
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=color, s=4, alpha=0.72, cmap=cmap, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/nhanes/processed/nhanes_phenome_raw.parquet")
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--steps", type=int, default=3200)
    p.add_argument("--batch-size", type=int, default=768)
    p.add_argument("--lr", type=float, default=0.012)
    p.add_argument("--l1", type=float, default=0.006)
    p.add_argument("--max-cols", type=int, default=1000)
    p.add_argument("--min-nonmissing", type=float, default=0.25)
    p.add_argument("--umap-sample", type=int, default=18000)
    p.add_argument("--keep-missingness", action="store_true")
    p.add_argument("--keep-demographics", action="store_true")
    p.add_argument("--include-prefixes", nargs="*", default=None)
    p.add_argument("--exclude-prefixes", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=67)
    p.add_argument("--out-dir", default="outputs/nhanes_mining")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)
    x_df, _ = prepare_matrix(df, args.min_nonmissing, args.max_cols)
    drop_cols = []
    if not args.keep_missingness:
        drop_cols.extend([c for c in x_df.columns if c.endswith("__MISSING")])
    if not args.keep_demographics:
        drop_cols.extend([c for c in x_df.columns if c.startswith("DEMO__")])
    if drop_cols:
        x_df = x_df.drop(columns=sorted(set(drop_cols)), errors="ignore")
    if args.include_prefixes:
        prefixes = tuple(f"{p}__" for p in args.include_prefixes)
        x_df = x_df[[c for c in x_df.columns if c.startswith(prefixes)]]
    if args.exclude_prefixes:
        prefixes = tuple(f"{p}__" for p in args.exclude_prefixes)
        x_df = x_df[[c for c in x_df.columns if not c.startswith(prefixes)]]
    if x_df.shape[1] < 10:
        raise SystemExit(f"Only {x_df.shape[1]} columns remain after filtering; relax prefix filters.")
    x = x_df.to_numpy(dtype=np.float32)

    model = train_sparse_autoencoder(x, args.hidden, args.steps, args.batch_size, args.lr, args.l1, args.seed)
    h = np.maximum(x @ model["w_enc"] + model["b_enc"], 0).astype(np.float32)
    active_rate = (h > 1e-3).mean(axis=0)
    live_units = active_rate > 0.003
    h_live = h[:, live_units]
    h_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(h_live)

    rng = np.random.default_rng(args.seed)
    sample_n = min(args.umap_sample, len(h_scaled))
    sample_idx = np.sort(rng.choice(len(h_scaled), size=sample_n, replace=False))
    h_sample = h_scaled[sample_idx]

    pca = PCA(n_components=min(30, h_sample.shape[1]), random_state=args.seed)
    h_pca = pca.fit_transform(h_sample)
    emb = UMAP(
        n_neighbors=35,
        min_dist=0.05,
        metric="cosine",
        random_state=args.seed,
        low_memory=False,
    ).fit_transform(h_pca)

    candidate_ks = list(range(5, 13))
    best = None
    for k in candidate_ks:
        km = KMeans(n_clusters=k, random_state=args.seed, n_init=25)
        labels = km.fit_predict(h_pca)
        score = silhouette_score(h_pca, labels, sample_size=min(6000, len(labels)), random_state=args.seed)
        if best is None or score > best["score"]:
            best = {"k": k, "score": float(score), "labels": labels, "model": km}
    labels = best["labels"]

    anchors = anchor_frame(x_df).iloc[sample_idx]
    cluster_anchor = anchors.assign(cluster=labels).groupby("cluster").mean()
    cluster_anchor_z = (cluster_anchor - anchors.mean(axis=0)) / (anchors.std(axis=0) + 1e-8)
    cluster_sizes = pd.Series(labels).value_counts().sort_index()

    field_corr = corrcoef(h_live, x)
    unit_strength = h[:, live_units][sample_idx].mean(axis=0)
    active_units = np.argsort(unit_strength)[::-1][:32]

    cluster_cards = []
    for cid, row in cluster_anchor_z.iterrows():
        mask = labels == cid
        field_delta = x_df.iloc[sample_idx[mask]].mean(axis=0) - x_df.iloc[sample_idx].mean(axis=0)
        top_pos = field_delta.sort_values(ascending=False).head(10)
        top_neg = field_delta.sort_values().head(10)
        cluster_cards.append({
            "cluster": int(cid),
            "name": cluster_name(row),
            "n": int(cluster_sizes.loc[cid]),
            "pct": float(cluster_sizes.loc[cid] / len(labels)),
            "anchor_z": {str(k): float(v) for k, v in row.items()},
            "top_positive_fields": {str(k): float(v) for k, v in top_pos.items()},
            "top_negative_fields": {str(k): float(v) for k, v in top_neg.items()},
        })

    make_umap_panel(emb, labels, f"NHANES phenome SAE UMAP: {best['k']} clusters", out / "umap_clusters.png", categorical=True)
    for anchor in anchors.columns:
        vals = anchors[anchor].to_numpy()
        lo, hi = np.nanpercentile(vals, [2, 98])
        clipped = np.clip(vals, lo, hi)
        make_umap_panel(emb, clipped, f"UMAP colored by {PRETTY_ANCHORS.get(anchor, anchor)}", out / f"umap_{anchor}.png", cmap="magma")

    plt.figure(figsize=(11, 6), facecolor="#f6f3ec")
    ax = plt.gca()
    ax.set_facecolor("#f6f3ec")
    mat = cluster_anchor_z.to_numpy()
    im = ax.imshow(mat, cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_yticks(range(len(cluster_anchor_z)))
    ax.set_yticklabels([f"{cid}: {cluster_cards[i]['name']}" for i, cid in enumerate(cluster_anchor_z.index)], fontsize=9)
    ax.set_xticks(range(len(cluster_anchor_z.columns)))
    ax.set_xticklabels([PRETTY_ANCHORS.get(c, c) for c in cluster_anchor_z.columns], rotation=35, ha="right", fontsize=9)
    ax.set_title("Cluster archetypes by anchor z-score", loc="left", fontsize=16, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    plt.tight_layout()
    plt.savefig(out / "cluster_archetypes.png", dpi=220)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor="#f6f3ec")
    preferred_showcase = ["adiposity_body_size", "mental_health_sleep", "blood_pressure", "lipids_glucose"]
    showcase = [a for a in preferred_showcase if a in anchors.columns]
    showcase += [a for a in anchors.columns if a not in showcase]
    for ax, anchor in zip(axes.ravel(), showcase[:4]):
        vals = anchors[anchor].to_numpy()
        lo, hi = np.nanpercentile(vals, [2, 98])
        ax.scatter(emb[:, 0], emb[:, 1], c=np.clip(vals, lo, hi), s=3, alpha=0.65, cmap="magma", linewidths=0)
        ax.set_title(PRETTY_ANCHORS.get(anchor, anchor), loc="left", fontsize=13, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.ravel()[len(showcase[:4]):]:
        ax.axis("off")
    fig.suptitle("A phenome map from sparse autoencoder activations", x=0.06, y=0.995, ha="left", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "tweetable_four_panel.png", dpi=240)
    plt.close()

    summary = {
        "args": vars(args),
        "n_participants": int(x_df.shape[0]),
        "n_features": int(x_df.shape[1]),
        "hidden": args.hidden,
        "live_units": int(live_units.sum()),
        "mean_active_rate": float(active_rate.mean()),
        "cluster_k": int(best["k"]),
        "cluster_silhouette": float(best["score"]),
        "cluster_cards": cluster_cards,
        "plots": {
            "umap_clusters": str(out / "umap_clusters.png"),
            "cluster_archetypes": str(out / "cluster_archetypes.png"),
            "tweetable_four_panel": str(out / "tweetable_four_panel.png"),
            **{f"umap_{a}": str(out / f"umap_{a}.png") for a in anchors.columns},
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
