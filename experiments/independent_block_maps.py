#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP

sys.path.append(str(Path(__file__).resolve().parent))
from multiview_nhanes_umap import BLOCKS, PRETTY, select_block_columns
from nhanes_phenome_sae import encode_sparse_autoencoder, prepare_matrix, train_sparse_autoencoder


def embed_block(x, hidden, steps, batch_size, lr, l1, seed, optimizer, activation, topk, weight_decay, device):
    model = train_sparse_autoencoder(
        x,
        hidden,
        steps,
        batch_size,
        lr,
        l1,
        seed,
        optimizer=optimizer,
        activation=activation,
        topk=topk,
        weight_decay=weight_decay,
        device=device,
    )
    h = encode_sparse_autoencoder(x, model, activation, topk)
    active = (h > 1e-3).mean(axis=0)
    live = active > 0.005
    if live.sum() >= 3:
        z = h[:, live]
        source = "sae"
    else:
        z = x
        source = "direct_fields"
    z = StandardScaler().fit_transform(z).astype(np.float32)
    return z, source, int(live.sum()), float(active.mean()), model.get("trainer", {})


def choose_clusters(z_pca, seed):
    max_k = min(8, max(3, len(z_pca) // 500))
    best = None
    for k in range(3, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=30).fit_predict(z_pca)
        score = silhouette_score(z_pca, labels, sample_size=min(5000, len(labels)), random_state=seed)
        if best is None or score > best["score"]:
            best = {"k": k, "labels": labels, "score": float(score)}
    return best


def plot_umap(emb, labels, title, path):
    plt.figure(figsize=(8, 7), facecolor="#f6f3ec")
    ax = plt.gca()
    ax.set_facecolor("#f6f3ec")
    for cid in sorted(np.unique(labels)):
        mask = labels == cid
        ax.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.75, linewidths=0, label=str(cid))
    ax.legend(markerscale=3, fontsize=8, ncols=2, frameon=False)
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_cluster_fields(cluster_delta, title, path):
    plt.figure(figsize=(11, 5.5), facecolor="#f6f3ec")
    ax = plt.gca()
    mat = cluster_delta.to_numpy()
    vmax = max(1.0, float(np.nanpercentile(np.abs(mat), 98)))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(cluster_delta)))
    ax.set_yticklabels([f"cluster {i}" for i in cluster_delta.index])
    ax.set_xticks(range(len(cluster_delta.columns)))
    ax.set_xticklabels([c.replace("__", " / ").replace("_", " ") for c in cluster_delta.columns], rotation=60, ha="right", fontsize=8)
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/nhanes/processed/nhanes_phenome_raw.parquet")
    p.add_argument("--out-dir", default="outputs/nhanes_independent_blocks")
    p.add_argument("--blocks", nargs="*", default=list(BLOCKS))
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--batch-size", type=int, default=768)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--l1", type=float, default=0.001)
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    p.add_argument("--activation", choices=["relu_l1", "topk"], default="relu_l1")
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--umap-sample", type=int, default=18000)
    p.add_argument("--seed", type=int, default=131)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)
    x_df, _ = prepare_matrix(df, min_nonmissing=0.25, max_cols=1000)
    x_df = x_df[[c for c in x_df.columns if not c.endswith("__MISSING") and not c.startswith("DEMO__")]]

    rng = np.random.default_rng(args.seed)
    sample_n = min(args.umap_sample, len(x_df))
    sample_idx = np.sort(rng.choice(len(x_df), size=sample_n, replace=False))
    all_cards = []

    for bi, block in enumerate(args.blocks):
        if block not in BLOCKS:
            continue
        cols = select_block_columns(x_df, BLOCKS[block])
        if len(cols) < 2:
            continue
        block_dir = out / block
        block_dir.mkdir(parents=True, exist_ok=True)
        xb_df = x_df[cols]
        xb = StandardScaler().fit_transform(xb_df.to_numpy(dtype=np.float32)).astype(np.float32)
        hidden = min(args.hidden, max(4, xb.shape[1] * 4))
        z, source, live_units, active, trainer = embed_block(
            xb,
            hidden,
            args.steps,
            args.batch_size,
            args.lr,
            args.l1,
            args.seed + bi,
            args.optimizer,
            args.activation,
            args.topk,
            args.weight_decay,
            args.device,
        )
        z_sample = z[sample_idx]
        n_pca = min(12, z_sample.shape[1], z_sample.shape[0] - 1)
        z_pca = PCA(n_components=n_pca, random_state=args.seed).fit_transform(z_sample)
        emb = UMAP(n_neighbors=35, min_dist=0.08, metric="cosine", random_state=args.seed + bi).fit_transform(z_pca)
        best = choose_clusters(z_pca, args.seed + bi)
        labels = best["labels"]
        sampled_fields = xb_df.iloc[sample_idx]
        cluster_means = sampled_fields.assign(cluster=labels).groupby("cluster").mean()
        cluster_delta = cluster_means - sampled_fields.mean(axis=0)

        plot_umap(emb, labels, f"{PRETTY.get(block, block)} independent map", block_dir / "umap_clusters.png")
        plot_cluster_fields(cluster_delta, f"{PRETTY.get(block, block)} cluster field shifts", block_dir / "cluster_fields.png")

        cards = []
        sizes = pd.Series(labels).value_counts().sort_index()
        for cid, row in cluster_delta.iterrows():
            cards.append({
                "cluster": int(cid),
                "n": int(sizes.loc[cid]),
                "pct": float(sizes.loc[cid] / len(labels)),
                "top_positive_fields": {str(k): float(v) for k, v in row.sort_values(ascending=False).head(8).items()},
                "top_negative_fields": {str(k): float(v) for k, v in row.sort_values().head(8).items()},
            })
        block_summary = {
            "block": block,
            "name": PRETTY.get(block, block),
            "n_columns": len(cols),
            "columns": cols,
            "embedding_source": source,
            "live_units": live_units,
            "mean_active_rate": active,
            "trainer": trainer,
            "cluster_k": int(best["k"]),
            "cluster_silhouette": float(best["score"]),
            "cards": cards,
            "plots": {
                "umap_clusters": str(block_dir / "umap_clusters.png"),
                "cluster_fields": str(block_dir / "cluster_fields.png"),
            },
        }
        (block_dir / "summary.json").write_text(json.dumps(block_summary, indent=2))
        all_cards.append(block_summary)

    summary = {
        "args": vars(args),
        "n_participants": int(x_df.shape[0]),
        "blocks": all_cards,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "out_dir": str(out),
        "n_participants": summary["n_participants"],
        "blocks": [
            {
                "block": b["block"],
                "columns": b["n_columns"],
                "source": b["embedding_source"],
                "k": b["cluster_k"],
                "silhouette": round(b["cluster_silhouette"], 3),
            }
            for b in all_cards
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
