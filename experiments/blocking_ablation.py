#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from multiview_nhanes_umap import BLOCKS, PRETTY, block_scores, select_block_columns
from nhanes_phenome_sae import encode_sparse_autoencoder, prepare_matrix, train_sparse_autoencoder


def zscore_frame(frame):
    return (frame - frame.mean(axis=0)) / (frame.std(axis=0) + 1e-8)


def curated_blocks(x_df):
    blocks = {name: select_block_columns(x_df, cols) for name, cols in BLOCKS.items()}
    return {k: v for k, v in blocks.items() if len(v) >= 2}


def prefix_blocks(cols):
    out = {}
    for col in cols:
        prefix = col.split("__", 1)[0]
        out.setdefault(prefix, []).append(col)
    singles = []
    keep = {}
    for name, group in out.items():
        if len(group) == 1:
            singles.extend(group)
        else:
            keep[f"prefix_{name}"] = group
    if singles:
        keep["prefix_singletons"] = singles
    return keep


def random_blocks(cols, sizes, seed):
    rng = np.random.default_rng(seed)
    shuffled = list(cols)
    rng.shuffle(shuffled)
    blocks = {}
    cursor = 0
    for i, size in enumerate(sizes):
        blocks[f"random_{i:02d}"] = shuffled[cursor: cursor + size]
        cursor += size
    if cursor < len(shuffled):
        blocks[f"random_{len(sizes) - 1:02d}"].extend(shuffled[cursor:])
    return {k: v for k, v in blocks.items() if v}


def correlation_blocks(x_train, cols, n_blocks):
    corr = np.corrcoef(x_train[cols].to_numpy(dtype=np.float32), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    try:
        clustering = AgglomerativeClustering(n_clusters=n_blocks, metric="precomputed", linkage="average")
    except TypeError:
        clustering = AgglomerativeClustering(n_clusters=n_blocks, affinity="precomputed", linkage="average")
    labels = clustering.fit_predict(dist)
    out = {}
    for col, label in zip(cols, labels):
        out.setdefault(f"corr_{label:02d}", []).append(col)
    return dict(sorted(out.items()))


def train_test_indices(n, seed, train_frac):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = int(n * train_frac)
    return np.sort(idx[:split]), np.sort(idx[split:])


def fit_block_representation(x_train_df, x_test_df, blocks, args, seed):
    train_parts = []
    test_parts = []
    metas = []
    for bi, (name, cols) in enumerate(blocks.items()):
        cols = [c for c in cols if c in x_train_df.columns]
        if not cols:
            continue
        scaler = StandardScaler()
        xb_train = scaler.fit_transform(x_train_df[cols].to_numpy(dtype=np.float32)).astype(np.float32)
        xb_test = scaler.transform(x_test_df[cols].to_numpy(dtype=np.float32)).astype(np.float32)

        if args.encoder == "pca":
            n_comp = min(args.pca_per_block, xb_train.shape[1], xb_train.shape[0] - 1)
            enc = PCA(n_components=max(1, n_comp), random_state=seed + bi)
            z_train = enc.fit_transform(xb_train).astype(np.float32)
            z_test = enc.transform(xb_test).astype(np.float32)
            trainer = {"backend": "pca", "explained_variance": enc.explained_variance_ratio_.tolist()}
            live = z_train.shape[1]
            active = None
        else:
            hidden = min(args.hidden_per_block, max(args.min_hidden, xb_train.shape[1] * args.hidden_multiplier))
            model = train_sparse_autoencoder(
                xb_train,
                hidden,
                args.steps,
                args.batch_size,
                args.lr,
                args.l1,
                seed + bi,
                optimizer=args.optimizer,
                activation=args.activation,
                topk=args.topk,
                weight_decay=args.weight_decay,
                device=args.device,
            )
            h_train = encode_sparse_autoencoder(xb_train, model, args.activation, args.topk)
            h_test = encode_sparse_autoencoder(xb_test, model, args.activation, args.topk)
            active_rates = (h_train > 1e-3).mean(axis=0)
            live_mask = active_rates > args.live_threshold
            if live_mask.sum() < 2:
                live_mask[:] = True
            z_train = h_train[:, live_mask]
            z_test = h_test[:, live_mask]
            trainer = model.get("trainer", {})
            live = int(live_mask.sum())
            active = float(active_rates.mean())

        z_scaler = StandardScaler()
        z_train = z_scaler.fit_transform(z_train).astype(np.float32)
        z_test = z_scaler.transform(z_test).astype(np.float32)
        z_train = z_train / np.sqrt(z_train.shape[1])
        z_test = z_test / np.sqrt(z_test.shape[1])
        train_parts.append(z_train)
        test_parts.append(z_test)
        metas.append(
            {
                "block": name,
                "n_columns": len(cols),
                "embedding_dim": int(z_train.shape[1]),
                "live_units": live,
                "mean_active_rate": active,
                "trainer": trainer,
                "columns": cols,
            }
        )
    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1), metas


def choose_kmeans(z_train, z_test, args, seed):
    n_comp = min(args.global_components, z_train.shape[1], z_train.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=seed)
    ztr = pca.fit_transform(z_train)
    zte = pca.transform(z_test)
    best = None
    for k in range(args.min_k, args.max_k + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=args.kmeans_n_init)
        train_labels = km.fit_predict(ztr)
        train_score = silhouette_score(ztr, train_labels, sample_size=min(args.silhouette_sample, len(train_labels)), random_state=seed)
        if best is None or train_score > best["train_silhouette"]:
            best = {
                "k": k,
                "model": km,
                "train_labels": train_labels,
                "train_silhouette": float(train_score),
            }
    test_labels = best["model"].predict(zte)
    if len(np.unique(test_labels)) > 1:
        test_score = silhouette_score(zte, test_labels, sample_size=min(args.silhouette_sample, len(test_labels)), random_state=seed)
    else:
        test_score = np.nan
    best.update({"test_labels": test_labels, "test_silhouette": float(test_score), "z_train_pca": ztr, "z_test_pca": zte})
    return best


def kmeans_ari_stability(z_train_pca, k, args, seed):
    labels = []
    for i in range(args.stability_runs):
        km = KMeans(n_clusters=k, random_state=seed + 1000 + i, n_init=1)
        labels.append(km.fit_predict(z_train_pca))
    scores = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            scores.append(adjusted_rand_score(labels[i], labels[j]))
    return float(np.mean(scores)) if scores else np.nan


def cluster_patterns(eval_scores, labels, k):
    frame = eval_scores.assign(cluster=labels)
    means = frame.groupby("cluster").mean()
    for missing in sorted(set(range(k)) - set(means.index)):
        means.loc[missing] = eval_scores.mean(axis=0)
    means = means.sort_index()
    return zscore_frame(means).to_numpy(dtype=np.float32)


def archetype_stability(z_train_pca, eval_train, k, seed):
    rng = np.random.default_rng(seed + 2027)
    idx = rng.permutation(len(z_train_pca))
    half = len(idx) // 2
    a_idx, b_idx = idx[:half], idx[half:]
    la = KMeans(n_clusters=k, random_state=seed + 7, n_init=20).fit_predict(z_train_pca[a_idx])
    lb = KMeans(n_clusters=k, random_state=seed + 8, n_init=20).fit_predict(z_train_pca[b_idx])
    pa = cluster_patterns(eval_train.iloc[a_idx], la, k)
    pb = cluster_patterns(eval_train.iloc[b_idx], lb, k)
    pa = pa - pa.mean(axis=1, keepdims=True)
    pb = pb - pb.mean(axis=1, keepdims=True)
    denom = (np.linalg.norm(pa, axis=1, keepdims=True) @ np.linalg.norm(pb, axis=1, keepdims=True).T) + 1e-8
    corr = (pa @ pb.T) / denom
    rows, cols = linear_sum_assignment(-corr)
    return float(corr[rows, cols].mean())


def evaluate_config(name, blocks, x_train_df, x_test_df, eval_train, args, seed):
    z_train, z_test, metas = fit_block_representation(x_train_df, x_test_df, blocks, args, seed)
    clustering = choose_kmeans(z_train, z_test, args, seed)
    ari = kmeans_ari_stability(clustering["z_train_pca"], clustering["k"], args, seed)
    archetype = archetype_stability(clustering["z_train_pca"], eval_train, clustering["k"], seed)
    return {
        "name": name,
        "n_blocks": len(blocks),
        "n_columns": int(sum(len(v) for v in blocks.values())),
        "embedding_dim": int(z_train.shape[1]),
        "cluster_k": int(clustering["k"]),
        "train_silhouette": clustering["train_silhouette"],
        "test_silhouette": clustering["test_silhouette"],
        "kmeans_ari_stability": ari,
        "archetype_stability": archetype,
        "blocks": metas,
    }


def make_composite(results, out, args):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "blocks"} for r in results])
    ordered = df.sort_values("test_silhouette", ascending=True)
    colors = ["#9aa0a6" if n.startswith("random") else "#4c78a8" for n in ordered["name"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="#f6f3ec")
    metrics = [
        ("test_silhouette", "Held-out cluster separation"),
        ("archetype_stability", "Split-half archetype stability"),
        ("kmeans_ari_stability", "KMeans seed stability"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        ax.barh(ordered["name"], ordered[metric], color=colors)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        for i, val in enumerate(ordered[metric]):
            ax.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=9)
    fig.suptitle("Blocking ablation: does the hand-made phenotype split survive controls?", x=0.03, y=0.98, ha="left", fontsize=20, fontweight="bold")
    fig.text(
        0.03,
        0.02,
        f"Same columns, encoder={args.encoder}, optimizer={args.optimizer}, activation={args.activation}. Curated blocks should beat random/module/correlation blocks before we claim more than analyst-imposed structure.",
        fontsize=10,
        color="#333333",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(out / "blocking_ablation_composite.png", dpi=230)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/nhanes/processed/nhanes_phenome_raw.parquet")
    p.add_argument("--out-dir", default="outputs/nhanes_blocking_ablation")
    p.add_argument("--encoder", choices=["sae", "pca"], default="sae")
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    p.add_argument("--activation", choices=["relu_l1", "topk"], default="topk")
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--hidden-per-block", type=int, default=32)
    p.add_argument("--hidden-multiplier", type=int, default=3)
    p.add_argument("--min-hidden", type=int, default=4)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--l1", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--live-threshold", type=float, default=0.005)
    p.add_argument("--pca-per-block", type=int, default=5)
    p.add_argument("--global-components", type=int, default=32)
    p.add_argument("--min-k", type=int, default=5)
    p.add_argument("--max-k", type=int, default=12)
    p.add_argument("--kmeans-n-init", type=int, default=25)
    p.add_argument("--silhouette-sample", type=int, default=6000)
    p.add_argument("--stability-runs", type=int, default=5)
    p.add_argument("--random-repeats", type=int, default=5)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--max-participants", type=int, default=None)
    p.add_argument("--seed", type=int, default=191)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)
    x_df, _ = prepare_matrix(df, min_nonmissing=0.25, max_cols=1000)
    x_df = x_df[[c for c in x_df.columns if not c.endswith("__MISSING") and not c.startswith("DEMO__")]]

    base_blocks = curated_blocks(x_df)
    base_cols = list(dict.fromkeys(col for cols in base_blocks.values() for col in cols))
    x_df = x_df[base_cols]
    if args.max_participants and args.max_participants < len(x_df):
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(x_df), size=args.max_participants, replace=False))
        x_df = x_df.iloc[keep]

    train_idx, test_idx = train_test_indices(len(x_df), args.seed, args.train_frac)
    x_train_df = x_df.iloc[train_idx]
    x_test_df = x_df.iloc[test_idx]
    eval_scores = block_scores(x_df, base_blocks)
    eval_train = eval_scores.iloc[train_idx]

    sizes = [len(v) for v in base_blocks.values()]
    configs = [
        ("monolithic", {"all_curated_fields": base_cols}),
        ("curated_clinical", base_blocks),
        ("nhanes_prefix", prefix_blocks(base_cols)),
        ("correlation_blocks", correlation_blocks(x_train_df, base_cols, len(base_blocks))),
    ]
    for i in range(args.random_repeats):
        configs.append((f"random_matched_{i}", random_blocks(base_cols, sizes, args.seed + 500 + i)))

    results = []
    for i, (name, blocks) in enumerate(configs):
        print(f"running {name} ({len(blocks)} blocks)")
        results.append(evaluate_config(name, blocks, x_train_df, x_test_df, eval_train, args, args.seed + i * 37))
        pd.DataFrame([{k: v for k, v in r.items() if k != "blocks"} for r in results]).to_csv(out / "ablation_scores.csv", index=False)
        (out / "summary.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2))

    make_composite(results, out, args)
    print(json.dumps({"out_dir": str(out), "results": [{k: v for k, v in r.items() if k != "blocks"} for r in results]}, indent=2))


if __name__ == "__main__":
    main()
