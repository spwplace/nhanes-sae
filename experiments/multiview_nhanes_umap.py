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
from nhanes_phenome_sae import encode_sparse_autoencoder, prepare_matrix, train_sparse_autoencoder


BLOCKS = {
    "body_size": ["BMX__BMXBMI", "BMX__BMXWT", "BMX__BMXWAIST", "BMX__BMXHIP", "BMX__BMXARMC", "BMX__BMXHT"],
    "blood_pressure": ["BPX__BPXSY1", "BPX__BPXSY2", "BPX__BPXSY3", "BPX__BPXDI1", "BPX__BPXDI2", "BPX__BPXDI3", "BPXO__BPXOSY1", "BPXO__BPXOSY2", "BPXO__BPXOSY3", "BPXO__BPXODI1", "BPXO__BPXODI2", "BPXO__BPXODI3"],
    "lipids_glucose": ["TCHOL__LBXTC", "HDL__LBDHDD", "TRIGLY__LBXTR", "TRIGLY__LBDLDL", "GLU__LBXGLU", "GHB__LBXGH", "BIOPRO__LBXSGL"],
    "renal_liver": ["BIOPRO__LBXSCR", "BIOPRO__LBXSBU", "BIOPRO__LBXSAL", "BIOPRO__LBXSAPSI", "BIOPRO__LBXSASSI", "BIOPRO__LBXSATSI", "BIOPRO__LBXSGTSI", "BIOPRO__LBXSUA"],
    "blood_counts": ["CBC__LBXWBCSI", "CBC__LBXRBCSI", "CBC__LBXHGB", "CBC__LBXHCT", "CBC__LBXPLTSI", "CBC__LBXRDW", "CBC__LBXLYPCT", "CBC__LBXNEPCT"],
    "diabetes_history": ["DIQ__DIQ010", "DIQ__DIQ050", "DIQ__DIQ070", "DIQ__DIQ160", "DIQ__DIQ170", "DIQ__DIQ180"],
    "cardio_history": ["BPQ__BPQ020", "BPQ__BPQ030", "BPQ__BPQ040A", "BPQ__BPQ050A", "MCQ__MCQ160B", "MCQ__MCQ160C", "MCQ__MCQ160D", "MCQ__MCQ160E", "MCQ__MCQ160F"],
    "resp_smoking": ["SMQ__SMQ020", "SMQ__SMQ040", "SMQ__SMQ890", "SMQ__SMQ900", "SMQ__SMQ910", "MCQ__MCQ010", "MCQ__MCQ160G", "MCQ__MCQ160K", "MCQ__MCQ160O"],
    "mood_sleep": ["DPQ__DPQ010", "DPQ__DPQ020", "DPQ__DPQ030", "DPQ__DPQ040", "DPQ__DPQ050", "DPQ__DPQ060", "DPQ__DPQ070", "DPQ__DPQ080", "DPQ__DPQ090", "DPQ__DPQ100", "SLQ__SLD012", "SLQ__SLQ050"],
    "utilization_meds": ["RXQ_RX__RXDUSE", "RXQ_RX__RXDCOUNT", "RXQ_RX__RXQSEEN", "RXQ_RX__RXDDAYS", "HSQ__HSD010", "HSQ__HSQ500", "HSQ__HSQ510"],
    "activity_alcohol": ["PAQ__PAQ605", "PAQ__PAQ620", "PAQ__PAQ635", "PAQ__PAQ650", "PAQ__PAQ665", "PAQ__PAD680", "ALQ__ALQ111", "ALQ__ALQ130", "ALQ__ALQ142", "ALQ__ALQ151", "ALQ__ALQ170"],
}

PRETTY = {
    "body_size": "Body size",
    "blood_pressure": "Blood pressure",
    "lipids_glucose": "Lipids/glucose",
    "renal_liver": "Renal/liver",
    "blood_counts": "Blood counts",
    "diabetes_history": "Diabetes history",
    "cardio_history": "Cardio history",
    "resp_smoking": "Resp/smoking",
    "mood_sleep": "Mood/sleep",
    "utilization_meds": "Utilization/meds",
    "activity_alcohol": "Activity/alcohol",
}


def invert_yes_no_codes(x_df):
    # NHANES questionnaire yes/no variables often code yes=1, no=2. Convert
    # common binary fields so high values mean more of the named condition.
    out = x_df.copy()
    for col in out.columns:
        vals = set(np.round(out[col].dropna().unique(), 6))
        # After scaling this is not available, so this function is here for
        # future raw-code use. The current pipeline receives scaled columns.
        if vals <= {1.0, 2.0}:
            out[col] = 2.0 - out[col]
    return out


def select_block_columns(x_df, block_cols):
    present = [c for c in block_cols if c in x_df.columns]
    aliases = []
    # Pull SI-unit duplicates only when canonical columns are absent.
    for c in block_cols:
        if c in x_df.columns:
            continue
        stem = c.rsplit("__", 1)[-1]
        matches = [col for col in x_df.columns if col.endswith(stem + "SI") or col.endswith(stem + "SI")]
        aliases.extend(matches[:1])
    cols = list(dict.fromkeys(present + aliases))
    return cols


def block_scores(x_df, block_columns):
    scores = {}
    for name, cols in block_columns.items():
        if cols:
            scores[name] = x_df[cols].mean(axis=1)
    return pd.DataFrame(scores, index=x_df.index)


def fit_block_sae(x_block, hidden, steps, batch_size, lr, l1, seed, optimizer, activation, topk, weight_decay, device):
    model = train_sparse_autoencoder(
        x_block,
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
    h = encode_sparse_autoencoder(x_block, model, activation, topk)
    active = (h > 1e-3).mean(axis=0)
    live = active > 0.005
    if live.sum() == 0:
        live[:] = True
    h = h[:, live]
    h = StandardScaler().fit_transform(h)
    return h, int(live.sum()), float(active.mean()), model.get("trainer", {})


def make_panel(emb, scores, out):
    names = [n for n in ["body_size", "mood_sleep", "blood_pressure", "lipids_glucose", "renal_liver", "utilization_meds"] if n in scores]
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 9), facecolor="#f6f3ec")
    for ax, name in zip(axes.ravel(), names):
        vals = scores[name].to_numpy()
        lo, hi = np.nanpercentile(vals, [2, 98])
        ax.scatter(emb[:, 0], emb[:, 1], c=np.clip(vals, lo, hi), s=3, alpha=0.65, cmap="magma", linewidths=0)
        ax.set_title(PRETTY.get(name, name), loc="left", fontsize=13, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Multi-view NHANES phenome map (block-balanced SAE activations)", x=0.04, y=0.995, ha="left", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "multiview_panel.png", dpi=240)
    plt.close()


def cluster_name(row):
    pos = row.sort_values(ascending=False)
    if pos.iloc[1] > 0.35:
        return f"{PRETTY.get(pos.index[0], pos.index[0])} + {PRETTY.get(pos.index[1], pos.index[1])}"
    return PRETTY.get(pos.index[0], pos.index[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/nhanes/processed/nhanes_phenome_raw.parquet")
    p.add_argument("--hidden-per-block", type=int, default=32)
    p.add_argument("--steps", type=int, default=2200)
    p.add_argument("--batch-size", type=int, default=768)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--l1", type=float, default=0.002)
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    p.add_argument("--activation", choices=["relu_l1", "topk"], default="relu_l1")
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--umap-sample", type=int, default=18000)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--out-dir", default="outputs/nhanes_multiview")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)
    x_df, _ = prepare_matrix(df, min_nonmissing=0.25, max_cols=1000)
    x_df = x_df[[c for c in x_df.columns if not c.endswith("__MISSING") and not c.startswith("DEMO__")]]

    block_columns = {name: select_block_columns(x_df, cols) for name, cols in BLOCKS.items()}
    block_columns = {k: v for k, v in block_columns.items() if len(v) >= 2}
    scores = block_scores(x_df, block_columns)

    embeddings = []
    block_meta = []
    for i, (name, cols) in enumerate(block_columns.items()):
        xb = x_df[cols].to_numpy(dtype=np.float32)
        xb = StandardScaler().fit_transform(xb).astype(np.float32)
        hidden = min(args.hidden_per_block, max(4, xb.shape[1] * 3))
        h, live, active, trainer = fit_block_sae(
            xb,
            hidden,
            args.steps,
            args.batch_size,
            args.lr,
            args.l1,
            args.seed + i,
            args.optimizer,
            args.activation,
            args.topk,
            args.weight_decay,
            args.device,
        )
        # Equalize block contribution regardless of field count or SAE live units.
        h = h / np.sqrt(h.shape[1])
        embeddings.append(h)
        block_meta.append({"block": name, "n_columns": len(cols), "hidden": hidden, "live_units": live, "mean_active_rate": active, "trainer": trainer, "columns": cols})

    z = np.concatenate(embeddings, axis=1).astype(np.float32)
    z = StandardScaler().fit_transform(z).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    sample_n = min(args.umap_sample, len(z))
    sample_idx = np.sort(rng.choice(len(z), size=sample_n, replace=False))
    z_sample = z[sample_idx]
    pca = PCA(n_components=min(40, z_sample.shape[1]), random_state=args.seed)
    z_pca = pca.fit_transform(z_sample)
    emb = UMAP(n_neighbors=45, min_dist=0.08, metric="cosine", random_state=args.seed).fit_transform(z_pca)
    scores_s = scores.iloc[sample_idx]

    best = None
    for k in range(6, 15):
        labels = KMeans(n_clusters=k, random_state=args.seed, n_init=30).fit_predict(z_pca)
        score = silhouette_score(z_pca, labels, sample_size=min(6000, len(labels)), random_state=args.seed)
        if best is None or score > best["score"]:
            best = {"k": k, "score": float(score), "labels": labels}
    labels = best["labels"]

    cluster_scores = scores_s.assign(cluster=labels).groupby("cluster").mean()
    cluster_z = (cluster_scores - scores_s.mean(axis=0)) / (scores_s.std(axis=0) + 1e-8)
    sizes = pd.Series(labels).value_counts().sort_index()
    cards = []
    for cid, row in cluster_z.iterrows():
        cards.append({
            "cluster": int(cid),
            "name": cluster_name(row),
            "n": int(sizes.loc[cid]),
            "pct": float(sizes.loc[cid] / len(labels)),
            "block_z": {str(k): float(v) for k, v in row.items()},
        })

    plt.figure(figsize=(9, 7), facecolor="#f6f3ec")
    ax = plt.gca()
    ax.set_facecolor("#f6f3ec")
    for cid in sorted(np.unique(labels)):
        mask = labels == cid
        ax.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.75, linewidths=0, label=str(cid))
    ax.legend(markerscale=3, fontsize=8, ncols=2, frameon=False)
    ax.set_title("Block-balanced SAE UMAP clusters", loc="left", fontsize=16, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "umap_clusters.png", dpi=220)
    plt.close()

    make_panel(emb, scores_s, out)

    plt.figure(figsize=(11, 7), facecolor="#f6f3ec")
    ax = plt.gca()
    mat = cluster_z.to_numpy()
    im = ax.imshow(mat, cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_yticks(range(len(cluster_z)))
    ax.set_yticklabels([f"{c['cluster']}: {c['name']}" for c in cards], fontsize=9)
    ax.set_xticks(range(len(cluster_z.columns)))
    ax.set_xticklabels([PRETTY.get(c, c) for c in cluster_z.columns], rotation=35, ha="right", fontsize=9)
    ax.set_title("Multi-view cluster archetypes", loc="left", fontsize=16, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    plt.tight_layout()
    plt.savefig(out / "cluster_archetypes.png", dpi=220)
    plt.close()

    summary = {
        "args": vars(args),
        "n_participants": int(x_df.shape[0]),
        "n_blocks": len(block_meta),
        "embedding_dim": int(z.shape[1]),
        "cluster_k": int(best["k"]),
        "cluster_silhouette": float(best["score"]),
        "blocks": block_meta,
        "cluster_cards": cards,
        "plots": {
            "umap_clusters": str(out / "umap_clusters.png"),
            "multiview_panel": str(out / "multiview_panel.png"),
            "cluster_archetypes": str(out / "cluster_archetypes.png"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
