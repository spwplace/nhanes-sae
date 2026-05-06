#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANCHORS = {
    "adiposity_body_size": ["BMX__BMXBMI", "BMX__BMXWT", "BMX__BMXWAIST"],
    "blood_pressure": ["BPX__BPXSY1", "BPX__BPXDI1", "BPXO__BPXOSY1", "BPXO__BPXODI1", "BPQ__BPQ020"],
    "lipids_glucose": ["TCHOL__LBXTC", "HDL__LBDHDD", "TRIGLY__LBXTR", "GLU__LBXGLU", "GHB__LBXGH"],
    "renal_liver_biochem": ["BIOPRO__LBXSCR", "BIOPRO__LBXSBU", "BIOPRO__LBXSAPSI", "BIOPRO__LBXSASSI", "BIOPRO__LBXSATSI"],
    "blood_counts": ["CBC__LBXWBCSI", "CBC__LBXRBCSI", "CBC__LBXHGB", "CBC__LBXPLTSI"],
    "diabetes": ["DIQ__DIQ010", "DIQ__DIQ050", "DIQ__DIQ070"],
    "cardio_history": ["MCQ__MCQ160B", "MCQ__MCQ160C", "MCQ__MCQ160E", "MCQ__MCQ160F"],
    "respiratory_smoking": ["SMQ__SMQ020", "SMQ__SMD641", "MCQ__MCQ160G", "MCQ__MCQ010"],
    "mental_health_sleep": ["DPQ__DPQ010", "DPQ__DPQ020", "DPQ__DPQ030", "SLQ__SLD012"],
    "general_health_utilization": ["HSQ__HSD010", "HSQ__HSQ500", "RXQ_RX__RXDCOUNT"],
}


def train_sparse_autoencoder(x, hidden, steps, batch_size, lr, l1, seed):
    rng = np.random.default_rng(seed)
    n, dim = x.shape
    w_enc = rng.normal(0, 0.04, (dim, hidden)).astype(np.float32)
    b_enc = np.zeros(hidden, dtype=np.float32)
    w_dec = rng.normal(0, 0.04, (hidden, dim)).astype(np.float32)
    b_dec = np.zeros(dim, dtype=np.float32)
    losses = []
    for step in range(steps):
        idx = rng.integers(0, n, size=batch_size)
        xb = x[idx]
        pre = xb @ w_enc + b_enc
        h = np.maximum(pre, 0)
        recon = h @ w_dec + b_dec
        err = recon - xb
        grad_recon = (2.0 / batch_size) * err
        grad_w_dec = h.T @ grad_recon
        grad_b_dec = grad_recon.sum(axis=0)
        grad_h = grad_recon @ w_dec.T + l1
        grad_pre = grad_h * (pre > 0)
        grad_w_enc = xb.T @ grad_pre
        grad_b_enc = grad_pre.sum(axis=0)
        for grad in (grad_w_enc, grad_b_enc, grad_w_dec, grad_b_dec):
            np.clip(grad, -1.0, 1.0, out=grad)
        w_enc -= lr * grad_w_enc
        b_enc -= lr * grad_b_enc
        w_dec -= lr * grad_w_dec
        b_dec -= lr * grad_b_dec
        if step % 25 == 0 or step == steps - 1:
            losses.append(float(np.mean(err * err) + l1 * np.mean(h)))
    return {"w_enc": w_enc, "b_enc": b_enc, "w_dec": w_dec, "b_dec": b_dec, "losses": losses}


def prepare_matrix(df, min_nonmissing, max_cols):
    df = df.copy()
    age_col = "DEMO__RIDAGEYR"
    if age_col in df:
        df = df[df[age_col] >= 18]
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.drop(columns=[c for c in ["SEQN"] if c in numeric], errors="ignore")
    design_prefixes = (
        "DEMO__WT", "DEMO__SDMV", "TCHOL__WT", "HDL__WT", "TRIGLY__WT",
        "GLU__WT", "GHB__WT", "BIOPRO__WT", "CBC__WT", "CRP__WT",
    )
    design_exact = {"DEMO__RIDSTATR", "DEMO__RIDEXMON"}
    numeric = numeric.drop(
        columns=[c for c in numeric.columns if c.startswith(design_prefixes) or c in design_exact],
        errors="ignore",
    )

    # NHANES questionnaire files often use repeated 7/9 codes for refused/don't know.
    # Treat large sentinel-style values as missing for compact ordinal fields.
    for col in numeric.columns:
        s = numeric[col]
        finite = s.dropna()
        if len(finite) == 0:
            continue
        if finite.max() <= 999 and finite.nunique() <= 25:
            numeric[col] = s.replace({7: np.nan, 9: np.nan, 77: np.nan, 99: np.nan, 777: np.nan, 999: np.nan})

    nonmissing = numeric.notna().mean(axis=0)
    numeric = numeric.loc[:, nonmissing >= min_nonmissing]
    nunique = numeric.nunique(dropna=True)
    numeric = numeric.loc[:, nunique >= 2]
    variance = numeric.var(axis=0, skipna=True).sort_values(ascending=False)
    if max_cols and len(variance) > max_cols:
        numeric = numeric[variance.head(max_cols).index]

    miss = numeric.isna().astype(np.float32)
    med = numeric.median(axis=0)
    filled = numeric.fillna(med)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0).replace(0, 1)
    scaled = ((filled - mean) / std).clip(-6, 6)

    # Include missingness indicators only for fields where absence itself varies.
    miss_cols = miss.loc[:, miss.mean(axis=0).between(0.02, 0.98)]
    miss_cols = miss_cols.add_suffix("__MISSING")
    x_df = pd.concat([scaled, miss_cols], axis=1)
    return x_df.astype(np.float32), numeric


def corrcoef(a, b):
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    return (a.T @ b) / ((len(a) - 1) * (a.std(axis=0)[:, None] + 1e-8) * (b.std(axis=0)[None, :] + 1e-8))


def anchor_scores(x_df):
    scores = {}
    for name, cols in ANCHORS.items():
        present = [c for c in cols if c in x_df.columns]
        if present:
            scores[name] = x_df[present].mean(axis=1).to_numpy(dtype=np.float32)
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/nhanes/processed/nhanes_phenome_raw.parquet")
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.014)
    p.add_argument("--l1", type=float, default=0.006)
    p.add_argument("--min-nonmissing", type=float, default=0.25)
    p.add_argument("--max-cols", type=int, default=900)
    p.add_argument("--seed", type=int, default=23)
    args = p.parse_args()

    out = Path("outputs/nhanes")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)
    x_df, observed = prepare_matrix(df, args.min_nonmissing, args.max_cols)
    x = x_df.to_numpy(dtype=np.float32)
    model = train_sparse_autoencoder(x, args.hidden, args.steps, args.batch_size, args.lr, args.l1, args.seed)
    h = np.maximum(x @ model["w_enc"] + model["b_enc"], 0)
    field_corr = corrcoef(h, x)
    anchors = anchor_scores(x_df)
    if anchors:
        anchor_mat = np.column_stack(list(anchors.values()))
        anchor_corr = corrcoef(h, anchor_mat)
        anchor_names = list(anchors)
    else:
        anchor_corr = np.zeros((args.hidden, 1), dtype=np.float32)
        anchor_names = ["none"]

    best_anchor = np.argmax(np.abs(anchor_corr), axis=1)
    best_score = anchor_corr[np.arange(args.hidden), best_anchor]
    active_rate = (h > 1e-3).mean(axis=0)
    # Prefer actual features over nearly dead units.
    live = np.where(active_rate > 0.002)[0]
    if len(live) == 0:
        live = np.arange(args.hidden)
    ranking = live[np.argsort(np.abs(best_score[live]))[::-1]]

    cards = []
    for unit in ranking[:24]:
        top_fields = np.argsort(np.abs(field_corr[unit]))[::-1][:12]
        cards.append({
            "unit": int(unit),
            "anchor": anchor_names[int(best_anchor[unit])],
            "anchor_correlation": float(best_score[unit]),
            "active_rate": float(active_rate[unit]),
            "top_fields": [
                {"field": str(x_df.columns[int(i)]), "correlation": float(field_corr[unit, i])}
                for i in top_fields
            ],
        })

    plt.figure(figsize=(7, 4))
    plt.plot(model["losses"], color="#19535f")
    plt.title("NHANES sparse autoencoder training loss")
    plt.xlabel("checkpoint")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(out / "loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.imshow(anchor_corr[ranking[:32]], aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.yticks(range(min(32, len(ranking))), [f"unit {u}" for u in ranking[:32]])
    plt.xticks(range(len(anchor_names)), anchor_names, rotation=45, ha="right")
    plt.colorbar(label="Pearson r")
    plt.tight_layout()
    plt.savefig(out / "anchor_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 7))
    top_field_cols = np.argsort(np.abs(field_corr[ranking[:24]]).max(axis=0))[::-1][:80]
    plt.imshow(field_corr[ranking[:24]][:, top_field_cols], aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.yticks(range(24), [f"unit {u}" for u in ranking[:24]])
    plt.xticks(range(len(top_field_cols)), [x_df.columns[i] for i in top_field_cols], rotation=90, fontsize=6)
    plt.colorbar(label="Pearson r")
    plt.tight_layout()
    plt.savefig(out / "field_heatmap.png", dpi=160)
    plt.close()

    summary = {
        "source": "nhanes",
        "args": vars(args),
        "n_participants": int(x_df.shape[0]),
        "n_features": int(x_df.shape[1]),
        "n_observed_fields": int(observed.shape[1]),
        "final_loss": model["losses"][-1],
        "mean_active_rate": float(active_rate.mean()),
        "anchor_names": anchor_names,
        "cards": cards,
        "plots": {
            "loss": "outputs/nhanes/loss.png",
            "anchor_heatmap": "outputs/nhanes/anchor_heatmap.png",
            "field_heatmap": "outputs/nhanes/field_heatmap.png",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
