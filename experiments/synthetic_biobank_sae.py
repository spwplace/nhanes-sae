#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LATENT_NAMES = [
    "adiposity_metabolic",
    "cardiovascular",
    "renal_impairment",
    "inflammation_autoimmune",
    "neurocognitive_frailty",
    "respiratory_smoking",
    "mental_health_pain_sleep",
    "healthcare_utilization",
]

FIELDS = [
    "height_cm", "bmi", "waist_hip_ratio", "systolic_bp", "diastolic_bp",
    "ldl", "hdl", "triglycerides", "hba1c", "fasting_glucose",
    "creatinine", "egfr", "crp", "alt", "ast", "ggt",
    "fev1", "pack_years", "sleep_hours", "cognitive_score",
    "diabetes_dx", "hypertension_dx", "cad_dx", "heart_failure_dx",
    "ckd_dx", "copd_dx", "asthma_dx", "alzheimers_dx", "depression_dx",
    "anxiety_dx", "autoimmune_dx", "chronic_pain_dx", "fall_history",
    "statin_rx", "antihypertensive_rx", "metformin_rx", "insulin_rx",
    "inhaler_rx", "antidepressant_rx", "opioid_rx", "steroid_rx",
    "hospital_admissions", "outpatient_visits", "procedure_count",
    "medication_count", "missingness_burden",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic_biobank(n, seed):
    rng = np.random.default_rng(seed)
    age = rng.normal(58, 12, n).clip(18, 90)
    sex = rng.binomial(1, 0.52, n)
    z = rng.normal(0, 1, (n, len(LATENT_NAMES))).astype(np.float32)
    met, cv, renal, inflam, neuro, resp, psych, util = z.T

    data = {}
    data["height_cm"] = 168 + 8 * (1 - sex) - 6 * sex + rng.normal(0, 6, n)
    data["bmi"] = 26 + 3.8 * met + 0.03 * (age - 55) + rng.normal(0, 2.2, n)
    data["waist_hip_ratio"] = 0.86 + 0.06 * met + 0.02 * cv + rng.normal(0, 0.035, n)
    data["systolic_bp"] = 122 + 8 * cv + 4 * met + 0.22 * (age - 55) + rng.normal(0, 8, n)
    data["diastolic_bp"] = 76 + 4 * cv + 2 * met + rng.normal(0, 5, n)
    data["ldl"] = 118 + 18 * cv + 7 * met - 10 * sigmoid(cv + util) + rng.normal(0, 18, n)
    data["hdl"] = 55 - 7 * met - 3 * resp + rng.normal(0, 8, n)
    data["triglycerides"] = 135 + 42 * met + 16 * cv + rng.normal(0, 35, n)
    data["hba1c"] = 5.3 + 0.45 * met + 0.12 * age / 10 + rng.normal(0, 0.28, n)
    data["fasting_glucose"] = 92 + 13 * met + rng.normal(0, 9, n)
    data["creatinine"] = 0.85 + 0.18 * renal + 0.05 * cv + rng.normal(0, 0.09, n)
    data["egfr"] = 92 - 11 * renal - 0.45 * (age - 55) + rng.normal(0, 7, n)
    data["crp"] = np.exp(0.3 + 0.45 * inflam + 0.25 * met + rng.normal(0, 0.35, n))
    data["alt"] = 22 + 7 * met + 4 * inflam + rng.normal(0, 6, n)
    data["ast"] = 21 + 4 * met + 4 * inflam + rng.normal(0, 5, n)
    data["ggt"] = 28 + 12 * met + 7 * resp + rng.normal(0, 10, n)
    data["fev1"] = 3.1 - 0.25 * resp - 0.015 * (age - 55) + rng.normal(0, 0.35, n)
    data["pack_years"] = np.maximum(0, 8 + 10 * resp + 3 * psych + rng.normal(0, 8, n))
    data["sleep_hours"] = 7.1 - 0.35 * psych - 0.12 * neuro + rng.normal(0, 0.7, n)
    data["cognitive_score"] = 0.2 - 0.7 * neuro - 0.02 * (age - 55) + rng.normal(0, 0.55, n)

    probs = {
        "diabetes_dx": -1.7 + 1.25 * met + 0.25 * age / 10,
        "hypertension_dx": -1.1 + 0.9 * cv + 0.5 * met + 0.25 * age / 10,
        "cad_dx": -2.4 + 1.05 * cv + 0.35 * met + 0.35 * age / 10,
        "heart_failure_dx": -3.0 + 0.9 * cv + 0.5 * renal + 0.35 * age / 10,
        "ckd_dx": -2.5 + 1.35 * renal + 0.3 * cv + 0.25 * age / 10,
        "copd_dx": -2.6 + 1.35 * resp + 0.2 * age / 10,
        "asthma_dx": -2.0 + 0.75 * resp + 0.35 * inflam,
        "alzheimers_dx": -4.0 + 1.2 * neuro + 0.55 * age / 10,
        "depression_dx": -1.5 + 1.15 * psych + 0.15 * inflam,
        "anxiety_dx": -1.4 + 1.05 * psych,
        "autoimmune_dx": -2.1 + 1.15 * inflam,
        "chronic_pain_dx": -1.7 + 0.8 * psych + 0.5 * inflam + 0.45 * util,
        "fall_history": -2.1 + 0.85 * neuro + 0.45 * util + 0.3 * age / 10,
    }
    for name, logit in probs.items():
        data[name] = rng.binomial(1, sigmoid(logit), n)

    rx_rules = {
        "statin_rx": -1.3 + 1.1 * data["cad_dx"] + 0.6 * cv + 0.25 * util,
        "antihypertensive_rx": -1.2 + 1.4 * data["hypertension_dx"] + 0.35 * util,
        "metformin_rx": -2.0 + 1.7 * data["diabetes_dx"] + 0.25 * met,
        "insulin_rx": -3.0 + 1.5 * data["diabetes_dx"] + 0.45 * util,
        "inhaler_rx": -1.8 + 1.2 * data["copd_dx"] + 0.8 * data["asthma_dx"],
        "antidepressant_rx": -1.7 + 1.45 * data["depression_dx"] + 0.55 * data["anxiety_dx"],
        "opioid_rx": -2.6 + 1.2 * data["chronic_pain_dx"] + 0.5 * util,
        "steroid_rx": -2.4 + 1.0 * data["autoimmune_dx"] + 0.45 * data["asthma_dx"],
    }
    for name, logit in rx_rules.items():
        data[name] = rng.binomial(1, sigmoid(logit), n)

    data["hospital_admissions"] = rng.poisson(np.exp(-0.25 + 0.45 * util + 0.25 * cv + 0.25 * neuro))
    data["outpatient_visits"] = rng.poisson(np.exp(1.4 + 0.35 * util + 0.15 * psych + 0.1 * inflam))
    data["procedure_count"] = rng.poisson(np.exp(0.8 + 0.35 * util + 0.2 * cv))
    med_cols = [c for c in data if c.endswith("_rx")]
    data["medication_count"] = np.sum([data[c] for c in med_cols], axis=0) + rng.poisson(np.exp(0.2 + 0.25 * util))
    data["missingness_burden"] = np.maximum(0, rng.normal(0.1 + 0.08 * util - 0.04 * psych, 0.06, n))

    x_raw = np.column_stack([data[f] for f in FIELDS]).astype(np.float32)
    mean = x_raw.mean(axis=0, keepdims=True)
    std = x_raw.std(axis=0, keepdims=True) + 1e-6
    x = (x_raw - mean) / std
    return x, x_raw, z, age, sex


def train_sparse_autoencoder(x, hidden, steps, batch_size, lr, l1, seed):
    rng = np.random.default_rng(seed)
    n, dim = x.shape
    w_enc = rng.normal(0, 0.05, (dim, hidden)).astype(np.float32)
    b_enc = np.zeros(hidden, dtype=np.float32)
    w_dec = rng.normal(0, 0.05, (hidden, dim)).astype(np.float32)
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


def corrcoef(a, b):
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    return (a.T @ b) / ((len(a) - 1) * (a.std(axis=0)[:, None] + 1e-8) * (b.std(axis=0)[None, :] + 1e-8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=8000)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.018)
    p.add_argument("--l1", type=float, default=0.003)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    x, x_raw, latent, age, sex = make_synthetic_biobank(args.n_samples, args.seed)
    model = train_sparse_autoencoder(x, args.hidden, args.steps, args.batch_size, args.lr, args.l1, args.seed)
    h = np.maximum(x @ model["w_enc"] + model["b_enc"], 0)
    latent_corr = corrcoef(h, latent)
    field_corr = corrcoef(h, x)
    best_latent = np.argmax(np.abs(latent_corr), axis=1)
    best_score = latent_corr[np.arange(args.hidden), best_latent]
    active_rate = (h > 1e-3).mean(axis=0)
    ranking = np.argsort(np.abs(best_score))[::-1]

    cards = []
    for unit in ranking[:16]:
        top_fields = np.argsort(np.abs(field_corr[unit]))[::-1][:8]
        cards.append({
            "unit": int(unit),
            "latent": LATENT_NAMES[int(best_latent[unit])],
            "latent_correlation": float(best_score[unit]),
            "active_rate": float(active_rate[unit]),
            "top_fields": [
                {"field": FIELDS[int(i)], "correlation": float(field_corr[unit, i])}
                for i in top_fields
            ],
        })

    plt.figure(figsize=(7, 4))
    plt.plot(model["losses"], color="#19535f")
    plt.title("Sparse autoencoder training loss")
    plt.xlabel("checkpoint")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(out / "loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.imshow(latent_corr[ranking[:24]], aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.yticks(range(24), [f"unit {u}" for u in ranking[:24]])
    plt.xticks(range(len(LATENT_NAMES)), LATENT_NAMES, rotation=45, ha="right")
    plt.colorbar(label="Pearson r")
    plt.tight_layout()
    plt.savefig(out / "latent_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.imshow(field_corr[ranking[:16]], aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.yticks(range(16), [f"unit {u}" for u in ranking[:16]])
    plt.xticks(range(len(FIELDS)), FIELDS, rotation=90, fontsize=7)
    plt.colorbar(label="Pearson r")
    plt.tight_layout()
    plt.savefig(out / "field_heatmap.png", dpi=160)
    plt.close()

    summary = {
        "args": vars(args),
        "fields": FIELDS,
        "latent_names": LATENT_NAMES,
        "final_loss": model["losses"][-1],
        "mean_active_rate": float(active_rate.mean()),
        "cards": cards,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
