# Phenome SAE biobank experiment

This repo is a research scaffold for asking:

> If we train sparse autoencoders on high-dimensional human phenome data, what
> do the learned features correspond to?

Here "phenome" means measured traits and health variables: LDL, height, BMI,
diagnosis codes, medication exposure, cognitive status, lab panels, vitals,
surveys, procedures, imaging-derived phenotypes, and longitudinal EHR events.

The default local experiment uses synthetic biobank-like tabular data with known
latent factors. That gives a privacy-safe validation target before touching
controlled-access resources such as UK Biobank or All of Us.

## Synthetic quickstart

```bash
python3 experiments/synthetic_biobank_sae.py --n-samples 8000 --steps 1200
python3 scripts/build_report.py
open site/index.html
```

Outputs are written to `outputs/` and the generated presentation page is
`site/index.html`.

## Real public-data quickstart

This uses CDC public-use NHANES files. Create the local venv once:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install pandas beautifulsoup4 lxml pyreadstat requests pyarrow matplotlib
```

Then download and train:

```bash
.venv/bin/python scripts/download_nhanes.py --cycles 2021-2023 2017-2018
.venv/bin/python experiments/nhanes_phenome_sae.py --steps 2200 --hidden 128 --l1 0.006
.venv/bin/python scripts/build_report.py
open site/index.html
```

The NHANES run writes raw XPT files under `data/nhanes/raw/`, the merged matrix
to `data/nhanes/processed/nhanes_phenome_raw.parquet`, and the report inputs to
`outputs/nhanes/`.

## Mining and UMAP visualization

After the NHANES data has been downloaded:

```bash
.venv/bin/python -m pip install scikit-learn umap-learn
.venv/bin/python experiments/mine_nhanes_clusters.py --hidden 256 --steps 4000 --l1 0.002 --out-dir outputs/nhanes_mining_clean
```

This writes tweetable maps and cluster summaries to
`outputs/nhanes_mining_clean/`.

## Scale-up idea

1. Build a participant-by-phenotype matrix from a biobank/EHR source.
2. Separate variable families: quantitative labs/vitals, binary diagnoses,
   medications, procedures, survey fields, imaging phenotypes, and genomics.
3. Train sparse autoencoders on normalized multimodal phenome vectors.
4. Interpret each sparse unit by enrichment over fields, ICD/PheCode groups,
   labs, meds, demographics, genetic PCs, and outcomes.
5. Compare SAE features to PCA/factor analysis/topic models and known clinical
   groupings.

See `docs/research_plan.md` for the longer plan with sources and guardrails.
