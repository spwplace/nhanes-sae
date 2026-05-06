# Tweet notes: phenome SAE mining pass

Suggested framing:

> Tried the "discover sigma/skibidi/67 from first principles" joke literally,
> but on public NHANES phenomes instead of internet slang.
>
> Train a sparse autoencoder on labs + body measures + questionnaire variables,
> UMAP the participant activation vectors, then color by derived anchors.
>
> The result is not crisp "types." It is a phenotype geography: smooth regions
> for adiposity, mood/sleep, blood pressure, lipids/glucose, renal/liver labs,
> medication/utilization, and blood counts.

Useful numbers:

- Data: CDC public-use NHANES 2017-2018 and 2021-2023.
- Adult participants after cleaning: 23,969.
- Clean mining matrix: 215 non-demographic, non-missingness phenotype features.
- SAE: 256 hidden units, 249 live units.
- UMAP sample: 18,000 participants.
- KMeans regions: 9.
- Silhouette: 0.062, so describe these as soft regions, not hard natural kinds.

Best images:

- `outputs/nhanes_mining_clean/tweetable_four_panel.png`
- `outputs/nhanes_mining_clean/cluster_archetypes.png`
- `outputs/nhanes_mining_clean/umap_clusters.png`

One-liner:

> Phenome memes are real-ish, but they look less like discrete archetypes and
> more like overlapping health gradients.

Caveat:

NHANES variables are public-use survey/exam fields, not a full biobank/EHR.
Some SAE features still capture measurement process, coding conventions, or
survey design artifacts. That is part of the result, not just noise.
