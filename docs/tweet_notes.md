# Tweet notes: phenome SAE mining pass

## Revised framing after blocking ablation

The hand-defined clinical block SAE is not the strongest claim. It injects a
large analyst prior. The more defensible result is an ablation:

- Same 95 curated NHANES phenotype columns.
- Same PyTorch SAE architecture: AdamW, decoder normalization, TopK sparse
  activations.
- Compare monolithic, hand clinical blocks, NHANES prefix blocks,
  data-derived correlation blocks, and matched random blocks.

Current full ablation:

- Correlation-derived blocks: held-out silhouette 0.363, archetype stability 0.884.
- Monolithic SAE: held-out silhouette 0.136, but low KMeans seed stability 0.078.
- Hand clinical blocks: held-out silhouette 0.064, archetype stability 0.860.
- Matched random blocks: held-out silhouette mostly 0.04-0.09.
- NHANES prefix blocks: held-out silhouette 0.090, archetype stability 0.987.

Tweetable conclusion:

> The hand-made clinical block split does *not* get to be the discovery. When I
> ablate the blocking rule, data-derived correlation blocks beat clinical blocks
> and random matched blocks on held-out separation. So the better experiment is:
> learn the views, then interpret them clinically.

Best current image:

- `outputs/nhanes_blocking_ablation_adamw_topk/blocking_ablation_composite.png`

Short reply to methodology critique:

> Fair critique. The clinical blocks are a strong prior. I reran this as a
> blocking ablation: monolithic vs clinical blocks vs NHANES-prefix blocks vs
> correlation-derived blocks vs matched random blocks, all with the same AdamW
> TopK SAE. Clinical blocks do not win; correlation-derived blocks do. That is
> the more defensible direction.

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
