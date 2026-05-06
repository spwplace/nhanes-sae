# Research plan: Sparse autoencoders for biobank-scale phenomes

## Corrected framing

The target is not facial phenotype. It is the measured human phenome:

- anthropometrics: height, weight, BMI, waist/hip ratio
- labs: LDL, HDL, triglycerides, HbA1c, creatinine, CRP, liver enzymes
- vitals: blood pressure, heart rate, spirometry
- diagnoses: ICD/PheCode/EHR condition flags, e.g. Alzheimer disease,
  diabetes, coronary artery disease
- medications and procedures
- lifestyle and survey variables
- cognitive tests, mental health measures, sleep, pain, function
- imaging-derived phenotypes
- longitudinal event counts and ages of onset

The research question becomes:

> Can sparse autoencoders trained on participant-by-phenotype matrices recover
> clinically and biologically meaningful latent phenome factors, and are those
> factors more interpretable or more predictive than PCA, NMF, factor analysis,
> topic models, or graph/community baselines?

## Sources checked

- UK Biobank: roughly 500k volunteers with biological, health, lifestyle,
  genomic, imaging, and EHR-linked data; Data Showcase organizes the phenotype
  fields. Use requires approved access, and public display has sensitive
  phenotype restrictions.
  <https://www.ukbiobank.ac.uk/>
  <https://biobank.ndph.ox.ac.uk/showcase/>
- All of Us: U.S. research resource with surveys, physical measurements, EHRs,
  biospecimens/genomics, and wearable-device data. Access tiers restrict what
  can be exported or published.
  <https://www.researchallofus.org/data/>
  <https://support.researchallofus.org/hc/en-us/articles/4619151535508-Data-Types-and-Organization>
- FinnGen: large Finnish biobank and registry resource with endpoint definitions
  useful for disease phenotyping and PheWAS-style work.
  <https://www.finngen.fi/en>
- MIMIC-IV: public credentialed critical-care/hospital EHR dataset. It is not a
  population biobank, but it is useful for developing EHR phenotyping machinery.
  <https://physionet.org/content/mimiciv/>
- NHANES: public-use U.S. health and nutrition survey with examinations,
  questionnaires, body measures, and laboratory tests. It is smaller than a
  biobank but excellent for a real open-data pilot.
  <https://www.cdc.gov/nchs/nhanes/>
- PheWAS/phecodes: standard way to aggregate ICD codes into higher-level disease
  phenotypes for phenome-wide association studies.
  <https://phewascatalog.org/>

## Feasibility

This is very feasible technically. The hard parts are data access, cleaning,
missingness, temporal leakage, privacy, and interpretation discipline.

Sparse autoencoders are a good fit because biobank phenome vectors are
heterogeneous and compositional. A participant may have only a small subset of
possible disease, medication, lab, and lifestyle signals active at once, while
common latent axes such as metabolic syndrome, renal dysfunction, inflammatory
burden, frailty, cardiovascular disease, psychiatric comorbidity, medication
intensity, or healthcare utilization may recur across many people.

## What SAE features might correspond to

Likely discovered feature families:

- metabolic syndrome: BMI, HbA1c, triglycerides, hypertension, diabetes meds
- dyslipidemia/cardiovascular risk: LDL, statins, CAD, hypertension, smoking
- renal impairment: creatinine/eGFR, anemia, diabetes, hypertension
- inflammation/autoimmune burden: CRP, steroid exposure, autoimmune codes
- frailty/healthcare utilization: many procedures, admissions, falls, pain,
  polypharmacy
- neurocognitive decline: dementia codes, cognitive tests, age, medications,
  care utilization
- respiratory disease: spirometry, smoking, COPD/asthma codes, inhalers
- liver/metabolic alcohol axis: ALT/AST/GGT, alcohol survey fields, obesity
- mental health/sleep/pain cluster: depression/anxiety, insomnia, pain meds,
  survey measures
- data-process artifacts: site, assay batch, missingness pattern, EHR density

That last category matters. Some SAE units will be real biology; others will be
healthcare access, measurement frequency, cohort ascertainment, or batch effects.

## Dream-big experiment

1. **Synthetic validation**
   Generate biobank-like records from known latent disease/lifestyle/healthcare
   factors. Train SAEs and verify recovery against ground truth.

2. **Public/EHR development**
   Use MIMIC-IV or another credentialed EHR dataset to build robust ETL:
   diagnoses, labs, meds, procedures, time windows, missingness indicators, and
   leakage-safe train/test splits.

3. **Biobank matrix**
   For UK Biobank/All of Us, build multiple matrices:
   baseline-only, longitudinal-ever, longitudinal-counts, and age-of-onset.
   Keep quantitative and binary fields distinct before final normalization.

4. **Representation models**
   Compare:
   PCA, NMF, sparse PCA, topic models, variational autoencoders, sparse
   autoencoders, and supervised embeddings for specific outcomes.

5. **SAE training**
   Train Top-K or JumpReLU SAEs with 4x, 8x, and 16x expansion. Use feature
   normalization, missingness masks, and modality-specific loss weighting so
   common binary fields do not dominate continuous labs.

6. **Feature reports**
   For each unit:
   top enriched fields, positive/negative quantitative shifts, participant
   activation distribution, enrichment by ICD/PheCode chapter, top medications,
   missingness profile, demographic/site sensitivity, and outcome associations.

7. **Genomic bridge**
   Treat SAE units as derived quantitative phenotypes. Run GWAS/PheWAS-style
   association tests, estimate heritability, and compare genetic correlations
   with known diseases and biomarkers.

8. **Causal caution**
   Interpret features as statistical structure, not causes. Use longitudinal
   models and external validation before claiming disease trajectories.

## M2 Max local scale

Your laptop can handle meaningful prototypes:

- 100k to 500k participants by 1k to 10k phenotype columns if stored as
  float32/sparse arrays and streamed in minibatches.
- Start with a dense 2k-column matrix and a 4x SAE.
- Move to PyTorch MPS for larger runs.
- Cache normalized shards on disk; train one model at a time; export feature
  cards as static HTML.

## Guardrails

- Do not export participant-level rows from controlled datasets.
- Suppress small counts and sensitive phenotypes in public reports.
- Never interpret a unit as destiny or identity.
- Audit site, ancestry, age, sex, socioeconomic, and measurement-density effects.
- Keep labels descriptive: "LDL/statin/CAD axis" is better than "heart-risk
  person type."
- Pre-register which variables are allowed in prediction tasks to avoid leakage.
