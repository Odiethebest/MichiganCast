# Project Background and Modeling Rationale

## A Forecasting Problem Grounded in Place

MichiganCast started with a simple question in a specific geography: can we forecast near-term precipitation events around Lake Michigan using the signals that forecasters actually watch, not just one data source in isolation.

Traverse City is a useful setting for this question. Lake-effect dynamics can shift local conditions quickly, and the same day may contain long stretches of no-rain punctuated by short, consequential events. In practice, that creates a hard modeling environment: sparse positives, noisy observations, and timing sensitivity.

## The Notebook Phase and What It Exposed

The early work happened in a single exploratory notebook:

- `notebooks/legacy/03_training/multimodal_rainfall_training_pipeline.ipynb`

That notebook did more than prototype models. It exposed operational facts that later became design constraints:

- Satellite sequences contained real structure, but quality was uneven.
- Daytime windows produced more stable visual signal than nighttime segments.
- Encoded 1D satellite fields required careful decoding and verification.
- Corrupted or missing image ranges could silently poison training.
- Coastline padding artifacts could leak false intensity into resized inputs.
- Rain classes were imbalanced enough to make plain accuracy misleading.

This was the turning point. The project moved from exploratory convenience to explicit data contracts and reproducible pipelines.

## From Detection to True Forecasting

A second turning point was task definition. Early classification-style framing was replaced with a leakage-safe forecasting setup:

- Features are built from history ending at anchor time `t`.
- Targets are defined at a future timestamp `y_time = t + horizon`.
- Data is split by time ranges, not random sampling.

This change made evaluation much harder but much more honest. It aligned the project with how forecasting systems are used outside notebooks.

## How the Engineering Stack Formed

Once data realities and task framing were clear, the system architecture followed naturally:

1. `src.data.contracts` to enforce schema and range assumptions before training.
2. `src.data.validate` to catch continuity, alignment, and image integrity issues.
3. `src.data.clean` to produce deterministic cleaned datasets.
4. `src.features.labeling` to enforce temporal ordering and prevent leakage.
5. `src.data.split` to apply stable year-based train, validation, and test partitions.

In other words, model work was intentionally downstream of data correctness.

## Model Choice as a Consequence of the Data

The modeling path was not chosen first and justified later. It emerged from the structure of the problem.

Baselines remained essential:

- Logistic Regression
- Random Forest
- Gradient Boosting

They set a transparent reference and exposed what tabular signals can and cannot do alone.

Multimodal temporal learning then became the logical next step:

- A visual sequence branch captures cloud motion and spatial evolution.
- A meteorological sequence branch captures tabular temporal dynamics.
- A fusion head produces a single event probability.

This is the rationale behind the current ConvLSTM style plus LSTM style architecture family in `src/models/multimodal`.

## What Counts as Progress in This Project

Given class imbalance and operational risk, progress is measured by event-aware behavior, not headline accuracy. Evaluation emphasizes:

- PR AUC
- F1
- Recall
- Recall at precision threshold
- Brier score

The project also treats reliability as part of model quality:

- Imbalance strategy comparisons (`weighted_bce`, `focal`, threshold moving)
- Stability checks across repeated runs with fixed seeds and early stopping
- Export verification through standalone TorchScript inference
- API-level monitoring of latency, score distribution, and input statistics

## The Current State

MichiganCast now operates as a reproducible local system rather than a notebook experiment:

- modular `src/` packages
- CLI-first execution
- standardized artifacts under `artifacts/`
- exportable model outputs
- lightweight inference service with monitoring hooks

The scope is intentionally bounded. The project does not currently target distributed training, managed cloud orchestration, or streaming ingestion. The focus is a clear, testable, end-to-end forecasting workflow that can be reasoned about technically and improved iteratively.
