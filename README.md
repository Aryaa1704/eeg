# EEG Motor-Imagery Classification

EEG motor-imagery classification project based on the **BCI Competition IV
Dataset 2a**. The project studies whether using an earlier training time
window that includes visual-presentation information changes classification
performance.

The repository contains two separate analyses:

1. **FBCSP operational comparison** — the original visual-cue versus
   without-visual-cue comparison.
2. **Paper-style sliding-window analysis** — a separate implementation inspired
   by the referenced IEEE Access paper.

The two analyses use separate scripts and separate result files. Their numbers
must not be mixed.

---

## Dataset

- **Dataset:** BCI Competition IV 2a
- **Subjects:** A01–A09
- **Classes:** Left Hand, Right Hand, Feet, Tongue
- **Sampling frequency:** 250 Hz
- **Training files:** `T.gdf`
- **Evaluation files:** `E.gdf` and the corresponding `E.mat` true labels

The dataset files are intentionally excluded from Git using `.gitignore`.

---

## Processing pipeline

1. Load GDF recordings and restore the original event codes.
2. Remove EOG channels and retain EEG channels.
3. Reject artifact-associated trials.
4. Apply an 8–30 Hz third-order Butterworth band-pass filter.
5. Extract the selected time window from each trial.
6. Extract CSP features.
7. Train an LDA classifier.
8. Evaluate on the clean evaluation trials.
9. Save subject-wise accuracy, statistics, graphs, and additional metrics.

### Important implementation fixes

- Updated MNE to resolve the NumPy/MNE compatibility problem.
- Corrected MNE annotation-to-event-code conversion in
  [`data_loader.py`](./data_loader.py).
- Corrected the CSP-to-classifier data flow in
  [`features_csp.py`](./features_csp.py).
- Added the FBCSP comparison pipeline in
  [`visual_comparison.py`](./visual_comparison.py).
- Added the paper-style sliding-window pipeline in
  [`paper_replication.py`](./paper_replication.py).

---

## Analysis 1: FBCSP operational comparison

### What it compares

This analysis uses four filter-bank bands:

- 8–12 Hz
- 12–16 Hz
- 16–24 Hz
- 24–30 Hz

Four CSP components are extracted per band and combined before shrinkage LDA
classification.

| Condition | Training window | Evaluation window |
|---|---:|---:|
| Without visual cue | 3.0–6.0 s | 3.0–6.0 s |
| With visual cue / early window | 2.0–5.0 s | 3.0–6.0 s |

Both conditions use the same clean evaluation trials. This is the original
operational comparison selected for the project.

### Main results

| Subject | Without visual cue | With visual cue | Improvement |
|---|---:|---:|---:|
| A01 | 52.16% | 70.86% | +18.71 pp |
| A02 | 57.86% | 63.21% | +5.36 pp |
| A03 | 74.81% | 82.22% | +7.41 pp |
| A04 | 34.67% | 43.11% | +8.44 pp |
| A05 | 29.67% | 37.73% | +8.06 pp |
| A06 | 39.62% | 45.75% | +6.13 pp |
| A07 | 20.44% | 72.26% | +51.82 pp |
| A08 | 58.21% | 63.06% | +4.85 pp |
| A09 | 16.48% | 46.36% | +29.89 pp |
| **Mean** | **42.66%** | **58.29%** | **+15.63 pp** |

- **Paired t-statistic:** 2.9572
- **Paired p-value:** 0.018224

### Additional classification metrics

These metrics are calculated from the actual four-class predictions. Sensitivity
and recall are macro-averaged one-vs-rest recall; specificity is macro-averaged
one-vs-rest specificity; F1 is macro F1; and kappa is Cohen's kappa.

| Condition | Accuracy | Sensitivity / Recall | Specificity | F1 | Kappa |
|---|---:|---:|---:|---:|---:|
| Without visual cue | 42.66% | 42.83% | 80.91% | 40.57% | 0.2364 |
| With visual cue | 58.29% | 58.30% | 86.09% | 56.16% | 0.4434 |

### FBCSP files

- [Main comparison table](./results/visual_comparison_final.csv)
- [Subject-wise metrics](./results/classification_metrics_all.csv)
- [Metrics summary](./results/classification_metrics_summary.csv)
- [Confusion-matrix counts](./results/confusion_matrices_aggregate.csv)
- [Aggregate confusion-matrix image](./results/confusion_matrices_aggregate.png)
- [Original comparison graph](./results/visual_comparison.png)

---

## Analysis 2: Paper-style sliding-window analysis

### Why this analysis was added

The referenced paper evaluates multiple three-second time windows and compares
their accuracy with a 3.0–6.0 s motor-imagery baseline. This analysis follows
that idea separately, without overwriting the FBCSP results above.

### Method

- Three-second training windows
- Start times: 2.0, 2.1, 2.2, ..., 3.5 s
- Baseline: 3.0–6.0 s
- Frequency range: 8–30 Hz
- Model: CSP + LDA
- Multiclass strategy: one-vs-one
- Subjects: A01–A09

For each subject, the window with the highest accuracy is selected as that
subject's best window.

### Paper-style subject-wise table

| Subject | Best window | Baseline 3–6 s | Best accuracy | Improvement |
|---|---|---:|---:|---:|
| A01 | 2.0–5.0 s | 38.49% | 72.66% | +34.17 pp |
| A02 | 2.1–5.1 s | 46.07% | 62.86% | +16.79 pp |
| A03 | 2.0–5.0 s | 64.07% | 82.22% | +18.15 pp |
| A04 | 2.1–5.1 s | 28.89% | 52.00% | +23.11 pp |
| A05 | 2.4–5.4 s | 26.74% | 37.36% | +10.62 pp |
| A06 | 2.5–5.5 s | 41.04% | 44.81% | +3.77 pp |
| A07 | 2.1–5.1 s | 18.98% | 51.82% | +32.85 pp |
| A08 | 2.0–5.0 s | 50.00% | 72.39% | +22.39 pp |
| A09 | 2.2–5.2 s | 25.29% | 29.89% | +4.60 pp |
| **Mean** | — | **37.73%** | **56.22%** | **+18.49 pp** |

### Paper-style summary

- **Baseline mean accuracy:** 37.73%
- **Best-window mean accuracy:** 56.22%
- **Mean improvement:** +18.49 percentage points
- **Paired t-statistic:** 5.0567
- **Paired p-value:** 0.000981

### Paper-style files

- [Paper replication script](./paper_replication.py)
- [Summary statistics](./results/paper_replication/summary.csv)
- [Subject-wise best-window table](./results/paper_replication/subject_best_windows.csv)
- [All tested windows](./results/paper_replication/all_window_accuracy.csv)
- [Mean accuracy by window](./results/paper_replication/mean_by_window.csv)
- [Accuracy-difference graph](./results/paper_replication/accuracy_difference_by_window.png)
- [Timing scheme](./results/paper_replication/timing_scheme_2a.png)
- [Subject-wise difference graph](./results/paper_replication/subject_window_differences.png)
- [Mean accuracy graph](./results/paper_replication/mean_accuracy_by_window.png)
- [One-second segment summary](./results/paper_replication/segment_accuracy_summary.csv)
- [One-second segment graph](./results/paper_replication/segment_accuracy_comparison.png)
- [T-test table](./results/paper_replication/t_test_table.csv)

### Graph interpretation

The graph shows the accuracy difference from the 3.0–6.0 s baseline for each
window start:

- Positive values mean improvement over baseline.
- Zero means the baseline window itself.
- Negative values mean lower accuracy than baseline.
- The x-axis is the start of each three-second training window.

---

## How the two tables differ

| Feature | FBCSP operational table | Paper-style table |
|---|---|---|
| Purpose | Current visual-cue comparison | Paper-inspired replication |
| Main script | `visual_comparison.py` | `paper_replication.py` |
| Model | FBCSP + shrinkage LDA | CSP + LDA + one-vs-one |
| Windows compared | Fixed 3–6 s vs 2–5 s | All 3-second windows from 2.0–3.5 s |
| Reported result | 42.66% → 58.29% | 37.73% → 56.22% |
| Output location | `results/` | `results/paper_replication/` |

The tables are both valid for their stated methods, but they are not the same
experiment and should not be combined into one result.

---

## Relation to the referenced paper

Referenced paper:

> K. Suemitsu and I. Nambu, “Effects of Data Including Visual Presentation
> and Rest Time on Classification of Motor Imagery of Using Brain-Computer
> Interface Competition Datasets,” IEEE Access, 2023.

- [DOI: 10.1109/ACCESS.2023.3285236](https://doi.org/10.1109/ACCESS.2023.3285236)

The paper primarily studies how selected time segments affect classification
accuracy and uses a 3.0–6.0 s baseline for BCI Competition IV 2a. The
paper-style pipeline follows its sliding-window idea. It is not an exact
numerical replication because the implementation, preprocessing details, and
classifier configuration are not identical.

---

## Important interpretation and limitation

The current BCI Competition IV 2a recordings do not contain separately
recorded, matched cue-absent trials. Therefore:

- The original “with visual cue” label represents the early-window condition
  used in the operational comparison.
- It should not be interpreted as proof of a literal cue-present versus
  cue-removed experiment.
- The paper-style analysis is a time-window replication of the paper's idea.

---

## Reports and downloadable material

- [Final technical report](./FINAL_REPORT.md)
- [Paper-style report](./PAPER_REPLICATION_REPORT.md)
- [Complete paper-style bundle](./paper_replication_materials.zip)

The ZIP contains the paper-style report, summary CSV, subject-wise table,
complete window results, mean-by-window CSV, and the graph.

---

## Reproducibility

Run the original comparison:

```bash
python visual_comparison.py
```

Run the paper-style sliding-window analysis:

```bash
python paper_replication.py
```

The paper-style outputs are written to:

```text
results/paper_replication/
```

To synchronize a Colab checkout:

```python
%cd /content/eeg
!git pull origin main
```

---

## Project status

- Dataset loading and event correction: complete
- CSP and FBCSP pipelines: complete
- Operational comparison: complete
- Confusion matrices and additional metrics: complete
- Paper-style sliding-window analysis: complete
- Reports, graphs, tables, and ZIP bundle: complete
- Matched cue-present/cue-absent experiment: not available in the current dataset
