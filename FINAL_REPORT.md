# Motor-Imagery EEG Classification with an Operational Visual-Cue Proxy

## 1. Objective

This study compares motor-imagery EEG classification performance using two
three-second training windows:

- **Baseline / without-cue proxy:** 3.0–6.0 s after the trial marker
- **Early-window / with-cue proxy:** 2.0–5.0 s after the trial marker

The purpose is to evaluate whether an earlier time window produces a measurable
change in classification accuracy.

> **Important interpretation:** BCI Competition IV 2a does not provide matched
> cue-present and cue-absent recordings for this analysis. Therefore, the
> comparison is an operational time-window proxy and must not be reported as a
> literal visual-cue removal experiment.

## 2. Dataset

The analysis uses the BCI Competition IV 2a dataset:

- Subjects: A01–A09
- Sampling frequency: 250 Hz
- Classes: left hand, right hand, feet, and tongue
- Training recordings: `T.gdf`
- Evaluation recordings: `E.gdf` with the supplied true labels

The evaluation labels are loaded from the corresponding `E.mat` files. Artifact
trials are excluded using the same peak-to-peak rejection rule for both
conditions.

## 3. Preprocessing and model

1. EEG channels are retained while EOG channels are removed.
2. Four third-order Butterworth band-pass filters are applied:
   8–12, 12–16, 16–24, and 24–30 Hz.
3. Each filtered signal is epoched for the selected three-second window and
   downsampled by taking every second sample.
4. Four-component regularized CSP features are extracted independently in each
   frequency band.
5. Features from all bands are concatenated.
6. A shrinkage LDA classifier is trained on the training recording.
7. Both models are evaluated on the same clean evaluation trials and the same
   3.0–6.0 s evaluation window.

The implementation is available in
[`visual_comparison.py`](./visual_comparison.py).

## 4. Results

| Subject | Baseline 3–6 s | Early window 2–5 s | Improvement (percentage points) |
|---|---:|---:|---:|
| A01 | 52.16% | 70.86% | +18.71 |
| A02 | 57.86% | 63.21% | +5.36 |
| A03 | 74.81% | 82.22% | +7.41 |
| A04 | 34.67% | 43.11% | +8.44 |
| A05 | 29.67% | 37.73% | +8.06 |
| A06 | 39.62% | 45.75% | +6.13 |
| A07 | 20.44% | 72.26% | +51.82 |
| A08 | 58.21% | 63.06% | +4.85 |
| A09 | 16.48% | 46.36% | +29.89 |
| **Mean** | **42.66%** | **58.29%** | **+15.63** |

A paired t-test across subjects gives:

- **t-statistic:** 2.9572
- **p-value:** 0.018224
- **Significance level:** 0.05

The early-window condition therefore achieved a statistically significant
subject-level improvement over the baseline window in this operational
comparison.

## 5. Confusion-matrix metrics

The predictions from both conditions were used to create four-class confusion
matrices for every subject. Sensitivity and recall are equivalent here because
they are calculated as macro-averaged one-vs-rest recall across the four
classes. Specificity is also macro-averaged one-vs-rest specificity. F1 is the
macro F1 score, and kappa is Cohen's kappa.

| Condition | Accuracy | Sensitivity / Recall | Specificity | F1 | Kappa |
|---|---:|---:|---:|---:|---:|
| Without visual cue | 42.66% | 42.83% | 80.91% | 40.57% | 0.2364 |
| With visual cue | 58.29% | 58.30% | 86.09% | 56.16% | 0.4434 |

These are means across the nine subject-level scores. The aggregate
four-class confusion matrices are shown in
[`results/confusion_matrices_aggregate.png`](./results/confusion_matrices_aggregate.png).
Machine-readable outputs are available in
[`results/classification_metrics_all.csv`](./results/classification_metrics_all.csv),
[`results/classification_metrics_summary.csv`](./results/classification_metrics_summary.csv),
and [`results/confusion_matrices_aggregate.csv`](./results/confusion_matrices_aggregate.csv).

## 6. Relation to the referenced paper

The study is conceptually related to:

> Suemitsu, K. and Nambu, I., “Effects of Data Including Visual Presentation
> and Rest Time on Classification of Motor Imagery of Using Brain-Computer
> Interface Competition Datasets,” *IEEE Access*, 2023.
> DOI: [10.1109/ACCESS.2023.3285236](https://doi.org/10.1109/ACCESS.2023.3285236)

Both studies investigate how visual presentation, rest/timing, and selected
EEG segments affect motor-imagery classification. However, the present work
uses a different FBCSP + LDA implementation and an operational window proxy.
Thus, the percentages above should not be presented as an exact replication
or a direct numerical reproduction of the paper.

## 7. Limitations and future work

- The current dataset has no separate matched no-cue recording.
- The label “with visual cue” refers to the earlier 2–5 s operational window,
  not proof that the visual cue itself caused the difference.
- The p-value is one overall paired test across nine subjects; the repeated
  p-value in the CSV is not a separate subject-level p-value.
- A literal cue-present versus cue-absent claim requires a new matched dataset
  containing both recording conditions.

The current main result is saved in
[`results/visual_comparison_final.csv`](./results/visual_comparison_final.csv).
The advanced subject-wise FBCSP results are retained separately as supporting
results and should not replace this main comparison.
