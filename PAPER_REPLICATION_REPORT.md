# Paper-Style EEG Sliding-Window Results

## Final result

This analysis follows the referenced paper's time-window comparison idea:
three-second training windows were tested at 0.1-second increments, with the
3.0–6.0 s motor-imagery window used as the baseline.

- Dataset: BCI Competition IV 2a
- Subjects: A01–A09
- Model: 8–30 Hz band-pass, CSP, LDA with one-vs-one multiclass classification
- Window starts tested: 2.0–3.5 s
- Window width: 3 s
- Window step: 0.1 s

## Summary

| Measure | Result |
|---|---:|
| Baseline mean accuracy (3.0–6.0 s) | 37.73% |
| Best-window mean accuracy | 56.22% |
| Mean improvement | +18.49 percentage points |
| Paired t-statistic | 5.0567 |
| Paired p-value | 0.000981 |
| Number of subjects | 9 |

## Subject-wise best windows

| Subject | Best window | Baseline | Best accuracy | Improvement |
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

## Interpretation

The best windows generally begin during the early visual-presentation period.
The best-window condition is higher than the 3.0–6.0 s baseline in all nine
subjects. The paired test indicates a statistically significant difference
across subjects at the 0.05 level.

This is a paper-style time-window comparison, not a separate recording with
visual cues removed. The earlier window is used as the visual-presentation
condition for comparison.

## Paper-style figures and additional segment analysis

The applicable BCI 2a CSP-LDA figures are available as:

- `timing_scheme_2a.png`: trial timing and sliding-window scheme
- `subject_window_differences.png`: subject-wise accuracy differences
- `accuracy_difference_by_window.png`: mean difference from baseline
- `mean_accuracy_by_window.png`: mean accuracy at each window start
- `segment_accuracy_comparison.png`: one-second segment comparison

The one-second training-segment means are:

| Training segment | Mean accuracy |
|---|---:|
| Visual presentation (2–3 s) | 54.28% |
| Motor imagery (3–4 s) | 53.54% |
| Rest (6–7 s) | 17.62% |

`segment_accuracy_summary.csv` and `segment_accuracy.csv` contain the numeric
values. The paper's CNN architecture figures and BCI 2b figures are not
recreated because this project uses the BCI 2a CSP-LDA pipeline only. The
paper's temporal-profile/topography figures are exploratory signal-analysis
figures, not classification outputs; reproducing them would require electrode
location metadata that is not available in the current recordings.

## Included files

- `summary.csv`: overall statistics
- `subject_best_windows.csv`: final subject-wise table
- `all_window_accuracy.csv`: every tested window for every subject
- `mean_by_window.csv`: mean accuracy and difference for each window start
- `accuracy_difference_by_window.png`: paper-style accuracy-difference graph
- `timing_scheme_2a.png`: timing diagram
- `subject_window_differences.png`: subject-wise difference graph
- `mean_accuracy_by_window.png`: mean accuracy graph
- `segment_accuracy_summary.csv`: one-second segment summary
- `segment_accuracy_comparison.png`: one-second segment graph
- `t_test_table.csv`: t-test table
