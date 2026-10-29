# eeg

## Project overview

This repository contains EEG motor-imagery classification analyses using the
BCI Competition IV 2a dataset (subjects A01–A09; four classes: left hand,
right hand, feet, and tongue).

## Completed work

- Corrected MNE/GDF event-code decoding and verified the expected trials.
- Fixed the CSP feature/classifier pipeline.
- Implemented FBCSP + shrinkage LDA analysis.
- Added the current visual-cue versus without-visual-cue proxy comparison.
- Added subject-wise confusion matrices and:
  accuracy, sensitivity/recall, specificity, macro F1, and Cohen's kappa.
- Implemented a separate paper-style CSP-LDA + one-vs-one sliding-window
  analysis without overwriting the earlier results.

## Results sets

### 1. Current FBCSP proxy comparison

This is the earlier operational comparison:

- Without visual cue: **42.66%**
- With visual cue: **58.29%**
- Improvement: **+15.63 percentage points**
- Paired p-value: **0.018224**

Files:

- [`visual_comparison.py`](./visual_comparison.py)
- [`results/visual_comparison_final.csv`](./results/visual_comparison_final.csv)
- [`results/classification_metrics_summary.csv`](./results/classification_metrics_summary.csv)
- [`results/confusion_matrices_aggregate.png`](./results/confusion_matrices_aggregate.png)

### 2. Paper-style sliding-window analysis

This separate analysis follows the referenced paper's time-window approach:
three-second windows starting every 0.1 s from 2.0 to 3.5 s, with 3.0–6.0 s
as the baseline. It uses 8–30 Hz filtering, CSP, LDA, and one-vs-one
four-class classification.

- Baseline mean accuracy: **37.73%**
- Best-window mean accuracy: **56.22%**
- Improvement: **+18.49 percentage points**
- Paired t-statistic: **5.0567**
- Paired p-value: **0.000981**

The paper-style analysis is implemented separately in
[`paper_replication.py`](./paper_replication.py), with outputs in
[`results/paper_replication/`](./results/paper_replication/):

- [`summary.csv`](./results/paper_replication/summary.csv)
- [`subject_best_windows.csv`](./results/paper_replication/subject_best_windows.csv)
- [`all_window_accuracy.csv`](./results/paper_replication/all_window_accuracy.csv)
- [`mean_by_window.csv`](./results/paper_replication/mean_by_window.csv)
- [`accuracy_difference_by_window.png`](./results/paper_replication/accuracy_difference_by_window.png)

## Reports and download bundle

- [`FINAL_REPORT.md`](./FINAL_REPORT.md): concise technical report
- [`PAPER_REPLICATION_REPORT.md`](./PAPER_REPLICATION_REPORT.md): paper-style
  results report
- [`paper_replication_materials.zip`](./paper_replication_materials.zip):
  report, tables, summary, and graph for download

## Interpretation note

The current BCI 2a recordings do not contain matched, separately recorded
cue-absent trials. The “with visual cue” label in the earlier comparison is
therefore an operational early-window label. The paper-style analysis is a
time-window replication of the paper's analysis idea, not an exact numerical
replication.

## Colab

To synchronize a Colab checkout:

```python
%cd /content/eeg
!git pull origin main
```
