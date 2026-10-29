# EEG Project Chat Changelog

This file tracks the main decisions, changes, results, and commands completed
during this chat.

## 1. Initial environment and data fixes

- MNE was upgraded from 1.7.0 to 1.13.2 because of a NumPy compatibility
  problem.
- Verified environment:
  - NumPy: 2.5.4
  - MNE: 1.13.2
- Fixed GDF event decoding in `data_loader.py`.
  MNE-generated annotation IDs are converted back to the original event codes
  769–772 and artifact code 1023.
- Verified that the dataset contains the expected 288 trials for each
  training/evaluation recording.
- Fixed the CSP pipeline in `features_csp.py` so CSP transformation happens
  before classification.

## 2. Main EEG analysis

- Dataset: BCI Competition IV 2a.
- Subjects: A01–A09.
- Classes: left hand, right hand, feet, and tongue.
- Sampling frequency: 250 Hz.
- Frequency range: 8–30 Hz.
- Main model: FBCSP + shrinkage LDA.
- Frequency bands:
  - 8–12 Hz
  - 12–16 Hz
  - 16–24 Hz
  - 24–30 Hz
- Four CSP components are extracted per frequency band.
- Artifact rejection and evaluation-label loading are included in the pipeline.

## 3. Main comparison table

The selected comparison uses two three-second **training** windows:

- `without_visual_cue`: 3–6 seconds
- `with_visual_cue_early_window`: 2–5 seconds

Both models are tested on the same clean evaluation trials using the 3–6
second evaluation window.

Main results:

- Without visual cue: **42.66% mean accuracy**
- With visual cue: **58.29% mean accuracy**
- Improvement: **+15.63 percentage points**
- Paired t-test: **t = 2.9572**
- Paired p-value: **0.018224**

The teacher-facing comparison is called:

**Visual Cue Accuracy vs Without Visual Cue Accuracy**

The technical report records that the current BCI 2a recordings do not contain
matched cue-absent trials. Therefore, the implementation is a time-window
comparison used as the current visual-cue proxy.

## 4. Final report

Created:

- `FINAL_REPORT.md`

The report contains:

- Objective and dataset
- Preprocessing and model
- Subject-wise accuracy table
- Mean accuracy comparison
- Paired statistical test
- Confusion-matrix metrics
- Paper relation
- Limitations and future work

Updated:

- `README.md` now links to `FINAL_REPORT.md`.

## 5. Confusion matrices and additional metrics

The file `visual_comparison.py` was extended to save predictions-derived
classification outputs for both conditions.

Added metrics:

- Accuracy
- Macro sensitivity
- Macro recall
- Macro specificity
- Macro F1-score
- Cohen's kappa

For this four-class problem, macro sensitivity and macro recall are equivalent
because both are the average one-vs-rest recall across classes.

Current mean metrics:

| Condition | Accuracy | Sensitivity / Recall | Specificity | F1 | Kappa |
|---|---:|---:|---:|---:|---:|
| Without visual cue | 42.66% | 42.83% | 80.91% | 40.57% | 0.2364 |
| With visual cue | 58.29% | 58.30% | 86.09% | 56.16% | 0.4434 |

Generated files:

- `results/classification_metrics_all.csv`
- `results/classification_metrics_summary.csv`
- `results/confusion_matrices_all.csv`
- `results/confusion_matrices_aggregate.csv`
- `results/confusion_matrices_aggregate.png`

The aggregate matrix contains raw counts pooled across the nine subjects.
Large off-diagonal values are expected because the overall accuracy is
42.66%/58.29%, not 90–100%. Class labels are categorical, so an error does
not need to occur only in an adjacent class. A row-normalized percentage
matrix is more suitable for visual interpretation than pooled raw counts.

## 6. Paper comparison

Referenced paper:

Suemitsu and Nambu, “Effects of Data Including Visual Presentation and Rest
Time on Classification of Motor Imagery of Using Brain-Computer Interface
Competition Datasets,” IEEE Access, 2023.

DOI:

`10.1109/ACCESS.2023.3285236`

The paper primarily reports classification accuracy while comparing training
data segments involving visual presentation/rest information and motor imagery.
The current project uses a different FBCSP + LDA implementation and additionally
reports confusion-matrix metrics. The current results should therefore not be
described as an exact numerical replication of the paper.

## 7. Repository and Colab synchronization

Important commits:

- `7dac487` — added `FINAL_REPORT.md` and updated `README.md`
- `7d56c4e` — added classification metrics and confusion matrices

The repository is currently synchronized:

```text
main...origin/main
```

Codespace push command:

```bash
git push origin main
```

Colab synchronization commands:

```python
%cd /content/eeg
!git pull origin main
```

The repository was cloned successfully in Colab and reported:

```text
Already up to date.
```

## 8. Useful Colab display commands

```python
import pandas as pd
from IPython.display import Image, display

display(pd.read_csv("results/visual_comparison_final.csv"))
display(pd.read_csv("results/classification_metrics_summary.csv"))
display(pd.read_csv("results/classification_metrics_all.csv"))
display(pd.read_csv("results/confusion_matrices_aggregate.csv"))
display(Image(filename="results/confusion_matrices_aggregate.png"))
```

To read the report:

```python
!cat FINAL_REPORT.md
```

## 9. Current status

- Dataset loading: complete
- Original CSP pipeline: complete
- FBCSP pipeline: complete
- Visual-cue/without-visual-cue comparison table: complete
- Accuracy statistics: complete
- Confusion matrices: complete
- Sensitivity, specificity, recall, F1, and kappa: complete
- Final report: complete
- GitHub/Colab synchronization: complete
- Actual matched cue-present versus cue-absent experiment: not available in
  the current dataset

No new model training is required unless the teacher asks for a different
methodology or a genuine matched no-cue dataset becomes available.
