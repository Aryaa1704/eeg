import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.decoding import CSP
from scipy.stats import ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsOneClassifier

from config import SUBJECTS, BASELINE_END, BASELINE_START, N_CSP_COMPONENTS
from data_loader import load_eval, load_train
from preprocessing import extract_epochs


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "bci_mi_project" / "data" / "BCI_2a"
OUT_DIR = ROOT / "results" / "paper_replication"
WINDOW_WIDTH = 3.0
WINDOW_STARTS = np.round(np.arange(2.0, 3.5 + 0.001, 0.1), 1)


def bandpass(raw):
    filtered = raw.copy()
    filtered.filter(
        l_freq=8.0,
        h_freq=30.0,
        method="iir",
        iir_params={"order": 3, "ftype": "butter"},
        verbose=False,
    )
    return filtered


def make_epochs(raw, events, artifacts, start, end):
    X, y = extract_epochs(raw, events, artifacts, start, end)
    return X[:, :, ::2].astype(np.float32), y


def keep_clean(X):
    limit = np.percentile(np.ptp(X, axis=2).max(axis=1), 99)
    return np.ptp(X, axis=2).max(axis=1) <= limit


def fit_predict(X_train, y_train, X_test):
    csp = CSP(
        n_components=N_CSP_COMPONENTS,
        reg=None,
        log=True,
        norm_trace=False,
    )
    features_train = csp.fit_transform(X_train, y_train)
    features_test = csp.transform(X_test)
    model = OneVsOneClassifier(LinearDiscriminantAnalysis())
    model.fit(features_train, y_train)
    return model.predict(features_test)


def run_subject(subject):
    train = load_train(str(DATA_DIR / f"{subject}T.gdf"))
    evaluation = load_eval(
        str(DATA_DIR / f"{subject}E.gdf"),
        str(DATA_DIR / f"{subject}E.mat"),
    )
    train_raw = bandpass(train["raw"])
    eval_raw = bandpass(evaluation["raw"])

    eval_X, eval_y = make_epochs(
        eval_raw,
        evaluation["events"],
        evaluation["artifact_pos"],
        BASELINE_START,
        BASELINE_END,
    )
    eval_keep = keep_clean(eval_X)
    eval_X, eval_y = eval_X[eval_keep], eval_y[eval_keep]

    rows = []
    for start in WINDOW_STARTS:
        end = round(float(start + WINDOW_WIDTH), 1)
        train_X, train_y = make_epochs(
            train_raw,
            train["events"],
            train["artifact_pos"],
            float(start),
            end,
        )
        train_keep = keep_clean(train_X)
        prediction = fit_predict(
            train_X[train_keep],
            train_y[train_keep],
            eval_X,
        )
        rows.append({
            "subject": subject,
            "window_start": float(start),
            "window_end": end,
            "accuracy": float(np.mean(prediction == eval_y)),
            "accuracy_percent": float(np.mean(prediction == eval_y) * 100),
            "train_epochs": int(train_keep.sum()),
            "test_epochs": len(eval_y),
        })
    return rows


def main():
    subjects = os.getenv("PAPER_REPLICATION_SUBJECTS", ",".join(SUBJECTS)).split(",")
    rows = []
    for subject in subjects:
        print(f"[PAPER REPLICATION] {subject}")
        rows.extend(run_subject(subject))

    all_windows = pd.DataFrame(rows)
    baseline = all_windows[
        (all_windows["window_start"] == BASELINE_START)
        & (all_windows["window_end"] == BASELINE_END)
    ][["subject", "accuracy"]].rename(columns={"accuracy": "baseline_accuracy"})
    all_windows = all_windows.merge(baseline, on="subject", how="left")
    all_windows["difference"] = all_windows["accuracy"] - all_windows["baseline_accuracy"]
    all_windows["difference_percent_points"] = all_windows["difference"] * 100

    best_index = all_windows.groupby("subject")["accuracy"].idxmax()
    best = all_windows.loc[best_index].copy()
    best["best_window"] = (
        best["window_start"].map(lambda value: f"{value:.1f}")
        + "–"
        + best["window_end"].map(lambda value: f"{value:.1f}")
    )

    baseline_values = baseline["baseline_accuracy"].to_numpy()
    best_values = best["accuracy"].to_numpy()
    t_stat, p_value = ttest_rel(best_values, baseline_values)
    summary = pd.DataFrame([{
        "baseline_window": "3.0–6.0",
        "best_window_mean_start": best["window_start"].mean(),
        "baseline_mean_accuracy": baseline_values.mean(),
        "best_mean_accuracy": best_values.mean(),
        "mean_improvement_percent_points": (best_values - baseline_values).mean() * 100,
        "t_statistic": t_stat,
        "p_value": p_value,
        "n_subjects": len(subjects),
    }])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_windows.to_csv(OUT_DIR / "all_window_accuracy.csv", index=False)
    best.to_csv(OUT_DIR / "subject_best_windows.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    mean_by_window = all_windows.groupby("window_start", as_index=False).agg(
        mean_accuracy=("accuracy", "mean"),
        mean_difference=("difference", "mean"),
    )
    mean_by_window["window_end"] = mean_by_window["window_start"] + WINDOW_WIDTH
    mean_by_window.to_csv(OUT_DIR / "mean_by_window.csv", index=False)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        mean_by_window["window_start"],
        mean_by_window["mean_difference"] * 100,
        marker="o",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Training window start (s)",
        ylabel="Accuracy difference from 3.0–6.0 baseline (percentage points)",
        title="Paper-style CSP-LDA sliding-window comparison",
    )
    figure.tight_layout()
    figure.savefig(OUT_DIR / "accuracy_difference_by_window.png", dpi=180)
    plt.close(figure)

    print(summary.to_string(index=False))
    print(best[[
        "subject", "best_window", "accuracy_percent",
        "baseline_accuracy", "difference_percent_points",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
