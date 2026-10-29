import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from mne.decoding import CSP

from config import SUBJECTS
from data_loader import load_train, load_eval
from preprocessing import extract_epochs


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "bci_mi_project" / "data" / "BCI_2a"
OUT_DIR = ROOT / "results"
BANDS = [(8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 30.0)]


def filtered(raw, low, high):
    result = raw.copy()
    result.filter(
        l_freq=low, h_freq=high, method="iir",
        iir_params={"order": 3, "ftype": "butter"},
        verbose=False,
    )
    return result


def epochs(raw, events, artifacts, start, end):
    X, y = extract_epochs(raw, events, artifacts, start, end)
    return X[:, :, ::2].astype(np.float32), y


def reject(X, reference):
    limit = np.percentile(np.ptp(reference, axis=2).max(axis=1), 99)
    return np.ptp(X, axis=2).max(axis=1) <= limit


def predict(train_bands, y_train, test_bands, components=4):
    train_features, test_features = [], []
    for X_train, X_test in zip(train_bands, test_bands):
        csp = CSP(
            n_components=components, reg="oas",
            log=True, norm_trace=False,
        )
        train_features.append(csp.fit_transform(X_train, y_train))
        test_features.append(csp.transform(X_test))
    model = Pipeline([
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])
    model.fit(np.concatenate(train_features, axis=1), y_train)
    return model.predict(np.concatenate(test_features, axis=1))


def run_subject(subject):
    train = load_train(str(DATA_DIR / f"{subject}T.gdf"))
    test = load_eval(
        str(DATA_DIR / f"{subject}E.gdf"),
        str(DATA_DIR / f"{subject}E.mat"),
    )
    train_raw = [filtered(train["raw"], *band) for band in BANDS]
    test_raw = [filtered(test["raw"], *band) for band in BANDS]
    test_bands = [
        epochs(raw, test["events"], test["artifact_pos"], 3.0, 6.0)
        for raw in test_raw
    ]
    test_keep = reject(test_bands[0][0], test_bands[0][0])
    test_X = [X[test_keep] for X, _ in test_bands]
    test_y = test_bands[0][1][test_keep]
    rows = []
    for condition, start, end in (
        ("without_visual_cue", 3.0, 6.0),
        ("with_visual_cue_early_window", 2.0, 5.0),
    ):
        train_bands = [
            epochs(raw, train["events"], train["artifact_pos"], start, end)
            for raw in train_raw
        ]
        train_keep = reject(train_bands[0][0], train_bands[0][0])
        X = [band[0][train_keep] for band in train_bands]
        y = train_bands[0][1][train_keep]
        prediction = predict(X, y, test_X)
        rows.append({
            "subject": subject,
            "condition": condition,
            "accuracy": float(np.mean(prediction == test_y)),
            "window_start": start,
            "window_end": end,
            "train_epochs": len(y),
            "test_epochs": len(test_y),
            "band_hz": "8-30_FBCSP",
        })
    return rows


def main():
    subjects = os.getenv("COMPARISON_SUBJECTS", ",".join(SUBJECTS)).split(",")
    rows = []
    for subject in subjects:
        print(f"[COMPARISON] {subject}")
        rows.extend(run_subject(subject))
    result = pd.DataFrame(rows)
    suffix = subjects[0] if len(subjects) == 1 else "all"
    result.to_csv(OUT_DIR / f"visual_comparison_{suffix}.csv", index=False)
    wide = result.pivot(index="subject", columns="condition", values="accuracy")
    wide["difference"] = (
        wide["with_visual_cue_early_window"] - wide["without_visual_cue"]
    )
    wide.reset_index().to_csv(OUT_DIR / f"visual_comparison_table_{suffix}.csv", index=False)
    if len(subjects) > 1:
        t, p = ttest_rel(
            wide["with_visual_cue_early_window"],
            wide["without_visual_cue"],
        )
        print(f"MEAN WITHOUT: {wide['without_visual_cue'].mean():.4f}")
        print(f"MEAN WITH: {wide['with_visual_cue_early_window'].mean():.4f}")
        print(f"MEAN DIFFERENCE: {wide['difference'].mean():+.4f}")
        print(f"PAIRED P-VALUE: {p:.6f}")
        print(f"T-STATISTIC: {t:.4f}")
        wide.to_csv(OUT_DIR / "visual_comparison_final.csv")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
