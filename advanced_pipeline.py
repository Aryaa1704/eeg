import warnings
import os
import gc
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mne.decoding import CSP

from config import SUBJECTS, BASELINE_START, BASELINE_END, WIN_WIDTH
from data_loader import load_train, load_eval
from preprocessing import extract_epochs


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "bci_mi_project" / "data" / "BCI_2a"
OUT_DIR = ROOT / "results"

BANDS = {
    "mu": (8.0, 12.0),
    "low_beta": (12.0, 16.0),
    "mid_beta": (16.0, 24.0),
    "high_beta": (24.0, 30.0),
}
WINDOWS = [float(value) for value in os.getenv(
    "ADVANCED_WINDOWS", "2.0,2.5,3.0,3.5"
).split(",")]
N_COMPONENTS = [int(value) for value in os.getenv(
    "ADVANCED_COMPONENTS", "4,6"
).split(",")]
MODELS = os.getenv("ADVANCED_MODELS", "LDA,SVM").split(",")


def filter_raw(raw, low, high):
    result = raw.copy()
    result.filter(
        l_freq=low, h_freq=high, method="iir",
        iir_params={"order": 3, "ftype": "butter"},
        verbose=False,
    )
    return result


def epoch_band(raw, events, artifacts, start, end, keep=None):
    X, y = extract_epochs(raw, events, artifacts, start, end)
    if keep is not None:
        X, y = X[keep], y[keep]
    # 125 Hz is sufficient for the 30 Hz upper band and halves memory use.
    return X[:, :, ::2].astype(np.float32), y


def reject_bad_epochs(X, reference):
    if not len(X):
        return np.array([], dtype=bool)
    peak_to_peak = np.ptp(X, axis=2).max(axis=1)
    reference_ptp = np.ptp(reference, axis=2).max(axis=1)
    limit = np.percentile(reference_ptp, 99)
    return peak_to_peak <= limit


def make_classifier(name):
    if name == "LDA":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "SVM":
        return Pipeline([
            ("scale", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
        ])
    raise ValueError(f"Unknown classifier: {name}")


def make_fbcsp_model(n_components, classifier):
    # Each band is fitted independently; their log-variance features are concatenated.
    return [
        (CSP(n_components=n_components, reg="oas", log=True, norm_trace=False),
         make_classifier(classifier))
        for _ in BANDS
    ]


def fbcsp_features(models, band_epochs, fit=False, y=None):
    features = []
    for (csp, _), X in zip(models, band_epochs):
        if fit:
            features.append(csp.fit_transform(X, y))
        else:
            features.append(csp.transform(X))
    return np.concatenate(features, axis=1)


def fit_predict_fbcsp(train_bands, y_train, test_bands, n_components, classifier):
    models = make_fbcsp_model(n_components, classifier)
    X_train = fbcsp_features(models, train_bands, fit=True, y=y_train)
    X_test = fbcsp_features(models, test_bands)
    estimator = make_classifier(classifier)
    estimator.fit(X_train, y_train)
    return estimator.predict(X_test)


def cv_score_fbcsp(band_epochs, y, n_components, classifier):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, valid_idx in cv.split(band_epochs[0], y):
        train_bands = [X[train_idx] for X in band_epochs]
        valid_bands = [X[valid_idx] for X in band_epochs]
        prediction = fit_predict_fbcsp(
            train_bands, y[train_idx], valid_bands, n_components, classifier
        )
        scores.append(np.mean(prediction == y[valid_idx]))
    return float(np.mean(scores))


def prepare_subject(subject):
    train = load_train(str(DATA_DIR / f"{subject}T.gdf"))
    evaluation = load_eval(
        str(DATA_DIR / f"{subject}E.gdf"),
        str(DATA_DIR / f"{subject}E.mat"),
    )
    train_bands = {
        name: filter_raw(train["raw"], *limits)
        for name, limits in BANDS.items()
    }
    eval_bands = {
        name: filter_raw(evaluation["raw"], *limits)
        for name, limits in BANDS.items()
    }
    return train, evaluation, train_bands, eval_bands


def run_subject(subject):
    train, evaluation, train_raws, eval_raws = prepare_subject(subject)
    train_epochs = {}
    eval_epochs = {}
    for start in sorted(set(WINDOWS + [BASELINE_START])):
        end = round(start + WIN_WIDTH, 1)
        train_epochs[start] = [
            epoch_band(
                train_raws[name], train["events"], train["artifact_pos"],
                start, end,
            )
            for name in BANDS
        ]
        eval_epochs[start] = [
            epoch_band(
                eval_raws[name], evaluation["events"], evaluation["artifact_pos"],
                BASELINE_START, BASELINE_END,
            )
            for name in BANDS
        ]

    reference = train_epochs[BASELINE_START][0][0]
    rows = []
    for start in WINDOWS:
        bands = train_epochs[start]
        y = bands[0][1]
        train_keep = reject_bad_epochs(bands[0][0], reference)
        bands = [X[train_keep] for X, _ in bands]
        y = y[train_keep]
        test_keep = reject_bad_epochs(eval_epochs[start][0][0], reference)
        score_bands = [X[test_keep] for X, _ in eval_epochs[start]]
        y_test = eval_epochs[start][0][1][test_keep]
        for components in N_COMPONENTS:
            for classifier in MODELS:
                cv = cv_score_fbcsp(bands, y, components, classifier)
                prediction = fit_predict_fbcsp(
                    bands, y, score_bands, components, classifier
                )
                rows.append({
                    "subject": subject,
                    "window_start": start,
                    "window_end": round(start + WIN_WIDTH, 1),
                    "components": components,
                    "classifier": classifier,
                    "cv_accuracy": cv,
                    "test_accuracy": np.mean(prediction == y_test),
                    "train_epochs": len(y),
                    "test_epochs": len(y_test),
                })
                gc.collect()
    return rows


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rows = []
    subjects = os.getenv("ADVANCED_SUBJECTS", ",".join(SUBJECTS)).split(",")
    for subject in subjects:
        print(f"[ADVANCED] {subject}")
        rows.extend(run_subject(subject))
    results = pd.DataFrame(rows)
    if len(subjects) == 1:
        results.to_csv(
            OUT_DIR / f"advanced_fbcsp_{subjects[0]}_results.csv", index=False
        )
    else:
        results.to_csv(OUT_DIR / "advanced_fbcsp_results.csv", index=False)
    best = results.loc[results.groupby("subject")["cv_accuracy"].idxmax()]
    if len(subjects) == 1:
        best.to_csv(
            OUT_DIR / f"advanced_fbcsp_{subjects[0]}_best.csv", index=False
        )
    else:
        best.to_csv(OUT_DIR / "advanced_fbcsp_subject_best.csv", index=False)
    print("\nSubject-wise selected configurations:")
    print(best.to_string(index=False))
    print("[SAVED] results/advanced_fbcsp_results.csv")
    print("[SAVED] results/advanced_fbcsp_subject_best.csv")


if __name__ == "__main__":
    main()
