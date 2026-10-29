import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
from mne.decoding import CSP
from scipy.stats import ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import SUBJECTS
from data_loader import load_train, load_eval
from preprocessing import extract_epochs


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "bci_mi_project" / "data" / "BCI_2a"
OUT_DIR = ROOT / "results" / "optimized"

BAND_SETS = {
    "mu_beta": [(8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 30.0)],
    "broad": [(8.0, 30.0)],
    "mu_only": [(8.0, 12.0)],
    "beta": [(12.0, 30.0)],
}
WINDOWS = [float(value) for value in os.getenv(
    "OPTIMIZED_WINDOWS", "2.0,2.5,3.0"
).split(",")]
COMPONENTS = [int(value) for value in os.getenv(
    "OPTIMIZED_COMPONENTS", "4,6"
).split(",")]
CLASSIFIERS = os.getenv("OPTIMIZED_MODELS", "LDA,SVM").split(",")


def bandpass(raw, low, high):
    result = raw.copy()
    result.filter(
        l_freq=low, h_freq=high, method="iir",
        iir_params={"order": 3, "ftype": "butter"},
        verbose=False,
    )
    return result


def make_classifier(name):
    if name == "LDA":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "SVM":
        return Pipeline([
            ("scale", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1.0, gamma="scale")),
        ])
    raise ValueError(f"Unknown classifier: {name}")


def get_epochs(raw, events, artifacts, start):
    X, y = extract_epochs(raw, events, artifacts, start, start + 3.0)
    return X[:, :, ::2].astype(np.float32), y


def bad_trial_mask(X, reference):
    limit = np.percentile(np.ptp(reference, axis=2).max(axis=1), 99)
    return np.ptp(X, axis=2).max(axis=1) <= limit


def features(train_bands, test_bands, y, components):
    train_features, test_features = [], []
    for X_train, X_test in zip(train_bands, test_bands):
        csp = CSP(
            n_components=components,
            reg="oas",
            log=True,
            norm_trace=False,
        )
        train_features.append(csp.fit_transform(X_train, y))
        test_features.append(csp.transform(X_test))
    return np.concatenate(train_features, axis=1), np.concatenate(test_features, axis=1)


def score(train_bands, y, test_bands, y_test, components, classifier):
    X_train, X_test = features(train_bands, test_bands, y, components)
    model = make_classifier(classifier)
    model.fit(X_train, y)
    return float(np.mean(model.predict(X_test) == y_test))


def cv_score(bands, y, components, classifier):
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, valid_idx in splitter.split(bands[0], y):
        train_bands = [X[train_idx] for X in bands]
        valid_bands = [X[valid_idx] for X in bands]
        scores.append(
            score(
                train_bands,
                y[train_idx],
                valid_bands,
                y[valid_idx],
                components,
                classifier,
            )
        )
    return float(np.mean(scores))


def load_subject(subject):
    train = load_train(str(DATA_DIR / f"{subject}T.gdf"))
    test = load_eval(
        str(DATA_DIR / f"{subject}E.gdf"),
        str(DATA_DIR / f"{subject}E.mat"),
    )
    return train, test


def run_subject(subject):
    train, test = load_subject(subject)
    rows = []
    for band_set_name, band_limits in BAND_SETS.items():
        train_raw = [bandpass(train["raw"], *band) for band in band_limits]
        test_raw = [bandpass(test["raw"], *band) for band in band_limits]
        for window_start in WINDOWS:
            train_bands = [
                get_epochs(raw, train["events"], train["artifact_pos"], window_start)
                for raw in train_raw
            ]
            test_bands = [
                get_epochs(raw, test["events"], test["artifact_pos"], 3.0)
                for raw in test_raw
            ]
            keep_train = bad_trial_mask(train_bands[0][0], train_bands[0][0])
            keep_test = bad_trial_mask(test_bands[0][0], train_bands[0][0])
            X_train = [X[keep_train] for X, _ in train_bands]
            y_train = train_bands[0][1][keep_train]
            X_test = [X[keep_test] for X, _ in test_bands]
            y_test = test_bands[0][1][keep_test]
            for components in COMPONENTS:
                for classifier in CLASSIFIERS:
                    cv = cv_score(X_train, y_train, components, classifier)
                    test_accuracy = score(
                        X_train, y_train, X_test, y_test, components, classifier
                    )
                    rows.append({
                        "subject": subject,
                        "band_set": band_set_name,
                        "window_start": window_start,
                        "window_end": window_start + 3.0,
                        "components": components,
                        "classifier": classifier,
                        "cv_accuracy": cv,
                        "test_accuracy": test_accuracy,
                        "train_epochs": len(y_train),
                        "test_epochs": len(y_test),
                    })
            del train_raw, test_raw, train_bands, test_bands
            gc.collect()
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects = os.getenv("OPTIMIZED_SUBJECTS", ",".join(SUBJECTS)).split(",")
    rows = []
    for subject in subjects:
        print(f"[OPTIMIZED] {subject}")
        rows.extend(run_subject(subject))
    results = pd.DataFrame(rows)
    best = results.loc[results.groupby("subject")["cv_accuracy"].idxmax()]
    suffix = subjects[0] if len(subjects) == 1 else "all"
    results.to_csv(OUT_DIR / f"optimized_{suffix}_all.csv", index=False)
    best.to_csv(OUT_DIR / f"optimized_{suffix}_best.csv", index=False)
    print(best.to_string(index=False))
    print(f"[SAVED] {OUT_DIR / f'optimized_{suffix}_best.csv'}")


if __name__ == "__main__":
    main()
