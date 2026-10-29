import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mne.decoding import CSP

from config import SUBJECTS, EVENT_ID, WIN_STARTS, WIN_WIDTH
from data_loader import load_train, load_eval
from preprocessing import extract_epochs


DATA_DIR = Path(__file__).resolve().parent / "bci_mi_project" / "data" / "BCI_2a"
OUT_DIR = Path(__file__).resolve().parent / "results"

BANDS = {
    "8-30": (8.0, 30.0),
    "4-30": (4.0, 30.0),
    "8-12": (8.0, 12.0),
    "12-30": (12.0, 30.0),
}


def filter_raw(raw, low, high):
    filtered = raw.copy()
    filtered.filter(
        l_freq=low,
        h_freq=high,
        method="iir",
        iir_params={"order": 3, "ftype": "butter"},
        verbose=False,
    )
    return filtered


def make_model(name):
    csp = CSP(n_components=4, reg="oas", log=True, norm_trace=False)
    if name == "CSP-LDA":
        classifier = OneVsOneClassifier(LinearDiscriminantAnalysis())
    elif name == "CSP-SVM":
        classifier = OneVsOneClassifier(
            Pipeline([
                ("scale", StandardScaler()),
                ("svm", SVC(kernel="linear", C=1.0)),
            ])
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("csp", csp), ("classifier", classifier)])


def evaluate(model_name, X_train, y_train, X_test, y_test):
    model = make_model(model_name)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return float(np.mean(prediction == y_test))


def main():
    rows = []
    for subject in SUBJECTS:
        print(f"[{subject}]")
        train = load_train(str(DATA_DIR / f"{subject}T.gdf"))
        evaluation = load_eval(
            str(DATA_DIR / f"{subject}E.gdf"),
            str(DATA_DIR / f"{subject}E.mat"),
        )
        for band_name, (low, high) in BANDS.items():
            train_raw = filter_raw(train["raw"], low, high)
            eval_raw = filter_raw(evaluation["raw"], low, high)
            X_test, y_test = extract_epochs(
                eval_raw, evaluation["events"], evaluation["artifact_pos"], 3.0, 6.0
            )
            for window_start in sorted(set([3.0, *WIN_STARTS])):
                window_end = round(window_start + WIN_WIDTH, 1)
                X_train, y_train = extract_epochs(
                    train_raw,
                    train["events"],
                    train["artifact_pos"],
                    window_start,
                    window_end,
                )
                if not len(y_train) or not len(y_test):
                    continue
                for model_name in ("CSP-LDA", "CSP-SVM"):
                    accuracy = evaluate(
                        model_name, X_train, y_train, X_test, y_test
                    )
                    rows.append({
                        "subject": subject,
                        "band_hz": band_name,
                        "model": model_name,
                        "window_start": window_start,
                        "window_end": window_end,
                        "accuracy": accuracy,
                    })

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "model_comparison_all.csv", index=False)
    summary = (
        results.groupby(["band_hz", "model", "window_start", "window_end"])
        .agg(mean_accuracy=("accuracy", "mean"), n_subjects=("accuracy", "size"))
        .reset_index()
    )
    baseline = results[results["window_start"] == 3.0].drop_duplicates(
        ["subject", "band_hz", "model"]
    )
    comparison_rows = []
    for keys, group in results[results["window_start"] != 3.0].groupby(
        ["band_hz", "model", "window_start", "window_end"]
    ):
        band, model, start, end = keys
        b = baseline[(baseline["band_hz"] == band) & (baseline["model"] == model)]
        v = group.set_index("subject")["accuracy"]
        b = b.set_index("subject")["accuracy"]
        b = b[~b.index.duplicated(keep="first")]
        v = v[~v.index.duplicated(keep="first")]
        common = b.index.intersection(v.index)
        t, p = stats.ttest_rel(v.loc[common], b.loc[common])
        comparison_rows.append({
            "band_hz": band,
            "model": model,
            "window_start": start,
            "window_end": end,
            "mean_baseline": b.loc[common].mean(),
            "mean_visual": v.loc[common].mean(),
            "mean_delta": (v.loc[common] - b.loc[common]).mean(),
            "p_value": p,
            "n_subjects": len(common),
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT_DIR / "model_comparison_statistics.csv", index=False)
    best = summary.loc[summary["mean_accuracy"].idxmax()]
    print("\nBEST CONFIGURATION")
    print(best.to_string())
    print("[SAVED] results/model_comparison_all.csv")
    print("[SAVED] results/model_comparison_statistics.csv")


if __name__ == "__main__":
    main()
