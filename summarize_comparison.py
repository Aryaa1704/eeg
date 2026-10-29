import pandas as pd
from scipy.stats import ttest_rel


results = pd.read_csv("results/model_comparison_all.csv")
baseline = results[results["window_start"] == 3.0].drop_duplicates(
    ["subject", "band_hz", "model"]
)
rows = []

for keys, group in results[results["window_start"] != 3.0].groupby(
    ["band_hz", "model", "window_start", "window_end"]
):
    band, model, start, end = keys
    b = baseline[
        (baseline["band_hz"] == band) & (baseline["model"] == model)
    ][["subject", "accuracy"]].rename(columns={"accuracy": "baseline"})
    v = group[["subject", "accuracy"]].drop_duplicates("subject").rename(
        columns={"accuracy": "visual"}
    )
    paired = b.merge(v, on="subject")
    _, p_value = ttest_rel(paired["visual"], paired["baseline"])
    rows.append({
        "band_hz": band,
        "model": model,
        "window_start": start,
        "window_end": end,
        "mean_baseline": paired["baseline"].mean(),
        "mean_visual": paired["visual"].mean(),
        "mean_delta": (paired["visual"] - paired["baseline"]).mean(),
        "p_value": p_value,
        "n_subjects": len(paired),
    })

comparison = pd.DataFrame(rows)
comparison.to_csv("results/model_comparison_statistics.csv", index=False)
best = comparison.loc[comparison["mean_visual"].idxmax()]
print(best.to_string())
print("[SAVED] results/model_comparison_statistics.csv")
