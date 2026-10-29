import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from config import (SUBJECTS, BASELINE_START, BASELINE_END,
                    WIN_WIDTH, WIN_STARTS, ALPHA, CLASS_NAMES)
from data_loader   import load_train, load_eval
from preprocessing import bandpass_filter, extract_epochs
from features_csp  import fit_predict

DATA_DIR = Path("bci_mi_project/data/BCI_2a")   # <-- GDF + MU.mat files yahan rakhna
OUT_DIR  = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)


def run_subject(sid):
    tr = load_train(str(DATA_DIR / f"{sid}T.gdf"))
    ev = load_eval(str(DATA_DIR / f"{sid}E.gdf"), str(DATA_DIR / f"{sid}E.mat"))
    tr_raw = bandpass_filter(tr["raw"])
    ev_raw = bandpass_filter(ev["raw"])

    rows = []
    class_rows = []

    def add_class_rows(condition, win_start, win_end, result):
        for label, name in CLASS_NAMES.items():
            mask = result["y_true"] == label
            class_rows.append({
                "subject": sid,
                "condition": condition,
                "win_start": win_start,
                "win_end": win_end,
                "class_id": label,
                "class_name": name,
                "accuracy": float(np.mean(result["y_pred"][mask] == label))
                if np.any(mask) else np.nan,
            })

    # Baseline
    Xb, yb = extract_epochs(tr_raw, tr["events"], tr["artifact_pos"], BASELINE_START, BASELINE_END)
    Xe, ye = extract_epochs(ev_raw, ev["events"], ev["artifact_pos"], BASELINE_START, BASELINE_END)
    baseline_result = fit_predict(Xb, yb, Xe, ye)
    bacc = baseline_result["accuracy"]
    add_class_rows("baseline_MI_only", BASELINE_START, BASELINE_END, baseline_result)
    print(f"\n[{sid}] BASELINE (3-6s): {bacc:.4f}")
    rows.append({"subject":sid,"condition":"baseline_MI_only",
                 "win_start":BASELINE_START,"win_end":BASELINE_END,
                 "accuracy":bacc,"delta_acc":0.0})

    # Sliding windows
    for ws in WIN_STARTS:
        we = round(ws + WIN_WIDTH, 1)
        Xt, yt = extract_epochs(tr_raw, tr["events"], tr["artifact_pos"], ws, we)
        Xe2,ye2= extract_epochs(ev_raw, ev["events"], ev["artifact_pos"], BASELINE_START, BASELINE_END)
        if not len(yt) or not len(ye2): continue
        result = fit_predict(Xt, yt, Xe2, ye2)
        acc = result["accuracy"]
        add_class_rows("visual_included", ws, we, result)
        delta = acc - bacc
        print(f"[{sid}] {ws:.1f}-{we:.1f}s | acc={acc:.4f} delta={delta:+.4f}")
        rows.append({"subject":sid,"condition":"visual_included",
                     "win_start":ws,"win_end":we,
                     "accuracy":acc,"delta_acc":delta})
    return pd.DataFrame(rows), pd.DataFrame(class_rows)


def run_all():
    dfs = []
    class_dfs = []
    for sid in SUBJECTS:
        print(f"\n{'='*55}\n  {sid}\n{'='*55}")
        try:
            df, class_df = run_subject(sid)
            df.to_csv(OUT_DIR / f"{sid}_results.csv", index=False)
            dfs.append(df)
            class_dfs.append(class_df)
        except Exception as ex:
            print(f"[{sid}] ERROR: {ex}")
    if not dfs:
        print("No results. Check DATA_DIR path."); return
    full = pd.concat(dfs, ignore_index=True)
    full.to_csv(OUT_DIR / "window_results_all.csv", index=False)
    pd.concat(class_dfs, ignore_index=True).to_csv(
        OUT_DIR / "class_accuracy_all.csv", index=False
    )
    print(f"\n[SAVED] window_results_all.csv ({len(full)} rows)")
    print("[SAVED] class_accuracy_all.csv")
    run_statistics(full)
    write_subject_comparison(full)
    print_summary(full)


def run_statistics(df):
    base = df[df["condition"]=="baseline_MI_only"]
    vis  = df[df["condition"]=="visual_included"]
    rows = []
    for ws in WIN_STARTS:
        we  = round(ws + WIN_WIDTH, 1)
        sub = vis[vis["win_start"]==ws]
        pb, pv = [], []
        for s in SUBJECTS:
            b = base[base["subject"]==s]["accuracy"].values
            v = sub[sub["subject"]==s]["accuracy"].values
            if len(b)==1 and len(v)==1:
                pb.append(b[0]); pv.append(v[0])
        if len(pb)<2: continue
        t, p = stats.ttest_rel(pv, pb)
        rows.append({"win_start":ws,"win_end":we,"n_subjects":len(pb),
                     "mean_baseline":round(np.mean(pb),4),
                     "mean_visual":round(np.mean(pv),4),
                     "mean_delta":round(np.mean(np.array(pv)-np.array(pb)),4),
                     "t_statistic":round(t,4),"p_value":round(p,4),
                     "significant": p < ALPHA})
    stat = pd.DataFrame(rows)
    stat.to_csv(OUT_DIR/"statistics.csv", index=False)
    print("\n"+"="*70)
    print("PAIRED t-TEST RESULTS | 8-30Hz | BCI IV 2a | CSP-LDA | OVO | 4-class")
    print("="*70)
    print(stat[["win_start","win_end","mean_baseline","mean_visual",
                "mean_delta","p_value","significant"]].to_string(index=False))
    best = stat.loc[stat["mean_visual"].idxmax()]
    print(f"\nBEST WINDOW: {best['win_start']:.1f}-{best['win_end']:.1f}s | "
          f"acc={best['mean_visual']:.4f} | delta={best['mean_delta']:+.4f} | "
          f"p={best['p_value']:.4f} | significant={best['significant']}")
    print("[SAVED] statistics.csv")


def write_subject_comparison(df):
    stat = pd.read_csv(OUT_DIR / "statistics.csv")
    best = stat.loc[stat["mean_visual"].idxmax()]
    visual = df[
        (df["condition"] == "visual_included")
        & (df["win_start"] == best["win_start"])
    ]
    baseline = df[df["condition"] == "baseline_MI_only"]
    rows = []
    for subject in SUBJECTS:
        b = baseline[baseline["subject"] == subject].iloc[0]
        v = visual[visual["subject"] == subject].iloc[0]
        rows.append({
            "subject": subject,
            "without_visual_cue_accuracy": b["accuracy"],
            "with_visual_cue_accuracy": v["accuracy"],
            "improvement": v["accuracy"] - b["accuracy"],
            "without_visual_target_85": b["accuracy"] >= 0.85,
            "with_visual_target_85": v["accuracy"] >= 0.85,
            "visual_window_start": best["win_start"],
            "visual_window_end": best["win_end"],
            "bandpass_hz": "8-30",
            "p_value": best["p_value"],
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "subject_comparison.csv", index=False)
    print("[SAVED] subject_comparison.csv")

    best_rows = []
    for subject in SUBJECTS:
        b = baseline[baseline["subject"] == subject].iloc[0]
        subject_visual = df[
            (df["subject"] == subject)
            & (df["condition"] == "visual_included")
        ]
        v = subject_visual.loc[subject_visual["accuracy"].idxmax()]
        best_rows.append({
            "subject": subject,
            "without_visual_cue_accuracy": b["accuracy"],
            "best_visual_cue_accuracy": v["accuracy"],
            "improvement": v["accuracy"] - b["accuracy"],
            "best_window_start": v["win_start"],
            "best_window_end": v["win_end"],
            "without_visual_target_85": b["accuracy"] >= 0.85,
            "best_visual_target_85": v["accuracy"] >= 0.85,
            "bandpass_hz": "8-30",
        })
    pd.DataFrame(best_rows).to_csv(
        OUT_DIR / "subject_best_comparison.csv", index=False
    )
    print("[SAVED] subject_best_comparison.csv")


def print_summary(df):
    base = df[df["condition"]=="baseline_MI_only"]
    vis  = df[df["condition"]=="visual_included"]
    print("\n"+"="*70)
    print(f"{'Subject':<10}{'Baseline':>10}{'Best Visual':>13}{'Window':>12}{'Delta':>10}")
    print("-"*70)
    for s in SUBJECTS:
        b = base[base["subject"]==s]["accuracy"].values
        v = vis[vis["subject"]==s]
        if not len(b) or not len(v): continue
        br = v.loc[v["accuracy"].idxmax()]
        print(f"{s:<10}{b[0]:>10.4f}{br['accuracy']:>13.4f}"
              f"  {br['win_start']:.1f}-{br['win_end']:.1f}s  {br['accuracy']-b[0]:>+10.4f}")
    print("="*70)


if __name__ == "__main__":
    run_all()
