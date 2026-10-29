"""
setup_project.py
Run this ONCE in your Codespace terminal:
    python setup_project.py
It will create all project files automatically.
"""

import os

BASE = os.path.dirname(os.path.abspath(__file__))

files = {}

# ─────────────────────────────────────────────────────────────────
files["config.py"] = '''
SUBJECTS = ['A01','A02','A03','A04','A05','A06','A07','A08','A09']

BANDPASS_LOW  = 8.0
BANDPASS_HIGH = 30.0
FILTER_ORDER  = 3
SFREQ = 250

BASELINE_START = 3.0
BASELINE_END   = 6.0

WIN_WIDTH  = 3.0
WIN_STARTS = [round(s * 0.1 + 2.0, 1) for s in range(16)]

N_CSP_COMPONENTS = 4

EVENT_ID = {769: 1, 770: 2, 771: 3, 772: 4}
CLASS_NAMES = {1: 'Left Hand', 2: 'Right Hand', 3: 'Feet', 4: 'Tongue'}

ARTIFACT_CODE = 1023
ALPHA = 0.05
RANDOM_SEED = 42
'''.strip()

# ─────────────────────────────────────────────────────────────────
files["data_loader.py"] = '''
import mne
import numpy as np
import scipy.io as sio
from pathlib import Path
from config import SFREQ, EVENT_ID, ARTIFACT_CODE, CLASS_NAMES


def load_train(gdf_path):
    raw = _load_raw(gdf_path)
    events, _ = mne.events_from_annotations(raw, verbose=False)
    artifact_positions = set(events[events[:, 2] == ARTIFACT_CODE, 0])
    mi_events = events[np.isin(events[:, 2], list(EVENT_ID.keys()))]
    audit = _make_audit(gdf_path, "TRAIN", raw, mi_events, artifact_positions)
    _print_audit(audit)
    return {"raw": raw, "events": mi_events,
            "artifact_pos": artifact_positions, "audit": audit}


def load_eval(gdf_path, mat_path):
    raw = _load_raw(gdf_path)
    events, _ = mne.events_from_annotations(raw, verbose=False)
    artifact_positions = set(events[events[:, 2] == ARTIFACT_CODE, 0])
    trial_events = _get_trial_positions(events)
    true_labels  = _load_mu_mat(mat_path)

    assert len(trial_events) == len(true_labels), (
        f"Mismatch: GDF {len(trial_events)} trials, MU.mat {len(true_labels)} labels"
    )

    code_map = {1: 769, 2: 770, 3: 771, 4: 772}
    mi_events = np.column_stack([
        trial_events[:, 0],
        np.zeros(len(trial_events), dtype=int),
        np.array([code_map[int(l)] for l in true_labels])
    ])

    audit = _make_audit(gdf_path, "EVAL", raw, mi_events, artifact_positions)
    _print_audit(audit)
    return {"raw": raw, "events": mi_events,
            "artifact_pos": artifact_positions, "audit": audit}


def _load_raw(gdf_path):
    gdf_path = Path(gdf_path)
    assert gdf_path.exists(), f"Not found: {gdf_path}"
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)
    assert abs(raw.info["sfreq"] - SFREQ) < 1, f"Bad sfreq: {raw.info[\'sfreq\']}"
    eeg_chs = [c for c in raw.ch_names if not c.upper().startswith("EOG")]
    raw.pick_channels(eeg_chs)
    return raw


def _get_trial_positions(events):
    for code in [768, 783, 769, 770, 771, 772]:
        sub = events[events[:, 2] == code]
        if len(sub) == 288:
            return sub
        if len(sub) > 0:
            return sub
    raise ValueError("Cannot find 288 trial markers in E.gdf")


def _load_mu_mat(mat_path):
    mat_path = Path(mat_path)
    assert mat_path.exists(), f"Not found: {mat_path}"
    mat = sio.loadmat(str(mat_path))
    for key in ["classlabel", "classLabel", "y", "labels", "label"]:
        if key in mat:
            labels = mat[key].flatten().astype(int)
            assert set(labels).issubset({1,2,3,4}), f"Bad labels: {set(labels)}"
            print(f"[MU.MAT] {mat_path.name} | key=\'{key}\' | n={len(labels)} | unique={sorted(set(labels))}")
            return labels
    available = [k for k in mat.keys() if not k.startswith("_")]
    raise KeyError(f"No classlabel in {mat_path.name}. Keys: {available}")


def _make_audit(gdf_path, session, raw, mi_events, artifact_positions):
    art = sum(1 for ev in mi_events if any(abs(ev[0]-ap)<=10 for ap in artifact_positions))
    cc  = {lid: int(np.sum(mi_events[:,2]==code)) for code, lid in EVENT_ID.items()}
    return {"file": Path(gdf_path).name, "session": session,
            "total_trials": len(mi_events), "artifact_trials": art,
            "clean_trials": len(mi_events)-art, "class_counts": cc,
            "n_channels": len(raw.ch_names), "sfreq": raw.info["sfreq"]}


def _print_audit(a):
    print(f"\\n[AUDIT] {a[\'file\']} ({a[\'session\']})")
    print(f"  Channels:{a[\'n_channels\']}  sfreq:{a[\'sfreq\']}Hz")
    print(f"  Trials: {a[\'total_trials\']} total | {a[\'artifact_trials\']} artifacts | {a[\'clean_trials\']} clean")
    for lid, cnt in a["class_counts"].items():
        print(f"    Class {lid} ({CLASS_NAMES[lid]}): {cnt}")
'''.strip()

# ─────────────────────────────────────────────────────────────────
files["preprocessing.py"] = '''
import numpy as np
import mne
from config import BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER, EVENT_ID


def bandpass_filter(raw):
    r = raw.copy()
    r.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH,
             method="iir", iir_params=dict(order=FILTER_ORDER, ftype="butter"),
             verbose=False)
    print(f"[FILTER] {BANDPASS_LOW}-{BANDPASS_HIGH} Hz Butterworth order {FILTER_ORDER}")
    return r


def extract_epochs(raw_f, events, artifact_pos, tmin, tmax):
    sfreq = raw_f.info["sfreq"]
    data, _ = raw_f[:, :]
    n_samp = int((tmax - tmin) * sfreq)
    X, y = [], []

    for ev in events:
        cue, code = ev[0], ev[2]
        label = EVENT_ID.get(code)
        if label is None:
            continue
        if any(abs(cue - ap) <= 10 for ap in artifact_pos):
            continue
        s = int(cue + tmin * sfreq)
        e = s + n_samp
        if s < 0 or e > data.shape[1]:
            continue
        X.append(data[:, s:e])
        y.append(label)

    X, y = np.array(X), np.array(y)
    if len(y):
        print(f"[EPOCHS] {tmin}-{tmax}s -> {len(y)} epochs | "
              + str({int(c): int(np.sum(y==c)) for c in np.unique(y)}))
    return X, y
'''.strip()

# ─────────────────────────────────────────────────────────────────
files["features_csp.py"] = '''
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from config import N_CSP_COMPONENTS


def make_csp_lda_ovo():
    csp  = CSP(n_components=N_CSP_COMPONENTS, reg=None, log=True, norm_trace=False)
    pipe = Pipeline([("csp", csp), ("lda", LDA())])
    return OneVsOneClassifier(pipe)


def fit_predict(X_train, y_train, X_test, y_test):
    model = make_csp_lda_ovo()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {"accuracy": float(np.mean(y_pred == y_test)),
            "y_pred": y_pred, "y_true": y_test}
'''.strip()

# ─────────────────────────────────────────────────────────────────
files["pipeline.py"] = '''
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from config import (SUBJECTS, BASELINE_START, BASELINE_END,
                    WIN_WIDTH, WIN_STARTS, ALPHA, CLASS_NAMES)
from data_loader   import load_train, load_eval
from preprocessing import bandpass_filter, extract_epochs
from features_csp  import fit_predict

DATA_DIR = Path("data/BCI_2a")   # <-- GDF + MU.mat files yahan rakhna
OUT_DIR  = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)


def run_subject(sid):
    tr = load_train(str(DATA_DIR / f"{sid}T.gdf"))
    ev = load_eval(str(DATA_DIR / f"{sid}E.gdf"), str(DATA_DIR / f"{sid}MU.mat"))
    tr_raw = bandpass_filter(tr["raw"])
    ev_raw = bandpass_filter(ev["raw"])

    rows = []

    # Baseline
    Xb, yb = extract_epochs(tr_raw, tr["events"], tr["artifact_pos"], BASELINE_START, BASELINE_END)
    Xe, ye = extract_epochs(ev_raw, ev["events"], ev["artifact_pos"], BASELINE_START, BASELINE_END)
    bacc   = fit_predict(Xb, yb, Xe, ye)["accuracy"]
    print(f"\\n[{sid}] BASELINE (3-6s): {bacc:.4f}")
    rows.append({"subject":sid,"condition":"baseline_MI_only",
                 "win_start":BASELINE_START,"win_end":BASELINE_END,
                 "accuracy":bacc,"delta_acc":0.0})

    # Sliding windows
    for ws in WIN_STARTS:
        we = round(ws + WIN_WIDTH, 1)
        Xt, yt = extract_epochs(tr_raw, tr["events"], tr["artifact_pos"], ws, we)
        Xe2,ye2= extract_epochs(ev_raw, ev["events"], ev["artifact_pos"], BASELINE_START, BASELINE_END)
        if not len(yt) or not len(ye2): continue
        acc   = fit_predict(Xt, yt, Xe2, ye2)["accuracy"]
        delta = acc - bacc
        print(f"[{sid}] {ws:.1f}-{we:.1f}s | acc={acc:.4f} delta={delta:+.4f}")
        rows.append({"subject":sid,"condition":"visual_included",
                     "win_start":ws,"win_end":we,
                     "accuracy":acc,"delta_acc":delta})
    return pd.DataFrame(rows)


def run_all():
    dfs = []
    for sid in SUBJECTS:
        print(f"\\n{\'=\'*55}\\n  {sid}\\n{\'=\'*55}")
        try:
            df = run_subject(sid)
            df.to_csv(OUT_DIR / f"{sid}_results.csv", index=False)
            dfs.append(df)
        except Exception as ex:
            print(f"[{sid}] ERROR: {ex}")
    if not dfs:
        print("No results. Check DATA_DIR path."); return
    full = pd.concat(dfs, ignore_index=True)
    full.to_csv(OUT_DIR / "window_results_all.csv", index=False)
    print(f"\\n[SAVED] window_results_all.csv ({len(full)} rows)")
    run_statistics(full)
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
    print("\\n"+"="*70)
    print("PAIRED t-TEST RESULTS | 8-30Hz | BCI IV 2a | CSP-LDA | OVO | 4-class")
    print("="*70)
    print(stat[["win_start","win_end","mean_baseline","mean_visual",
                "mean_delta","p_value","significant"]].to_string(index=False))
    best = stat.loc[stat["mean_visual"].idxmax()]
    print(f"\\nBEST WINDOW: {best[\'win_start\']:.1f}-{best[\'win_end\']:.1f}s | "
          f"acc={best[\'mean_visual\']:.4f} | delta={best[\'mean_delta\']:+.4f} | "
          f"p={best[\'p_value\']:.4f} | significant={best[\'significant\']}")
    print("[SAVED] statistics.csv")


def print_summary(df):
    base = df[df["condition"]=="baseline_MI_only"]
    vis  = df[df["condition"]=="visual_included"]
    print("\\n"+"="*70)
    print(f"{\'Subject\':<10}{\'Baseline\':>10}{\'Best Visual\':>13}{\'Window\':>12}{\'Delta\':>10}")
    print("-"*70)
    for s in SUBJECTS:
        b = base[base["subject"]==s]["accuracy"].values
        v = vis[vis["subject"]==s]
        if not len(b) or not len(v): continue
        br = v.loc[v["accuracy"].idxmax()]
        print(f"{s:<10}{b[0]:>10.4f}{br[\'accuracy\']:>13.4f}"
              f"  {br[\'win_start\']:.1f}-{br[\'win_end\']:.1f}s  {br[\'accuracy\']-b[0]:>+10.4f}")
    print("="*70)


if __name__ == "__main__":
    run_all()
'''.strip()

# ─────────────────────────────────────────────────────────────────
files["plot_results.py"] = '''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from config import SUBJECTS, WIN_STARTS, WIN_WIDTH

OUT_DIR = Path("results")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_window_accuracy():
    df  = pd.read_csv(OUT_DIR/"window_results_all.csv")
    vis = df[df["condition"]=="visual_included"]
    b   = df[df["condition"]=="baseline_MI_only"]
    fig, ax = plt.subplots(figsize=(12,6))
    for s in SUBJECTS:
        sub = vis[vis["subject"]==s].sort_values("win_start")
        ax.plot(sub["win_start"], sub["accuracy"], alpha=0.3, lw=1, color="steelblue")
    mean_acc = vis.groupby("win_start")["accuracy"].mean()
    ax.plot(mean_acc.index, mean_acc.values, color="darkblue", lw=2.5, label="Mean (all subjects)", zorder=5)
    bm = b["accuracy"].mean()
    ax.axhline(bm, color="red", ls="--", lw=2, label=f"Baseline MI-only (3-6s) = {bm:.3f}")
    ax.set_xlabel("Training Window Start (s)"); ax.set_ylabel("4-Class Accuracy")
    ax.set_title("CSP-LDA Accuracy vs Training Window\\n8-30 Hz | BCI IV 2a")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xticks(WIN_STARTS)
    plt.tight_layout()
    plt.savefig(FIG_DIR/"window_accuracy.png", dpi=150)
    print("[PLOT] window_accuracy.png saved"); plt.show()


def plot_subject_comparison():
    df   = pd.read_csv(OUT_DIR/"window_results_all.csv")
    stat = pd.read_csv(OUT_DIR/"statistics.csv")
    bws  = stat.loc[stat["mean_visual"].idxmax(), "win_start"]
    vb   = df[(df["condition"]=="visual_included")&(df["win_start"]==bws)]
    base = df[df["condition"]=="baseline_MI_only"]
    fig, ax = plt.subplots(figsize=(10,6))
    x  = np.arange(len(SUBJECTS))
    ba = [base[base["subject"]==s]["accuracy"].values[0] for s in SUBJECTS]
    va = [vb[vb["subject"]==s]["accuracy"].values[0] if len(vb[vb["subject"]==s]) else 0 for s in SUBJECTS]
    ax.bar(x-0.175, ba, 0.35, label="Baseline (3-6s)", color="salmon", edgecolor="black")
    ax.bar(x+0.175, va, 0.35, label=f"Visual window ({bws:.1f}-{bws+WIN_WIDTH:.1f}s)", color="steelblue", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(SUBJECTS)
    ax.set_xlabel("Subject"); ax.set_ylabel("4-Class Accuracy")
    ax.set_title("Per-Subject: Baseline vs Best Visual Window\\n8-30Hz | BCI IV 2a | CSP-LDA")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR/"subject_comparison.png", dpi=150)
    print("[PLOT] subject_comparison.png saved"); plt.show()


if __name__ == "__main__":
    plot_window_accuracy()
    plot_subject_comparison()
'''.strip()

# ─────────────────────────────────────────────────────────────────
files["requirements.txt"] = """mne==1.7.0
numpy>=1.24
scipy>=1.11
scikit-learn>=1.3
pandas>=2.0
matplotlib>=3.7""".strip()

files[".gitignore"] = """data/
*.gdf
*.mat
results/
__pycache__/
*.pyc
.env/
venv/""".strip()

# ─── CREATE ALL FILES ─────────────────────────────────────────────
created = []
for fname, content in files.items():
    fpath = os.path.join(BASE, fname)
    os.makedirs(os.path.dirname(fpath) if os.path.dirname(fpath) else BASE, exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content + "\n")
    created.append(fname)
    print(f"  [OK] {fname}")

# Create data folder
data_path = os.path.join(BASE, "data", "BCI_2a")
os.makedirs(data_path, exist_ok=True)
print(f"  [OK] data/BCI_2a/ (folder)")

os.makedirs(os.path.join(BASE, "results", "figures"), exist_ok=True)
print(f"  [OK] results/figures/ (folder)")

print(f"\n{'='*50}")
print(f"Setup complete! {len(created)} files created.")
print(f"{'='*50}")
print("""
NEXT STEPS:
1. Copy your GDF + MU.mat files to:
      bci_mi_project/data/BCI_2a/
   (A01T.gdf, A01E.gdf, A01MU.mat ... A09MU.mat)

2. Install dependencies:
      pip install -r requirements.txt

3. Run the pipeline:
      python pipeline.py

4. Plot results:
      python plot_results.py
""")