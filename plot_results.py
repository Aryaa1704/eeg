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
    ax.set_title("CSP-LDA Accuracy vs Training Window\n8-30 Hz | BCI IV 2a")
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
    ax.set_title("Per-Subject: Baseline vs Best Visual Window\n8-30Hz | BCI IV 2a | CSP-LDA")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR/"subject_comparison.png", dpi=150)
    print("[PLOT] subject_comparison.png saved"); plt.show()


if __name__ == "__main__":
    plot_window_accuracy()
    plot_subject_comparison()
