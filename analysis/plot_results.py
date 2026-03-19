"""
Publication-quality figures for RAG experiment comparison.
Outputs multiple PDFs/PNGs to analysis/figures/
Run from the repo root: python analysis/plot_results.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as pe
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
    "axes.linewidth":     0.8,
})

_HERE       = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_HERE, "output", "figures")
STATS_JSON  = os.path.join(_HERE, "output", "stats_detailed.json")

# Colour palette (colourblind-safe)
PALETTE = {
    "baseline":    "#2166ac",
    "no_reranker": "#d73027",
    "chunk256":    "#4dac26",
    "chunk1024":   "#b8e186",
    "token":       "#f1b6da",
    "paragraph":   "#d01c8b",
    "chat_model":  "#f46d43",
    "glm":         "#fdae61",
}

SHORT_LABELS = {
    "baseline":    "Baseline\n(Sent-512+RR)",
    "no_reranker": "No\nReranker",
    "chunk256":    "Chunk\n256",
    "chunk1024":   "Chunk\n1024",
    "token":       "Token\nChunk",
    "paragraph":   "Paragraph\nChunk",
    "chat_model":  "DeepSeek\nChat",
    "glm":         "GLM-4",
}

LONG_LABELS = {
    "baseline":    "Baseline (Sentence-512, Reranker)",
    "no_reranker": "No Reranker",
    "chunk256":    "Chunk Size 256",
    "chunk1024":   "Chunk Size 1024",
    "token":       "Token Chunking",
    "paragraph":   "Paragraph Chunking",
    "chat_model":  "DeepSeek-Chat",
    "glm":         "GLM-4",
}

ORDER = ["baseline", "no_reranker", "chunk256", "chunk1024",
         "token", "paragraph", "chat_model", "glm"]


def load_stats():
    with open(STATS_JSON) as f:
        data = json.load(f)
    return {d["name"]: d for d in data}


def sig_label(p, is_baseline=False):
    if is_baseline:
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ── Figure 1: Mean scores with 95% CI ─────────────────────────────────────────
def fig_mean_ci(stats):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    names   = [n for n in ORDER if n in stats]
    means   = [stats[n]["mean"]   for n in names]
    ci_lo   = [stats[n]["ci_lo"]  for n in names]
    ci_hi   = [stats[n]["ci_hi"]  for n in names]
    colors  = [PALETTE[n]         for n in names]
    xlabels = [SHORT_LABELS[n]    for n in names]

    x = np.arange(len(names))
    err_lo = [m - lo for m, lo in zip(means, ci_lo)]
    err_hi = [hi - m for m, hi in zip(means, ci_hi)]

    bars = ax.bar(x, means, color=colors, width=0.6, zorder=3,
                  edgecolor="white", linewidth=0.6)
    ax.errorbar(x, means, yerr=[err_lo, err_hi],
                fmt="none", color="black", capsize=4, capthick=1,
                linewidth=1.2, zorder=4)

    # Significance annotations above bars
    base_mean = stats["baseline"]["mean"]
    y_max = max(ci_hi) + 0.3
    for i, n in enumerate(names):
        p = stats[n]["wilcoxon_p"]
        label = sig_label(p, n == "baseline")
        if label:
            y_pos = ci_hi[i] + 0.15
            ax.text(x[i], y_pos, label, ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color="black" if label != "ns" else "#888888")

    # Baseline reference line
    ax.axhline(base_mean, color=PALETTE["baseline"], lw=1.2,
               ls="--", alpha=0.7, zorder=2, label=f"Baseline mean ({base_mean:.2f})")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.set_ylabel("Mean LLM Score (0–10)")
    ax.set_title("Mean RAG Answer Quality with 95% Bootstrap Confidence Intervals")
    ax.set_ylim(5.5, y_max + 0.4)
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.legend(loc="lower right", framealpha=0.9)
    ax.text(0.99, 0.02, "*** p<0.001  ** p<0.01  * p<0.05  ns: not significant",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="#555555", style="italic")

    fig.tight_layout()
    return fig


# ── Figure 2: Violin / box hybrid ─────────────────────────────────────────────
def fig_violin(stats):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))

    names   = [n for n in ORDER if n in stats]
    xlabels = [SHORT_LABELS[n] for n in names]
    x = np.arange(len(names))

    for i, n in enumerate(names):
        sc = np.array(stats[n]["scores"])
        color = PALETTE[n]

        # Violin
        parts = ax.violinplot([sc], positions=[i], widths=0.65,
                              showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.45)
            pc.set_edgecolor("none")

        # Box (IQR)
        q1, med, q3 = np.percentile(sc, [25, 50, 75])
        iqr = q3 - q1
        whisker_lo = max(sc.min(), q1 - 1.5 * iqr)
        whisker_hi = min(sc.max(), q3 + 1.5 * iqr)
        ax.plot([i, i], [whisker_lo, whisker_hi], color="black",
                lw=1, solid_capstyle="round", zorder=3)
        ax.add_patch(plt.Rectangle((i - 0.12, q1), 0.24, iqr,
                                   facecolor=color, edgecolor="black",
                                   linewidth=0.8, zorder=4))
        ax.plot([i - 0.12, i + 0.12], [med, med],
                color="white", lw=2, zorder=5)

        # Mean dot
        ax.scatter([i], [stats[n]["mean"]], color="black",
                   s=18, zorder=6, marker="D")

    # Significance brackets vs baseline
    base_idx = names.index("baseline")
    y_bracket = 10.6
    for i, n in enumerate(names):
        if n == "baseline":
            continue
        p = stats[n]["wilcoxon_p"]
        label = sig_label(p)
        if label != "ns":
            ax.annotate("", xy=(i, y_bracket), xytext=(base_idx, y_bracket),
                        arrowprops=dict(arrowstyle="-", color="#333333",
                                       lw=0.7, connectionstyle="arc3,rad=0"))
            ax.text((i + base_idx) / 2, y_bracket + 0.05, label,
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.set_ylabel("LLM Score (0–10)")
    ax.set_title("Score Distribution Across RAG Configurations")
    ax.set_ylim(-1, 11.8)
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    legend_handles = [
        mpatches.Patch(color="none", label="◆ = mean,  ─ = median"),
        mpatches.Patch(color="#aaaaaa", label="Box = IQR,  whiskers = 1.5×IQR"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              framealpha=0.85, fontsize=8)

    fig.tight_layout()
    return fig


# ── Figure 3: Score breakdown by query type ───────────────────────────────────
def fig_by_type(stats):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
    qtypes = ["abstractive", "extractive"]
    titles = ["Abstractive Queries", "Extractive Queries"]

    for ax, qt, title in zip(axes, qtypes, titles):
        names = [n for n in ORDER if n in stats and qt in stats[n]["by_type"]]
        x = np.arange(len(names))
        means = [stats[n]["by_type"][qt]["mean"]   for n in names]
        ci_lo = [stats[n]["by_type"][qt]["ci_lo"]  for n in names]
        ci_hi = [stats[n]["by_type"][qt]["ci_hi"]  for n in names]
        colors = [PALETTE[n] for n in names]

        err_lo = [m - lo for m, lo in zip(means, ci_lo)]
        err_hi = [hi - m for m, hi in zip(means, ci_hi)]

        bars = ax.bar(x, means, color=colors, width=0.6, zorder=3,
                      edgecolor="white", linewidth=0.5)
        ax.errorbar(x, means, yerr=[err_lo, err_hi],
                    fmt="none", color="black", capsize=3.5,
                    capthick=0.9, linewidth=1, zorder=4)

        # Significance labels
        base_qt = stats["baseline"]["by_type"].get(qt, {})
        for i, n in enumerate(names):
            p = stats[n]["by_type"][qt]["wilcoxon_p"]
            lbl = sig_label(p, n == "baseline")
            if lbl:
                ax.text(x[i], ci_hi[i] + 0.15, lbl, ha="center",
                        va="bottom", fontsize=8, fontweight="bold",
                        color="black" if lbl != "ns" else "#888888")

        if "baseline" in names:
            bm = stats["baseline"]["by_type"][qt]["mean"]
            ax.axhline(bm, color=PALETTE["baseline"], lw=1.2,
                       ls="--", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[n] for n in names],
                           fontsize=7.8, rotation=0)
        ax.set_title(title)
        ax.set_ylim(4.5, 10.5)

    axes[0].set_ylabel("Mean LLM Score (0–10)")
    fig.suptitle("Score by Query Type", fontsize=12, fontweight="bold", y=1.01)
    fig.text(0.5, -0.02,
             "*** p<0.001  ** p<0.01  * p<0.05  (Wilcoxon signed-rank vs. baseline)",
             ha="center", fontsize=7.5, color="#555555", style="italic")
    fig.tight_layout()
    return fig


# ── Figure 4: Cohen's d effect size ───────────────────────────────────────────
def fig_effect_size(stats):
    names = [n for n in ORDER if n in stats and n != "baseline"]
    d_vals = [stats[n]["cohens_d"] for n in names]
    colors_bar = [PALETTE[n] for n in names]
    labels = [LONG_LABELS[n] for n in names]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    y = np.arange(len(names))

    bars = ax.barh(y, d_vals, color=colors_bar, height=0.55,
                   edgecolor="white", linewidth=0.5, zorder=3)

    # Zero line
    ax.axvline(0, color="black", lw=0.8, zorder=4)
    # Threshold lines
    for val, ls, lbl in [(-0.2, "--", "small"), (0.2, "--", ""),
                         (-0.5, ":", "medium"), (0.5, ":", ""),
                         (-0.8, "-.", "large"), (0.8, "-.", "")]:
        ax.axvline(val, color="#999999", lw=0.7, ls=ls, alpha=0.7, zorder=2)
        if lbl:
            ax.text(val + 0.01, len(names) - 0.3, lbl, fontsize=6.5,
                    color="#777777", va="top")

    # Value labels
    for i, v in enumerate(d_vals):
        ax.text(v + (0.015 if v >= 0 else -0.015), i,
                f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Cohen's d  (positive = better than baseline)")
    ax.set_title("Effect Size vs. Baseline")
    ax.set_xlim(-0.45, 0.45)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ── Figure 5: Score heat-map (mean score by exp × query_type) ─────────────────
def fig_heatmap(stats):
    names  = [n for n in ORDER if n in stats]
    qtypes = ["abstractive", "extractive"]
    label_row = [LONG_LABELS[n] for n in names]

    data_matrix = np.zeros((len(names), 2))
    for i, n in enumerate(names):
        for j, qt in enumerate(qtypes):
            data_matrix[i, j] = stats[n]["by_type"].get(qt, {}).get("mean", np.nan)

    # Add overall column
    overall = np.array([[stats[n]["mean"]] for n in names])
    full_matrix = np.hstack([data_matrix, overall])
    col_labels = ["Abstractive", "Extractive", "Overall"]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    im = ax.imshow(full_matrix, cmap="RdYlGn", vmin=6.5, vmax=9.0, aspect="auto")

    ax.set_xticks(range(3))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(label_row, fontsize=8.5)

    for i in range(len(names)):
        for j in range(3):
            val = full_matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if val < 7.2 or val > 8.7 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Mean Score", fontsize=9)
    ax.set_title("Mean Score Heatmap by Query Type")
    fig.tight_layout()
    return fig


# ── Figure 6: Win / Tie / Loss stacked bar ────────────────────────────────────
def fig_wtl(stats):
    names = [n for n in ORDER if n in stats and n != "baseline"]
    labels = [SHORT_LABELS[n] for n in names]

    wins   = np.array([stats[n]["wins"]   for n in names])
    ties   = np.array([stats[n]["ties"]   for n in names])
    losses = np.array([stats[n]["losses"] for n in names])
    total  = wins + ties + losses

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(names))
    w = 0.55

    p1 = ax.bar(x, wins/total*100,   w, label="Win",  color="#4dac26", zorder=3)
    p2 = ax.bar(x, ties/total*100,   w, bottom=wins/total*100,
                label="Tie", color="#f7f7f7", edgecolor="#bbbbbb",
                linewidth=0.6, zorder=3)
    p3 = ax.bar(x, losses/total*100, w,
                bottom=(wins+ties)/total*100,
                label="Loss", color="#d73027", zorder=3)

    ax.axhline(50, color="black", lw=0.8, ls="--", alpha=0.5)

    for i in range(len(names)):
        ax.text(i, wins[i]/total[i]*100/2, f"{wins[i]}",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold")
        ax.text(i, (wins[i]+ties[i]/2)/total[i]*100, f"{ties[i]}",
                ha="center", va="center", fontsize=7.5, color="#444444")
        ax.text(i, (wins[i]+ties[i]+losses[i]/2)/total[i]*100, f"{losses[i]}",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-Question Win / Tie / Loss vs. Baseline")
    ax.legend(loc="upper right", ncol=3, framealpha=0.9)
    fig.tight_layout()
    return fig


# ── Figure 7: Timing comparison ───────────────────────────────────────────────
def fig_timing(stats):
    names = [n for n in ORDER if n in stats
             and stats[n]["mean_time_sec"] and not np.isnan(stats[n]["mean_time_sec"])]
    if not names:
        return None
    labels = [SHORT_LABELS[n] for n in names]
    times  = [stats[n]["mean_time_sec"] for n in names]
    colors = [PALETTE[n] for n in names]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = np.arange(len(names))
    ax.bar(x, times, color=colors, width=0.6, zorder=3,
           edgecolor="white", linewidth=0.5)
    for i, t in enumerate(times):
        ax.text(i, t + 0.1, f"{t:.1f}s", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Mean Query Time (seconds)")
    ax.set_title("Average Query Latency per Configuration")
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    stats = load_stats()

    figures = [
        ("fig1_mean_ci",       fig_mean_ci(stats),    "Mean scores with 95% CI"),
        ("fig2_violin",        fig_violin(stats),     "Violin + box plots"),
        ("fig3_by_type",       fig_by_type(stats),    "Score by query type"),
        ("fig4_effect_size",   fig_effect_size(stats),"Cohen's d effect sizes"),
        ("fig5_heatmap",       fig_heatmap(stats),    "Score heatmap"),
        ("fig6_wtl",           fig_wtl(stats),        "Win/Tie/Loss"),
        ("fig7_timing",        fig_timing(stats),     "Query latency"),
    ]

    for fname, fig, desc in figures:
        if fig is None:
            print(f"[SKIP] {fname}: no data")
            continue
        for ext in ("pdf", "png"):
            out = os.path.join(FIGURES_DIR, f"{fname}.{ext}")
            fig.savefig(out)
        plt.close(fig)
        print(f"[OK]   {fname}.pdf/.png  — {desc}")

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
