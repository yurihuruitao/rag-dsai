"""
Statistical comparison of RAG experiments vs baseline.
Outputs: analysis/stats_summary.csv, analysis/stats_detailed.json
Run from the repo root: python analysis/stats_analysis.py
"""

import json
import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, bootstrap
import warnings
warnings.filterwarnings("ignore")

_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_ROOT, "results")
OUT_DIR     = os.path.join(_HERE, "output")
BASELINE = "baseline"

EXP_LABELS = {
    "baseline":    "Baseline (Sentence-512, Reranker)",
    "no_reranker": "No Reranker",
    "chunk256":    "Chunk-256",
    "chunk1024":   "Chunk-1024",
    "token":       "Token Chunking",
    "paragraph":   "Paragraph Chunking",
    "chat_model":  "DeepSeek-Chat",
    "glm":         "GLM-4",
}


def load_eval(name):
    path = os.path.join(RESULTS_DIR, f"eval_{name}.json")
    with open(path) as f:
        data = json.load(f)
    return data


def load_bench(name):
    path = os.path.join(RESULTS_DIR, f"bench_{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data


def mean_ci(arr, confidence=0.95):
    """Bootstrap 95% CI for the mean."""
    arr = np.array(arr, dtype=float)
    n = len(arr)
    res = bootstrap((arr,), np.mean, confidence_level=confidence,
                    n_resamples=5000, random_state=42, method="percentile")
    return res.confidence_interval.low, res.confidence_interval.high


def analyze_experiment(name, baseline_data):
    data = load_eval(name)
    bench = load_bench(name)

    # Align by id
    base_map = {d["id"]: d["score"] for d in baseline_data if "score" in d}
    records = [(d["id"], d["score"], d.get("query_type", "unknown"))
               for d in data if "score" in d and d["id"] in base_map]

    ids, scores, qtypes = zip(*records)
    scores = np.array(scores, dtype=float)
    base_scores = np.array([base_map[i] for i in ids], dtype=float)

    # Timing
    if bench:
        bench_map = {d["id"]: d.get("time_sec", np.nan) for d in bench}
        times = np.array([bench_map.get(i, np.nan) for i in ids])
        valid_times = times[~np.isnan(times)]
        mean_time = float(np.mean(valid_times)) if len(valid_times) > 0 else np.nan
    else:
        mean_time = np.nan

    # Overall stats
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    ci_lo, ci_hi = mean_ci(scores)

    # Wilcoxon signed-rank test vs baseline
    diffs = scores - base_scores
    if np.all(diffs == 0):
        w_stat, w_p = np.nan, 1.0
    else:
        w_stat, w_p = wilcoxon(scores, base_scores, alternative="two-sided")

    # Effect size: Cohen's d
    pooled_std = np.sqrt((np.std(scores)**2 + np.std(base_scores)**2) / 2)
    cohens_d = (mean_score - np.mean(base_scores)) / pooled_std if pooled_std > 0 else 0.0

    # Win/tie/loss vs baseline
    wins = int(np.sum(scores > base_scores))
    ties = int(np.sum(scores == base_scores))
    losses = int(np.sum(scores < base_scores))

    # By query type
    by_type = {}
    for qt in ["abstractive", "extractive"]:
        mask = np.array([q == qt for q in qtypes])
        if mask.sum() == 0:
            continue
        qt_scores = scores[mask]
        qt_base = base_scores[mask]
        qt_ci_lo, qt_ci_hi = mean_ci(qt_scores)
        if np.all(qt_scores == qt_base):
            qt_p = 1.0
        else:
            _, qt_p = wilcoxon(qt_scores, qt_base, alternative="two-sided")
        by_type[qt] = {
            "n": int(mask.sum()),
            "mean": float(np.mean(qt_scores)),
            "std": float(np.std(qt_scores)),
            "ci_lo": float(qt_ci_lo),
            "ci_hi": float(qt_ci_hi),
            "wilcoxon_p": float(qt_p),
            "baseline_mean": float(np.mean(qt_base)),
        }

    return {
        "name": name,
        "label": EXP_LABELS.get(name, name),
        "n": len(scores),
        "mean": mean_score,
        "std": std_score,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "median": float(np.median(scores)),
        "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
        "wilcoxon_p": float(w_p),
        "cohens_d": float(cohens_d),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mean_time_sec": mean_time,
        "by_type": by_type,
        "scores": scores.tolist(),
        "base_scores": base_scores.tolist(),
    }


def significance_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    baseline_data = load_eval(BASELINE)
    base_scores_all = np.array([d["score"] for d in baseline_data if "score" in d])

    # All experiments including baseline itself
    exp_names = list(EXP_LABELS.keys())
    results = []
    for name in exp_names:
        try:
            r = analyze_experiment(name, baseline_data)
            results.append(r)
            print(f"[OK] {name}: mean={r['mean']:.3f} ± {r['std']:.3f}  "
                  f"p={r['wilcoxon_p']:.4f}{significance_stars(r['wilcoxon_p'])}  "
                  f"d={r['cohens_d']:.3f}")
        except FileNotFoundError:
            print(f"[SKIP] {name}: file not found")

    # Save detailed JSON
    detail_path = os.path.join(OUT_DIR, "stats_detailed.json")
    with open(detail_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed stats -> {detail_path}")

    # Build summary CSV
    rows = []
    for r in results:
        is_baseline = r["name"] == BASELINE
        row = {
            "Experiment": r["label"],
            "N": r["n"],
            "Mean Score": round(r["mean"], 3),
            "Std": round(r["std"], 3),
            "95% CI Low": round(r["ci_lo"], 3),
            "95% CI High": round(r["ci_hi"], 3),
            "Median": round(r["median"], 1),
            "vs Baseline (Wilcoxon p)": "-" if is_baseline else f"{r['wilcoxon_p']:.4f}",
            "Significance": "-" if is_baseline else significance_stars(r["wilcoxon_p"]),
            "Cohen's d": "-" if is_baseline else round(r["cohens_d"], 3),
            "W/T/L vs Baseline": "-" if is_baseline else f"{r['wins']}/{r['ties']}/{r['losses']}",
            "Mean Time (s)": round(r["mean_time_sec"], 2) if not np.isnan(r["mean_time_sec"]) else "-",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "stats_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV     -> {csv_path}")
    print("\n" + df.to_string(index=False))

    # Pairwise comparison matrix (all vs all, Mann-Whitney U)
    print("\n--- Pairwise Mann-Whitney U p-values ---")
    exp_means = [(r["name"], np.array(r["scores"])) for r in results]
    names = [r["name"] for r in results]
    matrix = np.ones((len(results), len(results)))
    for i, (n1, s1) in enumerate(exp_means):
        for j, (n2, s2) in enumerate(exp_means):
            if i != j:
                _, p = mannwhitneyu(s1, s2, alternative="two-sided")
                matrix[i, j] = p
    df_matrix = pd.DataFrame(matrix, index=names, columns=names).round(4)
    matrix_path = os.path.join(OUT_DIR, "pairwise_pvalues.csv")
    df_matrix.to_csv(matrix_path)
    print(f"Pairwise matrix -> {matrix_path}")

    return results


if __name__ == "__main__":
    main()
