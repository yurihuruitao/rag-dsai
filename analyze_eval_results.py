import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 results 目录下 eval JSON 的结果")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results"),
        help="包含 eval_*.json 的目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis"),
        help="统计结果输出目录",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="baseline",
        help="作为显著性比较基准的实验名，对应 eval_<name>.json",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="汇总图片分辨率 DPI",
    )
    return parser.parse_args()


def load_eval_files(input_dir: Path) -> pd.DataFrame:
    records: List[Dict] = []
    for path in sorted(input_dir.glob("eval_*.json")):
        experiment = path.stem.removeprefix("eval_")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            sources = item.get("sources") or []
            records.append(
                {
                    "experiment": experiment,
                    "index": item.get("index"),
                    "id": item.get("id"),
                    "query_type": item.get("query_type", "unknown") or "unknown",
                    "score": safe_float(item.get("score")),
                    "time_sec": safe_float(item.get("time_sec")),
                    "source_count": len(sources),
                    "has_eval_error": 1 if item.get("eval_error") else 0,
                    "has_sources": 1 if sources else 0,
                }
            )

    if not records:
        raise FileNotFoundError(f"未在 {input_dir} 中找到 eval_*.json")

    df = pd.DataFrame.from_records(records)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df["source_count"] = pd.to_numeric(df["source_count"], errors="coerce")
    return df


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def confidence_interval_95(values: pd.Series) -> tuple[float, float]:
    arr = values.dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return arr[0], arr[0]
    if np.allclose(arr, arr[0]):
        return arr[0], arr[0]
    sem = stats.sem(arr, nan_policy="omit")
    interval = stats.t.interval(0.95, df=arr.size - 1, loc=np.mean(arr), scale=sem)
    return float(interval[0]), float(interval[1])


def describe_scores(group: pd.DataFrame) -> pd.Series:
    scores = group["score"].dropna()
    time_values = group["time_sec"].dropna()
    source_values = group["source_count"].dropna()
    ci_low, ci_high = confidence_interval_95(scores)

    return pd.Series(
        {
            "n": int(scores.count()),
            "mean_score": scores.mean(),
            "std_score": scores.std(ddof=1),
            "median_score": scores.median(),
            "min_score": scores.min(),
            "max_score": scores.max(),
            "q1_score": scores.quantile(0.25),
            "q3_score": scores.quantile(0.75),
            "iqr_score": scores.quantile(0.75) - scores.quantile(0.25),
            "sem_score": scores.sem(ddof=1),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "score_skew": scores.skew(),
            "score_kurtosis": scores.kurtosis(),
            "pass_ge_6_rate": (scores >= 6).mean(),
            "pass_ge_8_rate": (scores >= 8).mean(),
            "pass_ge_9_rate": (scores >= 9).mean(),
            "perfect_10_rate": (scores >= 10).mean(),
            "avg_time_sec": time_values.mean(),
            "median_time_sec": time_values.median(),
            "p95_time_sec": time_values.quantile(0.95),
            "avg_source_count": source_values.mean(),
            "median_source_count": source_values.median(),
            "source_coverage_rate": (source_values > 0).mean(),
            "eval_error_rate": group["has_eval_error"].mean(),
        }
    )


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = (
        df.groupby("experiment", sort=True, dropna=False)
        .apply(describe_scores, include_groups=False)
        .reset_index()
    )
    by_type = (
        df.groupby(["experiment", "query_type"], sort=True, dropna=False)
        .apply(describe_scores, include_groups=False)
        .reset_index()
    )
    return overall, by_type


def paired_significance(df: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    baseline_df = df[df["experiment"] == baseline_name][["id", "score"]].rename(
        columns={"score": "baseline_score"}
    )
    if baseline_df.empty:
        raise ValueError(f"未找到 baseline 实验: {baseline_name}")

    rows = []
    experiments = [name for name in sorted(df["experiment"].unique()) if name != baseline_name]
    for experiment in experiments:
        current_df = df[df["experiment"] == experiment][["id", "score"]].rename(
            columns={"score": "experiment_score"}
        )
        paired = baseline_df.merge(current_df, on="id", how="inner").dropna()
        if paired.empty:
            continue

        diffs = paired["experiment_score"] - paired["baseline_score"]
        wins = int((diffs > 0).sum())
        ties = int((diffs == 0).sum())
        losses = int((diffs < 0).sum())

        ttest = stats.ttest_rel(
            paired["experiment_score"], paired["baseline_score"], nan_policy="omit"
        )
        try:
            wilcoxon = stats.wilcoxon(
                paired["experiment_score"],
                paired["baseline_score"],
                zero_method="wilcox",
                alternative="two-sided",
            )
            wilcoxon_stat = float(wilcoxon.statistic)
            wilcoxon_p = float(wilcoxon.pvalue)
        except ValueError:
            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan

        diff_std = diffs.std(ddof=1)
        cohen_dz = np.nan if diff_std == 0 or np.isnan(diff_std) else diffs.mean() / diff_std

        rows.append(
            {
                "experiment": experiment,
                "baseline": baseline_name,
                "paired_n": int(len(paired)),
                "baseline_mean": paired["baseline_score"].mean(),
                "experiment_mean": paired["experiment_score"].mean(),
                "mean_diff": diffs.mean(),
                "median_diff": diffs.median(),
                "win_rate": wins / len(paired),
                "tie_rate": ties / len(paired),
                "loss_rate": losses / len(paired),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "paired_t_stat": float(ttest.statistic),
                "paired_t_pvalue": float(ttest.pvalue),
                "wilcoxon_stat": wilcoxon_stat,
                "wilcoxon_pvalue": wilcoxon_p,
                "cohen_dz": cohen_dz,
            }
        )

    sig_df = pd.DataFrame(rows)
    if not sig_df.empty:
        sig_df["paired_t_significant_0.05"] = sig_df["paired_t_pvalue"] < 0.05
        sig_df["wilcoxon_significant_0.05"] = sig_df["wilcoxon_pvalue"] < 0.05
    return sig_df


def save_outputs(
    overall: pd.DataFrame,
    by_type: pd.DataFrame,
    significance: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(output_dir / "eval_summary_overall.csv", index=False)
    by_type.to_csv(output_dir / "eval_summary_by_query_type.csv", index=False)
    significance.to_csv(output_dir / "eval_significance_vs_baseline.csv", index=False)

    payload = {
        "overall": overall.round(6).replace({np.nan: None}).to_dict(orient="records"),
        "by_query_type": by_type.round(6).replace({np.nan: None}).to_dict(orient="records"),
        "significance_vs_baseline": significance.round(6)
        .replace({np.nan: None})
        .to_dict(orient="records"),
    }
    with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def plot_summary(
    df: pd.DataFrame,
    overall: pd.DataFrame,
    by_type: pd.DataFrame,
    significance: pd.DataFrame,
    baseline_name: str,
    output_path: Path,
    dpi: int,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#fbfbfc"

    ordered_experiments = overall.sort_values("mean_score", ascending=False)["experiment"].tolist()
    palette = sns.color_palette("viridis", n_colors=len(ordered_experiments))
    palette_map = dict(zip(ordered_experiments, palette))

    fig = plt.figure(figsize=(24, 18), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.1])

    ax1 = fig.add_subplot(grid[0, 0])
    mean_plot = overall.sort_values("mean_score", ascending=False).copy()
    mean_plot["err_low"] = mean_plot["mean_score"] - mean_plot["ci95_low"]
    mean_plot["err_high"] = mean_plot["ci95_high"] - mean_plot["mean_score"]
    ax1.bar(
        mean_plot["experiment"],
        mean_plot["mean_score"],
        yerr=np.vstack([mean_plot["err_low"], mean_plot["err_high"]]),
        color=[palette_map[name] for name in mean_plot["experiment"]],
        capsize=6,
    )
    ax1.set_title("Mean Score with 95% CI")
    ax1.set_xlabel("")
    ax1.set_ylabel("Score")
    ax1.tick_params(axis="x", rotation=20)
    for idx, (_, row) in enumerate(mean_plot.iterrows()):
        ax1.text(idx, row["mean_score"] + 0.06, f'{row["mean_score"]:.2f}', ha="center", fontsize=11)

    ax2 = fig.add_subplot(grid[0, 1])
    sns.boxplot(
        data=df,
        x="experiment",
        y="score",
        hue="experiment",
        order=ordered_experiments,
        palette=palette_map,
        ax=ax2,
        fliersize=2,
        legend=False,
    )
    ax2.set_title("Score Distribution")
    ax2.set_xlabel("")
    ax2.set_ylabel("Score")
    ax2.tick_params(axis="x", rotation=20)

    ax3 = fig.add_subplot(grid[1, 0])
    heatmap_df = (
        by_type.pivot(index="experiment", columns="query_type", values="mean_score")
        .reindex(ordered_experiments)
    )
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Mean Score"},
        ax=ax3,
    )
    ax3.set_title("Mean Score by Query Type")
    ax3.set_xlabel("")
    ax3.set_ylabel("")

    ax4 = fig.add_subplot(grid[1, 1])
    rate_plot = overall.melt(
        id_vars="experiment",
        value_vars=["pass_ge_8_rate", "pass_ge_9_rate", "perfect_10_rate"],
        var_name="metric",
        value_name="rate",
    )
    rate_plot["metric"] = rate_plot["metric"].map(
        {
            "pass_ge_8_rate": "Score >= 8",
            "pass_ge_9_rate": "Score >= 9",
            "perfect_10_rate": "Score = 10",
        }
    )
    sns.barplot(
        data=rate_plot,
        x="experiment",
        y="rate",
        hue="metric",
        order=ordered_experiments,
        ax=ax4,
    )
    ax4.set_title("High-Score Rates")
    ax4.set_xlabel("")
    ax4.set_ylabel("Rate")
    ax4.tick_params(axis="x", rotation=20)
    ax4.legend(title="")

    ax5 = fig.add_subplot(grid[2, 0])
    time_plot = overall.sort_values("avg_time_sec", ascending=False)
    ax5.bar(
        time_plot["experiment"],
        time_plot["avg_time_sec"],
        color=[palette_map[name] for name in time_plot["experiment"]],
    )
    ax5.set_title("Average Response Time")
    ax5.set_xlabel("")
    ax5.set_ylabel("Seconds")
    ax5.tick_params(axis="x", rotation=20)
    for idx, (_, row) in enumerate(time_plot.iterrows()):
        ax5.text(idx, row["avg_time_sec"] + 0.05, f'{row["avg_time_sec"]:.2f}s', ha="center", fontsize=11)

    ax6 = fig.add_subplot(grid[2, 1])
    ax6.axis("off")
    ax6.set_title(f"Paired Significance vs {baseline_name}", loc="left", pad=16)
    if significance.empty:
        ax6.text(0.02, 0.9, "No paired significance results available.", fontsize=14)
    else:
        sig_table = significance[
            [
                "experiment",
                "mean_diff",
                "paired_t_pvalue",
                "wilcoxon_pvalue",
                "wins",
                "ties",
                "losses",
            ]
        ].copy()
        sig_table["mean_diff"] = sig_table["mean_diff"].map(lambda x: f"{x:+.3f}")
        sig_table["paired_t_pvalue"] = sig_table["paired_t_pvalue"].map(lambda x: f"{x:.3g}")
        sig_table["wilcoxon_pvalue"] = sig_table["wilcoxon_pvalue"].map(
            lambda x: "nan" if pd.isna(x) else f"{x:.3g}"
        )
        table = ax6.table(
            cellText=sig_table.values,
            colLabels=[
                "Experiment",
                "Mean Diff",
                "Paired t p",
                "Wilcoxon p",
                "Wins",
                "Ties",
                "Losses",
            ],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.15, 1.9)

    fig.suptitle("Evaluation Summary", fontsize=28, y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_console_summary(
    overall: pd.DataFrame,
    significance: pd.DataFrame,
    baseline_name: str,
    output_dir: Path,
) -> None:
    print("\n=== Overall Score Summary ===")
    cols = [
        "experiment",
        "n",
        "mean_score",
        "median_score",
        "std_score",
        "pass_ge_8_rate",
        "pass_ge_9_rate",
        "perfect_10_rate",
        "avg_time_sec",
    ]
    print(overall[cols].round(4).to_string(index=False))

    print(f"\n=== Significance vs {baseline_name} ===")
    if significance.empty:
        print("No significance results available.")
    else:
        sig_cols = [
            "experiment",
            "paired_n",
            "mean_diff",
            "paired_t_pvalue",
            "wilcoxon_pvalue",
            "wins",
            "ties",
            "losses",
            "cohen_dz",
        ]
        print(significance[sig_cols].round(6).to_string(index=False))

    print(f"\n输出目录: {output_dir}")


def main() -> None:
    args = parse_args()
    df = load_eval_files(args.input_dir)
    overall, by_type = summarize(df)
    significance = paired_significance(df, args.baseline)
    save_outputs(overall, by_type, significance, args.output_dir)
    plot_summary(
        df=df,
        overall=overall,
        by_type=by_type,
        significance=significance,
        baseline_name=args.baseline,
        output_path=args.output_dir / "eval_summary.png",
        dpi=args.dpi,
    )
    print_console_summary(overall, significance, args.baseline, args.output_dir)


if __name__ == "__main__":
    main()
