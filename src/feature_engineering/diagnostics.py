"""Diagnostics: missingness, coverage over time, distributions, correlations,
outlier rates, feature stability, and universe coverage.

Correlations are computed on the **model-ready** (winsorized + rank-scaled)
panel rather than raw units — raw ratios have wildly different scales and
outlier sensitivity, which would dominate a Pearson correlation and obscure
the actual cross-feature relationships. This is a diagnostic-only choice;
it does not affect either saved feature panel.

Chart color follows the standard convention for each job: sequential
(magnitude — coverage/heatmap-of-counts) uses one hue light-to-dark
(`viridis`); correlation (polarity, -1..1) uses a diverging red-blue
colormap centered at zero (`RdBu_r`). No rainbow colormaps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_engineering.registry import feature_names_by_category


def missingness_report(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    n = len(df)
    rows = [
        {
            "feature": col,
            "n_non_null": int(df[col].notna().sum()),
            "pct_missing": round(100 * (1 - df[col].notna().sum() / n), 4),
        }
        for col in columns
        if col in df.columns
    ]
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False).reset_index(drop=True)


def coverage_by_year(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    year = df["MthCalDt"].dt.year
    present_cols = [c for c in columns if c in df.columns]
    return df.groupby(year)[present_cols].apply(lambda g: g.notna().mean() * 100)


def coverage_by_month(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    month = df["MthCalDt"].dt.to_period("M").astype(str)
    present_cols = [c for c in columns if c in df.columns]
    return df.groupby(month)[present_cols].apply(lambda g: g.notna().mean() * 100)


def summary_distributions(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    present_cols = [c for c in columns if c in df.columns]
    stats = df[present_cols].astype("float64").describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    return stats.reset_index().rename(columns={"index": "feature"})


def pairwise_correlations(model_ready_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    present_cols = [c for c in columns if c in model_ready_df.columns]
    return model_ready_df[present_cols].astype("float64").corr()


def outlier_diagnostics(df: pd.DataFrame, columns: list[str], n_std: float = 3.0) -> pd.DataFrame:
    """Percentage of non-null observations more than `n_std` standard
    deviations from that feature's own overall mean."""
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].astype("float64")
        valid = series.dropna()
        if valid.empty or valid.std() == 0:
            pct = 0.0
        else:
            z = (valid - valid.mean()) / valid.std()
            pct = round(100 * (z.abs() > n_std).mean(), 4)
        rows.append({"feature": col, "pct_beyond_3std": pct})
    return pd.DataFrame(rows).sort_values("pct_beyond_3std", ascending=False).reset_index(drop=True)


def feature_stability_through_time(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Monthly cross-sectional median per feature — a compact stability/drift diagnostic."""
    month = df["MthCalDt"].dt.to_period("M").astype(str)
    present_cols = [c for c in columns if c in df.columns]
    return df.groupby(month)[present_cols].median()


def universe_coverage_through_time(df: pd.DataFrame) -> pd.DataFrame:
    month = df["MthCalDt"].dt.to_period("M").astype(str)
    grouped = df.groupby(month)["is_investable"]
    return pd.DataFrame(
        {
            "n_total": grouped.size(),
            "n_investable": grouped.sum(),
            "pct_investable": grouped.mean() * 100,
        }
    )


def _save_heatmap(
    matrix: pd.DataFrame, title: str, path: Path, cmap: str, vmin: float, vmax: float
) -> None:
    n = len(matrix)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.4), max(5, n * 0.4)))
    im = ax.imshow(matrix.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_correlation_heatmaps(model_ready_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_category = feature_names_by_category()

    all_features = [f for cat in by_category.values() for f in cat]
    full_corr = pairwise_correlations(model_ready_df, all_features)
    _save_heatmap(
        full_corr, "All features (model-ready)", output_dir / "correlation_all.png", "RdBu_r", -1, 1
    )

    for category, names in by_category.items():
        present = [n for n in names if n in model_ready_df.columns]
        if len(present) < 2:
            continue
        corr = pairwise_correlations(model_ready_df, present)
        _save_heatmap(
            corr,
            f"{category} (model-ready)",
            output_dir / f"correlation_{category}.png",
            "RdBu_r",
            -1,
            1,
        )


def save_universe_coverage_plot(coverage: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(coverage))
    ax.plot(x, coverage["pct_investable"].to_numpy(), color="#1f6feb", linewidth=2)
    step = max(len(coverage) // 12, 1)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(coverage.index[::step], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("% investable")
    ax.set_title("Investable-universe coverage through time")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
