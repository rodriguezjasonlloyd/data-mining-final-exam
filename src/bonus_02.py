"""Bonus 2. Clustering Comparison Analysis."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel
from rich.table import Table
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from question_01 import console, get_clean_df
from question_10 import get_scaler_artifacts
from question_11 import NUM_CLUSTERS, SAMPLE_SIZE, sample_employees

DISTANCE_METRICS: list[str] = ["euclidean", "manhattan"]
LINKAGE_METHODS: list[str] = ["average", "complete", "ward"]
WARD_REQUIRES_EUCLIDEAN = "ward"

SCIPY_METRIC_ALIAS: dict[str, str] = {
    "euclidean": "euclidean",
    "manhattan": "cityblock",
}


class CombinationResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    linkage_method: str
    distance_metric: str
    linkage_matrix: np.ndarray
    cluster_labels: pd.Series
    cluster_sizes: dict[int, int]
    balance_ratio: float
    silhouette: float


def is_valid_combination(linkage_method: str, distance_metric: str) -> bool:
    return not (linkage_method == WARD_REQUIRES_EUCLIDEAN and distance_metric != "euclidean")


def valid_combinations() -> list[tuple[str, str]]:
    return [
        (linkage_method, distance_metric)
        for distance_metric in DISTANCE_METRICS
        for linkage_method in LINKAGE_METHODS
        if is_valid_combination(linkage_method, distance_metric)
    ]


def run_combination(
    scaled_sample: pd.DataFrame,
    linkage_method: str,
    distance_metric: str,
) -> CombinationResult:
    scipy_metric: str = SCIPY_METRIC_ALIAS[distance_metric]
    linkage_matrix: np.ndarray = linkage(scaled_sample, method=linkage_method, metric=scipy_metric)

    model = AgglomerativeClustering(
        n_clusters=NUM_CLUSTERS,
        linkage=linkage_method,
        metric=scipy_metric,
    )
    cluster_labels: pd.Series = pd.Series(model.fit_predict(scaled_sample), name="Cluster")

    cluster_sizes: dict[int, int] = cluster_labels.value_counts().sort_index().to_dict()
    sizes: list[int] = list(cluster_sizes.values())
    balance_ratio: float = max(sizes) / min(sizes) if min(sizes) > 0 else float("inf")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        silhouette: float = float(silhouette_score(scaled_sample, cluster_labels))

    return CombinationResult(
        linkage_method=linkage_method,
        distance_metric=distance_metric,
        linkage_matrix=linkage_matrix,
        cluster_labels=cluster_labels,
        cluster_sizes=cluster_sizes,
        balance_ratio=balance_ratio,
        silhouette=silhouette,
    )


def report_comparison(results: list[CombinationResult]) -> None:
    table = Table(title=f"Clustering Method Comparison (n={SAMPLE_SIZE}, k={NUM_CLUSTERS})", show_lines=True)
    table.add_column("Linkage", style="bright_cyan")
    table.add_column("Distance", style="bright_cyan")
    table.add_column("Cluster Sizes", justify="right")
    table.add_column("Balance Ratio", justify="right")
    table.add_column("Silhouette Score", justify="right")

    best_silhouette: float = max(r.silhouette for r in results)

    for result in results:
        sizes_str: str = " / ".join(str(result.cluster_sizes[k]) for k in sorted(result.cluster_sizes))
        silhouette_color = "bright_green" if result.silhouette == best_silhouette else "bright_yellow" if result.silhouette > 0.15 else "bright_red"
        balance_color = "bright_green" if result.balance_ratio < 2.0 else "bright_yellow" if result.balance_ratio < 4.0 else "bright_red"

        table.add_row(
            result.linkage_method.title(),
            result.distance_metric.title(),
            sizes_str,
            f"[{balance_color}]{result.balance_ratio:.2f}[/{balance_color}]",
            f"[{silhouette_color}]{result.silhouette:.4f}[/{silhouette_color}]",
        )

    console.print(table)


def report_recommendation(results: list[CombinationResult]) -> None:
    best: CombinationResult = max(results, key=lambda r: r.silhouette)
    sizes_str: str = " / ".join(str(best.cluster_sizes[k]) for k in sorted(best.cluster_sizes))

    console.print(
        f"[bold]Recommended setup:[/bold] [bright_green]{best.linkage_method.title()} Linkage + {best.distance_metric.title()} Distance[/bright_green]\n"
        f"[bold]Silhouette score:[/bold] {best.silhouette:.4f} — highest among all valid combinations, "
        f"indicating the most cohesive and well-separated clusters.\n"
        f"[bold]Balance ratio:[/bold] {best.balance_ratio:.2f} — cluster sizes ({sizes_str}) are "
        f"{'well-balanced, reducing bias toward the dominant cluster' if best.balance_ratio < 2.0 else 'moderately unbalanced but acceptable for HR segmentation'}.\n"
        f"[bold]HR rationale:[/bold] A higher silhouette score means employees within each cluster are more "
        f"similar to each other than to those in other clusters — making cluster-specific HR interventions "
        f"more targeted and actionable. Ward linkage minimizes within-cluster variance at each merge step, "
        f"which naturally produces compact, interpretable groups suitable for HR profiling.",
    )


def plot_dendrograms(results: list[CombinationResult]) -> None:
    num_combinations: int = len(results)
    ncols: int = 2 if num_combinations > 3 else num_combinations
    nrows: int = (num_combinations + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 9, nrows * 5))
    axes_flat = axes.flatten() if num_combinations > 1 else [axes]

    for index, result in enumerate(results):
        ax = axes_flat[index]
        dendrogram(
            result.linkage_matrix,
            ax=ax,
            truncate_mode="lastp",
            p=30,
            leaf_rotation=90,
            leaf_font_size=8,
            color_threshold=0,
        )
        ax.set_title(
            f"{result.linkage_method.title()} Linkage + {result.distance_metric.title()}\nSilhouette={result.silhouette:.4f}",
            fontsize=11,
        )
        ax.set_xlabel("Employee Index (last 30 merges)")
        ax.set_ylabel("Distance")

    for index in range(len(results), len(axes_flat)):
        axes_flat[index].set_visible(False)

    fig.suptitle(f"Dendrogram Comparison — {num_combinations} Valid Combinations (n={SAMPLE_SIZE})", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def main() -> None:
    scaler_artifacts = get_scaler_artifacts()
    df = get_clean_df()
    _sample_df, scaled_sample = sample_employees(df, scaler_artifacts)

    combinations = valid_combinations()

    console.print(f"[bold]Sample size:[/bold] {SAMPLE_SIZE}  [bold]Clusters:[/bold] {NUM_CLUSTERS}  [bold]Valid combinations:[/bold] {len(combinations)}\n")

    results: list[CombinationResult] = []
    for linkage_method, distance_metric in combinations:
        result = run_combination(scaled_sample, linkage_method, distance_metric)
        results.append(result)

    report_comparison(results)
    report_recommendation(results)
    plot_dendrograms(results)


if __name__ == "__main__":
    main()
