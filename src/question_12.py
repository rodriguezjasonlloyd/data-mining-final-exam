"""Q12. Linkage Method Comparison."""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel
from rich.panel import Panel
from rich.table import Table
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

from question_01 import console
from question_10 import SCALER_PATH, ScalerArtifacts
from question_11 import CLUSTER_PATH, NUM_CLUSTERS, SAMPLE_SIZE, ClusterArtifacts

COMPLETE_LINKAGE_METHOD = "complete"
DISTANCE_METRIC = "euclidean"
COMPLETE_CLUSTER_PATH = "clusters_complete.joblib"


class CompleteLinkageArtifacts(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    sample_df: pd.DataFrame
    scaled_sample: pd.DataFrame
    cluster_labels: pd.Series
    linkage_matrix: list[list[float]]
    num_clusters: int
    linkage_method: str


def load_artifacts() -> tuple[ClusterArtifacts, ScalerArtifacts]:
    average_artifacts = ClusterArtifacts(**joblib.load(CLUSTER_PATH))
    scaler_artifacts = ScalerArtifacts(**joblib.load(SCALER_PATH))
    return average_artifacts, scaler_artifacts


def cluster_complete(average_artifacts: ClusterArtifacts) -> CompleteLinkageArtifacts:
    scaled_sample: pd.DataFrame = average_artifacts.scaled_sample
    linkage_matrix: list[list[float]] = linkage(scaled_sample, method=COMPLETE_LINKAGE_METHOD, metric=DISTANCE_METRIC)

    model = AgglomerativeClustering(
        n_clusters=NUM_CLUSTERS,
        linkage=COMPLETE_LINKAGE_METHOD,
        metric=DISTANCE_METRIC,
    )
    cluster_labels: pd.Series = pd.Series(model.fit_predict(scaled_sample), name="Cluster")

    return CompleteLinkageArtifacts(
        sample_df=average_artifacts.sample_df,
        scaled_sample=scaled_sample,
        cluster_labels=cluster_labels,
        linkage_matrix=linkage_matrix,
        num_clusters=NUM_CLUSTERS,
        linkage_method=COMPLETE_LINKAGE_METHOD,
    )


def report_comparison(average_artifacts: ClusterArtifacts, complete_artifacts: CompleteLinkageArtifacts) -> None:
    for artifacts, method in zip(
        [average_artifacts, complete_artifacts],
        [average_artifacts.linkage_method, complete_artifacts.linkage_method],
        strict=False,
    ):
        profile_df = artifacts.sample_df.copy()
        profile_df["Cluster"] = artifacts.cluster_labels

        cluster_sizes = profile_df["Cluster"].value_counts().sort_index()
        min_size: int = int(cluster_sizes.min())
        max_size: int = int(cluster_sizes.max())
        balance_ratio: float = round(min_size / max_size, 3)

        table = Table(title=f"Cluster Profiles — {method.title()} Linkage ({DISTANCE_METRIC.title()})", show_lines=True)
        table.add_column("Cluster", style="cyan")
        table.add_column("Size", justify="right")
        table.add_column("Attrition %", justify="right")
        table.add_column("Avg Salary", justify="right")
        table.add_column("Avg Performance", justify="right")
        table.add_column("Avg Satisfaction", justify="right")
        table.add_column("Avg Tenure", justify="right")

        for cluster_id in sorted(profile_df["Cluster"].unique()):
            group = profile_df[profile_df["Cluster"] == cluster_id]
            attrition_pct: float = group["Attrition"].mean() * 100
            color = "red" if attrition_pct > 40 else "yellow" if attrition_pct > 20 else "green"

            table.add_row(
                str(cluster_id),
                str(len(group)),
                f"[{color}]{attrition_pct:.1f}%[/{color}]",
                f"{group['Monthly_Salary_PHP'].mean():,.0f}",
                f"{group['Performance_Score'].mean():.2f}",
                f"{group['Job_Satisfaction_Score'].mean():.2f}",
                f"{group['Tenure_Years'].mean():.2f}",
            )

        console.print(table)
        console.print(f"  [dim]Balance ratio (min/max cluster size): {balance_ratio:.3f} — closer to 1.0 is more balanced[/dim]\n")

    summary_table = Table(title="Linkage Method Comparison Summary", show_lines=True)
    summary_table.add_column("Method", style="cyan")
    summary_table.add_column("Cluster Sizes", justify="right")
    summary_table.add_column("Balance Ratio", justify="right")
    summary_table.add_column("Chaining?", justify="right")
    summary_table.add_column("Recommendation", justify="left")

    average_sizes = average_artifacts.sample_df.copy()
    average_sizes["Cluster"] = average_artifacts.cluster_labels
    average_counts = average_sizes["Cluster"].value_counts().sort_index()
    average_balance: float = round(int(average_counts.min()) / int(average_counts.max()), 3)
    average_chaining: bool = int(average_counts.max()) > SAMPLE_SIZE * 0.8

    complete_sizes = complete_artifacts.sample_df.copy()
    complete_sizes["Cluster"] = complete_artifacts.cluster_labels
    complete_counts = complete_sizes["Cluster"].value_counts().sort_index()
    complete_balance: float = round(int(complete_counts.min()) / int(complete_counts.max()), 3)
    complete_chaining: bool = int(complete_counts.max()) > SAMPLE_SIZE * 0.8

    summary_table.add_row(
        "Average",
        str(average_counts.tolist()),
        f"{average_balance:.3f}",
        "[red]Yes[/red]" if average_chaining else "[green]No[/green]",
        "[red]Not suitable — chaining detected[/red]" if average_chaining else "[green]Suitable[/green]",
    )
    summary_table.add_row(
        "Complete",
        str(complete_counts.tolist()),
        f"{complete_balance:.3f}",
        "[red]Yes[/red]" if complete_chaining else "[green]No[/green]",
        "[green]More balanced separation[/green]" if not complete_chaining else "[red]Not suitable — chaining detected[/red]",
    )

    console.print(summary_table)


def plot_dendrograms(average_artifacts: ClusterArtifacts, complete_artifacts: CompleteLinkageArtifacts) -> None:
    _fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    for ax, artifacts, method in zip(
        axes,
        [average_artifacts, complete_artifacts],
        [average_artifacts.linkage_method, complete_artifacts.linkage_method],
        strict=False,
    ):
        dendrogram(
            artifacts.linkage_matrix,
            ax=ax,
            truncate_mode="lastp",
            p=30,
            leaf_rotation=90,
            leaf_font_size=9,
            color_threshold=0,
        )
        ax.set_title(f"{method.title()} Linkage — Dendrogram\n(n={SAMPLE_SIZE}, truncated to last 30 merges)")
        ax.set_xlabel("Employee Index")
        ax.set_ylabel("Distance")

    plt.suptitle("Linkage Method Comparison — Average vs Complete", fontsize=13)
    plt.tight_layout()
    plt.show()


def main() -> None:
    average_artifacts, _scaler_artifacts = load_artifacts()

    console.print(
        Panel(
            f"[bold]Sample size:[/bold] {SAMPLE_SIZE}  "
            f"[bold]Clusters:[/bold] {NUM_CLUSTERS}  "
            f"[bold]Distance:[/bold] {DISTANCE_METRIC}\n"
            f"[dim]Comparing average linkage (Q11) vs complete linkage.[/dim]",
            title="Workforce Attrition — Q12 Linkage Method Comparison",
        ),
    )

    complete_artifacts = cluster_complete(average_artifacts)
    report_comparison(average_artifacts, complete_artifacts)
    plot_dendrograms(average_artifacts, complete_artifacts)

    joblib.dump(complete_artifacts.model_dump(), COMPLETE_CLUSTER_PATH)
    console.print(f"[dim]Complete linkage artifacts saved to {COMPLETE_CLUSTER_PATH}[/dim]")


if __name__ == "__main__":
    main()
