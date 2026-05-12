"""Q11. Agglomerative Hierarchical Clustering."""

from functools import cache

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel
from rich.panel import Panel
from rich.table import Table
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split

from question_01 import console, get_clean_df
from question_10 import ScalerArtifacts, get_scaler_artifacts

SAMPLE_SIZE = 300
RANDOM_STATE = 42
LINKAGE_METHOD = "average"
DISTANCE_METRIC = "euclidean"
NUM_CLUSTERS = 3


class ClusterArtifacts(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    sample_df: pd.DataFrame
    scaled_sample: pd.DataFrame
    cluster_labels: pd.Series
    linkage_matrix: list[list[float]]
    num_clusters: int
    linkage_method: str


@cache
def get_cluster_artifacts() -> ClusterArtifacts:
    df = get_clean_df()
    scaler_artifacts = get_scaler_artifacts()

    sample_df, scaled_sample = sample_employees(df, scaler_artifacts)
    cluster_labels, linkage_matrix = cluster(scaled_sample)

    return ClusterArtifacts(
        sample_df=sample_df,
        scaled_sample=scaled_sample,
        cluster_labels=cluster_labels,
        linkage_matrix=linkage_matrix,
        num_clusters=NUM_CLUSTERS,
        linkage_method=LINKAGE_METHOD,
    )


def sample_employees(df: pd.DataFrame, scaler_artifacts: ScalerArtifacts) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, sample_df = train_test_split(
        df,
        test_size=SAMPLE_SIZE,
        stratify=df["Attrition"],
        random_state=RANDOM_STATE,
    )
    sample_df = sample_df.reset_index(drop=True)

    scaled_sample = pd.DataFrame(
        scaler_artifacts.scaler.transform(sample_df[scaler_artifacts.feature_names]),
        columns=scaler_artifacts.feature_names,
    )
    return sample_df, scaled_sample


def cluster(scaled_sample: pd.DataFrame) -> tuple[pd.Series, list[list[float]]]:
    linkage_matrix: list[list[float]] = linkage(scaled_sample, method=LINKAGE_METHOD, metric=DISTANCE_METRIC)

    model = AgglomerativeClustering(
        n_clusters=NUM_CLUSTERS,
        linkage=LINKAGE_METHOD,
        metric=DISTANCE_METRIC,
    )
    labels: pd.Series = pd.Series(model.fit_predict(scaled_sample), name="Cluster")
    return labels, linkage_matrix


def report_clusters(sample_df: pd.DataFrame, cluster_labels: pd.Series) -> None:
    profile_df = sample_df.copy()
    profile_df["Cluster"] = cluster_labels

    table = Table(title=f"Cluster Profiles — {LINKAGE_METHOD.title()} Linkage ({DISTANCE_METRIC.title()})", show_lines=True)
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


def plot_dendrogram(linkage_matrix: list[list[float]]) -> None:
    _fig, ax = plt.subplots(figsize=(14, 6))

    dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=9,
        color_threshold=0,
    )

    ax.set_title(f"Dendrogram — {LINKAGE_METHOD.title()} Linkage ({DISTANCE_METRIC.title()}, n={SAMPLE_SIZE})")
    ax.set_xlabel("Employee Index (truncated to last 30 merges)")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = get_clean_df()
    scaler_artifacts = get_scaler_artifacts()

    console.print(
        Panel(
            f"[bold]Sample size:[/bold] {SAMPLE_SIZE}  "
            f"[bold]Linkage:[/bold] {LINKAGE_METHOD}  "
            f"[bold]Distance:[/bold] {DISTANCE_METRIC}  "
            f"[bold]Clusters:[/bold] {NUM_CLUSTERS}\n"
            f"[dim]Stratified on Attrition to preserve class balance.[/dim]",
            title="Workforce Attrition — Q11 Hierarchical Clustering",
        ),
    )

    sample_df, scaled_sample = sample_employees(df, scaler_artifacts)
    cluster_labels, linkage_matrix = cluster(scaled_sample)

    report_clusters(sample_df, cluster_labels)
    plot_dendrogram(linkage_matrix)


if __name__ == "__main__":
    main()
