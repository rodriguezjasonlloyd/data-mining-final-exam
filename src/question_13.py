"""Q13. Cluster Profiling."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.table import Table

from question_01 import console
from question_12 import CompleteLinkageArtifacts, get_complete_cluster_artifacts

BOX_PLOT_FEATURES = [
    "Monthly_Salary_PHP",
    "Performance_Score",
    "Job_Satisfaction_Score",
    "Work_Life_Balance_Score",
    "Tenure_Years",
    "Absences_YTD",
    "Training_Hours_YTD",
    "Overtime_Hours_Monthly",
]

CLUSTER_NAMES: dict[int, str] = {
    0: "Stable Performers",
    1: "Outlier Pair",
    2: "Flight Risk",
}

CLUSTER_INTERVENTIONS: dict[int, str] = {
    0: "Retention programs and career pathing to sustain engagement",
    1: "Individual case review — investigate anomalous profiles",
    2: "Urgent re-engagement: satisfaction surveys, salary review, workload audit",
}


def build_profile_frame(artifacts: CompleteLinkageArtifacts) -> pd.DataFrame:
    profile = artifacts.sample_df.copy()
    profile["Cluster"] = artifacts.cluster_labels
    profile["Cluster_Name"] = profile["Cluster"].map(CLUSTER_NAMES)
    return profile


def report_cluster_profiles(profile: pd.DataFrame) -> None:
    table = Table(title="Cluster Profiles — Complete Linkage with Descriptive Names", show_lines=True)
    table.add_column("Cluster", style="bright_cyan")
    table.add_column("Name", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Attrition %", justify="right")
    table.add_column("Avg Salary", justify="right")
    table.add_column("Avg Performance", justify="right")
    table.add_column("Avg Satisfaction", justify="right")
    table.add_column("Avg Tenure", justify="right")
    table.add_column("Avg Absences", justify="right")

    for cluster_id in sorted(profile["Cluster"].unique()):
        group = profile[profile["Cluster"] == cluster_id]
        attrition_pct: float = group["Attrition"].mean() * 100
        color = "bright_red" if attrition_pct > 60 else "bright_yellow" if attrition_pct > 40 else "bright_green"

        table.add_row(
            str(cluster_id),
            CLUSTER_NAMES[cluster_id],
            str(len(group)),
            f"[{color}]{attrition_pct:.1f}%[/{color}]",
            f"{group['Monthly_Salary_PHP'].mean():,.0f}",
            f"{group['Performance_Score'].mean():.2f}",
            f"{group['Job_Satisfaction_Score'].mean():.2f}",
            f"{group['Tenure_Years'].mean():.2f}",
            f"{group['Absences_YTD'].mean():.2f}",
        )

    console.print(table)


def report_interventions(profile: pd.DataFrame) -> None:
    table = Table(title="HR Interventions by Cluster", show_lines=True)
    table.add_column("Cluster", style="bright_cyan")
    table.add_column("Name", style="bold")
    table.add_column("Recommended Intervention")

    for cluster_id in sorted(profile["Cluster"].unique()):
        table.add_row(
            str(cluster_id),
            CLUSTER_NAMES[cluster_id],
            CLUSTER_INTERVENTIONS[cluster_id],
        )

    console.print(table)


def plot_box_plots(profile: pd.DataFrame) -> None:
    num_features = len(BOX_PLOT_FEATURES)
    num_cols = 2
    num_rows = (num_features + 1) // num_cols

    _, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 4))
    axes_flat = axes.flatten()

    for index, feature in enumerate(BOX_PLOT_FEATURES):
        ax = axes_flat[index]
        sns.boxplot(
            data=profile,
            x="Cluster_Name",
            y=feature,
            hue="Cluster_Name",
            palette={"Stable Performers": "steelblue", "Outlier Pair": "goldenrod", "Flight Risk": "tomato"},
            legend=False,
            ax=ax,
        )
        ax.set_title(feature)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(feature)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    for index in range(num_features, len(axes_flat)):
        axes_flat[index].set_visible(False)

    plt.suptitle("Feature Distributions by Cluster — Complete Linkage", fontsize=14)
    plt.tight_layout()
    plt.show()


def main() -> None:
    artifacts = get_complete_cluster_artifacts()
    profile = build_profile_frame(artifacts)

    report_cluster_profiles(profile)
    report_interventions(profile)
    plot_box_plots(profile)


if __name__ == "__main__":
    main()
