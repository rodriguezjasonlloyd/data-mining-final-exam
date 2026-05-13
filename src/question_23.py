"""Q23. Recommendation for the CHRO."""

import numpy as np
import pandas as pd
from rich.table import Table
from sklearn.metrics import accuracy_score

from question_01 import console, get_clean_df
from question_05 import ModelArtifacts, get_model_artifacts
from question_12 import CompleteLinkageArtifacts, get_complete_cluster_artifacts
from question_13 import CLUSTER_INTERVENTIONS, CLUSTER_NAMES
from question_18 import LassoArtifacts, get_lasso_artifacts
from question_21 import build_master_frame, get_flight_risk_ids


def compute_dt_accuracy(model_artifacts: ModelArtifacts) -> float:
    y_test_prediction: np.ndarray = model_artifacts.model.predict(model_artifacts.x_test)
    return float(accuracy_score(model_artifacts.y_test, y_test_prediction))


def build_cluster_summary(cluster_artifacts: CompleteLinkageArtifacts) -> pd.DataFrame:
    profile = cluster_artifacts.sample_df.copy()
    profile["Cluster"] = cluster_artifacts.cluster_labels

    rows: list[dict[str, object]] = []
    for cluster_id in sorted(profile["Cluster"].unique()):
        group = profile[profile["Cluster"] == cluster_id]
        rows.append(
            {
                "Cluster": cluster_id,
                "Name": CLUSTER_NAMES[cluster_id],
                "Size": len(group),
                "Attrition %": group["Attrition"].mean() * 100,
                "Avg Salary": group["Monthly_Salary_PHP"].mean(),
                "Avg Satisfaction": group["Job_Satisfaction_Score"].mean(),
                "Avg Absences": group["Absences_YTD"].mean(),
                "Avg Tenure": group["Tenure_Years"].mean(),
            },
        )
    return pd.DataFrame(rows).set_index("Cluster")


def report_model_performance(dt_accuracy: float, lasso_artifacts: LassoArtifacts) -> None:
    table = Table(title="Model Performance Summary", show_lines=True)
    table.add_column("Model", style="bright_cyan")
    table.add_column("Purpose", style="dim")
    table.add_column("Key Metric", justify="right")
    table.add_column("Value", justify="right")

    table.add_row("Decision Tree", "Attrition prediction", "Test Accuracy", f"{dt_accuracy:.1%}")
    table.add_row("Lasso Regression", "Salary benchmarking", "Test R²", f"{lasso_artifacts.test_r2:.4f}")
    table.add_row("Lasso Regression", "Salary benchmarking", "Test RMSE", f"PHP {lasso_artifacts.test_rmse:,.0f}")
    table.add_row("Hierarchical Clustering", "Employee segmentation", "Clusters", "3 (complete linkage)")

    console.print(table)


def report_cluster_interventions(cluster_summary: pd.DataFrame) -> None:
    table = Table(title="Cluster Profiles and Recommended HR Interventions", show_lines=True)
    table.add_column("Cluster", style="bright_cyan")
    table.add_column("Size", justify="right")
    table.add_column("Attrition %", justify="right")
    table.add_column("Avg Salary (PHP)", justify="right")
    table.add_column("Avg Satisfaction", justify="right")
    table.add_column("Avg Absences", justify="right")
    table.add_column("Recommended Intervention")

    for cluster_id, row in cluster_summary.iterrows():
        attrition_pct: float = float(row["Attrition %"])
        color = "bright_red" if attrition_pct > 60 else "bright_yellow" if attrition_pct > 40 else "bright_green"
        table.add_row(
            f"{cluster_id} — {row['Name']}",
            str(int(row["Size"])),
            f"[{color}]{attrition_pct:.1f}%[/{color}]",
            f"{row['Avg Salary']:,.0f}",
            f"{row['Avg Satisfaction']:.2f}",
            f"{row['Avg Absences']:.1f}",
            CLUSTER_INTERVENTIONS[int(str(cluster_id))],
        )

    console.print(table)


def report_salary_fairness(
    master: pd.DataFrame,
    flight_risk_ids: set[str],
) -> None:
    flight_risk_employees = master[master["Employee_ID"].isin(flight_risk_ids)]
    underpaid_flight_risk = flight_risk_employees[flight_risk_employees["salary_gap"] < 0]
    overall_underpaid = master[master["salary_gap"] < 0]

    avg_gap_flight_risk: float = float(flight_risk_employees["salary_gap"].mean())
    avg_gap_overall: float = float(master["salary_gap"].mean())

    table = Table(title="Salary Fairness Analysis", show_lines=True)
    table.add_column("Group", style="bright_cyan")
    table.add_column("Employees", justify="right")
    table.add_column("Underpaid", justify="right")
    table.add_column("Underpaid %", justify="right")
    table.add_column("Avg Salary Gap (PHP)", justify="right")

    table.add_row(
        "All Employees",
        str(len(master)),
        str(len(overall_underpaid)),
        f"{len(overall_underpaid) / len(master) * 100:.1f}%",
        f"{avg_gap_overall:+,.0f}",
    )
    table.add_row(
        "Flight Risk Cluster",
        str(len(flight_risk_employees)),
        f"[bright_red]{len(underpaid_flight_risk)}[/bright_red]",
        f"[bright_red]{len(underpaid_flight_risk) / len(flight_risk_employees) * 100:.1f}%[/bright_red]",
        f"[bright_red]{avg_gap_flight_risk:+,.0f}[/bright_red]",
    )

    console.print(table)


def report_chro_recommendation(
    cluster_summary: pd.DataFrame,
    master: pd.DataFrame,
    flight_risk_ids: set[str],
    dt_accuracy: float,
    lasso_artifacts: LassoArtifacts,
) -> None:
    flight_risk_row = cluster_summary[cluster_summary["Name"] == "Flight Risk"].iloc[0]
    flight_risk_attrition: float = float(flight_risk_row["Attrition %"])
    flight_risk_size: int = int(flight_risk_row["Size"])
    flight_risk_avg_salary: float = float(flight_risk_row["Avg Salary"])

    stable_row = cluster_summary[cluster_summary["Name"] == "Stable Performers"].iloc[0]
    stable_avg_salary: float = float(stable_row["Avg Salary"])

    flight_risk_employees = master[master["Employee_ID"].isin(flight_risk_ids)]
    avg_gap_flight_risk: float = float(flight_risk_employees["salary_gap"].mean())
    underpaid_count: int = int((flight_risk_employees["salary_gap"] < 0).sum())

    dt_predicted_leave: int = int(master["predicted_attrition"].sum())
    high_risk_intersection: int = int(((master["predicted_attrition"] == 1) & (master["Employee_ID"].isin(flight_risk_ids))).sum())
    contradiction_count: int = int(((master["predicted_attrition"] == 0) & (master["Employee_ID"].isin(flight_risk_ids))).sum())

    recommendation = f"""
[bold bright_cyan]EXECUTIVE SUMMARY[/bold bright_cyan]

A 15-year employee dataset of {len(master):,} records was analyzed using three complementary data mining approaches: Decision Tree classification for attrition prediction, Hierarchical Clustering for employee segmentation, and Lasso Regression for salary benchmarking. Together, these models surface actionable retention and compensation priorities for immediate HR attention.

[bold bright_cyan]1. RETENTION PRIORITIES[/bold bright_cyan]

The Decision Tree (test accuracy: [bright_green]{dt_accuracy:.1%}[/bright_green]) identified [bright_red]{dt_predicted_leave:,}[/bright_red] employees as likely to leave. Independently, hierarchical clustering segmented the workforce into three groups, with the [bright_red]Flight Risk[/bright_red] cluster comprising [bright_red]{flight_risk_size}[/bright_red] employees and a {flight_risk_attrition:.1f}% historical attrition rate — the highest of any segment.

[bold]{high_risk_intersection}[/bold] employees appear in both the DT's leave predictions and the Flight Risk cluster. These represent the [bold bright_red]highest-confidence departure risks[/bold bright_red] and should be the first cohort addressed through targeted retention programs.

A further [bold bright_yellow]{contradiction_count}[/bold bright_yellow] employees were predicted to stay by the Decision Tree but still belong to the Flight Risk cluster. These [italic]silent flight risks[/italic] share the structural profile of employees who have historically left — low satisfaction, elevated absences, and salary below expectations — without yet triggering the DT's explicit rules. Proactive engagement before resignation intent solidifies is critical for this group.

[bold bright_cyan]2. SALARY FAIRNESS[/bold bright_cyan]

The Lasso model (R² = [bright_green]{lasso_artifacts.test_r2:.4f}[/bright_green], RMSE = PHP {lasso_artifacts.test_rmse:,.0f}) provides a data-grounded salary benchmark. Within the Flight Risk cluster, [bright_red]{underpaid_count}[/bright_red] of {flight_risk_size} employees earn below their predicted salary, with an average shortfall of [bright_red]PHP {abs(avg_gap_flight_risk):,.0f}[/bright_red] per month. Their average actual salary (PHP {flight_risk_avg_salary:,.0f}) trails the Stable Performers segment (PHP {stable_avg_salary:,.0f}) — a gap that likely compounds disengagement.

Salary correction for underpaid Flight Risk employees is the highest-ROI retention lever available. The cost of a salary adjustment is substantially lower than the fully-loaded cost of turnover (recruitment, onboarding, lost productivity).

[bold bright_cyan]3. INTERVENTIONS BY SEGMENT[/bold bright_cyan]

See cluster intervention table above. Priority order for HR resource allocation:
  1. [bright_red]Flight Risk[/bright_red] — immediate salary review, satisfaction surveys, workload audit
  2. [bright_yellow]Outlier Pair[/bright_yellow] — individual case review for anomalous profiles
  3. [bright_green]Stable Performers[/bright_green] — career pathing and recognition programs to sustain engagement

[bold bright_cyan]4. MODELING LIMITATIONS[/bold bright_cyan]

The Lasso regression explains [bright_green]{lasso_artifacts.test_r2:.1%}[/bright_green] of salary variance, indicating that a substantial portion of compensation is driven by factors not captured in the dataset (e.g., negotiation history, market timing, role seniority within grade). Salary benchmarking results should be reviewed alongside job-level compensation bands before acting on individual cases.
"""

    console.print(recommendation.strip())


def main() -> None:
    df = get_clean_df()

    model_artifacts = get_model_artifacts()
    cluster_artifacts = get_complete_cluster_artifacts()
    lasso_artifacts = get_lasso_artifacts()

    dt_accuracy = compute_dt_accuracy(model_artifacts)
    cluster_summary = build_cluster_summary(cluster_artifacts)
    flight_risk_ids = get_flight_risk_ids(cluster_artifacts)
    master = build_master_frame(df, model_artifacts, lasso_artifacts)
    master["Employee_ID"] = master["Employee_ID"].astype(str)

    report_model_performance(dt_accuracy, lasso_artifacts)
    console.print()
    report_cluster_interventions(cluster_summary)
    console.print()
    report_salary_fairness(master, flight_risk_ids)
    console.print()
    report_chro_recommendation(cluster_summary, master, flight_risk_ids, dt_accuracy, lasso_artifacts)


if __name__ == "__main__":
    main()
