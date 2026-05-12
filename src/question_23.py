import joblib
import numpy as np
import pandas as pd
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from sklearn.metrics import accuracy_score

from question_01 import clean, console, load_raw
from question_05 import MODEL_PATH, ModelArtifacts
from question_12 import COMPLETE_CLUSTER_PATH, CompleteLinkageArtifacts
from question_13 import CLUSTER_INTERVENTIONS, CLUSTER_NAMES
from question_18 import LASSO_PATH, LassoArtifacts
from question_21 import build_master_frame, get_flight_risk_ids


def load_model_artifacts() -> ModelArtifacts:
    return ModelArtifacts(**joblib.load(MODEL_PATH))


def load_cluster_artifacts() -> CompleteLinkageArtifacts:
    return CompleteLinkageArtifacts(**joblib.load(COMPLETE_CLUSTER_PATH))


def load_lasso_artifacts() -> LassoArtifacts:
    return LassoArtifacts(**joblib.load(LASSO_PATH))


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
    table.add_column("Model", style="cyan")
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
    table.add_column("Cluster", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Attrition %", justify="right")
    table.add_column("Avg Salary (PHP)", justify="right")
    table.add_column("Avg Satisfaction", justify="right")
    table.add_column("Avg Absences", justify="right")
    table.add_column("Recommended Intervention")

    for cluster_id, row in cluster_summary.iterrows():
        attrition_pct: float = float(row["Attrition %"])
        color = "red" if attrition_pct > 60 else "yellow" if attrition_pct > 40 else "green"
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
    table.add_column("Group", style="cyan")
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
        f"[red]{len(underpaid_flight_risk)}[/red]",
        f"[red]{len(underpaid_flight_risk) / len(flight_risk_employees) * 100:.1f}%[/red]",
        f"[red]{avg_gap_flight_risk:+,.0f}[/red]",
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
[bold cyan]EXECUTIVE SUMMARY[/bold cyan]

A 15-year employee dataset of {len(master):,} records was analyzed using three complementary data mining approaches: Decision Tree classification for attrition prediction, Hierarchical Clustering for employee segmentation, and Lasso Regression for salary benchmarking. Together, these models surface actionable retention and compensation priorities for immediate HR attention.

[bold cyan]1. RETENTION PRIORITIES[/bold cyan]

The Decision Tree (test accuracy: [green]{dt_accuracy:.1%}[/green]) identified [red]{dt_predicted_leave:,}[/red] employees as likely to leave. Independently, hierarchical clustering segmented the workforce into three groups, with the [red]Flight Risk[/red] cluster comprising [red]{flight_risk_size}[/red] employees and a {flight_risk_attrition:.1f}% historical attrition rate — the highest of any segment.

[bold]{high_risk_intersection}[/bold] employees appear in both the DT's leave predictions and the Flight Risk cluster. These represent the [bold red]highest-confidence departure risks[/bold red] and should be the first cohort addressed through targeted retention programs.

A further [bold yellow]{contradiction_count}[/bold yellow] employees were predicted to stay by the Decision Tree but still belong to the Flight Risk cluster. These [italic]silent flight risks[/italic] share the structural profile of employees who have historically left — low satisfaction, elevated absences, and salary below expectations — without yet triggering the DT's explicit rules. Proactive engagement before resignation intent solidifies is critical for this group.

[bold cyan]2. SALARY FAIRNESS[/bold cyan]

The Lasso model (R² = [green]{lasso_artifacts.test_r2:.4f}[/green], RMSE = PHP {lasso_artifacts.test_rmse:,.0f}) provides a data-grounded salary benchmark. Within the Flight Risk cluster, [red]{underpaid_count}[/red] of {flight_risk_size} employees earn below their predicted salary, with an average shortfall of [red]PHP {abs(avg_gap_flight_risk):,.0f}[/red] per month. Their average actual salary (PHP {flight_risk_avg_salary:,.0f}) trails the Stable Performers segment (PHP {stable_avg_salary:,.0f}) — a gap that likely compounds disengagement.

Salary correction for underpaid Flight Risk employees is the highest-ROI retention lever available. The cost of a salary adjustment is substantially lower than the fully-loaded cost of turnover (recruitment, onboarding, lost productivity).

[bold cyan]3. INTERVENTIONS BY SEGMENT[/bold cyan]

See cluster intervention table above. Priority order for HR resource allocation:
  1. [red]Flight Risk[/red] — immediate salary review, satisfaction surveys, workload audit
  2. [yellow]Outlier Pair[/yellow] — individual case review for anomalous profiles
  3. [green]Stable Performers[/green] — career pathing and recognition programs to sustain engagement

[bold cyan]4. MODELING LIMITATIONS[/bold cyan]

The Lasso regression explains [green]{lasso_artifacts.test_r2:.1%}[/green] of salary variance, indicating that a substantial portion of compensation is driven by factors not captured in the dataset (e.g., negotiation history, market timing, role seniority within grade). Salary benchmarking results should be reviewed alongside job-level compensation bands before acting on individual cases.
"""

    console.print(Panel(recommendation.strip(), title="Q23 — CHRO Management Recommendation"))


def main() -> None:
    clean_df = clean(load_raw())

    console.print(
        Panel(
            f"[bold]Dataset:[/bold] {len(clean_df):,} employees over 15 years\n[bold]Models:[/bold] Decision Tree · Hierarchical Clustering · Lasso Regression",
            title="Workforce Attrition — Q23 CHRO Recommendation Report",
        ),
    )

    model_artifacts = load_model_artifacts()
    cluster_artifacts = load_cluster_artifacts()
    lasso_artifacts = load_lasso_artifacts()

    dt_accuracy = compute_dt_accuracy(model_artifacts)
    cluster_summary = build_cluster_summary(cluster_artifacts)
    flight_risk_ids = get_flight_risk_ids(cluster_artifacts)
    master = build_master_frame(clean_df, model_artifacts, lasso_artifacts)
    master["Employee_ID"] = master["Employee_ID"].astype(str)

    console.print(Rule("Model Performance"))
    report_model_performance(dt_accuracy, lasso_artifacts)

    console.print(Rule("Cluster Profiles and Interventions"))
    report_cluster_interventions(cluster_summary)

    console.print(Rule("Salary Fairness"))
    report_salary_fairness(master, flight_risk_ids)

    console.print(Rule("CHRO Recommendation"))
    report_chro_recommendation(cluster_summary, master, flight_risk_ids, dt_accuracy, lasso_artifacts)


if __name__ == "__main__":
    main()
