"""Q21. High-Risk Employee Analysis."""

import numpy as np
import pandas as pd
from rich.table import Table

from question_01 import console, get_clean_df
from question_05 import ModelArtifacts, get_model_artifacts, prepare_attrition
from question_12 import CompleteLinkageArtifacts, get_complete_cluster_artifacts
from question_13 import CLUSTER_NAMES
from question_15 import prepare_salary
from question_18 import LassoArtifacts, get_lasso_artifacts

FLIGHT_RISK_LABEL = "Flight Risk"
TOP_N = 10


def get_flight_risk_ids(cluster_artifacts: CompleteLinkageArtifacts) -> set[str]:
    flight_risk_cluster_id: int = next(k for k, v in CLUSTER_NAMES.items() if v == FLIGHT_RISK_LABEL)
    sample = cluster_artifacts.sample_df.copy()
    sample["Cluster"] = cluster_artifacts.cluster_labels
    return set(sample.loc[sample["Cluster"] == flight_risk_cluster_id, "Employee_ID"].astype(str))


def build_master_frame(
    clean_df: pd.DataFrame,
    model_artifacts: ModelArtifacts,
    lasso_artifacts: LassoArtifacts,
) -> pd.DataFrame:
    x_class, _y_class = prepare_attrition(clean_df)
    attrition_probability: np.ndarray = model_artifacts.model.predict_proba(x_class)[:, 1]

    x_salary, _y_salary = prepare_salary(clean_df)
    predicted_salary: np.ndarray = lasso_artifacts.model.predict(x_salary)

    master = clean_df[["Employee_ID", "Monthly_Salary_PHP"]].copy().reset_index(drop=True)
    master["attrition_probability"] = attrition_probability
    master["predicted_attrition"] = (attrition_probability >= 0.5).astype(int)
    master["predicted_salary"] = predicted_salary
    master["salary_gap"] = master["Monthly_Salary_PHP"] - master["predicted_salary"]
    master["Employee_ID"] = master["Employee_ID"].astype(str)
    return master


def identify_high_risk(master: pd.DataFrame, flight_risk_ids: set[str]) -> pd.DataFrame:
    high_risk = master[(master["predicted_attrition"] == 1) & (master["Employee_ID"].isin(flight_risk_ids))].copy()
    return high_risk.sort_values("attrition_probability", ascending=False).head(TOP_N)


def report_high_risk(high_risk: pd.DataFrame) -> None:
    table = Table(title=f"Top {TOP_N} High-Risk Employees — DT Predicted Leave + Flight Risk Cluster", show_lines=True)
    table.add_column("Employee_ID", style="bright_cyan")
    table.add_column("Actual Salary (PHP)", justify="right")
    table.add_column("Predicted Salary (PHP)", justify="right")
    table.add_column("Salary Gap (PHP)", justify="right")
    table.add_column("Attrition Probability", justify="right")

    for _, row in high_risk.iterrows():
        gap: float = row["salary_gap"]
        gap_color = "bright_red" if gap < -5000 else "bright_yellow" if gap < 0 else "bright_green"
        table.add_row(
            str(row["Employee_ID"]),
            f"{row['Monthly_Salary_PHP']:,.0f}",
            f"{row['predicted_salary']:,.0f}",
            f"[{gap_color}]{gap:+,.0f}[/{gap_color}]",
            f"{row['attrition_probability']:.3f}",
        )

    console.print(table)


def report_salary_summary(high_risk: pd.DataFrame) -> None:
    underpaid = high_risk[high_risk["salary_gap"] < 0]
    underpaid_count: int = len(underpaid)
    avg_gap: float = float(high_risk["salary_gap"].mean())
    avg_actual: float = float(high_risk["Monthly_Salary_PHP"].mean())
    avg_predicted: float = float(high_risk["predicted_salary"].mean())

    gap_color = "bright_red" if avg_gap < 0 else "bright_green"

    console.print(
        f"[bold]Underpaid (actual < predicted):[/bold] [bright_red]{underpaid_count}[/bright_red] of {len(high_risk)}\n"
        f"[bold]Average actual salary:[/bold] PHP {avg_actual:,.0f}\n"
        f"[bold]Average predicted salary:[/bold] PHP {avg_predicted:,.0f}\n"
        f"[bold]Average salary gap:[/bold] [{gap_color}]PHP {avg_gap:+,.0f}[/{gap_color}]",
    )


def main() -> None:
    df = get_clean_df()

    model_artifacts = get_model_artifacts()
    cluster_artifacts = get_complete_cluster_artifacts()
    lasso_artifacts = get_lasso_artifacts()

    flight_risk_ids = get_flight_risk_ids(cluster_artifacts)
    console.print(f"Flight Risk cluster size: {len(flight_risk_ids)} employees (from 300 employee sample)")

    master = build_master_frame(df, model_artifacts, lasso_artifacts)

    dt_predicted_leave: int = int(master["predicted_attrition"].sum())
    console.print(f"DT predicted leave (full dataset): {dt_predicted_leave} of {len(master)} employees")

    high_risk = identify_high_risk(master, flight_risk_ids)
    console.print(f"Intersection (both criteria): {len(high_risk)} employees — showing top {TOP_N}\n")

    report_high_risk(high_risk)
    report_salary_summary(high_risk)


if __name__ == "__main__":
    main()
