"""Q22. Contradiction Analysis."""

import joblib
import pandas as pd
from rich.panel import Panel
from rich.table import Table

from question_01 import clean, console, load_raw
from question_05 import MODEL_PATH, ModelArtifacts
from question_12 import COMPLETE_CLUSTER_PATH, CompleteLinkageArtifacts
from question_18 import LASSO_PATH, LassoArtifacts
from question_21 import build_master_frame, get_flight_risk_ids


def load_model_artifacts() -> ModelArtifacts:
    return ModelArtifacts(**joblib.load(MODEL_PATH))


def load_cluster_artifacts() -> CompleteLinkageArtifacts:
    return CompleteLinkageArtifacts(**joblib.load(COMPLETE_CLUSTER_PATH))


def load_lasso_artifacts() -> LassoArtifacts:
    return LassoArtifacts(**joblib.load(LASSO_PATH))


def identify_contradictions(master: pd.DataFrame, flight_risk_ids: set[str]) -> pd.DataFrame:
    contradictions = master[(master["predicted_attrition"] == 0) & (master["Employee_ID"].isin(flight_risk_ids))].copy()
    return contradictions.sort_values("attrition_probability", ascending=False)


def report_contradictions(contradictions: pd.DataFrame) -> None:
    table = Table(
        title="Contradiction Cases — DT Predicts Stay, Cluster = Flight Risk",
        show_lines=True,
    )
    table.add_column("Employee_ID", style="cyan")
    table.add_column("Actual Salary (PHP)", justify="right")
    table.add_column("Predicted Salary (PHP)", justify="right")
    table.add_column("Salary Gap (PHP)", justify="right")
    table.add_column("Attrition Probability", justify="right")
    table.add_column("Actual Attrition", justify="right")

    for _, row in contradictions.iterrows():
        gap: float = row["salary_gap"]
        gap_color = "red" if gap < -5000 else "yellow" if gap < 0 else "green"
        actual_color = "red" if row["Attrition"] == 1 else "green"
        table.add_row(
            str(row["Employee_ID"]),
            f"{row['Monthly_Salary_PHP']:,.0f}",
            f"{row['predicted_salary']:,.0f}",
            f"[{gap_color}]{gap:+,.0f}[/{gap_color}]",
            f"{row['attrition_probability']:.3f}",
            f"[{actual_color}]{'Yes' if row['Attrition'] == 1 else 'No'}[/{actual_color}]",
        )

    console.print(table)


def report_contradiction_summary(contradictions: pd.DataFrame) -> None:
    total: int = len(contradictions)
    actually_left: int = int(contradictions["Attrition"].sum())
    underpaid: int = int((contradictions["salary_gap"] < 0).sum())
    avg_probability: float = float(contradictions["attrition_probability"].mean())

    summary = (
        f"[bold]Contradiction cases found:[/bold] {total}\n"
        f"[bold]Of these, actually left:[/bold] [red]{actually_left}[/red] ({actually_left / total * 100:.1f}%) — "
        f"confirming the cluster captured real risk the DT missed\n"
        f"[bold]Underpaid (actual < predicted salary):[/bold] [yellow]{underpaid}[/yellow] of {total}\n"
        f"[bold]Average DT attrition probability:[/bold] {avg_probability:.3f} "
        f"[dim](below 0.5 — hence predicted stay)[/dim]\n\n"
        f"[bold]Why do the models disagree?[/bold]\n\n"
        f"The Decision Tree learns explicit threshold rules from labeled training data. "
        f"An employee is predicted to stay when their feature values do not cross any of the learned split points — "
        f"for example, if their satisfaction score or salary falls just above a threshold the tree treats as safe.\n\n"
        f"Hierarchical clustering is unsupervised — it groups employees purely by similarity in the feature space "
        f"(salary, satisfaction, absences, overtime, tenure, etc.) without ever seeing the attrition label. "
        f"An employee lands in the Flight Risk cluster because their overall profile resembles others who left, "
        f"even if no single feature is extreme enough to trigger the DT's decision rules.\n\n"
        f"[bold]What each model captures differently:[/bold]\n"
        f"• The DT captures [cyan]explicit, threshold-based attrition signals[/cyan] identified during supervised training.\n"
        f"• Clustering captures [yellow]latent structural similarity[/yellow] — employees who share a broader pattern "
        f"of disengagement across multiple dimensions simultaneously.\n\n"
        f"[bold]HR implication:[/bold] These contradiction employees are [red]silent flight risks[/red]. "
        f"The DT's rules give them a pass, but their overall profile places them among employees historically "
        f"prone to leaving. They warrant proactive monitoring — satisfaction check-ins, salary benchmarking, "
        f"and workload review — before their risk becomes visible in performance or attendance data."
    )

    console.print(Panel(summary, title="Q22 — Contradiction Analysis"))


def main() -> None:
    clean_df = clean(load_raw())

    console.print(
        Panel(
            "[bold]Criteria:[/bold] Decision Tree predicts Stay (probability < 0.5) AND assigned to Flight Risk cluster",
            title="Workforce Attrition — Q22 Contradiction Analysis",
        ),
    )

    model_artifacts = load_model_artifacts()
    cluster_artifacts = load_cluster_artifacts()
    lasso_artifacts = load_lasso_artifacts()

    flight_risk_ids = get_flight_risk_ids(cluster_artifacts)
    master = build_master_frame(clean_df, model_artifacts, lasso_artifacts)

    master = master.merge(
        clean_df[["Employee_ID", "Attrition"]].assign(Employee_ID=lambda frame: frame["Employee_ID"].astype(str)),
        on="Employee_ID",
        how="left",
    )

    contradictions = identify_contradictions(master, flight_risk_ids)

    console.print(
        f"[dim]Flight Risk cluster: {len(flight_risk_ids)} employees  |  DT predicted stay: {int((master['predicted_attrition'] == 0).sum())}  |  Contradictions: {len(contradictions)}[/dim]\n",
    )

    report_contradictions(contradictions)
    report_contradiction_summary(contradictions)


if __name__ == "__main__":
    main()
