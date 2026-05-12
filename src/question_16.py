"""Q16. Multicollinearity and VIF."""

import pandas as pd
import statsmodels.api as sm
from rich.panel import Panel
from rich.table import Table
from statsmodels.stats.outliers_influence import variance_inflation_factor

from question_01 import console
from question_15 import get_ols_artifacts

VIF_HIGH_THRESHOLD = 10.0
VIF_MODERATE_THRESHOLD = 5.0


def compute_vif(x_train: pd.DataFrame) -> pd.DataFrame:
    x_with_constant = sm.add_constant(x_train)
    vif_values: list[float] = [variance_inflation_factor(x_with_constant.values, index) for index in range(x_with_constant.shape[1])]
    feature_names: list[str] = list(x_with_constant.columns)
    return pd.DataFrame({"feature": feature_names, "vif": vif_values}).query("feature != 'const'").sort_values("vif", ascending=False).reset_index(drop=True)


def report_vif(vif_df: pd.DataFrame) -> None:
    table = Table(title="Variance Inflation Factor (VIF)", show_lines=True)
    table.add_column("Feature", style="cyan")
    table.add_column("VIF", justify="right")
    table.add_column("Multicollinearity Risk", justify="left")

    for _, row in vif_df.iterrows():
        vif: float = row["vif"]
        if vif >= VIF_HIGH_THRESHOLD:
            risk = "[red]High[/red]"
        elif vif >= VIF_MODERATE_THRESHOLD:
            risk = "[yellow]Moderate[/yellow]"
        else:
            risk = "[green]Low[/green]"
        table.add_row(str(row["feature"]), f"{vif:.4f}", risk)

    console.print(table)

    high_vif = vif_df[vif_df["vif"] >= VIF_HIGH_THRESHOLD]
    moderate_vif = vif_df[(vif_df["vif"] >= VIF_MODERATE_THRESHOLD) & (vif_df["vif"] < VIF_HIGH_THRESHOLD)]

    if not high_vif.empty:
        features = ", ".join(high_vif["feature"].tolist())
        console.print(f"\n[bold red]High multicollinearity (VIF ≥ {VIF_HIGH_THRESHOLD}):[/bold red] {features}")
        console.print("[dim]These predictors share substantial variance with others — regularization will shrink or eliminate them.[/dim]")

    if not moderate_vif.empty:
        features = ", ".join(moderate_vif["feature"].tolist())
        console.print(f"\n[bold yellow]Moderate multicollinearity (VIF ≥ {VIF_MODERATE_THRESHOLD}):[/bold yellow] {features}")

    if high_vif.empty and moderate_vif.empty:
        console.print("\n[green]No significant multicollinearity detected.[/green]")


def main() -> None:
    artifacts = get_ols_artifacts()

    console.print(
        Panel(
            f"[bold]Features:[/bold] {len(artifacts.feature_names)}  [bold]Train samples:[/bold] {artifacts.x_train.shape[0]}",
            title="Workforce Attrition — Q16 Multicollinearity and VIF",
        ),
    )

    vif_df = compute_vif(artifacts.x_train)
    report_vif(vif_df)


if __name__ == "__main__":
    main()
