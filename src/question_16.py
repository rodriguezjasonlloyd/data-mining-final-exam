"""Q16. Multicollinearity and VIF."""

import pandas as pd
import statsmodels.api as sm
from rich.table import Table
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor

from question_01 import console
from question_15 import get_ols_artifacts

VIF_HIGH_THRESHOLD = 10.0
VIF_MODERATE_THRESHOLD = 5.0

TENURE_FLOOR_YEARS = 0.5


def compute_vif(x_train: pd.DataFrame) -> pd.DataFrame:
    x_with_constant = sm.add_constant(x_train)
    vif_values: list[float] = [variance_inflation_factor(x_with_constant.values, index) for index in range(x_with_constant.shape[1])]
    feature_names: list[str] = list(x_with_constant.columns)
    return pd.DataFrame({"feature": feature_names, "vif": vif_values}).query("feature != 'const'").sort_values("vif", ascending=False).reset_index(drop=True)


def engineer_promotion_rate(x_train: pd.DataFrame) -> pd.DataFrame:
    result = x_train.copy()
    tenure_floored = result["Tenure_Years"].clip(lower=TENURE_FLOOR_YEARS)
    result["Promotion_Rate"] = result["Num_Promotions"] / tenure_floored
    return result.drop(columns=["Tenure_Years", "Num_Promotions"])


def fit_ols(x_train: pd.DataFrame, y_train: pd.Series) -> RegressionResultsWrapper:
    x_with_constant = sm.add_constant(x_train)
    return sm.OLS(y_train, x_with_constant).fit()


def report_vif(vif_df: pd.DataFrame, title: str) -> None:
    table = Table(title=title, show_lines=True)
    table.add_column("Feature", style="bright_cyan")
    table.add_column("VIF", justify="right")
    table.add_column("Multicollinearity Risk", justify="left")

    for _, row in vif_df.iterrows():
        vif: float = row["vif"]
        if vif >= VIF_HIGH_THRESHOLD:
            risk = "[bright_red]High[/bright_red]"
        elif vif >= VIF_MODERATE_THRESHOLD:
            risk = "[bright_yellow]Moderate[/bright_yellow]"
        else:
            risk = "[bright_green]Low[/bright_green]"
        table.add_row(str(row["feature"]), f"{vif:.4f}", risk)

    high_vif = vif_df[vif_df["vif"] >= VIF_HIGH_THRESHOLD]
    moderate_vif = vif_df[(vif_df["vif"] >= VIF_MODERATE_THRESHOLD) & (vif_df["vif"] < VIF_HIGH_THRESHOLD)]

    console.print(table)

    if not high_vif.empty:
        features = ", ".join(high_vif["feature"].tolist())
        console.print(f"[bold bright_red]High multicollinearity (VIF ≥ {VIF_HIGH_THRESHOLD}):[/bold bright_red] {features}")

    if not moderate_vif.empty:
        features = ", ".join(moderate_vif["feature"].tolist())
        console.print(f"[bold bright_yellow]Moderate multicollinearity (VIF ≥ {VIF_MODERATE_THRESHOLD}):[/bold bright_yellow] {features}")

    if high_vif.empty and moderate_vif.empty:
        console.print("[bright_green]No significant multicollinearity detected.[/bright_green]")


def report_ols_comparison(original_model: RegressionResultsWrapper, remediated_model: RegressionResultsWrapper) -> None:
    original_params: pd.Series = original_model.params.drop("const")
    remediated_params: pd.Series = remediated_model.params.drop("const")

    original_bse: pd.Series = original_model.bse.drop("const")
    remediated_bse: pd.Series = remediated_model.bse.drop("const")

    original_pvalues: pd.Series = original_model.pvalues.drop("const")
    remediated_pvalues: pd.Series = remediated_model.pvalues.drop("const")

    shared_features: list[str] = [feature for feature in original_params.index if feature in remediated_params.index]
    new_features: list[str] = [feature for feature in remediated_params.index if feature not in original_params.index]
    dropped_features: list[str] = [feature for feature in original_params.index if feature not in remediated_params.index]

    shared_table = Table(title="OLS Comparison — Shared Features (Original vs Remediated)", show_lines=True)
    shared_table.add_column("Feature", style="bright_cyan")
    shared_table.add_column("Orig Coef", justify="right")
    shared_table.add_column("Remed Coef", justify="right")
    shared_table.add_column("Orig SE", justify="right")
    shared_table.add_column("Remed SE", justify="right")
    shared_table.add_column("SE Δ", justify="right")
    shared_table.add_column("Orig p", justify="right")
    shared_table.add_column("Remed p", justify="right")

    for feature in shared_features:
        original_se: float = original_bse[feature]
        remediated_se: float = remediated_bse[feature]
        se_delta: float = remediated_se - original_se
        se_color = "bright_green" if se_delta < 0 else "bright_red" if se_delta > 0 else "dim"

        original_p: float = original_pvalues[feature]
        remediated_p: float = remediated_pvalues[feature]
        original_p_color = "bright_green" if original_p < 0.05 else "dim"
        remediated_p_color = "bright_green" if remediated_p < 0.05 else "dim"

        shared_table.add_row(
            feature,
            f"{original_params[feature]:,.2f}",
            f"{remediated_params[feature]:,.2f}",
            f"{original_se:,.2f}",
            f"{remediated_se:,.2f}",
            f"[{se_color}]{se_delta:+,.2f}[/{se_color}]",
            f"[{original_p_color}]{original_p:.4f}[/{original_p_color}]",
            f"[{remediated_p_color}]{remediated_p:.4f}[/{remediated_p_color}]",
        )

    console.print(shared_table)

    if dropped_features:
        dropped_table = Table(title="Dropped Features (replaced by Promotion_Rate)", show_lines=True)
        dropped_table.add_column("Feature", style="bright_red")
        dropped_table.add_column("Original Coef", justify="right")
        dropped_table.add_column("Original SE", justify="right")
        dropped_table.add_column("Original p", justify="right")
        for feature in dropped_features:
            dropped_table.add_row(
                feature,
                f"{original_params[feature]:,.2f}",
                f"{original_bse[feature]:,.2f}",
                f"{original_pvalues[feature]:.4f}",
            )
        console.print(dropped_table)

    if new_features:
        new_table = Table(title="New Features (Promotion_Rate engineered replacement)", show_lines=True)
        new_table.add_column("Feature", style="bright_green")
        new_table.add_column("Remediated Coef", justify="right")
        new_table.add_column("Remediated SE", justify="right")
        new_table.add_column("Remediated p", justify="right")
        for feature in new_features:
            p_color = "bright_green" if remediated_pvalues[feature] < 0.05 else "dim"
            new_table.add_row(
                feature,
                f"{remediated_params[feature]:,.2f}",
                f"{remediated_bse[feature]:,.2f}",
                f"[{p_color}]{remediated_pvalues[feature]:.4f}[/{p_color}]",
            )
        console.print(new_table)

    original_r2: float = original_model.rsquared
    remediated_r2: float = remediated_model.rsquared
    r2_delta: float = remediated_r2 - original_r2
    r2_color = "bright_green" if abs(r2_delta) <= 0.02 else "bright_yellow" if abs(r2_delta) <= 0.05 else "bright_red"

    console.print(
        f"\n[bold]R² — Original:[/bold] {original_r2:.4f}  "
        f"[bold]Remediated:[/bold] {remediated_r2:.4f}  "
        f"[bold]Δ:[/bold] [{r2_color}]{r2_delta:+.4f}[/{r2_color}]",
    )


def main() -> None:
    artifacts = get_ols_artifacts()

    vif_before = compute_vif(artifacts.x_train)
    report_vif(vif_before, "VIF — Before Remediation (Original Features)")

    console.print()

    x_train_remediated = engineer_promotion_rate(artifacts.x_train)
    vif_after = compute_vif(x_train_remediated)
    report_vif(vif_after, "VIF — After Remediation (Promotion_Rate)")

    console.print()

    original_ols = fit_ols(artifacts.x_train, artifacts.y_train)
    remediated_ols = fit_ols(x_train_remediated, artifacts.y_train)
    report_ols_comparison(original_ols, remediated_ols)


if __name__ == "__main__":
    main()
