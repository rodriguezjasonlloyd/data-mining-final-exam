
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import mean_squared_error, r2_score

from question_01 import console
from question_15 import OLS_PATH, OLSArtifacts
from question_17 import RIDGE_PATH, RidgeArtifacts
from question_18 import LASSO_PATH, LassoArtifacts
from question_19 import ELASTIC_NET_PATH, ElasticNetArtifacts

MODEL_COLORS: dict[str, str] = {
    "OLS": "steelblue",
    "Ridge": "gold",
    "Lasso": "tomato",
    "Elastic Net": "mediumseagreen",
}


def load_ols_artifacts() -> OLSArtifacts:
    return OLSArtifacts(**joblib.load(OLS_PATH))


def load_ridge_artifacts() -> RidgeArtifacts:
    return RidgeArtifacts(**joblib.load(RIDGE_PATH))


def load_lasso_artifacts() -> LassoArtifacts:
    return LassoArtifacts(**joblib.load(LASSO_PATH))


def load_elastic_net_artifacts() -> ElasticNetArtifacts:
    return ElasticNetArtifacts(**joblib.load(ELASTIC_NET_PATH))


def compute_ols_test_metrics(ols_artifacts: OLSArtifacts) -> tuple[float, float, int]:
    y_test_prediction: np.ndarray = ols_artifacts.model.predict(ols_artifacts.x_test)
    test_r2: float = float(r2_score(ols_artifacts.y_test, y_test_prediction))
    test_rmse: float = float(np.sqrt(mean_squared_error(ols_artifacts.y_test, y_test_prediction)))
    nonzero_count: int = int((np.array(ols_artifacts.model.coef_) != 0).sum())
    return test_r2, test_rmse, nonzero_count


def build_comparison_dataframe(
    ols_artifacts: OLSArtifacts,
    ridge_artifacts: RidgeArtifacts,
    lasso_artifacts: LassoArtifacts,
    elastic_net_artifacts: ElasticNetArtifacts,
) -> pd.DataFrame:
    ols_test_r2, ols_test_rmse, ols_nonzero_count = compute_ols_test_metrics(ols_artifacts)
    ridge_nonzero_count: int = int((ridge_artifacts.coefficients != 0).sum())

    records: list[dict[str, object]] = [
        {
            "Model": "OLS",
            "Test R²": ols_test_r2,
            "Test RMSE": ols_test_rmse,
            "Non-Zero Coefficients": ols_nonzero_count,
            "Optimal λ": None,
            "L1 Ratio": None,
        },
        {
            "Model": "Ridge",
            "Test R²": ridge_artifacts.test_r2,
            "Test RMSE": ridge_artifacts.test_rmse,
            "Non-Zero Coefficients": ridge_nonzero_count,
            "Optimal λ": ridge_artifacts.optimal_alpha,
            "L1 Ratio": None,
        },
        {
            "Model": "Lasso",
            "Test R²": lasso_artifacts.test_r2,
            "Test RMSE": lasso_artifacts.test_rmse,
            "Non-Zero Coefficients": lasso_artifacts.nonzero_count,
            "Optimal λ": lasso_artifacts.optimal_alpha,
            "L1 Ratio": None,
        },
        {
            "Model": "Elastic Net",
            "Test R²": elastic_net_artifacts.test_r2,
            "Test RMSE": elastic_net_artifacts.test_rmse,
            "Non-Zero Coefficients": elastic_net_artifacts.nonzero_count,
            "Optimal λ": elastic_net_artifacts.optimal_alpha,
            "L1 Ratio": elastic_net_artifacts.optimal_l1_ratio,
        },
    ]

    return pd.DataFrame(records).set_index("Model")


def report_comparison_table(comparison: pd.DataFrame) -> None:
    best_r2_model: str = str(comparison["Test R²"].idxmax())
    best_rmse_model: str = str(comparison["Test RMSE"].idxmin())

    table = Table(title="Regression Model Comparison — OLS vs Ridge vs Lasso vs Elastic Net", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("Test R²", justify="right")
    table.add_column("Test RMSE", justify="right")
    table.add_column("Non-Zero Coefficients", justify="right")
    table.add_column("Optimal λ", justify="right")
    table.add_column("L1 Ratio", justify="right")

    for model_name, row in comparison.iterrows():
        r2_color = "bold green" if model_name == best_r2_model else ""
        rmse_color = "bold green" if model_name == best_rmse_model else ""

        optimal_lambda = row["Optimal λ"]
        l1_ratio = row["L1 Ratio"]

        lambda_display = f"{optimal_lambda:.4f}" if pd.notna(optimal_lambda) else "[dim]N/A[/dim]"
        l1_display = f"{l1_ratio:.4f}" if pd.notna(l1_ratio) else "[dim]N/A[/dim]"

        r2_display = f"[{r2_color}]{row['Test R²']:.4f}[/{r2_color}]" if r2_color else f"{row['Test R²']:.4f}"
        rmse_display = f"[{rmse_color}]{row['Test RMSE']:,.2f}[/{rmse_color}]" if rmse_color else f"{row['Test RMSE']:,.2f}"

        table.add_row(
            str(model_name),
            r2_display,
            rmse_display,
            str(int(row["Non-Zero Coefficients"])),
            lambda_display,
            l1_display,
        )

    console.print(table)
    console.print(f"[dim]Bold green = best value. Best R²: {best_r2_model}. Best RMSE: {best_rmse_model}.[/dim]")


def report_recommendation(comparison: pd.DataFrame) -> None:
    best_r2_model: str = str(comparison["Test R²"].idxmax())
    best_rmse_model: str = str(comparison["Test RMSE"].idxmin())
    best_r2: float = float(comparison.loc[best_r2_model, "Test R²"])
    best_rmse: float = float(comparison.loc[best_rmse_model, "Test RMSE"])

    lasso_nonzero: int = int(comparison.loc["Lasso", "Non-Zero Coefficients"])
    elastic_net_nonzero: int = int(comparison.loc["Elastic Net", "Non-Zero Coefficients"])
    total_features: int = int(comparison["Non-Zero Coefficients"].max())

    recommendation = (
        f"[bold]Recommended model:[/bold] [green]{best_r2_model}[/green]\n\n"
        f"[bold]Justification:[/bold]\n"
        f"• [bold]{best_r2_model}[/bold] achieves the highest test R² of [cyan]{best_r2:.4f}[/cyan] "
        f"and the lowest test RMSE of [cyan]{best_rmse:,.2f} PHP[/cyan], indicating the strongest "
        f"out-of-sample predictive performance for monthly salary benchmarking.\n"
        f"• Lasso reduces the model to [cyan]{lasso_nonzero}[/cyan] of {total_features} features, "
        f"while Elastic Net retains [cyan]{elastic_net_nonzero}[/cyan] — both enable interpretable, "
        f"parsimonious salary models suited for HR reporting.\n"
        f"• Ridge retains all features and is preferred when multicollinearity is severe, "
        f"but offers no automatic feature selection.\n"
        f"• OLS is the baseline — if regularized models do not meaningfully outperform it, "
        f"OLS remains the most interpretable choice for stakeholders.\n\n"
        f"[bold]HR implication:[/bold] Use the recommended model to flag employees whose actual salary "
        f"deviates significantly from the predicted value — both underpaid employees (retention risk) "
        f"and overpaid outliers (budget risk)."
    )

    console.print(Panel(recommendation, title="Q20 — Model Recommendation for Salary Benchmarking"))


def plot_comparison(comparison: pd.DataFrame) -> None:
    models: list[str] = list(comparison.index)
    colors: list[str] = [MODEL_COLORS[model] for model in models]

    _fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    r2_values: list[float] = [float(comparison.loc[model, "Test R²"]) for model in models]
    axes[0].bar(models, r2_values, color=colors, edgecolor="black", linewidth=0.6)
    axes[0].set_title("Test R² by Model")
    axes[0].set_ylabel("Test R²")
    axes[0].set_ylim(0, 1)
    for index, value in enumerate(r2_values):
        axes[0].text(index, value + 0.01, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    rmse_values: list[float] = [float(comparison.loc[model, "Test RMSE"]) for model in models]
    axes[1].bar(models, rmse_values, color=colors, edgecolor="black", linewidth=0.6)
    axes[1].set_title("Test RMSE by Model (PHP)")
    axes[1].set_ylabel("Test RMSE (PHP)")
    for index, value in enumerate(rmse_values):
        axes[1].text(index, value + 0.01 * max(rmse_values), f"{value:,.0f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Regression Model Comparison — Salary Prediction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_nonzero_coefficients(comparison: pd.DataFrame) -> None:
    models: list[str] = list(comparison.index)
    nonzero_values: list[int] = [int(comparison.loc[model, "Non-Zero Coefficients"]) for model in models]

    plot_df = pd.DataFrame({"Model": models, "Non-Zero Coefficients": nonzero_values})

    _fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=plot_df,
        x="Model",
        y="Non-Zero Coefficients",
        hue="Model",
        palette=MODEL_COLORS,
        legend=False,
        ax=ax,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_title("Feature Retention by Model")
    ax.set_ylabel("Non-Zero Coefficients")
    for index, value in enumerate(nonzero_values):
        ax.text(index, value + 0.1, str(value), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.show()


def main() -> None:
    console.print(
        Panel(
            "Comparing OLS, Ridge, Lasso, and Elastic Net on test R², test RMSE, non-zero coefficients, and optimal λ.",
            title="Workforce Attrition — Q20 Regression Model Comparison",
        ),
    )

    ols_artifacts = load_ols_artifacts()
    ridge_artifacts = load_ridge_artifacts()
    lasso_artifacts = load_lasso_artifacts()
    elastic_net_artifacts = load_elastic_net_artifacts()

    comparison = build_comparison_dataframe(ols_artifacts, ridge_artifacts, lasso_artifacts, elastic_net_artifacts)

    report_comparison_table(comparison)
    report_recommendation(comparison)
    plot_comparison(comparison)
    plot_nonzero_coefficients(comparison)


if __name__ == "__main__":
    main()
