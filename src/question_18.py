"""Q18. Lasso Regression."""

from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from rich.panel import Panel
from rich.table import Table
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from question_01 import console
from question_15 import OLSArtifacts, get_ols_artifacts

ALPHA_GRID: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
CV_FOLDS = 5


@cache
def get_lasso_artifacts() -> LassoArtifacts:
    return train_lasso(get_ols_artifacts())


class LassoArtifacts(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    model: Lasso
    optimal_alpha: float
    coefficients: pd.Series
    train_r2: float
    test_r2: float
    train_rmse: float
    test_rmse: float
    nonzero_count: int


def train_lasso(ols_artifacts: OLSArtifacts) -> LassoArtifacts:
    grid_search = GridSearchCV(
        Lasso(max_iter=10000),
        param_grid={"alpha": ALPHA_GRID},
        cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
    )
    grid_search.fit(ols_artifacts.x_train, ols_artifacts.y_train)

    optimal_alpha: float = float(grid_search.best_params_["alpha"])
    model: Lasso = grid_search.best_estimator_

    y_train_prediction: np.ndarray = model.predict(ols_artifacts.x_train)
    y_test_prediction: np.ndarray = model.predict(ols_artifacts.x_test)

    train_r2: float = r2_score(ols_artifacts.y_train, y_train_prediction)
    test_r2: float = r2_score(ols_artifacts.y_test, y_test_prediction)
    train_rmse: float = float(np.sqrt(mean_squared_error(ols_artifacts.y_train, y_train_prediction)))
    test_rmse: float = float(np.sqrt(mean_squared_error(ols_artifacts.y_test, y_test_prediction)))

    coefficients = pd.Series(model.coef_, index=ols_artifacts.feature_names, name="Lasso")
    nonzero_count: int = int((coefficients != 0).sum())

    return LassoArtifacts(
        model=model,
        optimal_alpha=optimal_alpha,
        coefficients=coefficients,
        train_r2=train_r2,
        test_r2=test_r2,
        train_rmse=train_rmse,
        test_rmse=test_rmse,
        nonzero_count=nonzero_count,
    )


def report_metrics(lasso_artifacts: LassoArtifacts) -> None:
    table = Table(title="Lasso Regression — Performance Metrics", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Optimal λ (alpha)", f"{lasso_artifacts.optimal_alpha:.4f}")
    table.add_row("Train R²", f"{lasso_artifacts.train_r2:.4f}")
    table.add_row("Test R²", f"{lasso_artifacts.test_r2:.4f}")
    table.add_row("Train RMSE", f"{lasso_artifacts.train_rmse:,.2f}")
    table.add_row("Test RMSE", f"{lasso_artifacts.test_rmse:,.2f}")
    table.add_row("Non-zero Coefficients", str(lasso_artifacts.nonzero_count))
    table.add_row("Zeroed-out Features", str(len(lasso_artifacts.coefficients) - lasso_artifacts.nonzero_count))

    console.print(table)

    gap: float = lasso_artifacts.train_r2 - lasso_artifacts.test_r2
    color = "red" if gap > 0.05 else "green"
    console.print(f"\n[bold]Train-Test R² gap:[/bold] [{color}]{gap:.4f}[/{color}]")


def report_coefficient_sparsity(ols_artifacts: OLSArtifacts, lasso_artifacts: LassoArtifacts) -> None:
    ols_coefficients = pd.Series(ols_artifacts.model.coef_, index=ols_artifacts.feature_names, name="OLS")
    lasso_coefficients = lasso_artifacts.coefficients

    comparison = pd.DataFrame({"OLS": ols_coefficients, "Lasso": lasso_coefficients})
    comparison["Zeroed"] = comparison["Lasso"] == 0
    comparison = comparison.sort_values("OLS", key=abs, ascending=False)

    table = Table(title="OLS vs Lasso Coefficient Comparison", show_lines=True)
    table.add_column("Feature", style="cyan")
    table.add_column("OLS Coefficient", justify="right")
    table.add_column("Lasso Coefficient", justify="right")
    table.add_column("Eliminated", justify="right")

    for feature, row in comparison.iterrows():
        zeroed: bool = row["Zeroed"]
        color = "red" if zeroed else "green"
        table.add_row(
            str(feature),
            f"{row['OLS']:,.2f}",
            f"[{color}]{row['Lasso']:,.2f}[/{color}]",
            f"[{color}]{'Yes' if zeroed else 'No'}[/{color}]",
        )

    console.print(table)

    zeroed_features = comparison[comparison["Zeroed"]].index.tolist()
    retained_features = comparison[~comparison["Zeroed"]].index.tolist()

    if zeroed_features:
        console.print(f"\n[bold red]Eliminated features ({len(zeroed_features)}):[/bold red] {', '.join(zeroed_features)}")
    console.print(f"[bold green]Retained features ({len(retained_features)}):[/bold green] {', '.join(retained_features)}")
    console.print("[dim]Lasso's L1 penalty drives weak predictors to exactly zero, performing implicit feature selection.[/dim]")


def plot_coefficient_comparison(ols_artifacts: OLSArtifacts, lasso_artifacts: LassoArtifacts) -> None:
    ols_coefficients = pd.Series(ols_artifacts.model.coef_, index=ols_artifacts.feature_names, name="OLS")
    lasso_coefficients = lasso_artifacts.coefficients

    plot_df = pd.DataFrame({"OLS": ols_coefficients, "Lasso": lasso_coefficients}).reset_index()
    plot_df = plot_df.rename(columns={"index": "Feature"})
    plot_df = pd.melt(plot_df, id_vars="Feature", var_name="Model", value_name="Coefficient")

    _fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(
        data=plot_df,
        x="Coefficient",
        y="Feature",
        hue="Model",
        ax=ax,
        palette={"OLS": "steelblue", "Lasso": "tomato"},
    )
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"OLS vs Lasso Coefficients (λ={lasso_artifacts.optimal_alpha})")
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.show()


def plot_lasso_path(ols_artifacts: OLSArtifacts) -> None:
    alphas: list[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    coefficient_paths: dict[str, list[float]] = {feature: [] for feature in ols_artifacts.feature_names}

    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(ols_artifacts.x_train, ols_artifacts.y_train)
        for index, feature in enumerate(ols_artifacts.feature_names):
            coefficient_paths[feature].append(model.coef_[index])

    _fig, ax = plt.subplots(figsize=(12, 7))
    for feature, path in coefficient_paths.items():
        ax.plot(np.log10(alphas), path, label=feature, linewidth=1.2)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(x=np.log10(ols_artifacts.x_train.shape[0]), color="gray", linewidth=0.8, linestyle=":", label="log10(n)")
    ax.set_title("Lasso Coefficient Path — Coefficients vs log10(λ)")
    ax.set_xlabel("log10(λ)")
    ax.set_ylabel("Coefficient Value")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    plt.tight_layout()
    plt.show()


def main() -> None:
    ols_artifacts = get_ols_artifacts()

    console.print(
        Panel(
            f"[bold]Alpha grid:[/bold] {ALPHA_GRID}\n[bold]Cross-validation folds:[/bold] {CV_FOLDS}",
            title="Workforce Attrition — Q18 Lasso Regression",
        ),
    )

    lasso_artifacts = get_lasso_artifacts()
    report_metrics(lasso_artifacts)
    report_coefficient_sparsity(ols_artifacts, lasso_artifacts)
    plot_coefficient_comparison(ols_artifacts, lasso_artifacts)
    plot_lasso_path(ols_artifacts)


if __name__ == "__main__":
    main()
