import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from rich.panel import Panel
from rich.table import Table
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from q01_data_quality import console
from q15_ols_regression import OLS_PATH, OLSArtifacts
from q17_ridge import RIDGE_PATH, RidgeArtifacts
from q18_lasso import LASSO_PATH, LassoArtifacts

ALPHA_GRID: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
L1_RATIO_GRID: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
CV_FOLDS = 5
ELASTIC_NET_PATH = "elastic_net.joblib"


class ElasticNetArtifacts(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    model: ElasticNet
    optimal_alpha: float
    optimal_l1_ratio: float
    coefficients: pd.Series
    train_r2: float
    test_r2: float
    train_rmse: float
    test_rmse: float
    nonzero_count: int


def load_ols_artifacts() -> OLSArtifacts:
    return OLSArtifacts(**joblib.load(OLS_PATH))


def load_ridge_artifacts() -> RidgeArtifacts:
    return RidgeArtifacts(**joblib.load(RIDGE_PATH))


def load_lasso_artifacts() -> LassoArtifacts:
    return LassoArtifacts(**joblib.load(LASSO_PATH))


def train_elastic_net(ols_artifacts: OLSArtifacts) -> ElasticNetArtifacts:
    grid_search = GridSearchCV(
        ElasticNet(max_iter=10000),
        param_grid={"alpha": ALPHA_GRID, "l1_ratio": L1_RATIO_GRID},
        cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(ols_artifacts.x_train, ols_artifacts.y_train)

    optimal_alpha: float = float(grid_search.best_params_["alpha"])
    optimal_l1_ratio: float = float(grid_search.best_params_["l1_ratio"])
    model: ElasticNet = grid_search.best_estimator_

    y_train_prediction: np.ndarray = model.predict(ols_artifacts.x_train)
    y_test_prediction: np.ndarray = model.predict(ols_artifacts.x_test)

    train_r2: float = r2_score(ols_artifacts.y_train, y_train_prediction)
    test_r2: float = r2_score(ols_artifacts.y_test, y_test_prediction)
    train_rmse: float = float(np.sqrt(mean_squared_error(ols_artifacts.y_train, y_train_prediction)))
    test_rmse: float = float(np.sqrt(mean_squared_error(ols_artifacts.y_test, y_test_prediction)))

    coefficients = pd.Series(model.coef_, index=ols_artifacts.feature_names, name="ElasticNet")
    nonzero_count: int = int((coefficients != 0).sum())

    return ElasticNetArtifacts(
        model=model,
        optimal_alpha=optimal_alpha,
        optimal_l1_ratio=optimal_l1_ratio,
        coefficients=coefficients,
        train_r2=train_r2,
        test_r2=test_r2,
        train_rmse=train_rmse,
        test_rmse=test_rmse,
        nonzero_count=nonzero_count,
    )


def report_metrics(elastic_net_artifacts: ElasticNetArtifacts) -> None:
    table = Table(title="Elastic Net Regression — Performance Metrics", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Optimal λ (alpha)", f"{elastic_net_artifacts.optimal_alpha:.4f}")
    table.add_row("Optimal L1 Ratio", f"{elastic_net_artifacts.optimal_l1_ratio:.4f}")
    table.add_row("Train R²", f"{elastic_net_artifacts.train_r2:.4f}")
    table.add_row("Test R²", f"{elastic_net_artifacts.test_r2:.4f}")
    table.add_row("Train RMSE", f"{elastic_net_artifacts.train_rmse:,.2f}")
    table.add_row("Test RMSE", f"{elastic_net_artifacts.test_rmse:,.2f}")
    table.add_row("Non-Zero Coefficients", str(elastic_net_artifacts.nonzero_count))
    table.add_row("Zeroed-out Features", str(len(elastic_net_artifacts.coefficients) - elastic_net_artifacts.nonzero_count))

    console.print(table)

    gap: float = elastic_net_artifacts.train_r2 - elastic_net_artifacts.test_r2
    color = "red" if gap > 0.05 else "green"
    console.print(f"\n[bold]Train-Test R² gap:[/bold] [{color}]{gap:.4f}[/{color}]")


def report_coefficient_comparison(
    ols_artifacts: OLSArtifacts,
    ridge_artifacts: RidgeArtifacts,
    lasso_artifacts: LassoArtifacts,
    elastic_net_artifacts: ElasticNetArtifacts,
) -> None:
    ols_coefficients = pd.Series(ols_artifacts.model.coef_, index=ols_artifacts.feature_names, name="OLS")

    comparison = pd.DataFrame(
        {
            "OLS": ols_coefficients,
            "Ridge": ridge_artifacts.coefficients,
            "Lasso": lasso_artifacts.coefficients,
            "ElasticNet": elastic_net_artifacts.coefficients,
        },
    )
    comparison = comparison.sort_values("OLS", key=abs, ascending=False)

    table = Table(title="Coefficient Comparison: OLS vs Ridge vs Lasso vs Elastic Net", show_lines=True)
    table.add_column("Feature", style="cyan")
    table.add_column("OLS", justify="right")
    table.add_column("Ridge", justify="right", style="yellow")
    table.add_column("Lasso", justify="right", style="magenta")
    table.add_column("Elastic Net", justify="right", style="green")

    for feature, row in comparison.iterrows():
        lasso_display = "[dim]0.00[/dim]" if row["Lasso"] == 0.0 else f"{row['Lasso']:,.2f}"
        elastic_net_display = "[dim]0.00[/dim]" if row["ElasticNet"] == 0.0 else f"{row['ElasticNet']:,.2f}"
        table.add_row(
            str(feature),
            f"{row['OLS']:,.2f}",
            f"{row['Ridge']:,.2f}",
            lasso_display,
            elastic_net_display,
        )

    console.print(table)

    zeroed_features = comparison[comparison["ElasticNet"] == 0].index.tolist()
    retained_features = comparison[comparison["ElasticNet"] != 0].index.tolist()

    if zeroed_features:
        console.print(f"\n[bold red]Elastic Net eliminated ({len(zeroed_features)}):[/bold red] {', '.join(zeroed_features)}")
    console.print(f"[bold green]Retained ({len(retained_features)}):[/bold green] {', '.join(retained_features)}")
    console.print("[dim]Elastic Net blends L1 and L2 penalties — it selects features like Lasso while handling correlated predictors more stably like Ridge.[/dim]")


def report_model_comparison(
    ridge_artifacts: RidgeArtifacts,
    lasso_artifacts: LassoArtifacts,
    elastic_net_artifacts: ElasticNetArtifacts,
) -> None:
    ridge_nonzero_count: int = int((ridge_artifacts.coefficients != 0).sum())

    table = Table(title="Regularized Regression Comparison: Ridge vs Lasso vs Elastic Net", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Ridge", justify="right", style="yellow")
    table.add_column("Lasso", justify="right", style="magenta")
    table.add_column("Elastic Net", justify="right", style="green")

    table.add_row("Train R²", f"{ridge_artifacts.train_r2:.4f}", f"{lasso_artifacts.train_r2:.4f}", f"{elastic_net_artifacts.train_r2:.4f}")
    table.add_row("Test R²", f"{ridge_artifacts.test_r2:.4f}", f"{lasso_artifacts.test_r2:.4f}", f"{elastic_net_artifacts.test_r2:.4f}")
    table.add_row("Train RMSE", f"{ridge_artifacts.train_rmse:,.2f}", f"{lasso_artifacts.train_rmse:,.2f}", f"{elastic_net_artifacts.train_rmse:,.2f}")
    table.add_row("Test RMSE", f"{ridge_artifacts.test_rmse:,.2f}", f"{lasso_artifacts.test_rmse:,.2f}", f"{elastic_net_artifacts.test_rmse:,.2f}")
    table.add_row("Optimal λ", f"{ridge_artifacts.optimal_alpha:.4f}", f"{lasso_artifacts.optimal_alpha:.4f}", f"{elastic_net_artifacts.optimal_alpha:.4f}")
    table.add_row("L1 Ratio", "N/A", "1.0000", f"{elastic_net_artifacts.optimal_l1_ratio:.4f}")
    table.add_row("Non-Zero Coefficients", str(ridge_nonzero_count), str(lasso_artifacts.nonzero_count), str(elastic_net_artifacts.nonzero_count))

    console.print(table)


def plot_coefficient_comparison(
    ols_artifacts: OLSArtifacts,
    ridge_artifacts: RidgeArtifacts,
    lasso_artifacts: LassoArtifacts,
    elastic_net_artifacts: ElasticNetArtifacts,
) -> None:
    ols_coefficients = pd.Series(ols_artifacts.model.coef_, index=ols_artifacts.feature_names, name="OLS")

    plot_df = (
        pd.DataFrame(
            {
                "OLS": ols_coefficients,
                "Ridge": ridge_artifacts.coefficients,
                "Lasso": lasso_artifacts.coefficients,
                "Elastic Net": elastic_net_artifacts.coefficients,
            },
        )
        .reset_index()
        .rename(columns={"index": "Feature"})
    )
    plot_df = pd.melt(plot_df, id_vars="Feature", var_name="Model", value_name="Coefficient")

    _fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=plot_df,
        x="Coefficient",
        y="Feature",
        hue="Model",
        ax=ax,
        palette={"OLS": "steelblue", "Ridge": "gold", "Lasso": "tomato", "Elastic Net": "mediumseagreen"},
    )
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Coefficient Comparison — OLS vs Ridge vs Lasso vs Elastic Net (λ={elastic_net_artifacts.optimal_alpha}, l1_ratio={elastic_net_artifacts.optimal_l1_ratio})")
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.show()


def main() -> None:
    console.print(
        Panel(
            f"[bold]Alpha grid:[/bold] {ALPHA_GRID}\n[bold]L1 Ratio grid:[/bold] {L1_RATIO_GRID}\n[bold]Cross-validation folds:[/bold] {CV_FOLDS}",
            title="Workforce Attrition — Q19 Elastic Net Regression",
        ),
    )

    ols_artifacts = load_ols_artifacts()
    ridge_artifacts = load_ridge_artifacts()
    lasso_artifacts = load_lasso_artifacts()

    elastic_net_artifacts = train_elastic_net(ols_artifacts)

    report_metrics(elastic_net_artifacts)
    report_coefficient_comparison(ols_artifacts, ridge_artifacts, lasso_artifacts, elastic_net_artifacts)
    report_model_comparison(ridge_artifacts, lasso_artifacts, elastic_net_artifacts)
    plot_coefficient_comparison(ols_artifacts, ridge_artifacts, lasso_artifacts, elastic_net_artifacts)

    joblib.dump(elastic_net_artifacts.model_dump(), ELASTIC_NET_PATH)
    console.print(f"[dim]Elastic Net artifacts saved to {ELASTIC_NET_PATH}[/dim]")


if __name__ == "__main__":
    main()
