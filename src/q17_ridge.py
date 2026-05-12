import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from rich.panel import Panel
from rich.table import Table
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from q01_data_quality import console
from q15_ols_regression import OLS_PATH, OLSArtifacts

ALPHA_GRID: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
CV_FOLDS = 5
RIDGE_PATH = "ridge.joblib"


class RidgeArtifacts(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    model: Ridge
    optimal_alpha: float
    coefficients: pd.Series
    train_r2: float
    test_r2: float
    train_rmse: float
    test_rmse: float


def load_ols_artifacts() -> OLSArtifacts:
    return OLSArtifacts(**joblib.load(OLS_PATH))


def train_ridge(ols_artifacts: OLSArtifacts) -> RidgeArtifacts:
    grid_search = GridSearchCV(
        Ridge(),
        param_grid={"alpha": ALPHA_GRID},
        cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
    )
    grid_search.fit(ols_artifacts.x_train, ols_artifacts.y_train)

    optimal_alpha: float = float(grid_search.best_params_["alpha"])
    model: Ridge = grid_search.best_estimator_

    y_train_prediction: np.ndarray = model.predict(ols_artifacts.x_train)
    y_test_prediction: np.ndarray = model.predict(ols_artifacts.x_test)

    train_r2: float = r2_score(ols_artifacts.y_train, y_train_prediction)
    test_r2: float = r2_score(ols_artifacts.y_test, y_test_prediction)
    train_rmse: float = float(np.sqrt(mean_squared_error(ols_artifacts.y_train, y_train_prediction)))
    test_rmse: float = float(np.sqrt(mean_squared_error(ols_artifacts.y_test, y_test_prediction)))

    coefficients = pd.Series(model.coef_, index=ols_artifacts.feature_names, name="Ridge")

    return RidgeArtifacts(
        model=model,
        optimal_alpha=optimal_alpha,
        coefficients=coefficients,
        train_r2=train_r2,
        test_r2=test_r2,
        train_rmse=train_rmse,
        test_rmse=test_rmse,
    )


def report_metrics(ridge_artifacts: RidgeArtifacts) -> None:
    table = Table(title="Ridge Regression — Performance Metrics", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Optimal λ (alpha)", f"{ridge_artifacts.optimal_alpha:.4f}")
    table.add_row("Train R²", f"{ridge_artifacts.train_r2:.4f}")
    table.add_row("Test R²", f"{ridge_artifacts.test_r2:.4f}")
    table.add_row("Train RMSE", f"{ridge_artifacts.train_rmse:,.2f}")
    table.add_row("Test RMSE", f"{ridge_artifacts.test_rmse:,.2f}")

    console.print(table)

    gap: float = ridge_artifacts.train_r2 - ridge_artifacts.test_r2
    color = "red" if gap > 0.05 else "green"
    console.print(f"\n[bold]Train-Test R² gap:[/bold] [{color}]{gap:.4f}[/{color}]")


def report_coefficient_comparison(ols_artifacts: OLSArtifacts, ridge_artifacts: RidgeArtifacts) -> None:
    ols_coefficients = pd.Series(ols_artifacts.model.coef_, index=ols_artifacts.feature_names, name="OLS")
    ridge_coefficients = ridge_artifacts.coefficients

    comparison = pd.DataFrame({"OLS": ols_coefficients, "Ridge": ridge_coefficients})
    comparison["Shrinkage"] = (ols_coefficients - ridge_coefficients).abs()
    comparison = comparison.sort_values("Shrinkage", ascending=False)

    table = Table(title="OLS vs Ridge Coefficient Comparison", show_lines=True)
    table.add_column("Feature", style="cyan")
    table.add_column("OLS Coefficient", justify="right")
    table.add_column("Ridge Coefficient", justify="right")
    table.add_column("Shrinkage", justify="right")

    for feature, row in comparison.iterrows():
        shrinkage: float = row["Shrinkage"]
        color = "red" if shrinkage > 1000 else "yellow" if shrinkage > 100 else "green"
        table.add_row(
            str(feature),
            f"{row['OLS']:,.2f}",
            f"{row['Ridge']:,.2f}",
            f"[{color}]{shrinkage:,.2f}[/{color}]",
        )

    console.print(table)
    most_shrunk: str = str(comparison.index[0])
    console.print(f"\n[bold]Most shrunk feature:[/bold] [yellow]{most_shrunk}[/yellow] (shrinkage={comparison.iloc[0]['Shrinkage']:,.2f})")
    console.print("[dim]Ridge shrinks all coefficients toward zero but retains all features — no coefficient reaches exactly zero.[/dim]")


def plot_coefficient_comparison(ols_artifacts: OLSArtifacts, ridge_artifacts: RidgeArtifacts) -> None:
    ols_coefficients = pd.Series(ols_artifacts.model.coef_, index=ols_artifacts.feature_names, name="OLS")
    ridge_coefficients = ridge_artifacts.coefficients

    plot_df = pd.DataFrame({"OLS": ols_coefficients, "Ridge": ridge_coefficients}).reset_index()
    plot_df = plot_df.rename(columns={"index": "Feature"})
    plot_df = pd.melt(plot_df, id_vars="Feature", var_name="Model", value_name="Coefficient")

    _fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=plot_df, x="Coefficient", y="Feature", hue="Model", ax=ax, palette={"OLS": "steelblue", "Ridge": "tomato"})
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"OLS vs Ridge Coefficients (λ={ridge_artifacts.optimal_alpha})")
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.show()


def main() -> None:
    ols_artifacts = load_ols_artifacts()

    console.print(
        Panel(
            f"[bold]Alpha grid:[/bold] {ALPHA_GRID}\n[bold]Cross-validation folds:[/bold] {CV_FOLDS}",
            title="Workforce Attrition — Q17 Ridge Regression",
        ),
    )

    ridge_artifacts = train_ridge(ols_artifacts)
    report_metrics(ridge_artifacts)
    report_coefficient_comparison(ols_artifacts, ridge_artifacts)
    plot_coefficient_comparison(ols_artifacts, ridge_artifacts)

    joblib.dump(ridge_artifacts.model_dump(), RIDGE_PATH)
    console.print(f"[dim]Ridge artifacts saved to {RIDGE_PATH}[/dim]")


if __name__ == "__main__":
    main()
