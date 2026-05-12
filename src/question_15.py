import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel
from rich.panel import Panel
from rich.table import Table
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from question_01 import clean, console, load_raw

CATEGORICAL_COLUMNS = [
    "Gender",
    "Marital_Status",
    "Region",
    "Education_Level",
    "Department",
    "Employment_Type",
    "Shift",
    "Performance_Rating",
]

DROP_COLUMNS = [
    "Employee_ID",
    "Hire_Date",
    "Monthly_Salary_PHP",
    "Attrition",
]

TARGET_COLUMN = "Monthly_Salary_PHP"
OLS_PATH = "ols.joblib"


class OLSArtifacts(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    model: LinearRegression
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]


def encode(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    encoder = LabelEncoder()
    for column in CATEGORICAL_COLUMNS:
        if column in result.columns:
            result[column] = encoder.fit_transform(result[column].astype(str))
    return result


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    encoded = encode(df)
    x = encoded.drop(columns=DROP_COLUMNS)
    y = encoded[TARGET_COLUMN]
    return x, y


def train_ols(x: pd.DataFrame, y: pd.Series) -> OLSArtifacts:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    return OLSArtifacts(
        model=model,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(x.columns),
    )


def compute_metrics(artifacts: OLSArtifacts) -> dict[str, float]:
    y_train_prediction: np.ndarray = artifacts.model.predict(artifacts.x_train)
    y_test_prediction: np.ndarray = artifacts.model.predict(artifacts.x_test)

    train_r2: float = r2_score(artifacts.y_train, y_train_prediction)
    test_r2: float = r2_score(artifacts.y_test, y_test_prediction)
    train_rmse: float = float(np.sqrt(mean_squared_error(artifacts.y_train, y_train_prediction)))
    test_rmse: float = float(np.sqrt(mean_squared_error(artifacts.y_test, y_test_prediction)))

    n: int = len(artifacts.x_train)
    p: int = artifacts.x_train.shape[1]
    adjusted_r2: float = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)

    return {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "adjusted_r2": adjusted_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
    }


def report_metrics(metrics: dict[str, float]) -> None:
    table = Table(title="OLS Regression — Performance Metrics", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Train R²", f"{metrics['train_r2']:.4f}")
    table.add_row("Test R²", f"{metrics['test_r2']:.4f}")
    table.add_row("Adjusted R² (train)", f"{metrics['adjusted_r2']:.4f}")
    table.add_row("Train RMSE", f"{metrics['train_rmse']:,.2f}")
    table.add_row("Test RMSE", f"{metrics['test_rmse']:,.2f}")

    console.print(table)

    gap: float = metrics["train_r2"] - metrics["test_r2"]
    color = "red" if gap > 0.05 else "green"
    console.print(f"\n[bold]Train-Test R² gap:[/bold] [{color}]{gap:.4f}[/{color}]")
    console.print("[dim]A gap > 0.05 suggests overfitting.[/dim]")


def report_statsmodels_summary(x_train: pd.DataFrame, y_train: pd.Series) -> None:
    x_with_constant = sm.add_constant(x_train)
    ols_model = sm.OLS(y_train, x_with_constant).fit()

    table = Table(title="OLS Coefficients (statsmodels)", show_lines=True)
    table.add_column("Feature", style="cyan")
    table.add_column("Coefficient", justify="right")
    table.add_column("Std Error", justify="right")
    table.add_column("t-stat", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Significant", justify="right")

    for feature in ols_model.params.index:
        coefficient: float = ols_model.params[feature]
        std_error: float = ols_model.bse[feature]
        t_stat: float = ols_model.tvalues[feature]
        p_value: float = ols_model.pvalues[feature]
        significant: bool = p_value < 0.05
        color = "green" if significant else "dim"
        table.add_row(
            str(feature),
            f"[{color}]{coefficient:,.2f}[/{color}]",
            f"{std_error:,.2f}",
            f"{t_stat:.4f}",
            f"{p_value:.4f}",
            f"[{color}]{'Yes' if significant else 'No'}[/{color}]",
        )

    console.print(table)


def main() -> None:
    df = clean(load_raw())

    console.print(
        Panel(
            f"[bold]Cleaned dataset:[/bold] {df.shape[0]} rows x {df.shape[1]} columns\n[bold]Target:[/bold] {TARGET_COLUMN}",
            title="Workforce Attrition — Q15 OLS Regression",
        ),
    )

    x, y = prepare(df)
    artifacts = train_ols(x, y)
    metrics = compute_metrics(artifacts)

    report_metrics(metrics)
    report_statsmodels_summary(artifacts.x_train, artifacts.y_train)

    console.print(
        Panel(
            f"[bold]Train samples:[/bold] {artifacts.x_train.shape[0]}  [bold]Test samples:[/bold] {artifacts.x_test.shape[0]}  [bold]Features:[/bold] {len(artifacts.feature_names)}",
            title="Train/Test Split (70/30)",
        ),
    )

    joblib.dump(artifacts.model_dump(), OLS_PATH)
    console.print(f"[dim]OLS artifacts saved to {OLS_PATH}[/dim]")


if __name__ == "__main__":
    main()
