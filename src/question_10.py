"""Q10. Data Standardization."""

from functools import cache

import numpy as np
import pandas as pd
from pydantic import BaseModel
from rich.table import Table
from sklearn.preprocessing import StandardScaler

from question_01 import console, get_clean_df

CLUSTERING_FEATURES = [
    "Monthly_Salary_PHP",
    "Performance_Score",
    "Job_Satisfaction_Score",
    "Work_Life_Balance_Score",
    "Tenure_Years",
    "Absences_YTD",
    "Training_Hours_YTD",
    "Overtime_Hours_Monthly",
]


class ScalerArtifacts(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    scaler: StandardScaler
    scaled_features: np.ndarray
    feature_names: list[str]
    employee_ids: pd.Series


@cache
def get_scaler_artifacts() -> ScalerArtifacts:
    return standardize(get_clean_df())


def standardize(df: pd.DataFrame) -> ScalerArtifacts:
    subset = df[CLUSTERING_FEATURES].copy()
    scaler = StandardScaler()
    scaled_features: np.ndarray = scaler.fit_transform(subset)
    return ScalerArtifacts(
        scaler=scaler,
        scaled_features=scaled_features,
        feature_names=CLUSTERING_FEATURES,
        employee_ids=df["Employee_ID"].reset_index(drop=True),
    )


def report_standardization(df: pd.DataFrame, artifacts: ScalerArtifacts) -> None:
    scaled_df = pd.DataFrame(artifacts.scaled_features, columns=artifacts.feature_names)

    table = Table(title="Standardization Summary (Z-Score)", show_lines=True)
    table.add_column("Feature", style="bright_cyan")
    table.add_column("Original Mean", justify="right")
    table.add_column("Original Std", justify="right")
    table.add_column("Scaled Mean", justify="right")
    table.add_column("Scaled Std", justify="right")

    for feature in artifacts.feature_names:
        original_mean: float = df[feature].mean()
        original_std: float = df[feature].std()
        scaled_mean: float = scaled_df[feature].mean()
        scaled_std: float = scaled_df[feature].std()

        table.add_row(
            feature,
            f"{original_mean:.2f}",
            f"{original_std:.2f}",
            f"{scaled_mean:.4f}",
            f"{scaled_std:.4f}",
        )

    console.print(table)


def main() -> None:
    df = get_clean_df()

    scaler_artifacts = standardize(df)

    report_standardization(df, scaler_artifacts)


if __name__ == "__main__":
    main()
