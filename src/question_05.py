"""Q5. Build a Decision Tree Classifier."""

from functools import cache

import pandas as pd
from pydantic import BaseModel
from rich.panel import Panel
from rich.table import Table
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from question_01 import console, get_clean_df

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

ATTRITION_DROP_COLUMNS = [
    "Employee_ID",
    "Hire_Date",
    "Attrition",
]

_model_artifacts: ModelArtifacts | None = None


@cache
def get_model_artifacts() -> ModelArtifacts:
    x, y = prepare(get_clean_df())
    return train(x, y)


class ModelArtifacts(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    model: DecisionTreeClassifier
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def encode(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    encoder = LabelEncoder()
    for column in CATEGORICAL_COLUMNS:
        if column in result.columns:
            result[column] = encoder.fit_transform(result[column].astype(str))
    return result


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    encoded = encode(df)
    x = encoded.drop(columns=ATTRITION_DROP_COLUMNS)
    y = encoded["Attrition"]
    return x, y


def train(x: pd.DataFrame, y: pd.Series) -> ModelArtifacts:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(x_train, y_train)
    return ModelArtifacts(model=model, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def report_tree(artifacts: ModelArtifacts) -> None:
    root_feature = artifacts.x_train.columns[artifacts.model.tree_.feature[0]]

    table = Table(title="Decision Tree Structure", show_lines=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Max Depth", str(artifacts.model.get_depth()))
    table.add_row("Leaf Nodes", str(artifacts.model.get_n_leaves()))
    table.add_row("Root Split Feature", str(root_feature))
    table.add_row("Total Features", str(artifacts.x_train.shape[1]))
    table.add_row("Training Samples", str(artifacts.model.tree_.n_node_samples[0]))

    console.print(table)


def main() -> None:
    df = get_clean_df()

    console.print(
        Panel(
            f"[bold]Cleaned dataset:[/bold] {df.shape[0]} rows x {df.shape[1]} columns",
            title="Workforce Attrition — Q05 Decision Tree Classifier",
        ),
    )

    model_artifacts = get_model_artifacts()

    report_tree(model_artifacts)

    console.print(
        Panel(
            f"[bold]Train:[/bold] {model_artifacts.x_train.shape[0]} samples\n[bold]Test:[/bold] {model_artifacts.x_test.shape[0]} samples\n[bold]Features:[/bold] {model_artifacts.x_train.shape[1]}",
            title="Train/Test Split (70/30)",
        ),
    )


if __name__ == "__main__":
    main()
