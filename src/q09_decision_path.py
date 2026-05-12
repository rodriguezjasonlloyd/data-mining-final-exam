import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.panel import Panel
from rich.table import Table
from sklearn.tree import DecisionTreeClassifier, plot_tree

from q01_data_quality import console
from q05_decision_tree import MODEL_PATH, ModelArtifacts

PRUNED_DEPTH = 4

HYPOTHETICAL_EMPLOYEE: dict[str, float | int] = {
    "Age": 32,
    "Gender": 1,
    "Marital_Status": 0,
    "Region": 2,
    "Education_Level": 2,
    "Department": 3,
    "Employment_Type": 1,
    "Shift": 0,
    "Tenure_Years": 3.0,
    "Monthly_Salary_PHP": 28000.0,
    "Performance_Score": 3.0,
    "Performance_Rating": 2,
    "Training_Hours_YTD": 20,
    "Absences_YTD": 8,
    "Overtime_Hours_Monthly": 12,
    "Distance_Office_KM": 15,
    "Job_Satisfaction_Score": 4.0,
    "Work_Life_Balance_Score": 3.0,
    "Num_Promotions": 0,
    "Prev_Companies": 2,
}


def load_artifacts() -> ModelArtifacts:
    return ModelArtifacts(**joblib.load(MODEL_PATH))


def build_pruned_model(artifacts: ModelArtifacts) -> DecisionTreeClassifier:
    pruned = DecisionTreeClassifier(max_depth=PRUNED_DEPTH, random_state=42)
    pruned.fit(artifacts.x_train, artifacts.y_train)
    return pruned


def build_employee_frame(artifacts: ModelArtifacts) -> pd.DataFrame:
    return pd.DataFrame([HYPOTHETICAL_EMPLOYEE])[artifacts.x_train.columns]


def trace_decision_path(
    pruned_model: DecisionTreeClassifier,
    employee_frame: pd.DataFrame,
    feature_names: list[str],
) -> list[dict[str, str | float]]:
    node_indicator = pruned_model.decision_path(employee_frame)
    node_ids: np.ndarray = node_indicator.indices

    tree = pruned_model.tree_
    steps: list[dict[str, str | float]] = []

    for node_id in node_ids:
        is_leaf: bool = tree.children_left[node_id] == tree.children_right[node_id]

        if is_leaf:
            class_counts: np.ndarray = tree.value[node_id][0]
            predicted_class: int = int(np.argmax(class_counts))
            steps.append(
                {
                    "node": node_id,
                    "type": "leaf",
                    "feature": "—",
                    "threshold": "—",
                    "employee_value": "—",
                    "direction": "—",
                    "prediction": "Left" if predicted_class == 1 else "Stayed",
                },
            )
        else:
            feature_index: int = tree.feature[node_id]
            feature_name: str = feature_names[feature_index]
            threshold: float = round(float(tree.threshold[node_id]), 4)
            employee_value: float = round(float(employee_frame.iloc[0, feature_index]), 4)
            direction: str = "≤ (go left)" if employee_value <= threshold else "> (go right)"

            steps.append(
                {
                    "node": node_id,
                    "type": "split",
                    "feature": feature_name,
                    "threshold": threshold,
                    "employee_value": employee_value,
                    "direction": direction,
                    "prediction": "—",
                },
            )

    return steps


def report_decision_path(steps: list[dict[str, str | float]]) -> None:
    table = Table(title="Decision Path — Hypothetical Employee", show_lines=True)
    table.add_column("Node", justify="right", style="dim")
    table.add_column("Type", justify="left")
    table.add_column("Feature", style="cyan")
    table.add_column("Threshold", justify="right")
    table.add_column("Employee Value", justify="right")
    table.add_column("Direction", justify="left")
    table.add_column("Prediction", justify="left")

    final_prediction: str = "—"
    for step in steps:
        node_type: str = str(step["type"])
        prediction: str = str(step["prediction"])

        if node_type == "leaf":
            final_prediction = prediction
            color = "red" if prediction == "Left" else "green"
            table.add_row(
                str(step["node"]),
                f"[{color}]leaf[/{color}]",
                "—",
                "—",
                "—",
                "—",
                f"[bold {color}]{prediction}[/bold {color}]",
            )
        else:
            table.add_row(
                str(step["node"]),
                "split",
                str(step["feature"]),
                str(step["threshold"]),
                str(step["employee_value"]),
                str(step["direction"]),
                "—",
            )

    console.print(table)

    color = "red" if final_prediction == "Left" else "green"
    console.print(f"\n[bold]Final prediction:[/bold] [{color}]{final_prediction}[/{color}]")
    console.print("[dim]Interpretation: the pruned tree traverses each split using the employee's feature values until reaching a leaf node, where the majority class determines the prediction.[/dim]")


def plot_pruned_tree(pruned_model: DecisionTreeClassifier, feature_names: list[str]) -> None:
    _fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        pruned_model,
        feature_names=feature_names,
        class_names=["Stayed", "Left"],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
    )
    ax.set_title(f"Pruned Decision Tree (max_depth={PRUNED_DEPTH})")
    plt.tight_layout()
    plt.show()


def main() -> None:
    artifacts = load_artifacts()
    pruned_model = build_pruned_model(artifacts)
    feature_names: list[str] = list(artifacts.x_train.columns)
    employee_frame = build_employee_frame(artifacts)

    prediction: np.ndarray = pruned_model.predict(employee_frame)
    label: str = "Left" if prediction[0] == 1 else "Stayed"

    console.print(
        Panel(
            f"[bold]Pruned depth:[/bold] {PRUNED_DEPTH}  [bold]Leaf nodes:[/bold] {pruned_model.get_n_leaves()}  [bold]Prediction:[/bold] {label}",
            title="Workforce Attrition — Q09 Decision Path Interpretation",
        ),
    )

    steps = trace_decision_path(pruned_model, employee_frame, feature_names)
    report_decision_path(steps)
    plot_pruned_tree(pruned_model, feature_names)


if __name__ == "__main__":
    main()
