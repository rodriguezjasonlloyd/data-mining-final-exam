import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from question_01 import console
from question_05 import MODEL_PATH, ModelArtifacts


def load_artifacts() -> ModelArtifacts:
    return ModelArtifacts(**joblib.load(MODEL_PATH))


def report_metrics(artifacts: ModelArtifacts) -> np.ndarray:
    y_prediction = artifacts.model.predict(artifacts.x_test)
    report = classification_report(artifacts.y_test, y_prediction, target_names=["Stayed", "Left"], output_dict=True)
    accuracy = accuracy_score(artifacts.y_test, y_prediction)

    table = Table(title="Classification Report", show_lines=True)
    table.add_column("Class", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1-Score", justify="right")
    table.add_column("Support", justify="right")

    for label in ["Stayed", "Left"]:
        row = report[label]
        table.add_row(
            label,
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['f1-score']:.4f}",
            str(int(row["support"])),
        )

    for label in ["macro avg", "weighted avg"]:
        row = report[label]
        table.add_row(
            f"[dim]{label}[/dim]",
            f"[dim]{row['precision']:.4f}[/dim]",
            f"[dim]{row['recall']:.4f}[/dim]",
            f"[dim]{row['f1-score']:.4f}[/dim]",
            f"[dim]{int(row['support'])!s}[/dim]",
        )

    console.print(table)
    console.print(f"\n[bold]Accuracy:[/bold] {accuracy:.4f}")
    console.print("\n[bold]Most important metric:[/bold] [yellow]Recall (Left)[/yellow] — missing an employee who will leave is more costly than a false alarm.")

    return y_prediction


def plot_confusion_matrix(artifacts: ModelArtifacts, y_prediction: np.ndarray) -> None:
    cm = confusion_matrix(artifacts.y_test, y_prediction)

    _fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stayed", "Left"],
        yticklabels=["Stayed", "Left"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.show()


def main() -> None:
    artifacts = load_artifacts()

    console.print(
        Panel(
            f"[bold]Test samples:[/bold] {len(artifacts.y_test)}",
            title="Workforce Attrition — Q06 Model Evaluation",
        ),
    )

    y_prediction = report_metrics(artifacts)
    plot_confusion_matrix(artifacts, y_prediction)


if __name__ == "__main__":
    main()
