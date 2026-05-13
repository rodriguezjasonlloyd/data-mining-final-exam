"""Q8. Tree Pruning and Bias-Variance Tradeoff."""

import matplotlib.pyplot as plt
import pandas as pd
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from question_01 import console
from question_05 import ModelArtifacts, get_model_artifacts

PRUNED_DEPTH = 4


def train_at_depth(artifacts: ModelArtifacts, depth: int) -> tuple[float, float]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(artifacts.x_train, artifacts.y_train)
    train_accuracy: float = accuracy_score(artifacts.y_train, model.predict(artifacts.x_train))
    test_accuracy: float = accuracy_score(artifacts.y_test, model.predict(artifacts.x_test))
    return train_accuracy, test_accuracy


def compute_depth_curve(artifacts: ModelArtifacts) -> pd.DataFrame:
    max_depth: int = artifacts.model.get_depth()
    rows: list[dict[str, float | int]] = []
    for depth in range(1, max_depth + 1):
        train_accuracy, test_accuracy = train_at_depth(artifacts, depth)
        rows.append({"depth": depth, "train_accuracy": train_accuracy, "test_accuracy": test_accuracy})
    return pd.DataFrame(rows)


def report_pruned(artifacts: ModelArtifacts, curve: pd.DataFrame) -> None:
    unpruned_depth: int = artifacts.model.get_depth()
    unpruned_row = curve[curve["depth"] == unpruned_depth].iloc[0]

    pruned_model = DecisionTreeClassifier(max_depth=PRUNED_DEPTH, random_state=42)
    pruned_model.fit(artifacts.x_train, artifacts.y_train)
    pruned_row = curve[curve["depth"] == PRUNED_DEPTH].iloc[0]

    table = Table(title="Unpruned vs Pruned Decision Tree", show_lines=True)
    table.add_column("Model", style="bright_cyan")
    table.add_column("Max Depth", justify="right")
    table.add_column("Leaf Nodes", justify="right")
    table.add_column("Train Acc", justify="right")
    table.add_column("Test Acc", justify="right")

    table.add_row(
        "Unpruned",
        str(unpruned_depth),
        str(artifacts.model.get_n_leaves()),
        f"{unpruned_row['train_accuracy']:.4f}",
        f"{unpruned_row['test_accuracy']:.4f}",
    )
    table.add_row(
        f"Pruned (depth={PRUNED_DEPTH})",
        str(PRUNED_DEPTH),
        str(pruned_model.get_n_leaves()),
        f"{pruned_row['train_accuracy']:.4f}",
        f"{pruned_row['test_accuracy']:.4f}",
    )

    console.print(table)

    gap_unpruned: float = unpruned_row["train_accuracy"] - unpruned_row["test_accuracy"]
    gap_pruned: float = pruned_row["train_accuracy"] - pruned_row["test_accuracy"]
    console.print(f"\n[bold]Train-Test gap:[/bold] unpruned=[bright_red]{gap_unpruned:.4f}[/bright_red]  pruned=[bright_green]{gap_pruned:.4f}[/bright_green]")
    console.print("[dim]A smaller gap indicates reduced overfitting.[/dim]")


def plot_depth_curve(curve: pd.DataFrame) -> None:
    _fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(curve["depth"], curve["train_accuracy"], color="steelblue", marker="o", label="Training Accuracy")
    ax.plot(curve["depth"], curve["test_accuracy"], color="tomato", marker="o", label="Testing Accuracy")
    ax.axvline(x=PRUNED_DEPTH, color="gray", linestyle="--", linewidth=1.2, label=f"Recommended depth={PRUNED_DEPTH}")

    ax.set_title("Bias-Variance Tradeoff — Tree Depth vs Accuracy")
    ax.set_xlabel("Tree Depth")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_xticks(curve["depth"].tolist())
    plt.tight_layout()
    plt.show()


def main() -> None:
    artifacts = get_model_artifacts()

    console.print(
        Panel(
            f"[bold]Unpruned depth:[/bold] {artifacts.model.get_depth()}  [bold]Pruned depth:[/bold] {PRUNED_DEPTH}",
            title="Workforce Attrition - Q08 Pruning & Bias-Variance Tradeoff",
        ),
    )

    curve = compute_depth_curve(artifacts)
    report_pruned(artifacts, curve)
    plot_depth_curve(curve)


if __name__ == "__main__":
    main()
