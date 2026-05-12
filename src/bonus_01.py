"""Bonus 1. Decision Tree Optimization."""

import matplotlib.pyplot as plt
import pandas as pd
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from question_01 import console
from question_05 import ModelArtifacts, get_model_artifacts


def compute_depth_curve(artifacts: ModelArtifacts) -> pd.DataFrame:
    max_depth: int = artifacts.model.get_depth()
    rows: list[dict[str, float | int]] = []
    for depth in range(1, max_depth + 1):
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(artifacts.x_train, artifacts.y_train)
        training_accuracy: float = accuracy_score(artifacts.y_train, model.predict(artifacts.x_train))
        testing_accuracy: float = accuracy_score(artifacts.y_test, model.predict(artifacts.x_test))
        rows.append({"depth": depth, "training_accuracy": training_accuracy, "testing_accuracy": testing_accuracy})
    return pd.DataFrame(rows)


def find_optimal_depth(curve: pd.DataFrame) -> int:
    return int(curve.loc[curve["testing_accuracy"].idxmax(), "depth"])


def report_depth_curve(curve: pd.DataFrame, optimal_depth: int) -> None:
    table = Table(title="Tree Depth vs Accuracy — Training and Testing", show_lines=True)
    table.add_column("Depth", style="cyan", justify="right")
    table.add_column("Training Accuracy", justify="right")
    table.add_column("Testing Accuracy", justify="right")
    table.add_column("Train-Test Gap", justify="right")
    table.add_column("Note", justify="left")

    for _, row in curve.iterrows():
        depth: int = int(row["depth"])
        training_accuracy: float = row["training_accuracy"]
        testing_accuracy: float = row["testing_accuracy"]
        gap: float = training_accuracy - testing_accuracy

        gap_color = "red" if gap > 0.05 else "yellow" if gap > 0.02 else "green"
        note = "[bold green]★ Optimal[/bold green]" if depth == optimal_depth else ""

        table.add_row(
            str(depth),
            f"{training_accuracy:.4f}",
            f"{testing_accuracy:.4f}",
            f"[{gap_color}]{gap:.4f}[/{gap_color}]",
            note,
        )

    console.print(table)


def report_tradeoff_analysis(curve: pd.DataFrame, optimal_depth: int) -> None:
    optimal_row = curve[curve["depth"] == optimal_depth].iloc[0]
    max_depth: int = int(curve["depth"].max())
    full_row = curve[curve["depth"] == max_depth].iloc[0]

    underfitting_depths = curve[curve["testing_accuracy"] < curve["testing_accuracy"].quantile(0.33)]
    underfitting_range = f"depth 1-{int(underfitting_depths['depth'].max())}" if not underfitting_depths.empty else "none"

    overfitting_depths = curve[(curve["depth"] > optimal_depth) & (curve["training_accuracy"] - curve["testing_accuracy"] > 0.05)]
    overfitting_start = int(overfitting_depths["depth"].min()) if not overfitting_depths.empty else optimal_depth + 1

    console.print(
        Panel(
            f"[bold]Optimal depth:[/bold] [green]{optimal_depth}[/green]  "
            f"[bold]Testing accuracy at optimal:[/bold] {optimal_row['testing_accuracy']:.4f}  "
            f"[bold]Training accuracy at optimal:[/bold] {optimal_row['training_accuracy']:.4f}\n\n"
            f"[bold]Underfitting zone:[/bold] {underfitting_range} — low depth limits the tree's ability to capture "
            f"meaningful decision boundaries, resulting in both low training and testing accuracy (high bias).\n\n"
            f"[bold]Overfitting zone:[/bold] depth ≥ {overfitting_start} — the tree memorizes training noise. "
            f"At full depth ({max_depth}), training accuracy={full_row['training_accuracy']:.4f} but "
            f"testing accuracy={full_row['testing_accuracy']:.4f} "
            f"(gap={full_row['training_accuracy'] - full_row['testing_accuracy']:.4f}), indicating high variance.\n\n"
            f"[bold]Bias-variance tradeoff:[/bold] Shallow trees have high bias (underfit); deep trees have high variance "
            f"(overfit). Depth {optimal_depth} balances both — it achieves the highest generalizable testing accuracy "
            f"({optimal_row['testing_accuracy']:.4f}) before variance begins to dominate.",
            title="Bonus 1 — Bias-Variance Tradeoff Analysis",
        ),
    )


def plot_depth_curve(curve: pd.DataFrame, optimal_depth: int) -> None:
    _fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(curve["depth"], curve["training_accuracy"], color="steelblue", marker="o", linewidth=2, label="Training Accuracy")
    ax.plot(curve["depth"], curve["testing_accuracy"], color="tomato", marker="o", linewidth=2, label="Testing Accuracy")
    ax.axvline(x=optimal_depth, color="seagreen", linestyle="--", linewidth=1.5, label=f"Optimal Depth = {optimal_depth}")

    optimal_row = curve[curve["depth"] == optimal_depth].iloc[0]
    ax.annotate(
        f"Test={optimal_row['testing_accuracy']:.4f}",
        xy=(optimal_depth, optimal_row["testing_accuracy"]),
        xytext=(optimal_depth + 0.5, optimal_row["testing_accuracy"] - 0.02),
        fontsize=9,
        color="seagreen",
    )

    ax.set_title("Decision Tree — Tree Depth vs Training and Testing Accuracy")
    ax.set_xlabel("Tree Depth")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(curve["depth"].tolist())
    ax.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    artifacts = get_model_artifacts()

    console.print(
        Panel(
            f"[bold]Unpruned tree depth:[/bold] {artifacts.model.get_depth()}  [bold]Training samples:[/bold] {len(artifacts.x_train)}  [bold]Testing samples:[/bold] {len(artifacts.x_test)}",
            title="Workforce Attrition — Bonus 1: Decision Tree Optimization",
        ),
    )

    curve = compute_depth_curve(artifacts)
    optimal_depth = find_optimal_depth(curve)

    report_depth_curve(curve, optimal_depth)
    report_tradeoff_analysis(curve, optimal_depth)
    plot_depth_curve(curve, optimal_depth)


if __name__ == "__main__":
    main()
