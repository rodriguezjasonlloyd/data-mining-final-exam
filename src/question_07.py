"""Q7. Feature Importance."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.panel import Panel
from rich.table import Table

from question_01 import console
from question_05 import ModelArtifacts, get_model_artifacts

TOP_N = 10


def compute_importance(artifacts: ModelArtifacts) -> pd.DataFrame:
    importances: np.ndarray = artifacts.model.feature_importances_
    features: list[str] = list(artifacts.x_train.columns)

    df = pd.DataFrame({"feature": features, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(TOP_N).reset_index(drop=True)
    df["importance_pct"] = (df["importance"] * 100).round(2)
    return df


def report_importance(importance_df: pd.DataFrame) -> None:
    table = Table(title=f"Top {TOP_N} Feature Importances", show_lines=True)
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Feature", style="bright_cyan")
    table.add_column("Importance %", justify="right")
    table.add_column("Bar", justify="left")

    max_pct: float = importance_df["importance_pct"].max()

    for rank, (_, row) in enumerate(importance_df.iterrows(), start=1):
        pct: float = row["importance_pct"]
        bar = "█" * int((pct / max_pct) * 20)
        color = "bright_green" if pct >= 10 else "bright_yellow" if pct >= 5 else "dim"
        table.add_row(str(rank), str(row["feature"]), f"[{color}]{pct:.2f}%[/{color}]", f"[{color}]{bar}[/{color}]")

    console.print(table)


def plot_importance(importance_df: pd.DataFrame) -> None:
    plot_df = importance_df.sort_values("importance_pct", ascending=True)

    _fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(data=plot_df, y="feature", x="importance_pct", ax=ax, color="steelblue")

    for bar, (_, row) in zip(bars.patches, plot_df.iterrows(), strict=False):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{row['importance_pct']:.2f}%",
            va="center",
            fontsize=9,
        )

    ax.set_title(f"Top {TOP_N} Feature Importances — Decision Tree")
    ax.set_xlabel("Importance (%)")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.show()


def main() -> None:
    artifacts = get_model_artifacts()

    console.print(
        Panel(
            f"[bold]Model depth:[/bold] {artifacts.model.get_depth()}  [bold]Features:[/bold] {artifacts.x_train.shape[1]}",
            title="Workforce Attrition — Q07 Feature Importance",
        ),
    )

    importance_df = compute_importance(artifacts)
    report_importance(importance_df)
    plot_importance(importance_df)


if __name__ == "__main__":
    main()
