from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_core import load_panel, save_model_summary
from temporal_models import (
    forward_lagged_models,
    reverse_lagged_models,
    stacked_long_lag_models,
)
from plotting_core import (
    COLORS,
    LABELS,
    ORDER,
    WIDTH_INCHES,
    add_panel_letter,
    clean_axis,
    journal_style,
    save_figure,
)

import config


def draw_temporal_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    title: str,
    xlim: tuple[float, float],
    digits: int,
    show_labels: bool,
) -> None:
    data = frame.set_index("cohesion").loc[ORDER].reset_index()
    y = np.arange(3)[::-1]
    for position, (_, row) in zip(y, data.iterrows()):
        ax.errorbar(
            row["estimate"],
            position,
            xerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
            fmt="o",
            color=COLORS[row["cohesion"]],
            markersize=4.5,
            capsize=2,
            linewidth=1.3,
        )
        ax.annotate(
            (
                f"{row['estimate']:.{digits}f} "
                f"({row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f})"
            ),
            (row["ci_high"], position),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.2,
            color="#555555",
        )
    ax.set_yticks(y, [LABELS[name] for name in ORDER] if show_labels else ["", "", ""])
    ax.set_xlim(*xlim)
    ax.set_title(title, loc="left", fontsize=8.2, fontweight="bold", pad=10)
    clean_axis(ax)


def make_figure(forward: pd.DataFrame, five_year: pd.DataFrame, reverse: pd.DataFrame) -> None:
    journal_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(WIDTH_INCHES, 3.1),
        gridspec_kw={"wspace": 0.72},
    )
    panels = [
        (
            forward,
            "Previous perceived neighbourhood\ncohesion -> current GHQ-12",
            (-0.75, 0.75),
            2,
            True,
        ),
        (
            five_year,
            "Cohesion about 5 years earlier\n-> current GHQ-12",
            (-1.20, 0.55),
            2,
            False,
        ),
        (
            reverse,
            "Previous GHQ-12 -> current perceived\nneighbourhood cohesion",
            (-0.010, 0.015),
            3,
            False,
        ),
    ]
    for ax, (results, title, xlim, digits, show_labels) in zip(axes, panels):
        draw_temporal_panel(
            ax,
            results,
            title=title,
            xlim=xlim,
            digits=digits,
            show_labels=show_labels,
        )
    for letter, ax in zip("abc", axes):
        add_panel_letter(ax, letter)
    save_figure(fig, "figure5_temporal_order_reverse_lagged_checks")


def main() -> None:
    config.make_output_directories()
    panel = load_panel()
    forward = forward_lagged_models(panel)
    five_year = stacked_long_lag_models(
        panel,
        pairs=[("a", "f"), ("c", "i"), ("f", "l")],
        model_name="approximately_five_year_lag_fe",
    )
    reverse = reverse_lagged_models(panel)
    save_model_summary(forward, "figure5_forward_lagged_fe.csv")
    save_model_summary(five_year, "figure5_five_year_lagged_fe.csv")
    save_model_summary(reverse, "figure5_reverse_lagged_fe.csv")
    source = pd.concat(
        [
            forward.assign(panel="a_previous_cohesion_to_current_ghq"),
            five_year.assign(panel="b_five_year_cohesion_to_current_ghq"),
            reverse.assign(panel="c_previous_ghq_to_current_cohesion"),
        ],
        ignore_index=True,
    )
    source.to_csv(config.SOURCE_DIR / "figure5_temporal_order_checks.csv", index=False)
    make_figure(forward, five_year, reverse)


if __name__ == "__main__":
    main()
