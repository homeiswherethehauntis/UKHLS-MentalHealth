from __future__ import annotations

from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_core import (
    fit_between_measure,
    fit_within_measure,
    load_panel,
    primary_controls,
    save_model_summary,
)
from plotting_core import (
    COLORS,
    LABELS,
    TEXT_COLORS,
    ORDER,
    WIDTH_INCHES,
    add_panel_letter,
    ci_label,
    clean_axis,
    draw_sensitivity_panel,
    journal_style,
    legend_handles,
    save_figure,
)

config = import_module("00_config")


def non_movers(panel: pd.DataFrame) -> pd.DataFrame:
    ever_moved = panel["moved_since_last_wave"].eq(1).groupby(panel["pidp"]).transform("max")
    return panel[~ever_moved].copy()


def estimate_between_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_controls = primary_controls()
    specifications = [
        ("between_primary_person_mean", "ghq12_raw", panel, base_controls),
        ("between_sf12_mcs_outcome", "sf12mcs_raw", panel, base_controls),
        ("between_ghq_case_ge4_lpm", "ghq_case_ge4", panel, base_controls),
        (
            "between_pcs_adjusted",
            "ghq12_raw",
            panel,
            [*base_controls, "sf12pcs_raw"],
        ),
        (
            "between_age_squared_adjusted",
            "ghq12_raw",
            panel,
            [*base_controls, "age_sq"],
        ),
        (
            "between_excluding_observed_movers",
            "ghq12_raw",
            non_movers(panel),
            base_controls,
        ),
    ]
    for domain, label, measure in config.MEASURES:
        for model, outcome, model_panel, controls in specifications:
            rows.append(
                fit_between_measure(
                    model_panel,
                    domain=domain,
                    label=label,
                    measure=measure,
                    outcome=outcome,
                    controls=controls,
                    model=model,
                )
            )
    return pd.DataFrame(rows)


def estimate_within_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specifications = [
        ("primary_fe_1to5_age", "ghq12_raw", panel, []),
        ("sf12_mcs_outcome_sensitivity", "sf12mcs_raw", panel, []),
        ("ghq_case_ge4_lpm_sensitivity", "ghq_case_ge4", panel, []),
        ("pcs_adjusted_sensitivity", "ghq12_raw", panel, ["sf12pcs_raw"]),
        ("age_squared_sensitivity", "ghq12_raw", panel, ["age_sq"]),
        ("excluding_observed_movers", "ghq12_raw", non_movers(panel), []),
    ]
    for domain, label, measure in config.MEASURES:
        for model, outcome, model_panel, extra_controls in specifications:
            rows.append(
                fit_within_measure(
                    model_panel,
                    domain=domain,
                    label=label,
                    measure=measure,
                    outcome=outcome,
                    extra_controls=extra_controls,
                    model=model,
                )
            )
    return pd.DataFrame(rows)


def draw_three_outcomes(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    model: str,
    title: str,
    xlim: tuple[float, float],
    annotation_ax: plt.Axes,
    show_labels: bool = True,
) -> None:
    data = frame[frame["model"].eq(model)].set_index("cohesion").loc[ORDER].reset_index()
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
        annotation_ax.text(
            0.0,
            position,
            ci_label(row),
            transform=annotation_ax.get_yaxis_transform(),
            va="center",
            fontsize=6.8,
            color=TEXT_COLORS[row["cohesion"]],
        )
    ax.set_yticks(y, [LABELS[name] for name in ORDER] if show_labels else ["", "", ""])
    ax.set_xlim(*xlim)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=8.2, pad=7)
    clean_axis(ax)
    annotation_ax.set_ylim(ax.get_ylim())
    annotation_ax.set_axis_off()


def make_figure(between: pd.DataFrame, within: pd.DataFrame) -> None:
    journal_style()
    fig = plt.figure(figsize=(WIDTH_INCHES, 6.4))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.13, top=0.93)
    outer = fig.add_gridspec(2, 3, hspace=0.58, wspace=0.38)
    plot_axes: list[plt.Axes] = []
    annotation_axes: list[plt.Axes] = []
    for row in range(2):
        for column in range(3):
            inner = outer[row, column].subgridspec(
                1, 2, width_ratios=[1.0, 0.72], wspace=0.04
            )
            plot_axes.append(fig.add_subplot(inner[0, 0]))
            annotation_axes.append(fig.add_subplot(inner[0, 1]))

    draw_three_outcomes(
        plot_axes[0],
        between,
        model="between_sf12_mcs_outcome",
        title="SF-12 mental health outcome",
        xlim=(0.0, 3.7),
        annotation_ax=annotation_axes[0],
    )
    draw_three_outcomes(
        plot_axes[1],
        between,
        model="between_ghq_case_ge4_lpm",
        title="GHQ case >=4",
        xlim=(-0.12, 0.02),
        annotation_ax=annotation_axes[1],
        show_labels=False,
    )
    draw_sensitivity_panel(
        plot_axes[2],
        between,
        models=[
            ("between_primary_person_mean", "Primary model"),
            ("between_age_squared_adjusted", "Adjusted age squared"),
            ("between_pcs_adjusted", "Adjusted SF-12 PCS"),
            ("between_excluding_observed_movers", "Excluding movers"),
        ],
        xlim=(-2.2, 0.2),
        title="Extra adjustment and sample restriction",
        annotation_ax=annotation_axes[2],
    )
    draw_three_outcomes(
        plot_axes[3],
        within,
        model="sf12_mcs_outcome_sensitivity",
        title="SF-12 mental health outcome",
        xlim=(0.0, 2.1),
        annotation_ax=annotation_axes[3],
    )
    draw_three_outcomes(
        plot_axes[4],
        within,
        model="ghq_case_ge4_lpm_sensitivity",
        title="GHQ case >=4",
        xlim=(-0.085, 0.02),
        annotation_ax=annotation_axes[4],
        show_labels=False,
    )
    draw_sensitivity_panel(
        plot_axes[5],
        within,
        models=[
            ("primary_fe_1to5_age", "Primary FE"),
            ("age_squared_sensitivity", "Adjusted age squared"),
            ("pcs_adjusted_sensitivity", "Adjusted SF-12 PCS"),
            ("excluding_observed_movers", "Excluding movers"),
        ],
        xlim=(-1.45, 0.2),
        title="Extra adjustment and sample restriction",
        annotation_ax=annotation_axes[5],
    )
    for letter, ax in zip("abcdef", plot_axes):
        add_panel_letter(ax, letter)
    fig.text(0.01, 0.985, "Between-person models", fontweight="bold", fontsize=9.5, va="top")
    fig.text(0.01, 0.49, "Within-person fixed-effects models", fontweight="bold", fontsize=9.5, va="top")
    fig.legend(
        handles=legend_handles(),
        loc="lower left",
        bbox_to_anchor=(0.01, -0.015),
        frameon=False,
        ncol=3,
        fontsize=8,
    )
    save_figure(fig, "figure4_sensitivity_between_and_within")


def main() -> None:
    config.make_output_directories()
    panel = load_panel()
    between = estimate_between_sensitivity(panel)
    within = estimate_within_sensitivity(panel)
    save_model_summary(between, "figure4_between_person_sensitivity.csv")
    save_model_summary(within, "figure4_within_person_sensitivity.csv")
    source = pd.concat(
        [
            between.assign(analysis="between_person"),
            within.assign(analysis="within_person_fixed_effects"),
        ],
        ignore_index=True,
    )
    source.to_csv(
        config.SOURCE_DIR / "figure4_sensitivity_between_and_within.csv", index=False
    )
    make_figure(between, within)


if __name__ == "__main__":
    main()
