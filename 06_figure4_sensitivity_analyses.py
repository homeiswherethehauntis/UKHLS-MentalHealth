from __future__ import annotations

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
    ORDER,
    TEXT_COLORS,
    WIDTH_INCHES,
    add_panel_letter,
    ci_label,
    clean_axis,
    draw_sensitivity_panel,
    journal_style,
    legend_handles,
    save_figure,
)

import config


# Between-model name, within-model name, outcome, extra controls, exclude movers.
SENSITIVITY_MODELS = [
    ("between_primary_person_mean", "primary_fe_1to5_age", "ghq12_raw", [], False),
    (
        "between_sf12_mcs_outcome",
        "sf12_mcs_outcome_sensitivity",
        "sf12mcs_raw",
        [],
        False,
    ),
    (
        "between_ghq_case_ge4_lpm",
        "ghq_case_ge4_lpm_sensitivity",
        "ghq_case_ge4",
        [],
        False,
    ),
    (
        "between_pcs_adjusted",
        "pcs_adjusted_sensitivity",
        "ghq12_raw",
        ["sf12pcs_raw"],
        False,
    ),
    (
        "between_age_squared_adjusted",
        "age_squared_sensitivity",
        "ghq12_raw",
        ["age_sq"],
        False,
    ),
    (
        "between_excluding_observed_movers",
        "excluding_observed_movers",
        "ghq12_raw",
        [],
        True,
    ),
]


def remove_observed_movers(panel: pd.DataFrame) -> pd.DataFrame:
    moved = panel["moved_since_last_wave"].eq(1).groupby(panel["pidp"]).transform("max")
    return panel[~moved].copy()


def estimate_sensitivity_models(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the same six checks for between-person and within-person models."""
    samples = {
        False: panel,
        True: remove_observed_movers(panel),
    }
    between_rows = []
    within_rows = []

    for domain, label, measure in config.MEASURES:
        for (
            between_model,
            within_model,
            outcome,
            extra_controls,
            exclude_movers,
        ) in SENSITIVITY_MODELS:
            model_panel = samples[exclude_movers]

            between_rows.append(
                fit_between_measure(
                    model_panel,
                    domain=domain,
                    label=label,
                    measure=measure,
                    outcome=outcome,
                    controls=[*primary_controls(), *extra_controls],
                    model=between_model,
                )
            )
            within_rows.append(
                fit_within_measure(
                    model_panel,
                    domain=domain,
                    label=label,
                    measure=measure,
                    outcome=outcome,
                    extra_controls=extra_controls,
                    model=within_model,
                )
            )

    return pd.DataFrame(between_rows), pd.DataFrame(within_rows)


def draw_outcome_panel(
    ax: plt.Axes,
    annotation_ax: plt.Axes,
    results: pd.DataFrame,
    *,
    model: str,
    title: str,
    xlim: tuple[float, float],
    show_labels: bool,
) -> None:
    data = results[results["model"].eq(model)].set_index("cohesion").loc[ORDER]
    positions = np.arange(3)[::-1]

    for position, (_, result) in zip(positions, data.iterrows()):
        ax.errorbar(
            result["estimate"],
            position,
            xerr=[
                [result["estimate"] - result["ci_low"]],
                [result["ci_high"] - result["estimate"]],
            ],
            fmt="o",
            color=COLORS[result.name],
            markersize=4.5,
            capsize=2,
            linewidth=1.3,
        )
        annotation_ax.text(
            0,
            position,
            ci_label(result),
            transform=annotation_ax.get_yaxis_transform(),
            va="center",
            fontsize=6.8,
            color=TEXT_COLORS[result.name],
        )

    labels = [LABELS[name] for name in ORDER] if show_labels else ["", "", ""]
    ax.set_yticks(positions, labels)
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

    plot_axes = []
    annotation_axes = []
    for row in range(2):
        for column in range(3):
            inner = outer[row, column].subgridspec(
                1, 2, width_ratios=[1.0, 0.72], wspace=0.04
            )
            plot_axes.append(fig.add_subplot(inner[0, 0]))
            annotation_axes.append(fig.add_subplot(inner[0, 1]))

    outcome_panels = [
        (0, between, "between_sf12_mcs_outcome", "SF-12 mental health outcome", (0.0, 3.7), True),
        (1, between, "between_ghq_case_ge4_lpm", "GHQ case >=4", (-0.12, 0.02), False),
        (
            3,
            within,
            "sf12_mcs_outcome_sensitivity",
            "SF-12 mental health outcome",
            (0.0, 2.1),
            True,
        ),
        (4, within, "ghq_case_ge4_lpm_sensitivity", "GHQ case >=4", (-0.085, 0.02), False),
    ]
    for index, results, model, title, xlim, show_labels in outcome_panels:
        draw_outcome_panel(
            plot_axes[index],
            annotation_axes[index],
            results,
            model=model,
            title=title,
            xlim=xlim,
            show_labels=show_labels,
        )

    adjustment_models = [
        ("between_primary_person_mean", "Primary model"),
        ("between_age_squared_adjusted", "Adjusted age squared"),
        ("between_pcs_adjusted", "Adjusted SF-12 PCS"),
        ("between_excluding_observed_movers", "Excluding movers"),
    ]
    draw_sensitivity_panel(
        plot_axes[2],
        between,
        models=adjustment_models,
        xlim=(-2.2, 0.2),
        title="Extra adjustment and sample restriction",
        annotation_ax=annotation_axes[2],
    )

    adjustment_models = [
        ("primary_fe_1to5_age", "Primary FE"),
        ("age_squared_sensitivity", "Adjusted age squared"),
        ("pcs_adjusted_sensitivity", "Adjusted SF-12 PCS"),
        ("excluding_observed_movers", "Excluding movers"),
    ]
    draw_sensitivity_panel(
        plot_axes[5],
        within,
        models=adjustment_models,
        xlim=(-1.45, 0.2),
        title="Extra adjustment and sample restriction",
        annotation_ax=annotation_axes[5],
    )

    for letter, ax in zip("abcdef", plot_axes):
        add_panel_letter(ax, letter)

    fig.text(0.01, 0.985, "Between-person models", fontweight="bold", fontsize=9.5, va="top")
    fig.text(
        0.01,
        0.49,
        "Within-person fixed-effects models",
        fontweight="bold",
        fontsize=9.5,
        va="top",
    )
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
    between, within = estimate_sensitivity_models(load_panel())

    save_model_summary(between, "figure4_between_person_sensitivity.csv")
    save_model_summary(within, "figure4_within_person_sensitivity.csv")
    pd.concat(
        [
            between.assign(analysis="between_person"),
            within.assign(analysis="within_person_fixed_effects"),
        ],
        ignore_index=True,
    ).to_csv(
        config.SOURCE_DIR / "figure4_sensitivity_between_and_within.csv",
        index=False,
    )
    make_figure(between, within)


if __name__ == "__main__":
    main()
