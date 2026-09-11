from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_core import (
    add_wave_dummies,
    fit_within,
    linear_combination,
    load_panel,
    primary_controls,
    save_model_summary,
)
from plotting_core import (
    WIDTH_INCHES,
    add_panel_letter,
    draw_grouped_estimates,
    journal_style,
    legend_handles,
    save_figure,
)

import config


MODIFIERS = {
    "care_intensity": {
        "column": "care_intensity",
        "groups": [
            "Low (<20h)",
            "Moderate (20-99h)",
            "Very High (>=100h / continuous)",
        ],
        "labels": {
            "Low (<20h)": "Low caregiving\n(Reference group)",
            "Moderate (20-99h)": "Moderate caregiving",
            "Very High (>=100h / continuous)": "Very high caregiving",
        },
        "remove_controls": {"care_mod", "care_high"},
    },
    "childcare": {
        "column": "childcare_group",
        "groups": ["No", "Yes"],
        "labels": {
            "No": "No childcare responsibility\n(Reference group)",
            "Yes": "Childcare responsibility",
        },
        "remove_controls": {"childcare_yes"},
    },
    "employment_status": {
        "column": "employment_group",
        "groups": ["Not Employed", "Employed"],
        "labels": {
            "Not Employed": "Not employed\n(Reference group)",
            "Employed": "Employed",
        },
        "remove_controls": {"employed"},
    },
    "life_stage": {
        "column": "life_stage",
        "groups": ["18-34", "35-49", "50-65", "66+"],
        "labels": {
            "18-34": "Age 18-34\n(Reference group)",
            "35-49": "Age 35-49",
            "50-65": "Age 50-65",
            "66+": "Age 66+",
        },
        "remove_controls": set(),
    },
}


def add_context_groups(panel: pd.DataFrame) -> pd.DataFrame:
    """Create the four caregiving-context variables used in Figure 3."""
    work = panel.copy()
    work["childcare_group"] = np.where(work["childcare_yes"].eq(1), "Yes", "No")
    work["employment_group"] = np.where(
        work["employed"].eq(1), "Employed", "Not Employed"
    )
    work["life_stage"] = pd.cut(
        work["age"],
        bins=[18, 35, 50, 66, np.inf],
        labels=["18-34", "35-49", "50-65", "66+"],
        right=False,
    ).astype("string")
    return work


def estimate_context_moderation(panel: pd.DataFrame) -> pd.DataFrame:
    """Estimate subgroup associations and differences from each reference group."""
    panel = add_context_groups(panel)
    rows = []

    for domain, cohesion_label, measure in config.MEASURES:
        for modifier, specification in MODIFIERS.items():
            work, wave_terms = add_wave_dummies(panel)
            group_column = specification["column"]
            groups = specification["groups"]
            reference_group = groups[0]

            group_terms = []
            interaction_terms = {}
            for number, group in enumerate(groups[1:], start=1):
                group_term = f"{modifier}_group_{number}"
                interaction_term = f"{measure}_{modifier}_{number}"
                work[group_term] = work[group_column].eq(group).astype(float)
                work[interaction_term] = work[measure] * work[group_term]
                group_terms.append(group_term)
                interaction_terms[group] = interaction_term

            controls = [
                variable
                for variable in primary_controls(wave_terms)
                if variable not in specification["remove_controls"]
            ]
            design = [
                measure,
                *group_terms,
                *interaction_terms.values(),
                *controls,
            ]
            result, model_data = fit_within(work, "ghq12_raw", design)

            for group in groups:
                slope_weights = {measure: 1.0}
                if group != reference_group:
                    slope_weights[interaction_terms[group]] = 1.0
                slope = linear_combination(result, slope_weights)

                if group == reference_group:
                    difference = {
                        "estimate": 0.0,
                        "std_error": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "p_value": np.nan,
                    }
                else:
                    difference = linear_combination(
                        result, {interaction_terms[group]: 1.0}
                    )

                group_rows = model_data[group_column].astype(str).eq(group)
                rows.append(
                    {
                        "domain": domain,
                        "cohesion": cohesion_label,
                        "modifier": modifier,
                        "group": group,
                        "reference_group": reference_group,
                        "rows": int(group_rows.sum()),
                        "people": int(
                            model_data.loc[group_rows, "pidp"].nunique()
                        ),
                        "slope_estimate": slope["estimate"],
                        "slope_std_error": slope["std_error"],
                        "slope_ci_low": slope["ci_low"],
                        "slope_ci_high": slope["ci_high"],
                        "slope_p_value": slope["p_value"],
                        "difference_vs_reference": difference["estimate"],
                        "difference_std_error": difference["std_error"],
                        "difference_ci_low": difference["ci_low"],
                        "difference_ci_high": difference["ci_high"],
                        "difference_p_value": difference["p_value"],
                    }
                )

    return pd.DataFrame(rows)


def make_figure(results: pd.DataFrame) -> None:
    journal_style()
    fig = plt.figure(figsize=(WIDTH_INCHES, 7.3))
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.13, top=0.98)
    outer = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.34)

    plot_axes = []
    annotation_axes = []
    for row in range(2):
        for column in range(2):
            inner = outer[row, column].subgridspec(
                1, 2, width_ratios=[1.0, 0.92], wspace=0.03
            )
            plot_axes.append(fig.add_subplot(inner[0, 0]))
            annotation_axes.append(fig.add_subplot(inner[0, 1]))

    for index, (modifier, specification) in enumerate(MODIFIERS.items()):
        draw_grouped_estimates(
            plot_axes[index],
            results[results["modifier"].eq(modifier)],
            groups=specification["groups"],
            labels=specification["labels"],
            xlim=(-1.75, 0.50),
            annotation_ax=annotation_axes[index],
        )
        add_panel_letter(plot_axes[index], "abcd"[index])

    for ax in plot_axes[2:]:
        ax.set_xlabel("Association with GHQ-12 per 1-point higher cohesion")

    fig.legend(
        handles=legend_handles(),
        loc="lower left",
        bbox_to_anchor=(0.19, 0.01),
        frameon=False,
        ncol=3,
        fontsize=8,
    )
    save_figure(fig, "figure3_context_moderation_within_fe")


def main() -> None:
    results = estimate_context_moderation(load_panel())
    save_model_summary(results, "figure3_context_moderation.csv")
    results.to_csv(config.SOURCE_DIR / "figure3_context_moderation.csv", index=False)
    make_figure(results)


if __name__ == "__main__":
    main()
