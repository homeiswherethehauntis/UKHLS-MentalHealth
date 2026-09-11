from __future__ import annotations

from importlib import import_module

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

config = import_module("00_config")


def modifier_specification(
    work: pd.DataFrame, measure: str, modifier: str, wave_columns: list[str]
) -> tuple[pd.DataFrame, list[str], str, str, list[tuple[str, dict[str, float], dict[str, float] | None]]]:
    if modifier == "care_intensity":
        work[f"{measure}_care_mod"] = work[measure] * work["care_mod"]
        work[f"{measure}_care_high"] = work[measure] * work["care_high"]
        design = [
            measure,
            "care_mod",
            "care_high",
            "childcare_yes",
            f"{measure}_care_mod",
            f"{measure}_care_high",
            "age",
            "log_income",
            "partnered",
            "employed",
            *wave_columns,
        ]
        groups = [
            ("Low (<20h)", {measure: 1.0}, None),
            (
                "Moderate (20-99h)",
                {measure: 1.0, f"{measure}_care_mod": 1.0},
                {f"{measure}_care_mod": 1.0},
            ),
            (
                "Very High (>=100h / continuous)",
                {measure: 1.0, f"{measure}_care_high": 1.0},
                {f"{measure}_care_high": 1.0},
            ),
        ]
        return work, design, "care_intensity", "Low (<20h)", groups

    if modifier == "childcare":
        interaction = f"{measure}_childcare"
        work[interaction] = work[measure] * work["childcare_yes"]
        design = [
            measure,
            "childcare_yes",
            interaction,
            "care_mod",
            "care_high",
            "age",
            "log_income",
            "partnered",
            "employed",
            *wave_columns,
        ]
        groups = [
            ("No", {measure: 1.0}, None),
            ("Yes", {measure: 1.0, interaction: 1.0}, {interaction: 1.0}),
        ]
        return work, design, "childcare_group", "No", groups

    if modifier == "employment_status":
        interaction = f"{measure}_employed"
        work[interaction] = work[measure] * work["employed"]
        design = [
            measure,
            "employed",
            interaction,
            "care_mod",
            "care_high",
            "childcare_yes",
            "age",
            "log_income",
            "partnered",
            *wave_columns,
        ]
        groups = [
            ("Not Employed", {measure: 1.0}, None),
            ("Employed", {measure: 1.0, interaction: 1.0}, {interaction: 1.0}),
        ]
        return work, design, "employment_group", "Not Employed", groups

    if modifier == "life_stage":
        work["life_stage"] = pd.cut(
            work["age"],
            bins=[18, 35, 50, 66, np.inf],
            labels=["18-34", "35-49", "50-65", "66+"],
            right=False,
        ).astype("string")
        group_terms = {
            "35-49": "life_35_49",
            "50-65": "life_50_65",
            "66+": "life_66_plus",
        }
        for group, term in group_terms.items():
            work[term] = work["life_stage"].eq(group).astype(float)
            work[f"{measure}_{term}"] = work[measure] * work[term]
        design = [
            measure,
            *group_terms.values(),
            *(f"{measure}_{term}" for term in group_terms.values()),
            "care_mod",
            "care_high",
            "childcare_yes",
            "age",
            "log_income",
            "partnered",
            "employed",
            *wave_columns,
        ]
        groups = [("18-34", {measure: 1.0}, None)]
        for group, term in group_terms.items():
            interaction = f"{measure}_{term}"
            groups.append(
                (group, {measure: 1.0, interaction: 1.0}, {interaction: 1.0})
            )
        return work, design, "life_stage", "18-34", groups

    raise ValueError(f"Unknown modifier: {modifier}")


def estimate_context_moderation(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain, label, measure in config.MEASURES:
        for modifier in ["care_intensity", "childcare", "employment_status", "life_stage"]:
            work, wave_columns = add_wave_dummies(panel)
            work["childcare_group"] = np.where(work["childcare_yes"].eq(1), "Yes", "No")
            work["employment_group"] = np.where(work["employed"].eq(1), "Employed", "Not Employed")
            work, design, group_column, reference, groups = modifier_specification(
                work, measure, modifier, wave_columns
            )
            result, model_data = fit_within(work, "ghq12_raw", design)
            for group, slope_weights, difference_weights in groups:
                slope = linear_combination(result, slope_weights)
                difference = (
                    linear_combination(result, difference_weights)
                    if difference_weights is not None
                    else {
                        "estimate": 0.0,
                        "std_error": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "p_value": np.nan,
                    }
                )
                group_mask = model_data[group_column].astype(str).eq(group)
                rows.append(
                    {
                        "domain": domain,
                        "cohesion": label,
                        "modifier": modifier,
                        "group": group,
                        "reference_group": reference,
                        "rows": int(group_mask.sum()),
                        "people": int(model_data.loc[group_mask, "pidp"].nunique()),
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
    plot_axes: list[plt.Axes] = []
    annotation_axes: list[plt.Axes] = []
    for row in range(2):
        for column in range(2):
            inner = outer[row, column].subgridspec(
                1, 2, width_ratios=[1.0, 0.92], wspace=0.03
            )
            plot_axes.append(fig.add_subplot(inner[0, 0]))
            annotation_axes.append(fig.add_subplot(inner[0, 1]))

    specifications = [
        (
            plot_axes[0],
            annotation_axes[0],
            "a",
            "care_intensity",
            ["Low (<20h)", "Moderate (20-99h)", "Very High (>=100h / continuous)"],
            {
                "Low (<20h)": "Low caregiving\n(Reference group)",
                "Moderate (20-99h)": "Moderate caregiving",
                "Very High (>=100h / continuous)": "Very high caregiving",
            },
        ),
        (
            plot_axes[1],
            annotation_axes[1],
            "b",
            "childcare",
            ["No", "Yes"],
            {
                "No": "No childcare responsibility\n(Reference group)",
                "Yes": "Childcare responsibility",
            },
        ),
        (
            plot_axes[2],
            annotation_axes[2],
            "c",
            "employment_status",
            ["Not Employed", "Employed"],
            {
                "Not Employed": "Not employed\n(Reference group)",
                "Employed": "Employed",
            },
        ),
        (
            plot_axes[3],
            annotation_axes[3],
            "d",
            "life_stage",
            ["18-34", "35-49", "50-65", "66+"],
            {
                "18-34": "Age 18-34\n(Reference group)",
                "35-49": "Age 35-49",
                "50-65": "Age 50-65",
                "66+": "Age 66+",
            },
        ),
    ]
    for ax, annotation_ax, letter, modifier, groups, labels in specifications:
        draw_grouped_estimates(
            ax,
            results[results["modifier"].eq(modifier)],
            groups=groups,
            labels=labels,
            xlim=(-1.75, 0.50),
            annotation_ax=annotation_ax,
        )
        add_panel_letter(ax, letter)
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
    config.make_output_directories()
    results = estimate_context_moderation(load_panel())
    save_model_summary(results, "figure3_context_moderation.csv")
    results.to_csv(config.SOURCE_DIR / "figure3_context_moderation.csv", index=False)
    make_figure(results)


if __name__ == "__main__":
    main()
