from __future__ import annotations

from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_core import (
    add_wave_dummies,
    cluster_ols,
    fit_within,
    linear_combination,
    load_panel,
    primary_controls,
    save_model_summary,
    tidy_target,
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

config = import_module("00_config")


def forward_lagged_models(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, label, measure in config.MEASURES:
        work = panel.sort_values(["pidp", "wave_order"]).copy()
        lagged_measure = f"lag_{measure}"
        work[lagged_measure] = work.groupby("pidp")[measure].shift(1)
        work, wave_columns = add_wave_dummies(work)
        result, model_data = fit_within(
            work,
            "ghq12_raw",
            [lagged_measure, *primary_controls(wave_columns)],
        )
        rows.append(
            tidy_target(
                result,
                {lagged_measure: 1.0},
                model="previous_cohesion_to_current_ghq_fe",
                outcome="ghq12_raw",
                cohesion=label,
                domain=domain,
                rows=len(model_data),
                people=model_data["pidp"].nunique(),
            )
        )
    return pd.DataFrame(rows)


def stacked_long_lag_models(
    panel: pd.DataFrame,
    *,
    pairs: list[tuple[str, str]],
    model_name: str,
) -> pd.DataFrame:
    origin_columns = [
        "pidp",
        "wave",
        "ghq12_raw",
        "nbr_1to5",
        "cohesion_psychological_sense_community",
        "cohesion_neighbouring",
        "care_mod",
        "care_high",
        "childcare_yes",
        "age",
        "log_income",
        "partnered",
        "employed",
    ]
    rename = {
        "wave": "origin_wave",
        "ghq12_raw": "origin_ghq12_raw",
        "care_mod": "origin_care_mod",
        "care_high": "origin_care_high",
        "childcare_yes": "origin_childcare_yes",
        "age": "origin_age",
        "log_income": "origin_log_income",
        "partnered": "origin_partnered",
        "employed": "origin_employed",
    }
    origins = panel[origin_columns].rename(columns=rename)
    future = panel[["pidp", "wave", "ghq12_raw"]].rename(
        columns={"wave": "future_wave", "ghq12_raw": "future_ghq12_raw"}
    )
    pieces = []
    for origin_wave, future_wave in pairs:
        piece = origins[origins["origin_wave"].eq(origin_wave)].copy()
        piece["future_wave"] = future_wave
        piece["lag_pair"] = f"{config.WAVE_ORDER[origin_wave]}->{config.WAVE_ORDER[future_wave]}"
        pieces.append(piece)
    stacked = pd.concat(pieces, ignore_index=True).merge(
        future, on=["pidp", "future_wave"], how="left"
    )
    pair_dummies = pd.get_dummies(
        stacked["lag_pair"], prefix="pair_fe", drop_first=True, dtype=float
    )
    stacked = pd.concat(
        [stacked.reset_index(drop=True), pair_dummies.reset_index(drop=True)], axis=1
    )
    controls = [
        "origin_ghq12_raw",
        "origin_care_mod",
        "origin_care_high",
        "origin_childcare_yes",
        "origin_age",
        "origin_log_income",
        "origin_partnered",
        "origin_employed",
        *pair_dummies.columns.tolist(),
    ]
    rows = []
    for domain, label, measure in config.MEASURES:
        result, model_data = fit_within(
            stacked, "future_ghq12_raw", [measure, *controls]
        )
        row = tidy_target(
            result,
            {measure: 1.0},
            model=model_name,
            outcome="future_ghq12_raw",
            cohesion=label,
            domain=domain,
            rows=len(model_data),
            people=model_data["pidp"].nunique(),
        )
        row["lag_pairs"] = "; ".join(
            f"{config.WAVE_ORDER[start]}->{config.WAVE_ORDER[end]}" for start, end in pairs
        )
        rows.append(row)
    return pd.DataFrame(rows)


def reverse_lagged_models(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.sort_values(["pidp", "wave_order"]).copy()
    work["lag_ghq12_raw"] = work.groupby("pidp")["ghq12_raw"].shift(1)
    work, wave_columns = add_wave_dummies(work)
    rows = []
    for domain, label, measure in config.MEASURES:
        result, model_data = fit_within(
            work,
            measure,
            ["lag_ghq12_raw", *primary_controls(wave_columns)],
        )
        rows.append(
            tidy_target(
                result,
                {"lag_ghq12_raw": 1.0},
                model="previous_ghq_to_current_cohesion_fe",
                outcome=measure,
                cohesion=label,
                domain=domain,
                rows=len(model_data),
                people=model_data["pidp"].nunique(),
            )
        )
    return pd.DataFrame(rows)


def baseline_prospective_models(panel: pd.DataFrame) -> pd.DataFrame:
    origin = panel[panel["wave"].eq("a")].copy().rename(
        columns={
            "ghq12_raw": "origin_ghq12_raw",
            "care_mod": "origin_care_mod",
            "care_high": "origin_care_high",
            "childcare_yes": "origin_childcare_yes",
            "age": "origin_age",
            "log_income": "origin_log_income",
            "partnered": "origin_partnered",
            "employed": "origin_employed",
        }
    )
    future = panel[panel["wave"].eq("l")][["pidp", "ghq12_raw"]].rename(
        columns={"ghq12_raw": "future_ghq12_raw"}
    )
    work = origin.merge(future, on="pidp", how="inner")
    controls = [
        "origin_ghq12_raw",
        "origin_care_mod",
        "origin_care_high",
        "origin_childcare_yes",
        "origin_age",
        "origin_log_income",
        "origin_partnered",
        "origin_employed",
    ]
    rows = []
    for domain, label, measure in config.MEASURES:
        model_data = work[["pidp", "future_ghq12_raw", measure, *controls]].dropna()
        result = cluster_ols(
            model_data["future_ghq12_raw"],
            model_data[[measure, *controls]],
            model_data["pidp"],
        )
        row = tidy_target(
            result,
            {measure: 1.0},
            model="wave1_to_wave12_baseline_prospective_ols",
            outcome="wave12_ghq12_raw",
            cohesion=label,
            domain=domain,
            rows=len(model_data),
            people=model_data["pidp"].nunique(),
        )
        row["lag_pairs"] = "1->12"
        rows.append(row)
    return pd.DataFrame(rows)


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
            f"{row['estimate']:.{digits}f} ({row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f})",
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
    draw_temporal_panel(
        axes[0],
        forward,
        title="Previous perceived neighbourhood\ncohesion -> current GHQ-12",
        xlim=(-0.75, 0.75),
        digits=2,
        show_labels=True,
    )
    draw_temporal_panel(
        axes[1],
        five_year,
        title="Cohesion about 5 years earlier\n-> current GHQ-12",
        xlim=(-1.20, 0.55),
        digits=2,
        show_labels=False,
    )
    draw_temporal_panel(
        axes[2],
        reverse,
        title="Previous GHQ-12 -> current perceived\nneighbourhood cohesion",
        xlim=(-0.010, 0.015),
        digits=3,
        show_labels=False,
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
