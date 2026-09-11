from __future__ import annotations

import pandas as pd

import config
from analysis_core import (
    add_wave_dummies,
    fit_within,
    load_panel,
    primary_controls,
    tidy_target,
)

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
    cohesion_columns = [measure for _, _, measure in config.MEASURES]
    origin_controls = ["ghq12_raw", *primary_controls()]
    origin_columns = ["pidp", "wave", *cohesion_columns, *origin_controls]
    origins = panel[origin_columns].rename(
        columns={
            "wave": "origin_wave",
            **{name: f"origin_{name}" for name in origin_controls},
        }
    )
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
        *(f"origin_{name}" for name in origin_controls),
        *pair_dummies.columns,
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
