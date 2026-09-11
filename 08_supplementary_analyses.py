from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_core import (
    cluster_ols,
    fit_within_measure,
    load_panel,
    save_model_summary,
    tidy_target,
    zscore,
)

import config
from temporal_models import stacked_long_lag_models


def cronbach_alpha(frame: pd.DataFrame, columns: list[str]) -> float:
    complete = frame[columns].dropna()
    k = len(columns)
    total_variance = complete.sum(axis=1).var(ddof=1)
    if len(complete) < 2 or k < 2 or total_variance == 0:
        return np.nan
    return float(k / (k - 1) * (1 - complete.var(ddof=1).sum() / total_variance))


def reliability_summary(panel: pd.DataFrame) -> pd.DataFrame:
    specifications = [
        (
            "total_buckner",
            "Total cohesion",
            [f"{item}_pos" for item in config.COHESION_ITEMS],
            "nbr_1to5",
        ),
        (
            "psychological_sense_community",
            "Psychological sense of community",
            config.PSYCHOLOGICAL_SENSE_ITEMS,
            "cohesion_psychological_sense_community",
        ),
        (
            "neighbouring",
            "Neighbouring",
            config.NEIGHBOURING_ITEMS,
            "cohesion_neighbouring",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "domain": domain,
                "cohesion": label,
                "items": len(items),
                "person_wave_observations": int(panel[score].notna().sum()),
                "individuals": int(panel.loc[panel[score].notna(), "pidp"].nunique()),
                "mean": float(panel[score].mean()),
                "sd": float(panel[score].std(ddof=1)),
                "cronbach_alpha": cronbach_alpha(panel, items),
            }
            for domain, label, items, score in specifications
        ]
    )


def standardised_within_models(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    rows = []
    for domain, label, measure in config.MEASURES:
        standardised = f"{measure}_z"
        work[standardised] = zscore(work[measure])
        row = fit_within_measure(
            work,
            domain=domain,
            label=label,
            measure=standardised,
            model="supplementary_standardised_fe_age",
        )
        row["scale"] = "standard deviation"
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_prospective_models(panel: pd.DataFrame) -> pd.DataFrame:
    """Relate wave 1 cohesion to wave 12 GHQ-12, adjusting for wave 1 values."""
    baseline = panel[panel["wave"].eq("a")].copy()
    follow_up = panel.loc[
        panel["wave"].eq("l"), ["pidp", "ghq12_raw"]
    ].rename(columns={"ghq12_raw": "follow_up_ghq12"})

    work = baseline.merge(follow_up, on="pidp", how="inner")
    controls = [
        "ghq12_raw",
        "care_mod",
        "care_high",
        "childcare_yes",
        "age",
        "log_income",
        "partnered",
        "employed",
    ]
    rows = []

    for domain, label, measure in config.MEASURES:
        model_data = work[["pidp", "follow_up_ghq12", measure, *controls]].dropna()
        result = cluster_ols(
            model_data["follow_up_ghq12"],
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


def main() -> None:
    config.make_output_directories()
    panel = load_panel()
    reliability = reliability_summary(panel)
    standardised = standardised_within_models(panel)
    longer_lag = stacked_long_lag_models(
        panel,
        pairs=[("a", "i"), ("c", "l")],
        model_name="approximately_eight_year_lag_fe",
    )
    baseline = baseline_prospective_models(panel)

    reliability.to_csv(config.TABLE_DIR / "supplementary_reliability.csv", index=False)
    save_model_summary(standardised, "supplementary_standardised_within_models.csv")
    save_model_summary(longer_lag, "supplementary_longer_lag_fe_models.csv")
    save_model_summary(baseline, "supplementary_wave1_to_wave12_prospective_ols.csv")


if __name__ == "__main__":
    main()
