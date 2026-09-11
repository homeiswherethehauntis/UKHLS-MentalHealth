from __future__ import annotations

from math import erfc, sqrt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import config


def p_from_z(z_value: float) -> float:
    return float(erfc(abs(z_value) / sqrt(2))) if np.isfinite(z_value) else np.nan


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std(ddof=1)
    if pd.isna(sd) or sd == 0:
        return values * np.nan
    return (values - values.mean()) / sd


def _numeric(df: pd.DataFrame, *candidates: str) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _strip_wave_prefix(df: pd.DataFrame, wave: str) -> pd.DataFrame:
    prefix = f"{wave}_"
    return df.rename(
        columns={
            column: str(column)[len(prefix) :]
            for column in df
            if str(column).startswith(prefix)
        }
    )


def _care_intensity(value: float) -> object:
    if pd.isna(value) or int(value) == 97:
        return pd.NA
    code = int(value)
    if code in [1, 2, 3, 8]:
        return "Low (<20h)"
    if code in [4, 5, 6, 9]:
        return "Moderate (20-99h)"
    if code == 7:
        return "Very High (>=100h / continuous)"
    return pd.NA


def _education_group(value: float) -> object:
    if pd.isna(value):
        return pd.NA
    code = int(value)
    if code in [1, 2]:
        return "High"
    if code in [3, 4]:
        return "Medium"
    if code in [5, 9]:
        return "Low"
    return pd.NA


def _ghq_likert(df: pd.DataFrame) -> pd.Series:
    for name in ["scghq1_dv", "scghq01_dv"]:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    candidates = sorted(
        name for name in df.columns if str(name).startswith("scghq") and str(name).endswith("_dv")
    )
    if candidates:
        return pd.to_numeric(df[candidates[0]], errors="coerce")
    items = sorted(
        name
        for name in df.columns
        if len(str(name)) == 6
        and str(name).startswith("scghq")
        and str(name)[-1] in "abcdefghijkl"
    )
    if len(items) != 12:
        raise ValueError("No usable GHQ-12 Likert score or complete 12-item set was found.")
    recoded = df[items].apply(
        lambda column: pd.to_numeric(column, errors="coerce").map(
            lambda value: value - 1 if 1 <= value <= 4 else np.nan
        )
    )
    return recoded.sum(axis=1, min_count=12)


def read_wave(wave: str) -> pd.DataFrame:
    path = config.DATA_DIR / f"{wave}_indresp.dta"
    df = _strip_wave_prefix(pd.read_stata(path, convert_categoricals=False), wave)
    # Column-wise replacement avoids a pandas block-manager failure observed
    # with wide UKHLS Stata files while preserving the original missing-code rule.
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column] = df[column].replace(config.MISSING_CODES, np.nan)

    partner = _numeric(df, "mastat_dv", "marstat_dv", "mastat", "marstat")
    employment = _numeric(df, "jbstat_dv", "jbstat")
    children = _numeric(df, "nchresp", "nchund18resp")
    ghq_case_score = _numeric(df, "scghq2_dv")

    out = pd.DataFrame(
        {
            "pidp": _numeric(df, "pidp").astype("Int64"),
            "wave": wave,
            "wave_order": config.WAVE_ORDER[wave],
            "sex": _numeric(df, "sex_dv", "sex"),
            "age": _numeric(df, "dvage", "age_dv", "age"),
            "care_inside_household": _numeric(df, "aidhh"),
            "care_outside_household": _numeric(df, "aidxhh"),
            "aidhrs": _numeric(df, "aidhrs"),
            "nbr_1to5": _numeric(df, "nbrsnci_dv"),
            "ghq12_raw": _ghq_likert(df),
            "ghq_caseness_score": ghq_case_score,
            "sf12mcs_raw": _numeric(df, "sf12mcs_dv", "mcs12"),
            "sf12pcs_raw": _numeric(df, "sf12pcs_dv", "pcs12"),
            "income_gross_monthly": _numeric(df, "fimngrs_dv"),
            "moved_raw": _numeric(df, "addrmov_dv"),
            "child_resp_count": children,
            "partnered": np.where(
                partner.isin([2, 3, 10]),
                1.0,
                np.where(partner.notna(), 0.0, np.nan),
            ),
            "employed": np.where(
                employment.isin([1, 2]),
                1.0,
                np.where(employment.notna(), 0.0, np.nan),
            ),
            "education_group": _numeric(df, "hiqual_dv").apply(_education_group),
        }
    )
    for item in config.COHESION_ITEMS:
        if item not in df.columns:
            raise ValueError(f"Wave {wave} is missing cohesion item {item}.")
        out[f"{item}_pos"] = 6 - pd.to_numeric(df[item], errors="coerce")
    return out


def prepare_analysis_panel() -> pd.DataFrame:
    config.check_restricted_inputs()
    panel = pd.concat([read_wave(wave) for wave in config.TARGET_WAVES], ignore_index=True)
    panel = panel[
        panel["sex"].eq(2)
        & panel["age"].ge(18)
        & (panel["care_inside_household"].eq(1) | panel["care_outside_household"].eq(1))
    ].copy()

    panel["care_intensity"] = panel["aidhrs"].apply(_care_intensity)
    panel["care_mod"] = panel["care_intensity"].eq("Moderate (20-99h)").astype(float)
    panel["care_high"] = (
        panel["care_intensity"]
        .eq("Very High (>=100h / continuous)")
        .astype(float)
    )
    panel["childcare_yes"] = np.where(
        panel["child_resp_count"].gt(0),
        1.0,
        np.where(panel["child_resp_count"].eq(0), 0.0, np.nan),
    )
    panel["age_sq"] = panel["age"] ** 2
    non_negative_income = panel["income_gross_monthly"].where(
        panel["income_gross_monthly"].ge(0)
    )
    panel["log_income"] = np.log1p(non_negative_income)
    panel["moved_since_last_wave"] = np.where(
        panel["moved_raw"].eq(1), 1.0, np.where(panel["moved_raw"].eq(2), 0.0, np.nan)
    )
    panel["ghq_case_ge4"] = np.where(
        panel["ghq_caseness_score"].ge(4),
        1.0,
        np.where(panel["ghq_caseness_score"].between(0, 12), 0.0, np.nan),
    )
    panel["cohesion_psychological_sense_community"] = panel[
        config.PSYCHOLOGICAL_SENSE_ITEMS
    ].mean(axis=1, skipna=False)
    panel["cohesion_neighbouring"] = panel[config.NEIGHBOURING_ITEMS].mean(
        axis=1, skipna=False
    )

    # Keep the complete-case rule used to define the manuscript sample.
    # Education determines sample eligibility but is not a primary-model coefficient.
    required = [
        "pidp",
        "ghq12_raw",
        "nbr_1to5",
        "age",
        "log_income",
        "partnered",
        "employed",
        "care_intensity",
        "childcare_yes",
        "education_group",
    ]
    panel = panel.dropna(subset=required).copy()
    observation_counts = panel.groupby("pidp").size()
    panel = panel[panel["pidp"].isin(observation_counts[observation_counts.ge(2)].index)]
    return panel.sort_values(["pidp", "wave_order"]).reset_index(drop=True)


def load_panel() -> pd.DataFrame:
    if not config.PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Restricted analysis panel not found: {config.PANEL_PATH}. "
            "Run 01_prepare_analysis_panel.py first."
        )
    return pd.read_csv(config.PANEL_PATH).sort_values(["pidp", "wave_order"])


def add_wave_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dummies = pd.get_dummies(df["wave"], prefix="wave_fe", drop_first=True, dtype=float)
    work = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return work, dummies.columns.tolist()


def primary_controls(wave_columns: list[str] | None = None) -> list[str]:
    controls = [
        "care_mod",
        "care_high",
        "childcare_yes",
        "age",
        "log_income",
        "partnered",
        "employed",
    ]
    return controls + (wave_columns or [])


def cluster_ols(
    y: pd.Series,
    x: pd.DataFrame,
    groups: pd.Series,
    *,
    add_constant: bool = True,
):
    """Fit OLS with standard errors clustered by respondent."""
    design = x.loc[:, x.var(numeric_only=True).gt(0)].astype(float)
    if add_constant:
        design = sm.add_constant(design, has_constant="add")
    return sm.OLS(y.astype(float), design).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": True},
    )


def fit_within(
    df: pd.DataFrame,
    outcome: str,
    design: list[str],
):
    """Fit a person fixed-effects model by demeaning within respondents."""
    work = df.dropna(subset=["pidp", outcome, *design]).copy()
    observations = work.groupby("pidp").size()
    work = work[work["pidp"].isin(observations[observations.ge(2)].index)].copy()

    person = work.groupby("pidp")
    demeaned_outcome = work[outcome] - person[outcome].transform("mean")
    demeaned_design = work[design] - person[design].transform("mean")
    result = cluster_ols(
        demeaned_outcome,
        demeaned_design,
        work["pidp"],
        add_constant=False,
    )
    return result, work


def linear_combination(result, weights: dict[str, float]) -> dict[str, float]:
    names = result.params.index.tolist()
    vector = pd.Series(0.0, index=names)
    missing = [name for name in weights if name not in vector.index]
    if missing:
        return {name: np.nan for name in ["estimate", "std_error", "ci_low", "ci_high", "p_value"]}
    for name, weight in weights.items():
        vector.loc[name] = weight
    estimate = float(vector @ result.params)
    covariance = result.cov_params().loc[names, names].to_numpy()
    variance = float(vector.to_numpy() @ covariance @ vector.to_numpy())
    standard_error = sqrt(max(variance, 0.0))
    z_value = estimate / standard_error if standard_error > 0 else np.nan
    return {
        "estimate": estimate,
        "std_error": standard_error,
        "ci_low": estimate - 1.96 * standard_error,
        "ci_high": estimate + 1.96 * standard_error,
        "p_value": p_from_z(z_value),
    }


def person_mean_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ghq12_raw",
        "ghq_case_ge4",
        "sf12mcs_raw",
        "sf12pcs_raw",
        "nbr_1to5",
        "cohesion_psychological_sense_community",
        "cohesion_neighbouring",
        "age",
        "age_sq",
        "log_income",
        "partnered",
        "employed",
        "care_mod",
        "care_high",
        "childcare_yes",
    ]
    person = df.groupby("pidp")[columns].mean().reset_index()
    moved = (
        df["moved_since_last_wave"].eq(1).groupby(df["pidp"]).max().rename("ever_moved")
    )
    return person.merge(moved.reset_index(), on="pidp", how="left")


def tidy_target(
    result: SimpleNamespace,
    weights: dict[str, float],
    *,
    model: str,
    outcome: str,
    cohesion: str,
    domain: str,
    rows: int,
    people: int,
) -> dict[str, object]:
    return {
        "domain": domain,
        "cohesion": cohesion,
        "model": model,
        "outcome": outcome,
        "scale": "1-5",
        "rows": rows,
        "people": people,
        **linear_combination(result, weights),
    }


def fit_between_measure(
    panel: pd.DataFrame,
    *,
    domain: str,
    label: str,
    measure: str,
    outcome: str = "ghq12_raw",
    controls: list[str] | None = None,
    model: str = "between_primary_person_mean",
) -> dict[str, object]:
    person = person_mean_frame(panel)
    controls = controls or primary_controls()
    work = person[["pidp", outcome, measure, *controls]].dropna()
    result = cluster_ols(work[outcome], work[[measure, *controls]], work["pidp"])
    return tidy_target(
        result,
        {measure: 1.0},
        model=model,
        outcome=outcome,
        cohesion=label,
        domain=domain,
        rows=len(work),
        people=work["pidp"].nunique(),
    )


def fit_within_measure(
    panel: pd.DataFrame,
    *,
    domain: str,
    label: str,
    measure: str,
    outcome: str = "ghq12_raw",
    extra_controls: list[str] | None = None,
    model: str = "primary_fe_1to5_age",
) -> dict[str, object]:
    work, wave_columns = add_wave_dummies(panel)
    design = [measure, *primary_controls(wave_columns), *(extra_controls or [])]
    result, model_data = fit_within(work, outcome, design)
    return tidy_target(
        result,
        {measure: 1.0},
        model=model,
        outcome=outcome,
        cohesion=label,
        domain=domain,
        rows=len(model_data),
        people=model_data["pidp"].nunique(),
    )


def save_model_summary(frame: pd.DataFrame, filename: str) -> None:
    config.make_output_directories()
    frame.to_csv(config.MODEL_DIR / filename, index=False)
