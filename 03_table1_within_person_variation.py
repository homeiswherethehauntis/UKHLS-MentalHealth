from __future__ import annotations

import pandas as pd

from analysis_core import load_panel
import config


def summarise_within_person_variation(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, label, measure in config.MEASURES:
        work = panel[["pidp", measure]].dropna()
        person = (
            work.groupby("pidp")[measure]
            .agg(n_observations="size", mean="mean", sd="std", minimum="min", maximum="max")
            .reset_index()
        )
        repeated = person[person["n_observations"].ge(2)].copy()
        repeated["range"] = repeated["maximum"] - repeated["minimum"]
        person_mean = work.groupby("pidp")[measure].transform("mean")
        within_deviation = work[measure] - person_mean
        rows.append(
            {
                "domain": domain,
                "Cohesion measure": label,
                "Individuals with repeated measures": int(len(repeated)),
                "Within-person SD": float(within_deviation.std(ddof=1)),
                "Median within-person range": float(repeated["range"].median()),
                "Any within-person change, %": float(repeated["range"].gt(0).mean() * 100),
                "Range >=0.5, %": float(repeated["range"].ge(0.5).mean() * 100),
                "Range >=1.0, %": float(repeated["range"].ge(1.0).mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(table: pd.DataFrame) -> str:
    display = table.drop(columns="domain").copy()
    for column in ["Within-person SD"]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    for column in ["Median within-person range"]:
        display[column] = display[column].map(lambda value: f"{value:.2f}")
    for column in [
        "Any within-person change, %",
        "Range >=0.5, %",
        "Range >=1.0, %",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.1f}")
    lines = [
        "Table 1. Within-person variation in perceived neighbourhood cohesion on the 1-5 scale.",
        "",
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row[column]) for column in display.columns) + " |"
        for _, row in display.iterrows()
    )
    return "\n".join(lines)


def main() -> None:
    config.make_output_directories()
    table = summarise_within_person_variation(load_panel())
    table.drop(columns="domain").to_csv(
        config.TABLE_DIR / "table1_within_person_variation.csv", index=False
    )
    (config.TABLE_DIR / "table1_within_person_variation.md").write_text(
        markdown_table(table), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
