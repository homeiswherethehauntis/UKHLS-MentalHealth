from __future__ import annotations

import pandas as pd

from analysis_core import fit_between_measure, load_panel, save_model_summary
from plotting_core import forest_three
import config


def estimate_between_person_associations(panel: pd.DataFrame) -> pd.DataFrame:
    rows = [
        fit_between_measure(
            panel,
            domain=domain,
            label=label,
            measure=measure,
        )
        for domain, label, measure in config.MEASURES
    ]
    return pd.DataFrame(rows)


def main() -> None:
    config.make_output_directories()
    results = estimate_between_person_associations(load_panel())
    save_model_summary(results, "figure1_between_person_associations.csv")
    results.to_csv(
        config.SOURCE_DIR / "figure1_between_person_associations.csv", index=False
    )
    forest_three(
        results,
        xlabel=(
            "Difference in average GHQ-12 per 1-point higher average "
            "cohesion (1-5 scale)"
        ),
        stem="figure1_between_person_associations",
        xlim=(-2.15, 0.25),
    )


if __name__ == "__main__":
    main()
