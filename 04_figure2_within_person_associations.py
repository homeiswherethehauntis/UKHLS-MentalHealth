from __future__ import annotations

import pandas as pd

from analysis_core import fit_within_measure, load_panel, save_model_summary
from plotting_core import forest_three
import config


def estimate_within_person_associations(panel: pd.DataFrame) -> pd.DataFrame:
    rows = [
        fit_within_measure(
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
    results = estimate_within_person_associations(load_panel())
    save_model_summary(results, "figure2_within_person_associations.csv")
    results.to_csv(
        config.SOURCE_DIR / "figure2_within_person_associations.csv", index=False
    )
    forest_three(
        results,
        xlabel=(
            "Difference in GHQ-12 when cohesion is 1 point above the "
            "individual's own mean (1-5 scale)"
        ),
        stem="figure2_within_person_associations",
        xlim=(-1.45, 0.20),
    )


if __name__ == "__main__":
    main()
