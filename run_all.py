from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
STEPS = [
    "01_prepare_analysis_panel.py",
    "02_figure1_between_person_associations.py",
    "03_table1_within_person_variation.py",
    "04_figure2_within_person_associations.py",
    "05_figure3_context_moderation.py",
    "06_figure4_sensitivity_analyses.py",
    "07_figure5_temporal_order_checks.py",
]


def main() -> None:
    for filename in STEPS:
        print(f"\n==> {filename}", flush=True)
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / filename)],
            cwd=SCRIPT_DIR.parent,
            check=True,
        )


if __name__ == "__main__":
    main()
