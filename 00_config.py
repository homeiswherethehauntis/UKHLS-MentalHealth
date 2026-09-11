from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("UKHLS_DATA_DIR", ROOT / "survey_responses"))
OUTPUT_DIR = Path(
    os.environ.get("UKHLS_OUTPUT_DIR", ROOT / "outputs" / "journal_reproduction")
)
RESTRICTED_DIR = OUTPUT_DIR / "restricted"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
SOURCE_DIR = OUTPUT_DIR / "figure_sources"
MODEL_DIR = OUTPUT_DIR / "model_summaries"

PANEL_PATH = RESTRICTED_DIR / "analysis_panel.csv"

TARGET_WAVES = ["a", "c", "f", "i", "l"]
WAVE_ORDER = {"a": 1, "c": 3, "f": 6, "i": 9, "l": 12}
MISSING_CODES = [-1, -2, -7, -8, -9, -10, -11, -20, -21]

MEASURES = [
    ("total_buckner", "Total cohesion", "nbr_1to5"),
    (
        "psychological_sense_community",
        "Psychological sense of community",
        "cohesion_psychological_sense_community",
    ),
    ("neighbouring", "Neighbouring", "cohesion_neighbouring"),
]

COHESION_ITEMS = [
    "scopngbha",
    "scopngbhb",
    "scopngbhc",
    "scopngbhd",
    "scopngbhe",
    "scopngbhf",
    "scopngbhg",
    "scopngbhh",
]

PSYCHOLOGICAL_SENSE_ITEMS = [
    "scopngbha_pos",
    "scopngbhb_pos",
    "scopngbhe_pos",
    "scopngbhg_pos",
]
NEIGHBOURING_ITEMS = [
    "scopngbhc_pos",
    "scopngbhd_pos",
    "scopngbhh_pos",
]


def make_output_directories() -> None:
    for directory in [
        RESTRICTED_DIR,
        TABLE_DIR,
        FIGURE_DIR,
        SOURCE_DIR,
        MODEL_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def check_restricted_inputs() -> None:
    missing = [DATA_DIR / f"{wave}_indresp.dta" for wave in TARGET_WAVES]
    missing = [path for path in missing if not path.exists()]
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required licensed UKHLS files were not found. Set UKHLS_DATA_DIR "
            f"or add the following files locally:\n{paths}"
        )
