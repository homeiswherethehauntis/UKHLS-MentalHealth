from __future__ import annotations

import json
from analysis_core import prepare_analysis_panel

import config


def main() -> None:
    config.make_output_directories()
    panel = prepare_analysis_panel()
    panel.to_csv(config.PANEL_PATH, index=False)

    manifest = {
        "licensed_input_directory": "Set locally using UKHLS_DATA_DIR",
        "waves": config.TARGET_WAVES,
        "wave_numbers": [config.WAVE_ORDER[wave] for wave in config.TARGET_WAVES],
        "person_wave_observations": int(len(panel)),
        "individuals": int(panel["pidp"].nunique()),
        "restricted_panel": str(config.PANEL_PATH.relative_to(config.ROOT)),
        "restricted_data_included_in_repository": False,
    }
    (config.OUTPUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        "Prepared the restricted analysis panel: "
        f"{manifest['individuals']:,} individuals and "
        f"{manifest['person_wave_observations']:,} person-wave observations."
    )


if __name__ == "__main__":
    main()
