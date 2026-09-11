# Reproduction code

This repository has been prepared for anonymous peer review. It contains no
author names, affiliations, contact details, respondent-level data, or local
absolute paths.

This directory contains the code used to reproduce the tables and figures for
the study of perceived neighbourhood cohesion and psychological distress among
female informal carers in the UK Household Longitudinal Study (UKHLS).

UKHLS respondent records are distributed under licence and are not included.
The scripts expect authorised users to obtain the data from the UK Data Service.
No respondent-level data, identifiers, or protected geography should be
committed to this repository.

## Required restricted files

Set the environment variable `UKHLS_DATA_DIR` to a local directory containing:

```text
a_indresp.dta
c_indresp.dta
f_indresp.dta
i_indresp.dta
l_indresp.dta
```

If the environment variable is not set, the scripts look in
`survey_responses/` at the repository root. The household response files are
not required by this final workflow.

## Software

Python 3.12 or newer is recommended. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run in manuscript order

```powershell
.\.venv\Scripts\python.exe run_all.py
```

The numbered scripts can also be run separately:

1. `01_prepare_analysis_panel.py` constructs the restricted analysis panel.
2. `02_figure1_between_person_associations.py` fits person-mean OLS models and
   creates Figure 1.
3. `03_table1_within_person_variation.py` creates Table 1.
4. `04_figure2_within_person_associations.py` fits individual fixed-effects
   models and creates Figure 2.
5. `05_figure3_context_moderation.py` fits the fixed-effects interaction models
   and creates Figure 3.
6. `06_figure4_sensitivity_analyses.py` repeats the between-person and
   within-person sensitivity analyses and creates Figure 4.
7. `07_figure5_temporal_order_checks.py` runs the forward, approximately
   five-year, and reverse-lagged checks and creates Figure 5.
8. `08_supplementary_analyses.py` creates the remaining supplementary model
   summaries.

All generated files are written beneath `outputs/journal_reproduction/`, which
is excluded from version control. The private analysis panel is stored in its
`restricted/` subdirectory. Figure-source CSV files contain aggregate model
results only.

## Interpretation

The fixed-effects models estimate contemporaneous within-person associations.
The lagged models are checks on temporal ordering and do not establish a causal
effect or eliminate residual time-varying confounding.


## macOS and Linux

The equivalent commands are:

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_all.py
~~~
