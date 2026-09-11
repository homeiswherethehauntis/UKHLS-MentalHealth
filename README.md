# Reproduction code

This repository contains the code used to reproduce the tables and figures for
the study of perceived neighbourhood cohesion and psychological distress among
female informal carers in the UK Household Longitudinal Study (UKHLS).

It has been prepared for anonymous peer review. It contains no author names,
affiliations, contact details, respondent-level data, or local absolute paths.

## Data access

UKHLS respondent records are distributed under licence and are not included.
Authorised users must obtain the following files from the UK Data Service:

~~~text
a_indresp.dta
c_indresp.dta
f_indresp.dta
i_indresp.dta
l_indresp.dta
~~~

Place the files in `survey_responses/`, or set `UKHLS_DATA_DIR` to their local
directory. No household response files are required.

## Software

Python 3.12 or newer is recommended. Install the packages listed in
`requirements.txt`.

Windows PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_all.py
~~~

macOS or Linux:

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_all.py
~~~

## Analysis order

The numbered scripts follow the order of the manuscript:

1. `01_prepare_analysis_panel.py` constructs the restricted analysis panel.
2. `02_figure1_between_person_associations.py` creates Figure 1 from
   person-mean OLS models.
3. `03_table1_within_person_variation.py` creates Table 1.
4. `04_figure2_within_person_associations.py` creates Figure 2 from individual
   fixed-effects models.
5. `05_figure3_context_moderation.py` creates Figure 3 from fixed-effects
   interaction models.
6. `06_figure4_sensitivity_analyses.py` creates Figure 4 from the between-person
   and within-person sensitivity analyses.
7. `07_figure5_temporal_order_checks.py` creates Figure 5 from the forward,
   approximately five-year, and reverse-lagged checks.
8. `08_supplementary_analyses.py` creates the supplementary model summaries.

`run_all.py` runs these scripts in sequence. Shared data and model functions are
in `analysis_core.py`; shared temporal models are in `temporal_models.py`;
figure formatting is in `plotting_core.py`; and paths and variable lists are in
`config.py`.

OLS models are fitted with `statsmodels`. Fixed-effects models are implemented
by subtracting each respondent's own mean from the outcome and model variables.
Standard errors are clustered by respondent.

## Outputs and interpretation

Generated files are written to `outputs/journal_reproduction/`, which is
excluded from version control. The derived respondent-level panel is stored in
the ignored `restricted/` subdirectory. No generated data are committed.

The fixed-effects models estimate contemporaneous within-person associations.
The lagged analyses check temporal ordering; they do not establish causality or
remove residual time-varying confounding.
