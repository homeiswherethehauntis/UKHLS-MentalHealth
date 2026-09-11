# Reproduction code

This repository contains the code used to reproduce the tables and figures for
the study titled Perceived neighbourhood cohesion and psychological distress among female informal carers: a longitudinal study from the UK.

## Data access

UKHLS respondent records are distributed under special licence and are not included.
Authorised users must obtain the following files from the UK Data Service:

~~~text
a_indresp.dta
c_indresp.dta
f_indresp.dta
i_indresp.dta
l_indresp.dta
~~~

Place the files in `survey_responses/`, or set `UKHLS_DATA_DIR` to their local
directory.

## Software

Install the packages listed in `requirements.txt`.

## Analysis steps

The numbered scripts follow the order of the manuscript:

1. `01_prepare_analysis_panel.py` constructs the restricted analysis panel.
2. `02_figure1_between_person_associations.py` creates Figure 1 from OLS models.
3. `03_table1_within_person_variation.py` creates Table 1.
4. `04_figure2_within_person_associations.py` creates Figure 2 from individual fixed-effects models.
5. `05_figure3_context_moderation.py` creates Figure 3 from fixed-effects interaction models.
6. `06_figure4_sensitivity_analyses.py` creates Figure 4 from the between-person and within-person sensitivity analyses.
7. `07_figure5_temporal_order_checks.py` creates Figure 5 from the forward, approximately five-year, and reverse-lagged checks.

`run_all.py` runs these scripts in sequence. Shared data and model functions are in `analysis_core.py`; shared temporal models are in `temporal_models.py`;
figure formatting is in `plotting_core.py`; and paths and variable lists are in `config.py`
