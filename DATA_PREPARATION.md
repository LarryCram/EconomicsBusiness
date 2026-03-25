# Data Preparation

# Preparation
- Re-read CLAUDE.md
- Re-read main.pdf

# Aim
- Two outcomes:
- First, a revision of 03_processing.tex to be more precise about what we are doing
- Second, a DuckDB database at WORKING/econ_bus/duckdb/ containing edge lists that can be read to build CSR matrices and determine spectral rankings

# Parameters
- Make a way to comprehensively manage all parameters
- There are corpus-construction parameters (time window, field subset, tau_U) and ranking parameters (alpha, delta, m, chi)
- Persist a set of DuckDB edge list tables for each corpus-construction parameter combination, labelled uniquely
- Do not assume that all permutations of parameters are run. Drive parameter selection from a hard-coded selection table in YAML
- Note that it will take seconds to build a parameter set and its spectral ranking provided we pre-select all the possible works by journal and overall publication_year span, filtering out paratext and retractions
- The ranking parameters (alpha, delta, m, chi) are applied at read/ranking time from the stored edge lists and do not require separate edge list files

# Review
- Earlier we coded what is in prepare_data.py (now load_corpus_entities.py)
- Review this code and extend as required by revisions in main.pdf

# Census and target windows as parameters
- Define 5 symmetric windows where t^c = t^t: 2000-04 / 2005-09 / 2010-14 / 2015-19 / 2020-24.
  Take all the works in the census window and keep all references pointing to works in the same target window.
- Define 2 asymmetric windows:
  - Case 6: t^c = 2024 only (publication_year = 2024 exactly), t^t = 2000-2024. Analogue of a 5-year JIF.
  - Case 7: t^c = 2020-24, t^t = 2020 only (publication_year = 2020 exactly). Analogue of reference spectroscopy — traces where recent papers send their citations back to.
- Introduce a parameter t_x in {1,...,7} to select the case
- The works_per_year denominator used to compute mean annual institutional output must equal the length of the census window in years (e.g. 5 for the 5-year cases, 1 for case 6, 5 for case 7), not a global constant

# Economics Business as parameters
- Current data extraction constructs a single final journal table (source_master.parquet)
- Add a flag "E", "B", or "A" to source_master to indicate whether the OpenAlex topic counts are:
  - "E": only in Field 14 'Economics, Econometrics and Finance'
  - "B": only in Field 20 'Business, Management and Accounting'
  - "A": both fields have some counts
- Introduce a parameter F in {E, B, A} to select the journal subset for a given run
- F=A uses all sources in source_master; F=E or F=B filters to the respective flag

# Institution filtering computed first
- There is a tau_U institutional filter whose appropriate value may depend on the work set (time window, field subset)
- The expectation is that the shape of the frequency distribution will not change much across parameter cases due to the very long tail of institutions with few publications; this remains to be verified
- Make two institution retention diagnostic tables covering all 7 time windows and 3 field subsets (21 cases)
- Both tables have tau_U threshold (2, 4, 6, 8, 10 mean works/institution/year) as column labels and the 21 parameter cases as row labels
- For each cell, the retention baseline is the total works in that specific (t_x, F) case (not the full 2000-2024 corpus)
- In one table, the entry is the number of institutions retained at the tau_U threshold where 80% of that case's works are retained
- In the other table, the same but for 70% of that case's works retained
- Do not treat the calculation of tau_U as part of the pipeline; this is a standalone diagnostic script
- We will choose one value of tau_U (possibly two if the results suggest it) for the entire project after reviewing these tables. We do not explore a cross-section of tau_U in the rankings.
- Since tau_U is fixed before building edge lists, the edge lists for a given (t_x, F) combination are unique and do not need a tau_U dimension

# alpha
- alpha is a parameter of the spectral ranker and does not need to be present in the edge lists
- m and chi are factors in the assembled citation matrix and are applied after the raw edge list blocks are read

# Edge lists (not CSR)
- Persist edge lists to DuckDB, not CSR matrices
- Store the four raw block edge lists (SS, SI, IS, II) separately or with a block label column for each corpus-construction parameter combination
- Schema: (citer_unit_idx, cited_unit_idx, block, raw_weight) where raw_weight accumulates delta_i * b^X_i * b^Y_j over references
- The chi scaling and m masking are applied at read time when assembling C(chi, m, delta)
- delta (full vs fixed reference count) can also be applied at read time since it requires R_bar which is computed from the edge list itself
- The citation matrix convention is C_ij = attention from i (citing, row) to j (cited, column); the iterated matrix H = D_r^{-1} C is row-stochastic
- Each (t_x, F) combination with a fixed tau_U produces one set of four block edge lists; the total number of edge list sets is 7 * 3 = 21
- Unit index tables (mapping source_idx and institution_idx to contiguous unit indices 1..N_s and N_s+1..N) must be persisted alongside the edge lists for each case

# Continue to plan
- Once you have done this, provide extensive comments and suggestions on the completeness of the instructions and the way you propose to approach the tasks
