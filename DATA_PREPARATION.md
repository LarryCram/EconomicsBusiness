# Data Preparation

# Preparation
- Re-read CLAUDE.md
- Re-read main.pdf

# Aim
- outcome is a duckdb database in new path working/econ_bus/duckdb of edge lists that can be read to build CSR and determine spectral rankings

# Parameters
- make a way to comprehensively manage all the parameters 
- one way to manage parameters is to persist a duckdb table of edge list for each combination labelled uniquely.  
- do not assume that all permutations of parameters are run. drive the parameter selection from hard-coded selection table such as a YAML.
- Note that it will take seconds to build a parameter set and its spectral ranking provided we pre-select all the possible works by journal and overall publication_year span filtering out paratext and retractions.

# Review
- earlier we coded what is in prepare_data
- review this code and extend as required by revisions in main.pdf

# census and target windows
- define target as a window
- define 5 windows in which t^c = t^t - 2000-04/2005-09/2010-14/2015-19/2020-24. We take all the works in these census windows and keep all references to works in these target windows. 
- define 2 windows as t^c = 2024 only, t^t = 2000-2024 and t^c = 2020-24 and t^t=2020 only. These are analogues to 5-year JIF and reference spectroscopy respectively.
- introduce a parameter to select the 7 cases - like t_x.

# Economics Business 
- current data extraction constructs a single final journal table.
- add a flag "E" or "B" or NULL to the table to indicate whether the only openalex field name is "E": 'Economics, Econometrics and Finance' or only "B": 'Business, Management and Accounting' or NULL if both have some counts.
- introduce a new parameter F (i.e. field) for this filter.

# Institution filtering
- check how the \tau_U institutional filter works with reduced time windows. 
- the form of the frequency distribution may not change
- draw the institution retention plot for all 7 time windows
- do not treat the calculation of \tau_u as a part of the pipeline - do it seprately from the 2020-2024 works/authorship parquets

# alpha
- \alpha is a parameter of the spectral ranker and do not need to be present in the edge lists
- \m and \chi are factors in the projected matrix and can be included after the raw edge list blocks are constructed.

# Edge list or CSR
- Advise on whether it will be better to persist edge lists or CSRs to duckdb.
- Note that the citation matrix layout will be C_ij is references from i to j and that the iterated matrix will be row stochastic

# Continue to plan
- once you have done this, provide extensive comments and suggestions on the completeness of the instructions and the way you propose to approach the tasks

