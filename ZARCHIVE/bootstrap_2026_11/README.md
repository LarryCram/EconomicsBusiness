Archived 2026-11-26 (QSS revision).

- bootstrap_baseline.py (+ test): old 80%-subsample bootstrap, superseded by the
  `resample` filter of spectral_ranking_bootstrap/bootstrap_oa_errors.py.
  Its helpers moved: _bipartite_core → bootstrap_oa_errors.py;
  compute_r_bar, create_tmp_el → spectral_ranking_works/el_helpers.py.
- bootstrap_deviation_stats.py, diagnose_bootstrap_outliers.py, fig_7.py: built on
  the combined ("all") error run, which was removed as conceptually unjustified
  cross-talk between error types. None produced a published figure.
