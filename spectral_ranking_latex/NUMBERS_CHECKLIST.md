# In-text numbers: paper vs submitted data vs corrected baseline

Regenerated 2026-11-24.

- **Corrected baseline:** `econ_bus_rev` (identity corrections, restored type-excluded institutions, journal merges), run with `ECONBUS_CONFIG=config.rev.yaml`.
- **"Submitted data":** the same scripts run on the locked `econ_bus` folder. Figures and tables from that run are in `plots/qss_submitted/` and `tables/qss_submitted/`; the locked data was not written.
- **Bootstrap and metadata-error figures** (`fig_boot_envelope_resample`, `fig_boot_ratio_facet`): **regenerated 2026-11-26** from the error studies re-run on the corrected baseline with ω = 0 replicates (`run_error_studies.sh`). Submitted versions are kept in `plots/qss_submitted/`. Benchmark summaries (`data/benchmark_*_summary.csv`) use the new resample bands.

**Applied 2026-11-24** to `04_results.tex` and `05_discussion.tex`: every *update* and *stale* item, plus "OAJ" → "OAS" and "anticipted" → "anticipated". The link-density table was applied later the same day using the `paper_numbers.py` definition. The submitted values could not be reproduced:
- the stated definition gives a mean error of 0.23 dex or more;
- defining Rest as *v* < 1 journals does not reproduce them either;
- neither does the older τ<sub>I</sub>/τ<sub>S</sub> = 10/5 corpus, rebuilt for the test;
- neither do binary links, ρ = 1, or other choices of top-ten.

The new ratios are 74× and 70×, and Top-B → Top-E is 5.1×. Not applied: the survivorship addition. Ritzberger now reads one journal not in the corpus (Games and Economic Behavior, absent from the corpus), J. Mathematical Economics with *v* < 1, and one ambiguous rather than four.

**Status codes**

| Code | Meaning |
|---|---|
| update | Number changes with the corrected baseline. |
| stale | The paper value already disagreed with the submitted data. |
| ok | Unchanged. |
| check | Needs a decision. |

**Sources** (script → output):
- `fig_3.py` → log;
- `table_kernel_structure.py` → `tables/table_kernel_structure.tex`;
- `fig_stability.py` → log;
- `fig_5.py` → `plots/fig_5_stability.csv`;
- `analyse_enclaves.py` → `tables/table_enclaves.tex` plus log;
- `paper_numbers.py` (clique, link density, top-20, ABDC, Ritzberger);
- `fig_6.py` → log.

## §4 Results: opening and Fig. 1 (`04_results.tex` l.10)

| Quantity | Paper | Submitted data | Corrected | Status |
|---|---|---|---|---|
| Fraction of sources with *v* ≥ 1 (rank position of *v* = 1) | ≈15% | 19.2% | 19.4% | stale → "≈19%" |
| Fraction of institutions with *v* ≥ 1 | ≈25% | 27.4% | 26.5% | stale → "≈27%" |

## Kernel structure table (l.53–60) and text (l.65)

| Quantity | Paper | Submitted | Corrected | Status |
|---|---|---|---|---|
| C<sub>II</sub> fan-out *K̄* | 256 | 256 | 255 | update |
| C<sub>IS</sub> *J*<sub>col</sub> / *K̄* | 0.84 / 97 | 0.84 / 97 | 0.83 / 95 | update |
| M<sub>I</sub> *J̄* / *K̄* | 0.91 / 960 | 0.91 / 960 | 0.90 / 951 | update |
| Fan-out ratio C<sub>II</sub>/C<sub>SS</sub> | 7.9× | 7.9× | 7.88× | ok |
| Text: "*J*<sub>col</sub>: 0.83 c.f. 0.90" for M<sub>S</sub> | 0.83 | table 0.84 | 0.84 | stale → 0.84 |
| Text: M<sub>I</sub> "*J̄* = 0.90" | 0.90 | table 0.91 | 0.90 | ok after update |

All other cells are unchanged.

## §4.1 Business and Economics

| Quantity | Paper | Submitted | Corrected | Status |
|---|---|---|---|---|
| Clique share of π, E+B journal-only | 25.9% | 25.9% | 26.1% | update |
| Clique share of π, E-only journal-only | 38.5% | 38.5% | 38.8% | update |
| Link-density table (`tab:link_density`) and ratios "182×", "161×", "4.4×" | Top-E/Top-E 49.61, Rest-E/Rest-E 0.272, All-B 0.307 … | `paper_numbers.py` gives Top-E/Top-E 52.44, Rest-E/Rest-E 0.713, All-B 0.765 | 53.32, 0.719, 0.767 | **check** |
| Top-20 institutions (E+B, institution-only) designated E | 19 of 20 | 19 | 19 | ok |

**Link-density table: the submitted values cannot be reproduced with the definition stated in the text** (ρ = 0 citation weights, top ten by field-specific journal-only ranking, per-directed-pair average).
- The ratios differ: in the reproduction Top-E is about 74× Rest-E, not 182×.
- The cell "0.4ten" is corrupted.
- The script that made the table is not in the repository.

**Decide:**
- either adopt the `paper_numbers.py` definition and values, with ratios recomputed;
- or find the original definition (possibly binary links or different groups).

## Figures in §4.1

`fig_3eb` and `fig_3ef` are regenerated.
- Separate-vs-combined medians, journal-only: E −0.155, B +0.189 (submitted +0.191).
- Bipartite (right panel of `fig_3ef`): E −0.012 / B +0.018.

## §4.2 Time dependence (l.127–131, Fig. 5 `fig_stability_scatter`)

| Quantity | Paper | Submitted | Corrected | Status |
|---|---|---|---|---|
| Units present in 2000–04 and 2020–24 | 651 sources, 417 institutions | 651, 417 | 651, 411 | update (caption too) |
| *v* > 1 in 2000–04 retaining it in 2020–24 | 86%, 96% | 85.8%, 95.8% | 86.6%, 95.5% | update → "87%, 95%" |
| Share of units with *v* > 1 in each window | ≈15%, ≈25% | sources 16–19%, institutions ≈27% | same | stale → "about one sixth to one fifth of sources and a quarter of institutions" |

Add the survivorship-free counts (referee 3.2) from `data/survivorship_transitions.csv`.

## §4.3 Parameter sensitivity (l.168–179; `plots/fig_5_stability.csv`)

| Quantity | Paper | Submitted | Corrected | Status |
|---|---|---|---|---|
| τ = 40: corpus size | ≈40% S, 57% I | 42%, 57% (663 / 1,125) | 42%, 57% (664 / 1,134) | caption *N* update |
| τ = 40: Spearman | > 0.97 | 0.995, 0.975 | 0.995, 0.974 | ok |
| τ = 40: "lowering by about 0.1 in log *v* (0.06 institutions)" | 0.1 / 0.06 | **median** −0.039 / −0.076; **s.d.** 0.107 / 0.062 | −0.039 / −0.077; 0.106 / 0.062 | **stale**: the paper's figures are the s.d., not the shift |
| τ = 40: median rank shift | 7.5%, 6.4% | 7.5%, 6.4% | 7.4%, 6.6% | update |
| ρ = 1: mean shift sources | −0.07 | −0.075 | −0.075 | ok |
| ρ = 1: s.d. | 0.2 S, 0.1 I | 0.18, 0.11 | 0.18, 0.12 | ok |
| ρ = 1: median rank change | 5% | 5.4%, 5.2% | 5.3%, 5.3% | ok |
| Census 2024 only: s.d. | 0.2, 0.1 | 0.21, 0.10 | 0.21, 0.10 | ok |
| ω: institution s.d. | 0.04 | 0.044 | 0.045 | ok |
| ω: median rank change | < 1.6% | 1.6% | 1.8% | update → "< 2%" |
| ε = 1: "median shift in log *v* 0.3 S, 0.25 I" | 0.3 / 0.25 | **median** 0.037 / 0.155; **s.d.** 0.30 / 0.24 | 0.039 / 0.159; 0.30 / 0.24 | **stale**: the paper's figures are the s.d. |
| ε = 1: rank shift | ≈10% | 8.7%, 11.9% | 8.7%, 12.2% | ok |
| Self-edges removed (α = 0.85): shift for *v* < 1 | ≈ −0.04 | −0.036 S, −0.045 I (no-self vs with-self, μ = 1/N) | −0.034, −0.045 | ok |
| Fig. 5 caption: "baseline (τ<sub>I</sub> = 10, τ<sub>S</sub> = 5 …)" | 10 / 5 | baseline is 20 / 10 | 20 / 10 | **stale** caption |

## §4.4 Errors and uncertainties

Regenerated 2026-11-26 from the re-run (corrected baseline, ω = 0 replicates, each error type separate). The section moves to Supplement 2 (plan §6).

Half-widths are of the 2.5–97.5% interval of (replicate − baseline) fractional rank; "rest" = units below rank 100. S = sources, I = institutions.

| Quantity | Paper | Submitted | Corrected | Status |
|---|---|---|---|---|
| Units in the error studies | — | 1,565 S, 1,957 I | 1,564 S, 1,979 I | — |
| Resample: top ≈100 "narrow" (median half-width) | narrow | 0.006 S, 0.006 I | 0.006 S, 0.006 I | ok |
| Resample: lower ranks "≈±15%" (median / 90th pct half-width, rest) | ≈ ±15% | 0.063/0.134 S, 0.055/0.106 I | 0.062/0.133 S, 0.058/0.112 I | ok |
| Resample: median band half-width in log *v* | — | 0.123 S, 0.065 I | 0.121 S, 0.067 I | ok |
| Errors: "insubstantial among top ≈10%" (95th pct half-width, top 10%; year_v2 / ref / inst / source) | insubstantial | S 0.011/0.003/0.003/0.001, I 0.018/0.008/0.006/0.001 | S 0.011/0.003/0.003/0.001, I 0.015/0.007/0.007/0.001 | ok |
| "1000 replicates" | 1000 | 1000 | 1000 | ok |

**Text changed 2026-11-26** (`04_results.tex`, metadata-error paragraph), because the method changed:
- "Combining a random Stage 1 replicate … with a random Stage 2 replicate …" → each error type is propagated separately (the combined run was removed).
- "To avoid the need to model author uncertainties, we adopt institutional fractionation …" → replicates use the baseline's author-fractional weights, so a zero-error replicate reproduces the baseline exactly.

**Resolved 2026-11-26 — year-uncertainty text.** The paper said 2% of works from 2018–2021 are dropped (2020–21) or inserted (2018–19). The `year_v2` code draws 2% of a 2015–2024 pool, deletes in-window draws and inserts pre-window draws as 2020 works; but the inserted works never appear in the replicate references (0 of ~418k), so the model is in effect a 2% deletion of window works (≈6,000 of 299,810 per replicate). §4.4 text and the Table 3 row now say exactly that, and that misdating into the window is not modelled.

## §4.5 Enclaves (l.263, `tab:enclaves`)

| Quantity | Paper | Submitted | Corrected | Status |
|---|---|---|---|---|
| HCW 2014–2023 | 8,675 | 9,151 | 9,151 | **stale** |
| HCW with ranked institution and source | — | 8,539 | 8,558 | (new) |

Enclave table, *n*<sub>HCW</sub> / works / median *v* by cluster:

| Cluster | Submitted | Corrected |
|---|---|---|
| Sustainability | 666 / 39,654 | 755 / 45,911 |
| SLR | 332 / 17,478 | 311 / 16,658 |
| Digital & China | 304 / 13,140 | 302 / 13,615 |
| COVID | 265 / 14,807 | 219 / 9,687 |
| Resource curse | 160 / 8,744 | 163 / 8,751 |
| Bitcoin | 70 / 4,492 | 63 / 4,138 |

- Update (`tables/table_enclaves.tex` is regenerated; the paper has the table inline).
- Six clusters, with resource curse in two connected components: ok.

## §5 Discussion (`05_discussion.tex` l.7–9)

| Quantity | Paper | Submitted | Corrected | Status |
|---|---|---|---|---|
| ABDC 2025 A\* journals | 219 | 219 | 219 | ok |
| … in corpus | 128 | 135 | 135 | stale |
| … among the top-*N* by *v* | 92 of 128 | 95 of 135 | 98 of 135 | update |
| … with *v* < 1 | 9 | 5 | 5 | stale |
| Ritzberger A+/A journals | 25 | 25 | 25 | ok |
| … not in corpus | 2 | 1 (Games & Econ. Behavior unmatched by name; check) | 1 | check |
| … with *v* > 1 | 22 | 23 | 23 | stale |
| … with *v* < 1 | 1 (math. econometrics) | J. Mathematical Economics | same | ok in substance |
| "OAJ corpus" (referee 2) | OAJ | → "OAS" | | fix |
| Franceschet paragraph: "≈0.2 (s.d. in log), ≈5%" | | ρ / census s.d. 0.18–0.21; rank 5% | same | ok |

## Figures regenerated in `plots/`

`*.pdf` is gitignored. The submitted versions are reproduced in `plots/qss_submitted/`.

- `fig_3by_latex.pdf` (Fig. 1)
- `fig_3eb_latex.pdf` (Fig. 2)
- `fig_3ef_latex.pdf` (Fig. 3)
- `fig_6_latex.pdf` (Fig. 4)
- `fig_stability_scatter_latex.pdf` (Fig. 5)
- `fig_5_latex.pdf` and `fig_5d_latex.pdf` (sensitivity)
- `enclave_referer_v.pdf`
- also `fig_3a/3bz/3c`, `fig_5a–f`
