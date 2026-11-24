# QSS revision lock (created 2026-11-24)

Record of the exact state behind the paper submitted to *Quantitative Science Studies* on 2026-06-12,
so the referee revision does not drift onto newer OpenAlex data or library versions.

## Code and paper
- Git tag `qss-submitted` → commit `75e3f67` ("final version before submission").
- Revision work happens on branch `qss-revision`. Do not work on `main` for the revision.
- `origin/main` has two later commits (`399ef1e`, `065ed1a`, 2026-06-22) from the other machine. They regenerate
  `data/source_master.csv`, `comprehensive_journal_list.csv` and several plot CSVs with small differences,
  and revert `tables/table_kernel_structure.tex` to an older format. They are **not** the submitted state and
  are not on `qss-revision`.

## Working data (`WORKING` in config.yaml)
- Live: `/home/lc/m/working/econ_bus` (22 GB, 3,907 files; nothing modified after 2026-06-12 at lock time).
- Frozen reference copy: `/home/lc/m/working/econ_bus_qss_submitted`. Do not write to it.
- `working_econ_bus.sha256`: sha256 of every file at lock time. Check either folder with
  `cd <folder> && sha256sum --quiet -c /home/lc/Projects/EconomicsBusiness/qss_lock/working_econ_bus.sha256`
  (silent = identical).

## OpenAlex snapshot (`OPENALEX` in config.yaml)
- The paper used the Feb-2026 snapshot, converted with OpenAlex_ETL to numeric-id parquet (re-run commit
  `6ed9621`, 2026-03-15; source_idx added in `2cd4e24`, 2026-04-02). That converted copy
  (`/home/lc/m/openalex_feb26/parquet`) **no longer exists**.
- Surviving copies:
  - raw gz: `/media/s-drive/openalex_feb26/data` and the 5 TB USB backup `DataDisk_NTFS_5Tb/openalex_feb26/data`
  - parquet with text ids (pre-idx, 2026-03-05): `/media/s-drive/openalex_feb26/parquet`
- `OPENALEX` is deliberately left pointing at the missing path, so any re-extraction fails loudly instead of
  silently using `openalex_jul26` / `openalex_oct26`.
- Only `prepare_data/journal_filter_match_oa.py`, `load_corpus_entities.py` and `exogenous_source_count.py`
  read OpenAlex. Everything downstream reads the frozen corpus parquets in `WORKING/parquet` (2026-05-17).
- If a referee request needs re-extraction: rebuild an idx copy from the s-drive pre-idx parquet, regenerate the
  corpus into a scratch folder, and diff against `econ_bus_qss_submitted/parquet` before using it.

## Python environment
- `requirements-qss.txt`: full `pip freeze` of `.venv` (Python 3.12.3). The venv was last changed on 2026-06-09
  (pytest only), so this is the environment that produced the submitted results.
- Note: the repo's `requirements.txt` pins *different* versions (e.g. duckdb 1.5.0, numpy 2.2.6, scipy 1.16.0)
  than this venv (duckdb 1.3.2, numpy 1.26.4, scipy 1.13.1). Use `requirements-qss.txt` for the revision on
  both machines.

## Config
- `config.qss-submitted.yaml`: copy of this machine's `config.yaml` at lock time.

## Corrected baseline for the revision (added 2026-11-24)
- **Location:** `/home/lc/m/working/econ_bus_rev`, selected with `ECONBUS_CONFIG=config.rev.yaml`.
- **What it is:** the frozen corpus parquets plus explicit correction tables. No OpenAlex access.
  - `data/institution_corrections.csv` contains 35 merges, 6 drops and 58 includes. The includes are units passing τ that were removed by the OpenAlex institution-type filter; the rule is to keep everything except companies and hospitals. The three former `INSTITUTION_EXCLUSIONS` drops became merges.
  - `data/source_corrections.csv` contains 2 journal merges: Eurasian Economic Review, and Review of Finance (filed by OpenAlex under the old-name record "European Finance Review").
- **Applied by:** `prepare_data/apply_institution_corrections.py --source /home/lc/m/working/econ_bus/parquet`, then build_edge_lists → build_institution_field_eb → filter_mode_units → run_rankings.
- **Reproduction check:** with `--empty`, the rerun reproduces all 36 locked ranking tables to ≤ 6e-11 dex (`spectral_ranking/compare_rankings_db.py`). The empty-run rankings are kept in `econ_bus_rev/empty_check/`.
- **Other folders:**
  - `econ_bus_rev_fin` is the finance-as-business variant (referee 3.1);
  - `revision.duckdb` holds the split-institution test (referee 1).
- **Locked data:** the checksums of the locked `rankings.duckdb`, `edge_lists.duckdb` and `econ.duckdb` were re-verified after the submitted figures were reproduced from them.
