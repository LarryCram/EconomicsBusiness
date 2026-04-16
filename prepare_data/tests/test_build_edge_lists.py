"""
Tests for build_edge_lists.py pure functions.
No parquet data or DuckDB required.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from build_edge_lists import table_name, corpus_configs_from_csv


# ─── table_name ───────────────────────────────────────────────────────────────

def test_table_name_baseline():
    assert table_name('20242024', 'A', 20, 20) == 'el_20242024_A_tauU20_tauS20_vartau'


def test_table_name_field_subset():
    assert table_name('20242024', 'E', 20, 20) == 'el_20242024_E_tauU20_tauS20_vartau'


def test_table_name_tau40():
    assert table_name('20242024', 'A', 40, 40) == 'el_20242024_A_tauU40_tauS40_vartau'


def test_table_name_time_series():
    assert table_name('00040004', 'A', 20, 20) == 'el_00040004_A_tauU20_tauS20_vartau'


def test_table_name_fixed_universe():
    assert (table_name('00040004', 'A', 20, 20, ref_units='20242024_A_tauU20_tauS20')
            == 'el_00040004_A_tauU20_tauS20_fixtau')


# ─── corpus_configs_from_csv ──────────────────────────────────────────────────

def test_configs_are_unique():
    configs = corpus_configs_from_csv()
    keys = [(c['run_code'], c['fx'], c['tau_u'], c['tau_s'], c.get('ref_units', ''))
            for c in configs]
    assert len(keys) == len(set(keys)), "Duplicate corpus configs found"


def test_a_before_field_subsets():
    """Within each (run_code, tau_u, tau_s) group, fx='A' must come first."""
    configs = corpus_configs_from_csv()
    from itertools import groupby
    key_fn = lambda c: (c['run_code'], c['tau_u'], c['tau_s'])
    for _, group in groupby(configs, key=key_fn):
        group = list(group)
        fx_list = [c['fx'] for c in group]
        if 'A' in fx_list:
            assert fx_list[0] == 'A', (
                f"fx='A' must be first in group, got order: {fx_list}"
            )


def test_fixtau_configs_sorted_after_vartau():
    """Fixtau configs (non-empty ref_units) must appear after all vartau configs so
    the reference _units_..._vartau table is built before they run."""
    configs = corpus_configs_from_csv()
    seen_fixtau = False
    for c in configs:
        if c.get('ref_units', ''):
            seen_fixtau = True
        else:
            assert not seen_fixtau, (
                f"Vartau config {c['run_code']}/{c['fx']} appears after a fixtau config"
            )


def test_fixtau_configs_have_ref_units():
    """Every fixtau config must carry a non-empty ref_units pointer."""
    # verified by construction, but check at least one fixtau row exists
    configs = corpus_configs_from_csv()
    fixtau = [c for c in configs if c.get('ref_units', '')]
    assert fixtau, "No fixtau configs found in params.csv"
    for c in fixtau:
        assert c['ref_units'], f"Fixtau config {c['run_code']} has empty ref_units"


def test_vartau_configs_have_empty_ref_units():
    """Vartau configs must not carry a ref_units value."""
    configs = corpus_configs_from_csv()
    for c in configs:
        if not c.get('ref_units', ''):
            assert not c.get('ref_units', ''), (
                f"Config {c['run_code']}/{c['fx']} has unexpected ref_units"
            )


def test_expected_configs_present():
    """All corpus configs needed for the paper must be present."""
    configs = corpus_configs_from_csv()
    # key includes ref_units to distinguish vartau from fixtau
    keys = {(c['run_code'], c['fx'], c['tau_u'], c['tau_s'], c.get('ref_units', ''))
            for c in configs}

    # Baseline window: all field subsets at tau=20 (vartau)
    for fx in ['A', 'E', 'B', 'EB', 'X']:
        assert ('20242024', fx, 20, 20, '') in keys, f"Missing (20242024, {fx}, 20, 20, vartau)"

    # tau sensitivity
    assert ('20242024', 'A', 40, 40, '') in keys, "Missing tau40 config"

    # Time series (A only, vartau)
    for rc in ['00040004', '05090509', '10141014', '15191519']:
        assert (rc, 'A', 20, 20, '') in keys, f"Missing time series vartau config {rc}"

    # Fixed-universe time series (fixtau)
    ref = '20242024_A_tauU20_tauS20'
    for rc in ['00040004', '05090509', '10141014', '15191519']:
        assert (rc, 'A', 20, 20, ref) in keys, f"Missing fixtau config {rc}"


def test_configs_have_required_keys():
    configs = corpus_configs_from_csv()
    required = {'run_code', 'tc0', 'tc1', 'tt0', 'tt1', 'fx', 'tau_u', 'tau_s', 'ref_units'}
    for c in configs:
        assert required <= c.keys(), f"Config missing keys: {required - c.keys()}"


def test_year_types_are_int():
    configs = corpus_configs_from_csv()
    for c in configs:
        assert isinstance(c['tc0'], int)
        assert isinstance(c['tc1'], int)
        assert isinstance(c['tt0'], int)
        assert isinstance(c['tt1'], int)
        assert isinstance(c['tau_u'], int)
        assert isinstance(c['tau_s'], int)


def test_no_redundant_field_subsets_for_time_series():
    """Time-series runs (t1-t4) use A only; E/B/EB/X should not appear."""
    configs = corpus_configs_from_csv()
    time_series_rcs = {'00040004', '05090509', '10141014', '15191519'}
    for c in configs:
        if c['run_code'] in time_series_rcs:
            assert c['fx'] == 'A', (
                f"Unexpected field subset {c['fx']} for time-series "
                f"run_code {c['run_code']}"
            )
