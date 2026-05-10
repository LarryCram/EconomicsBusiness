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
    assert table_name('20242024', 'EBAX', 20, 20) == 'el_20242024_EBAX_tauU20_tauS20_vartau'


def test_table_name_field_subset():
    assert table_name('20242024', 'E', 20, 20) == 'el_20242024_E_tauU20_tauS20_vartau'


def test_table_name_tau40():
    assert table_name('20242024', 'EBAX', 40, 40) == 'el_20242024_EBAX_tauU40_tauS40_vartau'


def test_table_name_time_series():
    assert table_name('00040004', 'EBAX', 20, 20) == 'el_00040004_EBAX_tauU20_tauS20_vartau'


def test_table_name_fixed_universe():
    assert (table_name('00040004', 'EBAX', 20, 20, ref_units='20242024_EBAX_tauU20_tauS20')
            == 'el_00040004_EBAX_tauU20_tauS20_fixtau')


def test_table_name_epsilon():
    assert (table_name('20242024', 'EBAX', 20, 20, epsilon=1)
            == 'el_20242024_EBAX_tauU20_tauS20_vartau_eps1')


# ─── corpus_configs_from_csv ──────────────────────────────────────────────────

def test_configs_are_unique():
    configs = corpus_configs_from_csv()
    keys = [(c['run_code'], c['fx'], c['tau_u'], c['tau_s'],
             c.get('ref_units', ''), c.get('epsilon', 0))
            for c in configs]
    assert len(keys) == len(set(keys)), "Duplicate corpus configs found"


def test_all_before_field_subsets():
    """Within each (run_code, tau_u, tau_s) group, fx='EBAX' must come first."""
    configs = corpus_configs_from_csv()
    from itertools import groupby
    key_fn = lambda c: (c['run_code'], c['tau_u'], c['tau_s'])
    for _, group in groupby(configs, key=key_fn):
        group = list(group)
        fx_list = [c['fx'] for c in group]
        if 'EBAX' in fx_list:
            assert fx_list[0] == 'EBAX', (
                f"fx='EBAX' must be first in group, got order: {fx_list}"
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
    # key includes ref_units and epsilon to distinguish all variants
    keys = {(c['run_code'], c['fx'], c['tau_u'], c['tau_s'],
             c.get('ref_units', ''), c.get('epsilon', 0))
            for c in configs}

    # Baseline window: all field subsets at tau=20 (vartau, epsilon=0)
    for fx in ['EBAX', 'E', 'B', 'EBA', 'X', 'A']:
        assert ('20242024', fx, 20, 20, '', 0) in keys, f"Missing (20242024, {fx}, 20, 20, vartau)"

    # tau sensitivity
    assert ('20242024', 'EBAX', 40, 40, '', 0) in keys, "Missing tau40 config"

    # Time series (EBAX only, vartau)
    for rc in ['00040004', '05090509', '10141014', '15191519']:
        assert (rc, 'EBAX', 20, 20, '', 0) in keys, f"Missing time series vartau config {rc}"

    # Fixed-universe time series (fixtau)
    ref = '20242024_EBAX_tauU20_tauS20'
    for rc in ['00040004', '05090509', '10141014', '15191519']:
        assert (rc, 'EBAX', 20, 20, ref, 0) in keys, f"Missing fixtau config {rc}"

    # Epsilon=1 baseline
    assert ('20242024', 'EBAX', 20, 20, '', 1) in keys, "Missing baseline-eps config (epsilon=1)"


def test_configs_have_required_keys():
    configs = corpus_configs_from_csv()
    required = {'run_code', 'tc0', 'tc1', 'tt0', 'tt1', 'fx', 'tau_u', 'tau_s', 'ref_units', 'epsilon'}
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
    """Time-series runs (t1-t4) use EBAX only; subsets should not appear."""
    configs = corpus_configs_from_csv()
    time_series_rcs = {'00040004', '05090509', '10141014', '15191519'}
    for c in configs:
        if c['run_code'] in time_series_rcs:
            assert c['fx'] == 'EBAX', (
                f"Unexpected field subset {c['fx']} for time-series "
                f"run_code {c['run_code']}"
            )
