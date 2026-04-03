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
    assert table_name('20242024', 'A', 20, 20) == 'el_20242024_A_tauU20_tauS20'


def test_table_name_field_subset():
    assert table_name('20242024', 'E', 20, 20) == 'el_20242024_E_tauU20_tauS20'


def test_table_name_tau40():
    assert table_name('20242024', 'A', 40, 40) == 'el_20242024_A_tauU40_tauS40'


def test_table_name_time_series():
    assert table_name('00040004', 'A', 20, 20) == 'el_00040004_A_tauU20_tauS20'


# ─── corpus_configs_from_csv ──────────────────────────────────────────────────

def test_configs_are_unique():
    configs = corpus_configs_from_csv()
    keys = [(c['run_code'], c['fx'], c['tau_u'], c['tau_s']) for c in configs]
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


def test_expected_configs_present():
    """All corpus configs needed for the paper must be present."""
    configs = corpus_configs_from_csv()
    keys = {(c['run_code'], c['fx'], c['tau_u'], c['tau_s']) for c in configs}

    # Stage 1: baseline window, all field subsets at tau=20
    for fx in ['A', 'E', 'B', 'EB', 'NEB']:
        assert ('20242024', fx, 20, 20) in keys, f"Missing (20242024, {fx}, 20, 20)"

    # Stage 1: tau sensitivity
    assert ('20242024', 'A', 40, 40) in keys, "Missing tau40 config"

    # Stage 2: time series (A only)
    for rc in ['00040004', '05090509', '10141014', '15191519']:
        assert (rc, 'A', 20, 20) in keys, f"Missing time series config {rc}"


def test_configs_have_required_keys():
    configs = corpus_configs_from_csv()
    required = {'run_code', 'tc0', 'tc1', 'tt0', 'tt1', 'fx', 'tau_u', 'tau_s'}
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


def test_no_redundant_field_subsets_in_stage2():
    """Stage-2 runs are all fx='A'; no E/B/EB/NEB should appear for t1-t4."""
    configs = corpus_configs_from_csv()
    stage2_rcs = {'00040004', '05090509', '10141014', '15191519'}
    for c in configs:
        if c['run_code'] in stage2_rcs:
            assert c['fx'] == 'A', (
                f"Non-A field subset {c['fx']} found for stage-2 run_code {c['run_code']}"
            )
