"""Tests for run_rankings.py pure functions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from run_rankings import RunParams, table_name, _dense_rank_desc, runs_from_csv


# ─── table_name ───────────────────────────────────────────────────────────────

def test_table_name_baseline():
    p = RunParams(run_code='20242024', fx='A', tau_u=20, tau_s=20,
                  rho=0, m=(0,1,1,0), chi=0.5, alpha=1.0, label='baseline')
    assert table_name(p) == 'rk_20242024_A_tauU20_tauS20_rho0_m0110_chi50_alpha100'


def test_table_name_chi_star():
    p = RunParams(run_code='20242024', fx='A', tau_u=20, tau_s=20,
                  rho=0, m=(1,1,1,1), chi=-1.0, alpha=1.0, label='full-joint-chi-star')
    assert table_name(p) == 'rk_20242024_A_tauU20_tauS20_rho0_m1111_chiSTAR_alpha100'


def test_table_name_time_series():
    p = RunParams(run_code='00040004', fx='A', tau_u=20, tau_s=20,
                  rho=0, m=(0,1,1,0), chi=0.5, alpha=1.0, label='t1')
    assert table_name(p) == 'rk_00040004_A_tauU20_tauS20_rho0_m0110_chi50_alpha100'


def test_table_name_tau40():
    p = RunParams(run_code='20242024', fx='A', tau_u=40, tau_s=40,
                  rho=0, m=(0,1,1,0), chi=0.5, alpha=1.0, label='tau40')
    assert table_name(p) == 'rk_20242024_A_tauU40_tauS40_rho0_m0110_chi50_alpha100'


def test_table_name_rho1():
    p = RunParams(run_code='20242024', fx='A', tau_u=20, tau_s=20,
                  rho=1, m=(0,1,1,0), chi=0.5, alpha=1.0, label='rho1')
    assert table_name(p) == 'rk_20242024_A_tauU20_tauS20_rho1_m0110_chi50_alpha100'


# ─── _dense_rank_desc ─────────────────────────────────────────────────────────

def test_dense_rank_desc_basic():
    ranks = _dense_rank_desc(np.array([3.0, 1.0, 2.0]))
    assert list(ranks) == [1, 3, 2]


def test_dense_rank_desc_ties():
    ranks = _dense_rank_desc(np.array([2.0, 2.0, 1.0]))
    assert ranks[0] == 1
    assert ranks[1] == 1
    assert ranks[2] == 2   # dense: no gap after tie


def test_dense_rank_desc_all_equal():
    ranks = _dense_rank_desc(np.array([5.0, 5.0, 5.0]))
    assert list(ranks) == [1, 1, 1]


def test_dense_rank_desc_single():
    ranks = _dense_rank_desc(np.array([42.0]))
    assert list(ranks) == [1]


# ─── runs_from_csv ────────────────────────────────────────────────────────────

def test_runs_from_csv_stage1_count():
    runs = runs_from_csv(stage=1)
    assert len(runs) == 11


def test_runs_from_csv_stage2_count():
    runs = runs_from_csv(stage=2)
    assert len(runs) == 4


def test_runs_from_csv_stage2_labels():
    runs = runs_from_csv(stage=2)
    labels = [p.label for p in runs]
    assert set(labels) == {'t1', 't2', 't3', 't4'}
    assert 'baseline' not in labels


def test_runs_from_csv_m_is_tuple():
    runs = runs_from_csv(stage=1)
    assert all(isinstance(p.m, tuple) for p in runs)
    baseline = next(p for p in runs if p.label == 'baseline')
    assert baseline.m == (0, 1, 1, 0)


def test_runs_from_csv_chi_star():
    runs = runs_from_csv(stage=1)
    chi_star = [p for p in runs if p.chi == -1.0]
    assert len(chi_star) == 1
    assert chi_star[0].label == 'full-joint-chi-star'


def test_runs_from_csv_run_code_format():
    runs = runs_from_csv()
    for p in runs:
        assert len(p.run_code) == 8
        assert p.run_code.isdigit()
