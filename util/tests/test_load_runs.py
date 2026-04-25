"""Tests for load_runs()."""
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from util import load_runs

RUNS = load_runs()


def test_all_rows_loaded():
    assert len(RUNS) == 32


def test_skip_filters():
    content = (
        "skip,run_code,tc0,tc1,tt0,tt1,fx,tau_u,tau_s,rho,m,chi,alpha,mu_type,label\n"
        "0,20242024,2020,2024,2020,2024,A,20,20,0,0110,0.5,1.0,,baseline\n"
        "1,20242024,2020,2024,2020,2024,E,20,20,0,0110,0.5,1.0,,F=E\n"
    )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content)
        tmp = Path(f.name)
    try:
        runs = load_runs(tmp)
        assert len(runs) == 1
        assert runs[0]['label'] == 'baseline'
    finally:
        os.unlink(tmp)


def test_types():
    r = RUNS[0]
    assert isinstance(r['tc0'],   int),   f"tc0 should be int, got {type(r['tc0'])}"
    assert isinstance(r['tc1'],   int),   f"tc1 should be int, got {type(r['tc1'])}"
    assert isinstance(r['tt0'],   int),   f"tt0 should be int, got {type(r['tt0'])}"
    assert isinstance(r['tt1'],   int),   f"tt1 should be int, got {type(r['tt1'])}"
    assert isinstance(r['tau_u'], int),   f"tau_u should be int, got {type(r['tau_u'])}"
    assert isinstance(r['tau_s'], int),   f"tau_s should be int, got {type(r['tau_s'])}"
    assert isinstance(r['rho'],   int),   f"rho should be int, got {type(r['rho'])}"
    assert isinstance(r['chi'],   float), f"chi should be float, got {type(r['chi'])}"
    assert isinstance(r['alpha'], float), f"alpha should be float, got {type(r['alpha'])}"
    assert isinstance(r['m'],     str),   f"m should be str, got {type(r['m'])}"
    assert isinstance(r['run_code'], str), f"run_code should be str, got {type(r['run_code'])}"


def test_chi_star_is_float():
    chi_star_rows = [r for r in RUNS if r['chi'] == -1.0]
    assert len(chi_star_rows) == 1
    assert chi_star_rows[0]['label'] == 'full-joint-chi-star'
    assert isinstance(chi_star_rows[0]['chi'], float)


def test_m_is_string():
    assert RUNS[0]['m'] == '0110'


def test_run_code_matches_years():
    """run_code in CSV must equal last-2-digits of tc0,tc1,tt0,tt1."""
    for r in RUNS:
        expected = (f"{r['tc0'] % 100:02d}{r['tc1'] % 100:02d}"
                    f"{r['tt0'] % 100:02d}{r['tt1'] % 100:02d}")
        assert r['run_code'] == expected, (
            f"run_code mismatch for {r['label']}: "
            f"got {r['run_code']!r}, expected {expected!r}"
        )
