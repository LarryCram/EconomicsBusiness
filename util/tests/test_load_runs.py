"""Tests for load_runs()."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from util import load_runs

RUNS = load_runs()


def test_all_rows_not_skipped():
    # All rows in params.csv currently have skip=0
    assert len(RUNS) == 15


def test_skip_filters():
    import csv, tempfile, os
    content = (
        "skip,run_code,tc0,tc1,tt0,tt1,fx,tau_u,tau_s,rho,m,chi,alpha,mu_type,label,stage\n"
        "0,20242024,2020,2024,2020,2024,A,20,20,0,0110,0.5,1.0,,baseline,1\n"
        "1,20242024,2020,2024,2020,2024,E,20,20,0,0110,0.5,1.0,,F=E,1\n"
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
    assert isinstance(r['tau_u'], int)
    assert isinstance(r['tau_s'], int)
    assert isinstance(r['rho'],   int)
    assert isinstance(r['stage'], int)
    assert isinstance(r['chi'],   float)
    assert isinstance(r['alpha'], float)
    assert isinstance(r['m'],     str)
    assert isinstance(r['run_code'], str)


def test_chi_star_is_float():
    chi_star_rows = [r for r in RUNS if r['chi'] == -1.0]
    assert len(chi_star_rows) == 1
    assert chi_star_rows[0]['label'] == 'full-joint-chi-star'
    assert isinstance(chi_star_rows[0]['chi'], float)


def test_m_is_string():
    # m must not be parsed as int (leading zero would be lost)
    assert RUNS[0]['m'] == '0110'
