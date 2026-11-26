"""Tests for the cardinal AIS comparison in benchmark_journals.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from benchmark_journals import ais_summary, rma_line


def test_rma_line_recovers_exact_line():
    x = np.array([0., 1., 2., 3.])
    r, b, a = rma_line(x, 0.5 + 2 * x)
    assert np.isclose(r, 1) and np.isclose(b, 2) and np.isclose(a, 0.5)


def test_ais_summary_classes_against_band():
    la = np.linspace(-1, 1, 12)
    lv = la.copy()
    lv[0] += 0.05          # within its band
    lv[-1] -= 0.5          # far below the line
    d = pd.DataFrame({'lv': lv, 'hw': 0.1, 'ais': 10 ** la, 'field_eb': 'E'})
    d.loc[11, 'ais'] = np.nan          # unmatched journal is left out
    out = ais_summary(d, 'bipartite')
    assert d.loc[11, 'ais_class'] is np.nan or pd.isna(d.loc[11, 'ais_class'])
    allrow = out[(out.group_type == 'field') & (out.group == 'all')].iloc[0]
    assert allrow.n == 11 and allrow.in_band == 11
    d2 = pd.DataFrame({'lv': lv, 'hw': 0.1, 'ais': 10 ** la, 'field_eb': 'E'})
    ais_summary(d2, 'bipartite')
    assert d2.loc[11, 'ais_class'] == 'far' and d2.loc[0, 'ais_class'] == 'in'
