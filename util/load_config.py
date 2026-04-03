"""
util/load_config.py — Shared loaders for the EconomicsBusiness project.

  load_config() → Paths   reads config.yaml  (machine-specific, gitignored)
  load_params() → dict    reads params.yaml  (model parameters, version-controlled)

Usage:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from util import load_config, load_params

    paths  = load_config()
    params = load_params()
    # paths.data, paths.parquet, ...
    # params['time_windows'], params['tau_u_floor']
"""

import csv
from dataclasses import dataclass
from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'
_PARAMS_PATH = Path(__file__).parent.parent / 'params.yaml'
_RUNS_PATH   = Path(__file__).parent.parent / 'params.csv'


@dataclass(frozen=True)
class Paths:
    project_root: Path   # PROJECT_ROOT in config.yaml
    data: Path           # PROJECT_ROOT / DATA  (small files, git-tracked)
    tables: Path         # PROJECT_ROOT / TABLES  (LaTeX table fragments)
    working: Path        # WORKING  (SSD root for large parquets)
    openalex: Path       # OPENALEX (OA parquet snapshot)
    parquet: Path        # WORKING / parquet  (pipeline intermediates)
    plots: Path          # PROJECT_ROOT / PLOTS


def load_runs(runs_path: Path = _RUNS_PATH) -> list[dict]:
    """
    Read params.csv and return one dict per non-skipped run.

    Type conversions:
        skip, tau_u, tau_s, rho, stage  → int
        chi, alpha                       → float
        m                                → str  (e.g. '0110')
        all others                       → str
    """
    int_cols   = {'skip', 'tc0', 'tc1', 'tt0', 'tt1', 'tau_u', 'tau_s', 'rho', 'stage'}
    float_cols = {'chi', 'alpha'}

    runs = []
    with open(runs_path, newline='') as f:
        for row in csv.DictReader(f):
            for col in int_cols:
                row[col] = int(row[col])
            for col in float_cols:
                row[col] = float(row[col])
            if row['skip']:
                continue
            runs.append(row)
    return runs


def load_params(params_path: Path = _PARAMS_PATH) -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def load_config(config_path: Path = _CONFIG_PATH) -> Paths:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = Path(cfg['PROJECT_ROOT'])
    working = Path(cfg['WORKING'])
    return Paths(
        project_root=project_root,
        data=project_root / cfg['DATA'],
        tables=project_root / cfg['TABLES'],
        working=working,
        openalex=Path(cfg['OPENALEX']),
        parquet=working / 'parquet',
        plots=project_root / cfg['PLOTS'],
    )
