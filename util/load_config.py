"""
util/load_config.py — Shared configuration loader for the EconomicsBusiness project.

Reads config.yaml from the project root and exposes all standard paths as a
frozen dataclass. Scripts import this and use only the fields they need.

Usage:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from util import load_config

    paths = load_config()
    # paths.data, paths.working, paths.parquet, paths.openalex, paths.plots
"""

from dataclasses import dataclass
from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'


@dataclass(frozen=True)
class Paths:
    project_root: Path   # PROJECT_ROOT in config.yaml
    data: Path           # PROJECT_ROOT / DATA  (small files, git-tracked)
    working: Path        # WORKING  (SSD root for large parquets)
    openalex: Path       # OPENALEX (OA parquet snapshot)
    parquet: Path        # WORKING / parquet  (pipeline intermediates)
    plots: Path          # PROJECT_ROOT / plots


def load_config(config_path: Path = _CONFIG_PATH) -> Paths:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = Path(cfg['PROJECT_ROOT'])
    working = Path(cfg['WORKING'])
    return Paths(
        project_root=project_root,
        data=project_root / cfg['DATA'],
        working=working,
        openalex=Path(cfg['OPENALEX']),
        parquet=working / 'parquet',
        plots=project_root / 'plots',
    )
