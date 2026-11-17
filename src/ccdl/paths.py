"""Canonical in-repo locations for data, plot-source data, and figures.

Every path is resolved relative to the repository root. The code contains no hardcoded absolute paths.
"""
from pathlib import Path

# This file is src/ccdl/paths.py, so the repository root is two levels above
# the package directory (…/repo/src/ccdl/paths.py -> …/repo).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR        = PROJECT_ROOT / "data"          # datasets + training-run outputs (DATADIR / WORKDIR)
FIGURES_DIR     = PROJECT_ROOT / "figures"       # rendered figures (formerly PLOTSDIR/final_plots)
SOURCE_DATA_DIR = PROJECT_ROOT / "source_data"           # per-figure source data (formerly WORKDIR/plots_data)
ASSETS_DIR = Path(__file__).resolve().parent
PRETRAINED_DIR = ASSETS_DIR / "assets/pretrained"          # shipped with the package