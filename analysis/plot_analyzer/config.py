"""Field Plot Analyzer v1 — shared config and plot design loading.

Consolidated from the analysis notebook cells (reference-ground cell, batch
extraction cell, QC/time-series cell). Unlike reconstruction/odm_light_cuda_v11
and orthomosaic/compositor_v5_2, this package was pulled together into
importable modules for the first time here — the source lived as evolving
Colab cells, not a single script. Treat this as a v1 starting point and diff
it against your live notebook before relying on it in place of the notebook.
"""
from pathlib import Path

import pandas as pd

# ── path conventions (match reconstruction/orthomosaic outputs) ────────────
SUB = "pipeline_output"
DSM_NAME = "dsm.tif"
CHM_NAME = "chm.tif"
ORTHO_NAME = "orthophoto.tif"
SUMMARY_DIRNAME = "plot_analysis_summary"   # sibling of pipeline_output/

# ── plot design workbook ────────────────────────────────────────────────────
# e.g. Aussaat_Salez_05_2026.xlsx, sheet "6m2 Plots": columns plot_id, line_name
DESIGN_XLSX = None      # set per field, e.g. Path("Aussaat_Salez_05_2026.xlsx")
DESIGN_SHEET = "6m2 Plots"


def flight_key(flight_dir: Path) -> str:
    """Stable key for a flight folder, used to align QC/growth-curve rows
    across the reference-ground, batch, and QC cells."""
    return Path(flight_dir).name


def summary_dir(root: Path) -> Path:
    d = Path(root) / SUMMARY_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_plot_design(xlsx_path: Path = None, sheet: str = DESIGN_SHEET) -> pd.DataFrame:
    """Load the plot design workbook: plot_id -> line_name (+ any other
    trial-design columns present). Returns a DataFrame keyed by plot_id.
    """
    xlsx_path = Path(xlsx_path or DESIGN_XLSX)
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    if "plot_id" not in df.columns:
        raise ValueError(f"{xlsx_path}::{sheet} has no 'plot_id' column "
                         f"(found {list(df.columns)})")
    return df.set_index("plot_id", drop=False)


def merge_line_names(df: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    """Left-join line_name (and other design columns) onto a plot-level
    dataframe by plot_id, without duplicating plot_id."""
    cols = [c for c in design.columns if c != "plot_id"]
    return df.merge(design[["plot_id"] + cols], on="plot_id", how="left")
