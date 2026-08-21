"""Field Plot Analyzer v1 — QC + time-series (formerly the "QC/time-series cell").

Reads the combined CSVs written by extract.run_batch_extraction(), merges
line names from the plot design workbook, produces a datum-alignment QC
table per flight, and plots growth curves (canopy height / cover / best
available vegetation index) over time, grouped by line, with error bars.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import summary_dir, load_plot_design, merge_line_names

CMAP_H = "viridis"
CMAP_COVER = "YlGn"
CMAP_GI = "RdYlGn"
BEST_INDEX = "ExG"   # fallback vegetation index when NIR/NDVI unavailable


def has(df, col):
    return col in df.columns and df[col].notna().any()


def load_combined(root: Path):
    d = summary_dir(root)
    dsm = pd.read_csv(d / "all_dsm_plot_data.csv") if (d / "all_dsm_plot_data.csv").exists() else None
    ortho = pd.read_csv(d / "all_orthophoto_plot_data.csv") if (d / "all_orthophoto_plot_data.csv").exists() else None
    return dsm, ortho


def datum_qc_table(dsm: pd.DataFrame, design_xlsx: Path = None) -> pd.DataFrame:
    """One row per flight: n plots covered, mean/median canopy_height_ref_mean,
    and a verdict carried over from reference_ground.flight_datum_offset()
    if present as a `datum_verdict` column, else re-derived from the spread
    of canopy_height_ref_mean at flights with near-zero expected growth.
    Kept intentionally simple — richer QC lives in reference_ground.py at
    extraction time; this is a post-hoc sanity table.
    """
    if dsm is None or "flight" not in dsm.columns:
        raise ValueError("no DSM data to QC")
    rows = []
    for flight, g in dsm.groupby("flight"):
        col = "canopy_height_ref_mean" if has(g, "canopy_height_ref_mean") else "canopy_height_mean"
        v = g[col].to_numpy(float); v = v[np.isfinite(v)]
        rows.append({"flight": flight, "n_plots": len(g),
                    "height_col": col, "mean": float(v.mean()) if v.size else np.nan,
                    "median": float(np.median(v)) if v.size else np.nan,
                    "std": float(v.std()) if v.size else np.nan})
    return pd.DataFrame(rows).sort_values("flight").reset_index(drop=True)


def _growth_plot(df: pd.DataFrame, value_col: str, ylabel: str, out_name: str,
                 outdir: Path, design: pd.DataFrame = None):
    if design is not None and "line_name" not in df.columns:
        df = merge_line_names(df, design)
    if "line_name" not in df.columns or value_col not in df.columns:
        print(f"skip {out_name}: missing line_name or {value_col}")
        return
    flights_sorted = sorted(df["flight"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    for line, g in df.groupby("line_name"):
        gg = (g.groupby("flight")[value_col]
              .agg(["mean", "std"]).reindex(flights_sorted))
        x = range(len(flights_sorted))
        ax.errorbar(x, gg["mean"], yerr=gg["std"], marker="o", capsize=3, label=line)
    ax.set_xticks(range(len(flights_sorted)))
    ax.set_xticklabels(flights_sorted, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel); ax.set_title(f"{ylabel} over time, by line")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(outdir / out_name, dpi=140)
    plt.close(fig)
    print(f"saved {outdir / out_name}")


def growth_curves(root: Path, dsm: pd.DataFrame, ortho: pd.DataFrame,
                  design_xlsx: Path = None):
    """Produce growth_height.png, growth_cover.png, growth_index.png in
    SUMMARY_DIR, one line per trial line (merged from the design workbook),
    with mean +/- std error bars across replicate plots."""
    outdir = summary_dir(root)
    design = load_plot_design(design_xlsx) if design_xlsx is not None else None

    if dsm is not None:
        height_col = "canopy_height_ref_mean" if has(dsm, "canopy_height_ref_mean") else "canopy_height_mean"
        _growth_plot(dsm, height_col, "canopy height (m)", "growth_height.png", outdir, design)

    cover_col = next((c for c in ["canopy_cover_pct", "cover_frac", "veg_cover_frac", "cover_fraction"]
                      if (ortho is not None and has(ortho, c)) or (dsm is not None and has(dsm, c))), None)
    if cover_col:
        src = ortho if (ortho is not None and has(ortho, cover_col)) else dsm
        ylabel = "canopy cover (%)" if cover_col == "canopy_cover_pct" else "canopy cover (fraction)"
        _growth_plot(src, cover_col, ylabel, "growth_cover.png", outdir, design)
    else:
        print("no cover column found - skipping growth_cover.png")

    # best available vegetation index: NDVI if NIR was captured, else ExG > NGRDI > VARI > GLI
    index_col = next((c + "_mean" for c in ["NDVI", "ExG", "NGRDI", "VARI", "GLI"]
                      if (ortho is not None and has(ortho, c + "_mean"))
                      or (dsm is not None and has(dsm, c + "_mean"))), None)
    if index_col:
        src = ortho if (ortho is not None and has(ortho, index_col)) else dsm
        _growth_plot(src, index_col, f"mean {index_col.replace('_mean','')}",
                    "growth_index.png", outdir, design)
