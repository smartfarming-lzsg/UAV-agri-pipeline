"""Field Plot Analyzer v1 — per-plot heatmap visualizations.

Two panel variants, matching the notebook's summary figures:
  - per-pixel:    the raw raster with plot outlines drawn on top
  - per-plot mean: each plot polygon filled with its own mean value

And two colour-scale modes for a multi-flight summary grid (see the
attached "Bohne - orthophoto.tif - per-plot canopy cover" example):
  - shared:     one colorbar/legend across all flights (absolute comparison)
  - individual: per-facet colour limits (relative, within-flight contrast)

Watch the PIL-vs-numpy array handling gotcha when mixing raster sources —
an RGB orthophoto read via PIL and a single-band DSM/CHM read via numpy
arrive with different shapes/dtypes; both paths are handled explicitly
below rather than assumed.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import rasterio

from .grid import PlotGrid


def _read_display_image(raster_path: Path):
    """Returns (array, is_single_band). Single-band -> (H, W) float for
    imshow with a colormap. Multi-band -> (H, W, 3) uint8 RGB."""
    with rasterio.open(raster_path) as src:
        if src.count == 1:
            return src.read(1, out_dtype="float32"), True
        arr = src.read([1, 2, 3])
        arr = np.moveaxis(arr, 0, -1)
        if arr.dtype != np.uint8:
            lo, hi = np.nanpercentile(arr, [2, 98])
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        return arr, False


def plot_heatmap(raster_path: Path, grid: PlotGrid, plot_values: dict,
                 title: str = None, mode: str = "per_pixel",
                 cmap: str = "viridis", vmin=None, vmax=None, ax=None):
    """mode='per_pixel': show the raster with plot outlines.
    mode='per_plot_mean': fill each plot polygon with its own value."""
    img, is_single = _read_display_image(raster_path)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 6))

    if mode == "per_pixel":
        if is_single:
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            if own_fig:
                plt.colorbar(im, ax=ax, label=title or "")
        else:
            ax.imshow(img)
        with rasterio.open(raster_path) as src:
            transform = src.transform
        for poly in grid.polygons:
            xs, ys = poly.exterior.xy
            rc = [~transform * (x, y) for x, y in zip(xs, ys)]
            ax.add_patch(MplPolygon(rc, fill=False, edgecolor="cyan", linewidth=0.8))
    elif mode == "per_plot_mean":
        with rasterio.open(raster_path) as src:
            transform = src.transform
            extent_img = np.full(src.shape, np.nan, np.float32)
        vals = [plot_values.get(pid, np.nan) for pid in (grid.plot_ids or range(len(grid)))]
        vmin = vmin if vmin is not None else np.nanpercentile(vals, 2)
        vmax = vmax if vmax is not None else np.nanpercentile(vals, 98)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cm = plt.get_cmap(cmap)
        ax.imshow(np.ones((*extent_img.shape, 3)), alpha=0)   # blank canvas, correct extent
        for poly, v in zip(grid.polygons, vals):
            xs, ys = poly.exterior.xy
            rc = [~transform * (x, y) for x, y in zip(xs, ys)]
            color = cm(norm(v)) if np.isfinite(v) else (0.7, 0.7, 0.7, 1.0)
            ax.add_patch(MplPolygon(rc, facecolor=color, edgecolor="k", linewidth=0.5))
        if own_fig:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
            plt.colorbar(sm, ax=ax, label=title or "")
    else:
        raise ValueError("mode must be 'per_pixel' or 'per_plot_mean'")

    ax.set_title(title or raster_path.name)
    ax.axis("off")
    if own_fig:
        fig.tight_layout()
        return fig, ax
    return ax


def summary_grid(flight_rasters: list, grid: PlotGrid, value_col_getter,
                 title: str, out_path: Path, share_scale: bool = False,
                 ncols: int = 3):
    """flight_rasters: list of (flight_name, raster_path, plot_values_dict).
    value_col_getter is unused directly here (values are pre-extracted into
    plot_values_dict by the caller from extract.py's combined table) but
    kept as a parameter for API symmetry with the notebook's summary cell.
    share_scale=True uses one global vmin/vmax across all flights (matches
    the attached example: 'Bohne - orthophoto.tif - per-plot canopy cover
    (%) (individual scale per flight)' used share_scale=False).
    """
    n = len(flight_rasters)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    vmin = vmax = None
    if share_scale:
        allv = np.concatenate([list(pv.values()) for _, _, pv in flight_rasters])
        allv = allv[np.isfinite(allv)]
        vmin, vmax = np.nanpercentile(allv, [2, 98])

    for ax, (flight, rpath, pv) in zip(axes, flight_rasters):
        plot_heatmap(rpath, grid, pv, title=flight, mode="per_plot_mean",
                    vmin=vmin, vmax=vmax, ax=ax)
    for ax in axes[len(flight_rasters):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
