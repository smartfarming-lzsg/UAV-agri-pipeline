"""Field Plot Analyzer v1 — plot grid construction.

Builds the per-plot polygon grid by bilinear interpolation between picked
corner points (originally driven by a JS canvas corner-picker cell in the
notebook: click the field's 4 corners on the orthophoto/DSM preview, get
back pixel or UTM coordinates, and this module turns that into an
n_rows x n_cols grid of plot polygons).
"""
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon


@dataclass
class PlotGrid:
    """n_rows x n_cols plot polygons in UTM (or whatever CRS the corners
    were given in), row-major, plus per-plot ids assigned externally via
    config.load_plot_design()."""
    polygons: list          # len == n_rows * n_cols, row-major
    n_rows: int
    n_cols: int
    corners: np.ndarray      # the 4 picked corners, (4, 2)
    plot_ids: list = field(default_factory=list)   # optional, set by caller

    def __len__(self):
        return len(self.polygons)

    def __iter__(self):
        return iter(self.polygons)


def _bilinear(corners, u, v):
    """corners ordered [top-left, top-right, bottom-right, bottom-left];
    u, v in [0, 1] along the two grid axes."""
    tl, tr, br, bl = corners
    top = tl + (tr - tl) * u
    bot = bl + (br - bl) * u
    return top + (bot - top) * v


def build_plot_grid(corners, n_rows: int, n_cols: int,
                     row_margin: float = 0.0, col_margin: float = 0.0) -> PlotGrid:
    """Bilinearly interpolate an n_rows x n_cols grid of plot polygons from
    4 picked corners.

    Parameters
    ----------
    corners : array-like, shape (4, 2)
        [top-left, top-right, bottom-right, bottom-left] in the picker's
        coordinate system (UTM if picked on a georeferenced preview).
    n_rows, n_cols : int
        Plot grid dimensions (e.g. 20 lines x 2 reps -> n_rows=20, n_cols=2,
        or however the trial design lays plots out).
    row_margin, col_margin : float
        Fraction (0-0.49) of each cell's extent to inset on every side,
        shrinking each plot polygon inward from the nominal grid line —
        keeps sampling away from alley/border pixels.
    """
    C = np.asarray(corners, float)
    assert C.shape == (4, 2), "corners must be 4 points: TL, TR, BR, BL"

    polys = []
    for r in range(n_rows):
        v0, v1 = r / n_rows, (r + 1) / n_rows
        v0 += row_margin * (v1 - v0); v1 -= row_margin * (v1 - v0)
        for c in range(n_cols):
            u0, u1 = c / n_cols, (c + 1) / n_cols
            u0 += col_margin * (u1 - u0); u1 -= col_margin * (u1 - u0)
            pts = [_bilinear(C, u0, v0), _bilinear(C, u1, v0),
                   _bilinear(C, u1, v1), _bilinear(C, u0, v1)]
            polys.append(Polygon(pts))
    return PlotGrid(polygons=polys, n_rows=n_rows, n_cols=n_cols, corners=C)


def assign_plot_ids(grid: PlotGrid, design_df, order: str = "row_major") -> PlotGrid:
    """Attach plot_id (and any other design columns) from the loaded design
    table onto each polygon in row-major order. `design_df` is expected
    sorted the same way the grid was walked when corners were picked —
    verify against a heatmap overlay before trusting the mapping blindly.
    """
    ids = list(design_df["plot_id"])
    if len(ids) != len(grid.polygons):
        raise ValueError(f"design has {len(ids)} plots, grid has "
                         f"{len(grid.polygons)} cells ({grid.n_rows}x{grid.n_cols})")
    grid.plot_ids = ids
    return grid
