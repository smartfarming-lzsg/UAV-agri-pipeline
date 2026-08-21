"""Field Plot Analyzer v1 — batch per-plot extraction (formerly "Cell B").

Walks every flight under ROOT, extracts windowed full-resolution per-plot
statistics from the DSM/CHM raster (canopy height) and the orthophoto
(canopy cover + vegetation indices), and writes the combined tables that
qc.py and viz.py consume.

Column names (confirmed against the actual notebook output, not guessed):
  height  : canopy_height_mean       (p5-of-plot baseline, legacy)
            canopy_height_ref_mean   (fixed-reference datum, PREFERRED —
                                       see reference_ground.py for why)
            canopy_height_p95, h_median, h_std, h_p90, h_max
  cover   : canopy_cover_pct         (0-100, ExG/NDVI-Otsu vegetation mask)
  indices : {ExG,NGRDI,VARI,GLI}_mean / _median   (RGB-only)
            NDVI_mean / _median                   (only if NIR band present)

Coverage is measured over the plot polygon area, not the padded sampling
grid — a narrow trial strip can fail a broad-canvas coverage threshold
while its actual plot-area coverage is fine.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

from .config import SUB, DSM_NAME, CHM_NAME, ORTHO_NAME, summary_dir
from .reference_ground import flight_datum_offset, REF_READY
from .grid import PlotGrid

MASK_INDEX = "ExG"          # index used to build the vegetation mask when SOIL_MASK=True
SOIL_MASK = True
MIN_COVERAGE_PLOT_AREA = 0.5   # fraction of PLOT polygon area that must be valid pixels


def indices_from_stack(bands: np.ndarray, has_nir: bool) -> dict:
    """bands: (H, W, C) float, band order R,G,B[,NIR]. Returns per-pixel
    index arrays. NDVI only computed if has_nir."""
    R, G, B = bands[..., 0], bands[..., 1], bands[..., 2]
    total = R + G + B + 1e-6
    out = {
        "ExG":   2 * (G / total) - (R / total) - (B / total),
        "NGRDI": (G - R) / (G + R + 1e-6),
        "VARI":  (G - R) / (G + R - B + 1e-6),
        "GLI":   (2 * G - R - B) / (2 * G + R + B + 1e-6),
    }
    if has_nir and bands.shape[-1] >= 4:
        NIR = bands[..., 3]
        out["NDVI"] = (NIR - R) / (NIR + R + 1e-6)
    return out


def _plot_window_stats_height(dsm, chm, ref_ground_win, datum_offset, rr, cc):
    v = dsm[rr, cc]; v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    rec = {
        "canopy_height_mean": float(np.mean(v)) - float(np.percentile(v, 5)),
        "h_median": float(np.median(v)), "h_std": float(v.std()),
        "h_p90": float(np.percentile(v, 90)), "h_p95": float(np.percentile(v, 95)),
        "h_max": float(v.max()),
    }
    if ref_ground_win is not None and datum_offset is not None:
        ch = dsm[rr, cc] - (ref_ground_win[rr, cc] + datum_offset)
        ch = ch[np.isfinite(ch)]
        if ch.size:
            rec["canopy_height_ref_mean"] = float(ch.mean())
            rec["canopy_height_ref_p95"] = float(np.percentile(ch, 95))
    if chm is not None:
        c = chm[rr, cc]; c = c[np.isfinite(c)]
        if c.size:
            rec["canopy_height_mean"] = float(c.mean())
            rec["canopy_height_p95"] = float(np.percentile(c, 95))
    return rec


def _extract_raster(raster_path: Path, grid: PlotGrid, *, is_height: bool,
                    has_nir: bool = False, veg_thr: float = None,
                    ref_ground_win=None, datum_offset=None) -> pd.DataFrame:
    rows_out = []
    with rasterio.open(raster_path) as src:
        arr = src.read(out_dtype="float32")
        if arr.shape[0] == 1:
            a = arr[0]
        else:
            a = np.moveaxis(arr, 0, -1)   # (H, W, C)

        for plot_id, poly in zip(grid.plot_ids or range(len(grid)), grid.polygons):
            mask = ~geometry_mask([poly], out_shape=src.shape,
                                  transform=src.transform, invert=False)
            rr, cc = np.nonzero(mask)
            plot_area_px = mask.sum()
            if plot_area_px == 0:
                continue
            rec = {"plot_id": plot_id}

            if is_height:
                chm = None  # caller passes a pre-read CHM stack separately if desired
                stats = _plot_window_stats_height(a, chm, ref_ground_win,
                                                   datum_offset, rr, cc)
                if stats is None:
                    continue
                rec.update(stats)
                coverage = np.isfinite(a[rr, cc]).mean()
            else:
                idx = indices_from_stack(a, has_nir)
                veg = ((idx[MASK_INDEX][rr, cc] > veg_thr)
                      if (SOIL_MASK and veg_thr is not None)
                      else np.ones(rr.size, bool))
                if SOIL_MASK:
                    rec["canopy_cover_pct"] = float(100.0 * veg.mean()) if veg.size else np.nan
                for k, vv in idx.items():
                    vals = vv[rr, cc][veg]; vals = vals[np.isfinite(vals)]
                    rec[f"{k}_mean"] = float(vals.mean()) if vals.size else np.nan
                    rec[f"{k}_median"] = float(np.median(vals)) if vals.size else np.nan
                coverage = np.isfinite(idx[MASK_INDEX][rr, cc]).mean()

            if coverage < MIN_COVERAGE_PLOT_AREA:
                continue
            rows_out.append(rec)

    return pd.DataFrame(rows_out).sort_values("plot_id").reset_index(drop=True)


def run_batch_extraction(root: Path, grid: PlotGrid, *, has_nir: bool = False,
                         veg_thr: float = 0.05, log=print) -> dict:
    """Walk every flight under root that has a DSM and/or orthophoto, extract
    per-plot stats, and write the combined CSVs to SUMMARY_DIR. Returns
    {"dsm": DataFrame, "orthophoto": DataFrame} of the COMBINED
    (all-flights) tables — the same tables written to disk.
    """
    root = Path(root)
    outdir = summary_dir(root)
    results, skipped = [], []

    flights = sorted(p for p in root.iterdir() if p.is_dir() and (p / SUB).is_dir())
    for flight in flights:
        for rname, is_height in ((DSM_NAME, True), (ORTHO_NAME, False)):
            rpath = flight / SUB / rname
            if not rpath.exists():
                skipped.append((flight.name, rname, "missing")); continue
            ref_win, offset = None, None
            if is_height and REF_READY:
                offset, verdict = flight_datum_offset(root, flight, log=lambda *_: None)
                if offset is None:
                    skipped.append((flight.name, rname, f"datum: {verdict}")); continue
            try:
                dfx = _extract_raster(rpath, grid, is_height=is_height,
                                      has_nir=has_nir, veg_thr=veg_thr,
                                      ref_ground_win=ref_win, datum_offset=offset)
            except Exception as e:
                skipped.append((flight.name, rname, f"{type(e).__name__}: {e}")); continue
            if dfx.empty:
                skipped.append((flight.name, rname, "0 plots passed coverage gate")); continue
            dfx.insert(0, "flight", flight.name)
            cov = len(dfx) / max(len(grid), 1)
            log(f"  ok   {flight.name}/{rname}   coverage {cov:.0%}   plots {len(dfx)}")
            results.append({"flight": flight.name, "raster": rname, "df": dfx,
                            "is_height": is_height})

    for fname, rname, reason in skipped:
        log(f"  skip {fname}/{rname}  ({reason})")

    combined = {}
    for rname, is_height in ((DSM_NAME, True), (ORTHO_NAME, False)):
        sub = [r["df"] for r in results if r["raster"] == rname]
        if not sub:
            continue
        allrows = pd.concat(sub, ignore_index=True)
        out_csv = outdir / f"all_{Path(rname).stem}_plot_data.csv"
        allrows.to_csv(out_csv, index=False)
        combined["dsm" if is_height else "orthophoto"] = allrows
        log(f"saved {out_csv}")

    return combined
