"""Field Plot Analyzer v1 — reference ground surface (formerly "Cell A").

Run once per field, before batch extraction. Produces a fixed ground datum
from bare soil NEAR the experiment so canopy-height referencing does not
drift as canopy closes over the season (see module-level rationale below).

WHY A FIXED REFERENCE INSTEAD OF PER-PLOT 5TH-PERCENTILE BASELINES
--------------------------------------------------------------------------
Per-plot p5-of-DSM baselines are biased once canopy cover increases: the
5th-percentile pixel shifts from actual soil to the lowest leaves, and that
shift correlates with the treatment effect you are trying to measure. A
fixed reference ground surface, established once from bare soil and
carried forward with only a per-flight scalar datum offset, has no such
correlation — soil does not move more than a few cm over a season.

SHIPPED dtm.tif IS NOT TRUSTED
--------------------------------------------------------------------------
Pipeline-generated dtm.tif files (see reconstruction/odm_light_cuda_v11's
_rasters(), a morphological opening of the DSM) disagreed with observed
bare soil by more than a meter on test flights — they are interpolated
surfaces, not observed ground, and are only used here as a QC cross-check,
never as the reference itself.

WHAT THE FILTER CHAIN REMOVES, IN ORDER
--------------------------------------------------------------------------
  1. distance band (NEAR_M)     -- drop bare-soil samples far from the plot
                                    block; distant terrain has no business
                                    informing ground under the plots.
  2. ExG-Otsu vegetation reject -- takes out green vegetation.
  3. cell-wise low-percentile   -- CELL_M x CELL_M cells, CELL_Q-th
     seeds                         percentile taken as a per-cell ground
                                    floor immune to isolated weeds.
  4. MAX_DEV_UP / MAX_DEV_DOWN  -- drop pixels above their cell floor
                                    (weeds/volunteers) or below it
                                    (ruts / reconstruction pits).
  5. roughness window           -- kills residual speckle.
  6. blob filter (MIN_BLOB_M2)  -- removes isolated bare-soil components
                                    below a size threshold.
  7. robust polynomial trend    -- degree-2 fit with sigma-clipped
                                    residuals; the smoothed residual field
                                    captures real (non-treatment) terrain
                                    undulation the trend missed.

The printed bare_frac_of_outside diagnostic tells you how aggressive this
was: below ~40% the thresholds are usually too tight for the field's soil
texture, and ROUGH_MAX is normally the first knob to loosen.

Ground *under* the plots still comes from the earliest (<5% cover) flight's
own DSM inside the plot polygons, not from extrapolating the perimeter fit
inward — with multi-metre plots that would be a long, untrustworthy
extrapolation. The bare-soil-derived surface does co-registration, QC, and
gap-fill for the perimeter; ref_extrap_frac in the output CSV flags any
plot where the assumption had to stretch further than usual.
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import (binary_dilation, gaussian_filter, uniform_filter,
                           distance_transform_edt, label)

from .config import SUB, DSM_NAME

# ---------------- config ----------------
REF_RES       = 0.10      # m, resolution of the reference ground grid
MARGIN_M      = 20.0      # m, grid padding beyond the plot grid (bare-ground sampling area)
BUFFER_M      = 1.0       # m, keep-out buffer around plots when sampling bare ground
NEAR_M        = 15.0      # m, only sample bare ground within this distance of the plot block
MIN_BLOB_M2   = 2.0       # m^2, drop isolated bare-soil components smaller than this
N_REF_FLIGHTS = 2         # how many earliest flights to mosaic into the reference
GROUND_PREF   = ["dtm.tif", "dsm.tif"]   # cross-check candidates only, never trusted blindly

# --- bare-ground outlier filtering ---
CELL_M        = 2.0       # m, coarse cell for low-percentile ground seeds
CELL_Q        = 15        # percentile within each cell taken as "ground"
CELL_MIN_PX   = 30        # min valid px per cell for a usable seed
CELL_MAX_IQR  = 0.15      # m, cells rougher than this are rejected (veg / machinery)
MAX_DEV_UP    = 0.12      # m, pixel above its cell floor by more than this -> not ground
MAX_DEV_DOWN  = 0.25      # m, pixel below its cell floor by more than this -> not ground
ROUGH_WIN_PX  = 5
ROUGH_MAX     = 0.05      # m, local std above this -> speckle, rejected
MIN_BARE_PX   = 8000      # hard floor: below this, flag INSUFFICIENT DATA rather than
                          # trust a thin/noisy per-flight datum correction

REF_READY = False
REF_CRS = None
GT = None                 # reference-grid affine transform
REF_GROUND = None         # (H, W) float32 ground elevation, reference grid
REF_EXTRAP = None         # (H, W) bool, True where ground had to be extrapolated
_LAST_GOOD = {}           # per-field fallback: last well-aligned flight's plane correction


def _exg_otsu_vegetation_mask(rgb):
    """Excess-Green index thresholded by Otsu -> boolean vegetation mask,
    True where vegetated. rgb is (H, W, 3) float in any consistent scale."""
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    total = r + g + b + 1e-6
    exg = 2 * (g / total) - (r / total) - (b / total)
    finite = exg[np.isfinite(exg)]
    if finite.size == 0:
        return np.zeros(rgb.shape[:2], bool)
    hist, edges = np.histogram(finite, bins=256)
    p = hist.astype(np.float64) / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    sigma_b2 = np.nan_to_num(sigma_b2, nan=-1)
    thr = edges[np.argmax(sigma_b2)]
    return exg > thr


def _cellwise_ground_seed(dsm, valid, cell_px):
    """Per-CELL_M-cell CELL_Q-th percentile as a local ground floor,
    rejecting cells rougher than CELL_MAX_IQR or with too few valid px."""
    h, w = dsm.shape
    floor = np.full_like(dsm, np.nan)
    for r0 in range(0, h, cell_px):
        for c0 in range(0, w, cell_px):
            r1, c1 = min(r0 + cell_px, h), min(c0 + cell_px, w)
            sub = dsm[r0:r1, c0:c1]; vmask = valid[r0:r1, c0:c1]
            vals = sub[vmask]
            if vals.size < CELL_MIN_PX:
                continue
            q1, q3 = np.percentile(vals, [25, 75])
            if (q3 - q1) > CELL_MAX_IQR:
                continue   # too rough for this cell to be a clean ground seed
            floor[r0:r1, c0:c1] = np.percentile(vals, CELL_Q)
    return floor


def _robust_poly_ground(X, Y, Z, degree=2, sigma=3.0, n_iter=3):
    """Degree-2 (default) polynomial trend surface with iterative
    sigma-clipping of residuals, plus a smoothed residual field capturing
    real terrain undulation the trend alone would miss."""
    keep = np.isfinite(Z)
    for _ in range(n_iter):
        A = np.column_stack([X[keep] ** i * Y[keep] ** j
                             for i in range(degree + 1) for j in range(degree + 1 - i)])
        coeffs, *_ = np.linalg.lstsq(A, Z[keep], rcond=None)
        pred_all = sum(coeffs[k] * X ** i * Y ** j
                       for k, (i, j) in enumerate((i, j) for i in range(degree + 1)
                                                  for j in range(degree + 1 - i)))
        resid = Z - pred_all
        s = np.nanstd(resid[keep])
        keep = keep & (np.abs(resid) < sigma * s)
    return coeffs, pred_all, resid, keep


def build_reference_ground(root: Path, plot_grid_polygons, log=print) -> bool:
    """Build REF_GROUND / REF_EXTRAP / REF_CRS / GT from the earliest
    N_REF_FLIGHTS flights under `root`. Returns True and sets REF_READY on
    success; False (REF_READY left False) if bare-ground coverage is too
    thin to trust.
    """
    global REF_READY, REF_CRS, GT, REF_GROUND, REF_EXTRAP
    REF_READY = False
    root = Path(root)

    flights = sorted(p for p in root.iterdir()
                     if p.is_dir() and (p / SUB / DSM_NAME).exists())
    if not flights:
        log(f"no flights with {DSM_NAME} under {root}"); return False
    ref_flights = flights[:N_REF_FLIGHTS]
    log(f"reference ground: mosaicking {len(ref_flights)} earliest flights "
        f"({', '.join(f.name for f in ref_flights)})")

    with rasterio.open(ref_flights[0] / SUB / DSM_NAME) as src0:
        REF_CRS = src0.crs
        b = src0.bounds
    xmin = b.left - MARGIN_M; xmax = b.right + MARGIN_M
    ymax = b.top + MARGIN_M;  ymin = b.bottom - MARGIN_M
    W = int((xmax - xmin) / REF_RES); H = int((ymax - ymin) / REF_RES)
    GT = from_origin(xmin, ymax, REF_RES, REF_RES)

    plot_mask = rasterize([(p, 1) for p in plot_grid_polygons], out_shape=(H, W),
                          transform=GT, fill=0, dtype="uint8").astype(bool)
    plot_mask_buf = binary_dilation(plot_mask, iterations=int(BUFFER_M / REF_RES))
    dist_to_plots = distance_transform_edt(~plot_mask, sampling=(REF_RES, REF_RES))
    near_band = dist_to_plots <= NEAR_M

    acc = np.full((H, W), np.nan, np.float32)
    acc_n = np.zeros((H, W), np.int32)

    for i, flight in enumerate(ref_flights):
        with rasterio.open(flight / SUB / DSM_NAME) as src:
            dsm = np.full((H, W), np.nan, np.float32)
            reproject(source=rasterio.band(src, 1), destination=dsm,
                     src_transform=src.transform, src_crs=src.crs,
                     dst_transform=GT, dst_crs=REF_CRS,
                     resampling=Resampling.bilinear)
        ortho_path = flight / SUB / "orthophoto.tif"
        veg = np.zeros((H, W), bool)
        if ortho_path.exists():
            with rasterio.open(ortho_path) as osrc:
                rgb = np.zeros((H, W, 3), np.float32)
                for b_i in range(min(3, osrc.count)):
                    band = np.full((H, W), np.nan, np.float32)
                    reproject(source=rasterio.band(osrc, b_i + 1), destination=band,
                             src_transform=osrc.transform, src_crs=osrc.crs,
                             dst_transform=GT, dst_crs=REF_CRS,
                             resampling=Resampling.bilinear)
                    rgb[..., b_i] = band
            veg = _exg_otsu_vegetation_mask(rgb)

        valid = np.isfinite(dsm) & near_band & (~plot_mask_buf) & (~veg)
        floor = _cellwise_ground_seed(dsm, valid, int(round(CELL_M / REF_RES)))
        dev = dsm - floor
        ground = (valid & np.isfinite(floor) &
                 (dev <= MAX_DEV_UP) & (dev >= -MAX_DEV_DOWN))
        rough = uniform_filter(dsm.astype(np.float64), size=ROUGH_WIN_PX)
        local_std = np.sqrt(np.clip(
            uniform_filter(dsm.astype(np.float64) ** 2, size=ROUGH_WIN_PX) - rough ** 2, 0, None))
        ground &= local_std <= ROUGH_MAX

        lbl, n = label(ground)
        if n:
            sizes = np.bincount(lbl.ravel())
            small = np.isin(lbl, np.nonzero(sizes < MIN_BLOB_M2 / REF_RES ** 2)[0])
            ground &= ~small

        # base flight (<5% cover): also sample ground INSIDE the plots
        if i == 0:
            inside = plot_mask & np.isfinite(dsm) & (~veg)
            ground = ground | inside

        acc = np.where(ground, np.where(np.isnan(acc), dsm, acc), acc)
        acc_n += ground.astype(np.int32)

    bare_frac = float((acc_n > 0).mean())
    n_bare_px = int((acc_n > 0).sum())
    log(f"bare_frac_of_outside: {bare_frac:.1%}  ({n_bare_px} px)")
    if n_bare_px < MIN_BARE_PX:
        log(f"  ! only {n_bare_px} bare-ground px (< MIN_BARE_PX={MIN_BARE_PX}) "
            f"-- reference NOT built; loosen ROUGH_MAX/MAX_DEV_* or widen NEAR_M")
        return False

    Yg, Xg = np.mgrid[0:H, 0:W]
    Xu = xmin + (Xg + 0.5) * REF_RES
    Yu = ymax - (Yg + 0.5) * REF_RES
    Zg = np.where(acc_n > 0, acc, np.nan)
    coeffs, trend, resid, keep = _robust_poly_ground(Xu, Yu, Zg)
    resid_smooth = gaussian_filter(np.where(keep, resid, 0.0), sigma=3.0)
    REF_GROUND = (trend + resid_smooth).astype(np.float32)
    REF_EXTRAP = ~(acc_n > 0)

    # QC cross-check against the shipped ground rasters (never trusted, just reported)
    for name in GROUND_PREF:
        gp = ref_flights[0] / SUB / name
        if not gp.exists():
            continue
        with rasterio.open(gp) as gsrc:
            shipped = np.full((H, W), np.nan, np.float32)
            reproject(source=rasterio.band(gsrc, 1), destination=shipped,
                     src_transform=gsrc.transform, src_crs=gsrc.crs,
                     dst_transform=GT, dst_crs=REF_CRS, resampling=Resampling.bilinear)
        diff = shipped[acc_n > 0] - REF_GROUND[acc_n > 0]
        diff = diff[np.isfinite(diff)]
        if diff.size:
            log(f"{name} vs bare-ground fit: median {np.median(diff):+.3f} m  "
                f"(large -> {name} is not trustworthy ground; the bare-ground "
                f"fit is preferred)")

    REF_READY = True
    log(f"reference ground ready: {W}x{H} @ {REF_RES} m, "
        f"{100*(~REF_EXTRAP).mean():.0f}% observed (rest extrapolated)")
    return True


def flight_datum_offset(root: Path, flight, log=print):
    """Per-flight scalar datum offset: align this flight's own bare-ground
    pixels (same filter chain, evaluated on THIS flight's DSM) to
    REF_GROUND by a constant vertical shift (a plane fit is deliberately
    NOT used -- soil cannot tilt more than a few cm over a season, so a
    plane would just be fitting noise). Falls back to the last well-aligned
    flight's offset (_LAST_GOOD) when this flight's own bare-ground sample
    is too thin (e.g. canopy has closed over the sampling ring near harvest).
    """
    global _LAST_GOOD
    if not REF_READY:
        raise RuntimeError("call build_reference_ground() first")

    flight = Path(flight)
    with rasterio.open(flight / SUB / DSM_NAME) as src:
        H, W = REF_GROUND.shape
        dsm = np.full((H, W), np.nan, np.float32)
        reproject(source=rasterio.band(src, 1), destination=dsm,
                 src_transform=src.transform, src_crs=src.crs,
                 dst_transform=GT, dst_crs=REF_CRS, resampling=Resampling.bilinear)

    valid = np.isfinite(dsm) & np.isfinite(REF_GROUND) & (~REF_EXTRAP)
    n_px = int(valid.sum())
    key = flight.name
    if n_px < MIN_BARE_PX:
        if "_offset" in _LAST_GOOD:
            log(f"{key}: only {n_px} bare px (<{MIN_BARE_PX}) -- falling back "
                f"to last well-aligned flight's datum offset "
                f"({_LAST_GOOD['_offset']:+.3f} m carried forward)")
            return _LAST_GOOD["_offset"], "INSUFFICIENT DATA (thin sample, fallback used)"
        return None, "INSUFFICIENT DATA (thin sample, no fallback available)"

    diff = (dsm - REF_GROUND)[valid]
    offset = float(np.median(diff))
    mad = float(np.median(np.abs(diff - offset)))
    verdict = ("ok" if mad < 0.03 else
              "SUSPECT" if mad < 0.08 else "NOT ALIGNED")
    _LAST_GOOD["_offset"] = offset
    log(f"{key}: datum offset {offset:+.3f} m  MAD {mad:.3f} m  n={n_px}  [{verdict}]")
    return offset, verdict
