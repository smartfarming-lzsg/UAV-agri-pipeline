# -*- coding: utf-8 -*-
"""
ODM-light-cuda v11 — DJI crop survey -> ortho-grade sparse -> dense -> DSM/DTM/CHM
LightGlue (tiled + CLAHE, heading-aware) -> COLMAP DB -> prior-BA -> RTK georef
-> dense (CUDA) -> DSM/DTM/CHM GeoTIFFs (true UTM).

═══════════════════════════════════════════════════════════════════════════
WHAT CHANGED IN v11  (all of it targets doming on low-texture / bare-soil blocks)
═══════════════════════════════════════════════════════════════════════════
Diagnosis that motivated this version (Zwiebel _008, 117 cams):
    Umeyama scale 1.00000  -> prior-BA WAS running (poses already metric)
    Z rms 0.471 m, Z max 1.098 m
    dz = +1.076e-03*r^2 ... , R^2 = 0.992, bowl +1.645 m over r_max 39.1 m
A radial fit that good is not noise. It is a classic focal/distortion bowl.

  1. use_prior_position=True on the mapper.  In v10 the pose_priors rows were
     written to the DB and audited, but IncrementalPipelineOptions was never
     told to use them (COLMAP defaults to False). Every BA inside the
     incremental loop was therefore pure photogrammetry on bare soil with
     self-calibrating distortion -> textbook doming conditions.

  2. coordinate_system = CARTESIAN (1), was UNDEFINED (-1).  The positions
     were always Cartesian metres (UTM minus centroid); only the label was
     wrong. Some prior-aware code paths reject or skip UNDEFINED.

  3. Anisotropic prior covariance, Z TIGHTER than XY (default 0.05 / 0.03 m,
     was isotropic 0.10). Doming is spatially correlated across the block
     while RTK vertical noise is not, so many cameras each pulling weakly
     toward their own height prior suppress a bowl far more effectively than
     the per-frame sigma alone would suggest.

  4. prior-BA may now refine focal length + extra params (radial distortion).
     v10 froze ALL intrinsics, so prior-BA could only drag poses onto the RTK
     while the bowl stayed baked into the structure -- exactly the 0.47 m
     residual observed. Letting the priors re-solve focal/distortion is the
     actual anti-doming mechanism. Principal point stays FIXED (weakly
     observable on bare soil; refining it lets BA hide error there).

  5. Robust loss on prior residuals, so one bad RTK epoch cannot lever the
     whole block while the rest pull honestly.

  6. prior-BA failures RAISE by default (prior_ba_strict=True). v10 swallowed
     every exception into a single grey log line and continued as if the
     geometry were constrained.

  7. Checkpoints carry a pipeline_version stamp. Anything built by v10 or
     earlier is auto-invalidated from the matching stage onward, because the
     old DB holds UNDEFINED priors with isotropic sigma and the old sfm/ holds
     domed poses. NOTE: force_dense=True is NOT sufficient for this -- it
     deliberately preserves the sparse checkpoints that must change.

  8. Doming is measured and logged automatically (before AND after prior-BA)
     so a regression is visible in the run log instead of in the ortho.
     Standalone diagnose_doming(src) re-checks any existing flight in seconds.

═══════════════════════════════════════════════════════════════════════════
LOCKED-INTRINSIC OPERATING MODE (recommended default; see wrapper scripts)
═══════════════════════════════════════════════════════════════════════════
Subsequent testing across the flight archive found that free intrinsic
refinement (even RTK-anchored, item 4 above) still lets BA trade focal length
against flying height on near-planar scenes with minimal camera-height
variation, inventing a DIFFERENT focal solution per flight (4074 / 3425 /
3808 / 2620 px on the same physical lens) while encoding a bowl in the
distortion terms. Passing camera_model="DJI_DEWARP" reads the factory
DewarpData calibration from XMP and holds fx/fy/cx/cy/k1/k2/p1/p2/k3 FIXED
across the whole block instead of self-calibrating them per flight. On the
same test flight this reduced Z rms from 0.469 m to 0.011 m -- a 43x
improvement over letting BA refine focal freely, RTK priors or not. RTK
priors alone (117 of them) cannot outvote ~600k reprojection observations;
they stabilise pose, but locking intrinsics is what removes the degeneracy
that lets a bowl form in the first place.

Recommended production call (see batch/run_pipeline_batch.py):
    run_pipeline(flight, camera_model="DJI_DEWARP",
                 ba_refine_focal_length=False, ba_refine_principal_point=False,
                 ba_refine_extra_params=False,      # locked: distortion is factory
                 prior_ba_refine_focal=True, prior_ba_refine_extra=True,
                 prior_ba_refine_pp=False,           # NEVER free the principal point
                 use_prior_position=True, prior_std_xy=0.05, prior_std_z=0.035,
                 force_sparse=True)

ba_refine_principal_point must stay False PERMANENTLY: freeing it let BA pull
focal length ~29% below the factory value with the worst vertical residuals
observed in the whole archive. prior_ba_refine_focal/extra stay True even
under DJI_DEWARP because prior-BA is RTK-anchored (unlike the mapper's
self-calibrating BA) -- letting it make a small, prior-constrained correction
to the locked starting point cut ~0.105 m of residual bowl on the bean flight
without reintroducing the per-flight focal drift.

Colab: Runtime -> Change runtime type -> T4 GPU BEFORE running.
Run CELL 1 once per session, then call run_pipeline(...) in CELL 2.
"""

# ═══════════════════════════════════════════════════════════════════════════
# CELL 1 — dependencies
# ═══════════════════════════════════════════════════════════════════════════
import importlib
import subprocess
import sys


def install_dependencies(cuda: bool = True, quiet: bool = True, force: bool = False):
    """Install non-preinstalled packages. Idempotent."""
    q = ["-q"] if quiet else []

    def pip_install(*pkgs):
        subprocess.run([sys.executable, "-m", "pip", "install", *q, *pkgs], check=False)

    def have(mod):
        return importlib.util.find_spec(mod) is not None

    if force or not have("lightglue"):
        print("installing LightGlue ...", flush=True)
        pip_install("git+https://github.com/cvg/LightGlue.git")

    need = [m for m in ("pyproj", "rasterio", "plyfile", "scipy", "psutil")
            if force or not have(m)]
    if need:
        print("installing", ", ".join(need), "...", flush=True)
        pip_install(*need)

    if force or not have("pycolmap"):
        print("installing pycolmap ...", flush=True)
        if cuda:
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *q,
                            "pycolmap"], check=False)
            pip_install("pycolmap-cuda")
        else:
            pip_install("pycolmap")
    print("dependencies ready.", flush=True)


def mount_drive(path="/content/drive"):
    """Mount Google Drive in Colab (no-op outside Colab)."""
    try:
        from google.colab import drive
        drive.mount(path)
    except Exception as e:
        print(f"drive mount skipped ({e})")


mount_drive()
install_dependencies()


# ═══════════════════════════════════════════════════════════════════════════
# CELL 2 — the pipeline
# ═══════════════════════════════════════════════════════════════════════════
import re
import math
import json
import time
import pickle
import shutil
import sqlite3
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import cv2
import torch
import pycolmap
from pyproj import Transformer
from PIL import Image as PilImage
from PIL.ExifTags import TAGS, GPSTAGS

PilImage.MAX_IMAGE_PIXELS = None
MAX_IMAGE_ID = 2 ** 31 - 1

# ── version stamp: bump this whenever a change invalidates old checkpoints ──
PIPELINE_VERSION = 11

# COLMAP PosePrior::CoordinateSystem enum
CS_UNDEFINED, CS_WGS84, CS_CARTESIAN = -1, 0, 1
_CS_NAME = {-1: "UNDEFINED", 0: "WGS84", 1: "CARTESIAN"}


# ── small stateless helpers ────────────────────────────────────────────────
def _dms(dms, ref):
    v = float(dms[0]) + float(dms[1]) / 60 + float(dms[2]) / 3600
    return -v if ref in ("S", "W") else v


def _get_gps(path):
    try:
        pil = PilImage.open(str(path)); exif = pil._getexif(); pil.close()
        if not exif:
            return None
        graw = next((v for t, v in exif.items() if TAGS.get(t) == "GPSInfo"), None)
        if not graw:
            return None
        g = {GPSTAGS.get(k, k): v for k, v in graw.items()}
        ld, lr = g.get("GPSLatitude"), g.get("GPSLatitudeRef", "N")
        nd, nr = g.get("GPSLongitude"), g.get("GPSLongitudeRef", "E")
        al = g.get("GPSAltitude", 0)
        if not ld or not nd:
            return None
        alt = float(al[0] / al[1]) if isinstance(al, tuple) else float(al)
        return _dms(ld, lr), _dms(nd, nr), alt
    except Exception:
        return None


def _get_gps_xmp(path):
    """FALLBACK GPS from DJI XMP (drone-dji:GpsLatitude/GpsLongitude/
    AbsoluteAltitude). Used ONLY when EXIF GPSInfo is absent, so it can never
    regress the EXIF-georeferenced frames but can rescue frames (e.g. an SD
    card / firmware where the second recording folder wrote RTK position to
    XMP but not EXIF). Handles the DJI firmware typo 'GpsLongtitude'."""
    try:
        raw = Path(path).read_bytes()
        s = raw.find(b"<x:xmpmeta"); e = raw.find(b"</x:xmpmeta")
        if s < 0 or e < 0:
            return None
        xmp = raw[s:e + 12].decode("utf-8", errors="ignore")

        def _f(*tags):
            for tag in tags:
                for pat in (rf'drone-dji:{tag}="([^"]+)"',
                            rf"<drone-dji:{tag}>([^<]+)</drone-dji:{tag}>"):
                    m = re.search(pat, xmp)
                    if m:
                        try:
                            return float(m.group(1))
                        except Exception:
                            pass
            return None

        lat = _f("GpsLatitude", "Latitude")
        lon = _f("GpsLongitude", "GpsLongtitude", "Longitude")
        alt = _f("AbsoluteAltitude", "RtkStdHgt")
        if lat is None or lon is None:
            return None
        return lat, lon, (alt if alt is not None else 0.0)
    except Exception:
        return None


def _xmp_text(path):
    """Raw XMP packet as text, or None."""
    try:
        raw = Path(path).read_bytes()
        s = raw.find(b"<x:xmpmeta"); e = raw.find(b"</x:xmpmeta")
        if s < 0 or e < 0:
            return None
        return raw[s:e + 12].decode("utf-8", errors="ignore")
    except Exception:
        return None


def _xmp_float(xmp, *tags):
    if not xmp:
        return None
    for tag in tags:
        for pat in (rf'drone-dji:{tag}="([^"]+)"',
                    rf"<drone-dji:{tag}>([^<]+)</drone-dji:{tag}>"):
            m = re.search(pat, xmp)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    pass
    return None


def _get_heading(path):
    return _xmp_float(_xmp_text(path), "FlightYawDegree", "GimbalYawDegree")


def _get_pitch(path):
    """GimbalPitchDegree from XMP. Nadir ~ -90 deg; obliques sit ~-45 to -60."""
    return _xmp_float(_xmp_text(path), "GimbalPitchDegree")


def _get_rtk_std(path):
    """Per-image RTK sigma (metres) + fix flag from DJI XMP.
    Returns dict(std_lon, std_lat, std_hgt, flag) with None where absent.
    v11 uses this only to REPORT prior quality; the covariance actually
    written is the configured prior_std_xy/prior_std_z (see _prior_cov)."""
    xmp = _xmp_text(path)
    return {"std_lon": _xmp_float(xmp, "RtkStdLon"),
            "std_lat": _xmp_float(xmp, "RtkStdLat"),
            "std_hgt": _xmp_float(xmp, "RtkStdHgt"),
            "flag": _xmp_float(xmp, "RtkFlag")}


def _get_dewarp(path):
    """Parse DJI factory calibration from XMP.
    DewarpData: "<date>;fx,fy,cx,cy,k1,k2,p1,p2,k3"
    cx,cy are principal-point OFFSET from image centre (DJI convention).
    """
    try:
        xmp = _xmp_text(path)
        if not xmp:
            return None
        m = (re.search(r'drone-dji:DewarpData="([^"]+)"', xmp)
             or re.search(r"<drone-dji:DewarpData>([^<]+)</drone-dji:DewarpData>", xmp))
        if not m:
            return None
        payload = m.group(1)
        if ";" in payload:
            payload = payload.split(";", 1)[1]
        nums = [float(x) for x in re.split(r"[,\s]+", payload.strip()) if x]
        if len(nums) < 9:
            return None
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = nums[:9]
        return dict(fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, p1=p1, p2=p2, k3=k3)
    except Exception:
        return None


_COLMAP_MODEL = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2,
                 "RADIAL": 3, "OPENCV": 4, "FULL_OPENCV": 6}


def _build_camera(camera_model, W0, H0, focal_px, dewarp, log):
    cx0, cy0 = W0 / 2.0, H0 / 2.0

    if camera_model == "DJI_DEWARP":
        if dewarp is None:
            log("DewarpData not found -> falling back to self-calibrating OPENCV")
            camera_model = "OPENCV"
        else:
            fx, fy = dewarp["fx"], dewarp["fy"]
            pcx, pcy = cx0 + dewarp["cx"], cy0 + dewarp["cy"]
            log(f"DJI DewarpData (FIXED): fx={fx:.1f} fy={fy:.1f} "
                f"pp=({pcx:.1f},{pcy:.1f})  k1={dewarp['k1']:.4f} k2={dewarp['k2']:.4f} "
                f"p1={dewarp['p1']:.5f} p2={dewarp['p2']:.5f} k3={dewarp['k3']:.4f}")
            if not (0.3 * W0 < pcx < 0.7 * W0 and 0.3 * H0 < pcy < 0.7 * H0):
                log("  ! principal point looks off-centre - verify DewarpData "
                    "convention (offset vs absolute) before trusting this run")
            params = np.array([fx, fy, pcx, pcy,
                               dewarp["k1"], dewarp["k2"], dewarp["p1"], dewarp["p2"],
                               dewarp["k3"], 0.0, 0.0, 0.0], np.float64)
            return _COLMAP_MODEL["FULL_OPENCV"], params, 1

    if camera_model == "PINHOLE":
        return 1, np.array([focal_px, focal_px, cx0, cy0], np.float64), 1
    if camera_model == "SIMPLE_RADIAL":
        return 2, np.array([focal_px, cx0, cy0, 0.0], np.float64), 1
    if camera_model == "OPENCV":
        return 4, np.array([focal_px, focal_px, cx0, cy0, 0., 0., 0., 0.], np.float64), 1
    raise ValueError(f"unknown camera_model {camera_model!r} "
                     f"(use OPENCV / SIMPLE_RADIAL / PINHOLE / DJI_DEWARP)")


def _set(o, a, v):
    if o is not None and hasattr(o, a):
        setattr(o, a, v)
        return True
    return False


def _set_logged(obj, attr, val, seen, label):
    """_set() that records whether the attribute actually existed, so the run
    log shows which knobs this pycolmap build really honoured."""
    if _set(obj, attr, val):
        seen[f"{label}.{attr}"] = val
        return True
    return False


def _get_pose(im):
    cfw = im.cam_from_world
    return cfw() if callable(cfw) else cfw


def _center(im):
    try:
        return np.asarray(im.projection_center(), float)
    except Exception:
        return np.asarray(_get_pose(im).inverse().translation, float)


def _pidf(i, j):
    if i > j:
        i, j = j, i
    return i * MAX_IMAGE_ID + j


def _blob(a):
    return np.ascontiguousarray(a).tobytes()


def _umeyama(S, D):
    mS, mD = S.mean(0), D.mean(0); Sc, Dc = S - mS, D - mD
    U, d, Vt = np.linalg.svd((Dc.T @ Sc) / len(S))
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    s = (d * [1, 1, np.sign(np.linalg.det(U @ Vt))]).sum() / (Sc ** 2).sum() * len(S)
    return s, R, mD - s * R @ mS


def _db_image_names(db_path):
    """Names the checkpointed DB was built from (for staleness detection)."""
    try:
        con = sqlite3.connect(str(db_path)); cur = con.cursor()
        rows = cur.execute("SELECT name FROM images").fetchall()
        con.close()
        return sorted(r[0] for r in rows)
    except Exception:
        return None


_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
  model INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, params BLOB, prior_focal_length INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS images (image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
  name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL,
  prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL, prior_tx REAL, prior_ty REAL, prior_tz REAL,
  CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
  FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
CREATE TABLE IF NOT EXISTS keypoints (image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB, FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS descriptors (image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB, FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS matches (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB);
CREATE TABLE IF NOT EXISTS two_view_geometries (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB, config INTEGER NOT NULL, F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB);
CREATE TABLE IF NOT EXISTS pose_priors (image_id INTEGER PRIMARY KEY NOT NULL, position BLOB, coordinate_system INTEGER NOT NULL, position_covariance BLOB, FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
"""


# ═══════════════════════════════════════════════════════════════════════════
# prior helpers (v11)
# ═══════════════════════════════════════════════════════════════════════════
def _prior_cov(std_xy, std_z):
    """Anisotropic diagonal covariance in metres^2, matching utm_local units.

    Z is deliberately tightened relative to the true per-frame RTK vertical
    sigma. Justification: doming is a low-frequency bowl correlated across the
    WHOLE block, whereas RTK vertical noise is independent per epoch. N cameras
    each pulling weakly toward their own height average the noise down by
    ~sqrt(N) while pulling coherently against the bowl. Over-tightening is a
    real risk though -- if the post-BA Z rms reported below stays well above
    std_z, loosen it rather than forcing BA to chase noise into focal length.
    """
    return np.diag([float(std_xy) ** 2,
                    float(std_xy) ** 2,
                    float(std_z) ** 2]).astype(np.float64)


def _cartesian_cs():
    """CARTESIAN enum across pycolmap variants; falls back to the raw int."""
    try:
        return pycolmap.PosePrior.CoordinateSystem.CARTESIAN
    except AttributeError:
        pass
    try:
        return pycolmap.PosePriorCoordinateSystem.CARTESIAN
    except AttributeError:
        return CS_CARTESIAN


def _fit_doming(centers_aligned, priors_xyz):
    """Fit dz = a*r^2 + b*r + c against radial distance from block centre.

    A high R^2 here is the doming fingerprint: random vertical error does not
    organise itself radially. Returns dict or None if too few cameras.
    """
    P = np.asarray(centers_aligned, float)
    D = np.asarray(priors_xyz, float)
    if len(P) < 8:
        return None
    res = P - D
    ctr = D[:, :2].mean(0)
    r = np.linalg.norm(D[:, :2] - ctr, axis=1)
    if np.ptp(r) < 1e-6:
        return None
    a, b, c = np.polyfit(r, res[:, 2], 2)
    fit = np.polyval([a, b, c], r)
    ss_res = float(((res[:, 2] - fit) ** 2).sum())
    ss_tot = float(((res[:, 2] - res[:, 2].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "n": int(len(P)),
        "a": float(a), "b": float(b), "c": float(c),
        "r2": float(r2),
        "r_max": float(r.max()),
        "bowl_m": float(a * r.max() ** 2),
        "xy_rms": float(np.sqrt((res[:, :2] ** 2).sum(1).mean())),
        "z_rms": float(np.sqrt((res[:, 2] ** 2).mean())),
        "z_max": float(np.abs(res[:, 2]).max()),
    }


def _doming_report(model, utm_local, log, label, warn_bowl_m=0.10):
    """Align the model to the RTK priors and report the radial bowl.

    Called before AND after prior-BA so a regression shows up in the run log
    instead of silently in the ortho. Never raises: this is instrumentation.
    """
    try:
        Sc, Dc = [], []
        for i in model.reg_image_ids():
            nm = model.images[i].name
            if nm in utm_local:
                Sc.append(_center(model.images[i]))
                Dc.append(utm_local[nm])
        if len(Sc) < 8:
            log(f"doming [{label}]: only {len(Sc)} cams with priors - skipped")
            return None
        Sc, Dc = np.array(Sc), np.array(Dc)
        s, R, t = _umeyama(Sc, Dc)
        P = (s * (R @ Sc.T).T + t)
        d = _fit_doming(P, Dc)
        if d is None:
            log(f"doming [{label}]: fit not possible")
            return None
        d["scale"] = float(s)
        log(f"doming [{label}]: {d['n']} cams  scale {s:.5f}  "
            f"XY rms {d['xy_rms']:.3f} m  Z rms {d['z_rms']:.3f} m  "
            f"Z max |{d['z_max']:.3f}| m")
        log(f"doming [{label}]: bowl {d['bowl_m']:+.3f} m over r_max "
            f"{d['r_max']:.1f} m   R2 {d['r2']:.3f}")
        if d["r2"] > 0.4 and abs(d["bowl_m"]) > warn_bowl_m:
            log(f"  ! SIGNIFICANT DOMING at stage '{label}' "
                f"({d['bowl_m']:+.2f} m, R2 {d['r2']:.2f})")
        return d
    except Exception as e:
        log(f"doming [{label}] skipped: {type(e).__name__}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# THE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def run_pipeline(
    src,
    *,
    src2=None,                  # optional second folder (battery-swap split flight)
    # --- geodesy / matching ------------------------------------------------
    utm_epsg: int = 32632,
    min_overlap: float = 0.10,
    min_matches: int = 15,
    # --- RTK pose priors (v11) --------------------------------------------
    prior_std_xy: float = 0.05,        # m, horizontal 1-sigma
    prior_std_z: float = 0.03,         # m, vertical 1-sigma (tight on purpose)
    prior_std: float = None,           # DEPRECATED isotropic alias (v10)
    use_prior_position: bool = True,   # THE flag v10 never set
    use_robust_prior_loss: bool = True,
    prior_loss_scale: float = 0.30,    # m, robust-loss knee
    prior_ba_refine_focal: bool = True,   # anti-doming lever
    prior_ba_refine_extra: bool = True,   # anti-doming lever
    prior_ba_refine_pp: bool = False,     # keep False on bare soil
    prior_ba_strict: bool = True,         # raise instead of silent skip
    doming_report: bool = True,
    doming_warn_m: float = 0.10,
    # --- oblique frames ----------------------------------------------------
    use_oblique: bool = True,
    nadir_pitch_thresh: float = -80.0,
    # --- camera model ------------------------------------------------------
    camera_model: str = "OPENCV",
    # --- bundle-adjustment (mapper) ---------------------------------------
    ba_refine_focal_length: bool = True,
    ba_refine_principal_point: bool = False,
    ba_refine_extra_params: bool = True,
    # --- feature extraction ------------------------------------------------
    tile_r: int = 3,
    tile_c: int = 4,
    kp_per_tile: int = 700,
    clahe_clip: float = 3.0,
    detection_threshold: float = 0.0001,
    nms_radius: int = 3,
    fp16: bool = True,
    # --- camera intrinsics guess ------------------------------------------
    focal_mm: float = 24.0,
    sensor_width_mm: float = 36.0,
    gsd_fallback: float = 0.00317,
    forward_step_frac: float = 0.20,
    # --- dense / raster ------------------------------------------------------
    do_dense: bool = True,
    dense_max_image_size: int = 1000,
    dense_geom_consistency: bool = False,
    grid: float = 0.05,
    chm_max: float = 4.0,
    iqr_factor: float = 3.0,
    dtm_window_m: float = 1.0,
    # --- runtime / checkpointing ------------------------------------------
    work=Path("/content/proj"),
    resume: bool = True,
    force_dense: bool = False,
    force_sparse: bool = False,
    cache_features: bool = False,
    make_plots: bool = True,
):
    """
    Run the full survey -> CHM pipeline on one or two image folders.

    Parameters (v11 additions)
    --------------------------
    prior_std_xy, prior_std_z : float
        RTK prior 1-sigma in metres, written as an anisotropic diagonal
        covariance. Z is tighter than XY deliberately (see _prior_cov).
        Passing the deprecated `prior_std` overrides BOTH with one isotropic
        value, reproducing v10 weighting.
    use_prior_position : bool
        Makes the incremental mapper actually USE the pose_priors rows. This
        is the single most important change in v11; with it False the DB
        priors are inert during mapping and doming is expected on bare soil.
        A warning is logged if the installed pycolmap lacks the option.
    prior_ba_refine_focal / prior_ba_refine_extra : bool
        Let prior-BA re-solve focal length / radial distortion. This is what
        actually removes a bowl: with intrinsics frozen (v10 behaviour),
        prior-BA can only drag poses onto the RTK while the curvature stays
        in the structure. Principal point stays fixed by default.
    prior_ba_strict : bool
        Raise if prior-BA fails. v10 logged and continued, so an unconstrained
        reconstruction looked identical to a constrained one.
    camera_model : str
        "DJI_DEWARP" is the recommended production setting: locks intrinsics
        to the factory calibration instead of self-calibrating (see module
        docstring "LOCKED-INTRINSIC OPERATING MODE"). "OPENCV" self-calibrates
        (v10 default; kept for comparison / non-DJI imagery).
    force_sparse : bool
        Drop the matched/verified DB, sfm_raw/, sfm/ AND all dense artefacts,
        forcing a rebuild from the matching stage. Required after changing any
        prior setting -- force_dense alone preserves exactly the sparse
        checkpoints that need to change.

    Resume checkpoints (in <src>/pipeline_output/_checkpoints)
    ----------------------------------------------------------
    feats_canon.pkl     after SuperPoint (optional, cache_features=True)
    database.db         after LightGlue matching (BEFORE verify_matches)
    verified.flag       sentinel: DB has been through verify_matches
    sfm_raw/            raw incremental model, pre-BA
    sfm/                refined sparse model, post-BA
    fused.ply           dense fusion output
    image_manifest.json image names + pipeline_version (staleness guard)

    Checkpoints written by pipeline_version < 11 are invalidated automatically
    from the matching stage onward: their DB carries UNDEFINED priors with
    isotropic sigma and their sfm/ carries domed poses.
    """
    t_start = time.time()
    src = Path(src)
    work = Path(work)
    assert src.is_dir(), f"src is not a directory: {src}"

    out = src / "pipeline_output"; out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "_checkpoints"; ckpt.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def status(stage, **extra):
        rec = {"stage": stage, "version": PIPELINE_VERSION,
               "time": time.strftime("%Y-%m-%d %H:%M:%S"),
               "elapsed_s": round(time.time() - t_start, 1), **extra}
        (out / "status.json").write_text(json.dumps(rec, indent=2))
        log(f"STAGE DONE: {stage}   {extra if extra else ''}")

    # deprecated isotropic alias
    if prior_std is not None:
        log(f"! prior_std={prior_std} is DEPRECATED (v10 isotropic weighting); "
            f"overriding prior_std_xy/prior_std_z with it")
        prior_std_xy = prior_std_z = float(prior_std)

    log(f"odm_light_cuda v{PIPELINE_VERSION}")
    log(f"src    : {src}")
    log(f"output : {out}")
    log(f"work   : {work}")
    log(f"device : {dev}  |  pycolmap {pycolmap.__version__}")
    log(f"priors : sigma_xy={prior_std_xy:.3f} m  sigma_z={prior_std_z:.3f} m  "
        f"use_prior_position={use_prior_position}  robust={use_robust_prior_loss}")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True

    eff_refine_extra = False if camera_model == "DJI_DEWARP" else ba_refine_extra_params
    eff_refine_focal = ba_refine_focal_length
    if camera_model == "DJI_DEWARP":
        log("camera_model=DJI_DEWARP -> distortion held FIXED at factory values")

    # ── STAGE 0: build image root (multi-folder merge, collision-safe) ─────
    origin_of = {}
    src2 = Path(src2) if src2 is not None else None
    if src2 is not None:
        assert src2.is_dir(), f"src2 is not a directory: {src2}"
        log(f"src2   : {src2}  (battery-swap merge)")
        merged_dir = work / "images_merged"
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        merged_dir.mkdir(parents=True)

        def _link_or_copy(p, dst):
            try:
                dst.symlink_to(p.resolve())
            except OSError:           # Google Drive doesn't support symlinks
                shutil.copy2(p, dst)

        src_jpgs = sorted(p for p in src.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg"))
        src2_jpgs = sorted(p for p in src2.iterdir()
                           if p.suffix.lower() in (".jpg", ".jpeg"))

        for p in src_jpgs:
            _link_or_copy(p, merged_dir / p.name)
            origin_of[p.name] = "src"
        n_collide = 0
        for p in src2_jpgs:
            name = p.name
            if (merged_dir / name).exists():
                name = "f2_" + p.name
                n_collide += 1
            _link_or_copy(p, merged_dir / name)
            origin_of[name] = "src2"
        images = merged_dir
        n_merged = sum(1 for q in merged_dir.iterdir()
                       if q.suffix.lower() in (".jpg", ".jpeg"))
        log(f"merged dir: {len(src_jpgs)} (src) + {len(src2_jpgs)} (src2) "
            f"= {n_merged} JPGs  [{n_collide} src2 collisions renamed f2_*]")
        assert n_merged == len(src_jpgs) + len(src2_jpgs), (
            f"merge lost frames: got {n_merged}, expected "
            f"{len(src_jpgs) + len(src2_jpgs)}")
    else:
        images = src

    jpgs = sorted(p for p in images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    assert len(jpgs) >= 5, f"need >=5 JPGs, found {len(jpgs)}"
    log(f"{len(jpgs)} JPGs found")

    recs = []
    no_gps = []
    for p in jpgs:
        g = _get_gps(p) or _get_gps_xmp(p)      # EXIF first, XMP-RTK fallback
        if g is None:
            no_gps.append(p.name)
            continue
        recs.append(dict(name=p.name, path=p, lat=g[0], lon=g[1], alt=g[2],
                         heading=_get_heading(p), pitch=_get_pitch(p)))
    assert len(recs) >= 5, f"only {len(recs)} images carried GPS"

    if origin_of:
        got = Counter(origin_of.get(r["name"], "src") for r in recs)
        tot = Counter(origin_of.values())
        log(f"GPS yield  src {got['src']}/{tot['src']}  "
            f"src2 {got['src2']}/{tot['src2']}")
        if no_gps:
            miss = Counter(origin_of.get(n, "src") for n in no_gps)
            log(f"  ! {len(no_gps)} frames without EXIF/XMP GPS "
                f"(src={miss['src']} src2={miss['src2']}); first: {no_gps[:5]}")
    elif no_gps:
        log(f"  ! {len(no_gps)} frames without EXIF/XMP GPS; first: {no_gps[:5]}")

    # ── RTK quality report (informational; does not change the covariance) ──
    _report_rtk_quality(recs, prior_std_xy, prior_std_z, log)

    for r in recs:
        r["oblique"] = (r["pitch"] is not None and r["pitch"] > nadir_pitch_thresh)
    if not use_oblique:
        recs = [r for r in recs if not r["oblique"]]
    n_obl = sum(r["oblique"] for r in recs)
    log(f"{len(recs)} images with GPS  ({len(recs)-n_obl} nadir, {n_obl} oblique)")
    if n_obl and not use_oblique:
        log("use_oblique=False -> oblique frames dropped")
    if n_obl == 0 and use_oblique:
        log("  ! NO oblique frames detected. Obliques are a primary anti-doming "
            "constraint (they break the focal/height ambiguity a pure nadir "
            "block cannot). Expect to lean entirely on the RTK priors here.")

    lat0, lon0 = recs[0]["lat"], recs[0]["lon"]
    cl = np.cos(np.radians(lat0))
    for r in recs:
        r["E"] = (r["lon"] - lon0) * 111320 * cl
        r["N"] = (r["lat"] - lat0) * 110540

    # ── STAGE 1: survey geometry + match graph ──────────────────────────────
    img0 = cv2.imread(str(recs[0]["path"])); H0, W0 = img0.shape[:2]; del img0
    nad = [r for r in recs if not r["oblique"]]
    assert len(nad) >= 5, "need >=5 nadir frames to define the survey frame"
    heads = [r["heading"] for r in nad if r["heading"] is not None]

    steps = []
    for i in range(len(nad) - 1):
        hi, hj = nad[i]["heading"] or 0.0, nad[i + 1]["heading"] or 0.0
        dh = abs(hi - hj) % 360; dh = min(dh, 360 - dh)
        if dh < 20:
            d = math.hypot(nad[i + 1]["E"] - nad[i]["E"],
                           nad[i + 1]["N"] - nad[i]["N"])
            if 0.1 < d < 20:
                steps.append(d)
    gsd = (np.percentile(steps, 25) / forward_step_frac / H0) if steps else gsd_fallback
    fp_h, fp_w = H0 * gsd, W0 * gsd

    h_rad2 = np.radians(2 * np.array(heads))
    A = 0.5 * math.atan2(np.mean(np.sin(h_rad2)), np.mean(np.cos(h_rad2)))
    along = np.array([math.sin(A), math.cos(A)])
    cross = np.array([math.cos(A), -math.sin(A)])
    for r in recs:
        r["at"] = r["E"] * along[0] + r["N"] * along[1]
        r["ct"] = r["E"] * cross[0] + r["N"] * cross[1]

    ct = np.array([r["ct"] for r in nad])
    bins = np.round((ct - ct.min()) / (fp_w * 0.25)).astype(int)
    centers = sorted(float(np.median(ct[bins == b])) for b in np.unique(bins)
                     if (bins == b).sum() >= 2)
    row_sp = float(np.median(np.diff(centers))) if len(centers) >= 2 else float("nan")

    log(f"axis azimuth {math.degrees(A):.1f} deg  |  GSD {gsd*100:.3f} cm/px")
    log(f"footprint along {fp_h:.2f} m  cross {fp_w:.2f} m  |  row spacing {row_sp:.2f} m")

    at_thr = fp_h * (1 - min_overlap)
    ct_thr = fp_w * (1 - min_overlap)
    pairs, deg = [], {r["name"]: 0 for r in recs}
    for i in range(len(nad)):
        for j in range(i + 1, len(nad)):
            if (abs(nad[i]["at"] - nad[j]["at"]) < at_thr and
                    abs(nad[i]["ct"] - nad[j]["ct"]) < ct_thr):
                pairs.append((nad[i]["name"], nad[j]["name"]))
                deg[nad[i]["name"]] += 1; deg[nad[j]["name"]] += 1

    obl = [r for r in recs if r["oblique"]]
    n_obl_pairs = 0
    if obl:
        existing = set(pairs)
        for k, o in enumerate(obl):
            for r in nad + obl[k + 1:]:
                if r["name"] == o["name"]:
                    continue
                pr = tuple(sorted((o["name"], r["name"])))
                if pr in existing:
                    continue
                existing.add(pr); pairs.append(pr)
                deg[pr[0]] += 1; deg[pr[1]] += 1
                n_obl_pairs += 1
        log(f"oblique pairs added: {n_obl_pairs} (will be geometrically verified)")

    log(f"candidate pairs {len(pairs)} (exhaustive {len(recs)*(len(recs)-1)//2}) | "
        f"neighbours min {min(deg.values())} med {int(np.median(list(deg.values())))} "
        f"max {max(deg.values())}")

    if origin_of:
        n_bridge = sum(1 for a, b in pairs
                       if origin_of.get(a, "src") != origin_of.get(b, "src"))
        log(f"cross-flight candidate pairs (src<->src2): {n_bridge}")
        if n_bridge == 0:
            log("  ! NO candidate pairs bridge the two flights - they will form "
                "separate submodels and src2 will be DROPPED. Increase overlap "
                "(min_overlap up) or check the two flights actually overlap.")

    if make_plots:
        oblique_names = {r["name"] for r in obl}
        _plot_match_graph(recs, pairs, fp_w, fp_h, A, out / "match_graph.jpg",
                          oblique_names)

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    utm = {r["name"]: np.array([*tr.transform(r["lon"], r["lat"]), r["alt"]], float)
           for r in recs}
    georef_origin = np.mean(list(utm.values()), axis=0)
    utm_local = {k: v - georef_origin for k, v in utm.items()}
    cov = _prior_cov(prior_std_xy, prior_std_z)
    name2id = {r["name"]: i for i, r in enumerate(recs, start=1)}

    DB = work / "database.db"
    SFM_RAW = work / "sfm_raw"
    SFM_DIR = work / "sfm"

    ck_db = ckpt / "database.db"
    ck_verified = ckpt / "verified.flag"
    ck_sfm = ckpt / "sfm"
    ck_sfm_raw = ckpt / "sfm_raw"
    ck_fused = ckpt / "fused.ply"
    ck_feats = ckpt / "feats_canon.pkl"
    ck_manifest = ckpt / "image_manifest.json"

    def _invalidate(reason):
        log(f"! {reason} -> invalidating extract/match/sfm/dense checkpoints")
        for pth in (ck_db, ck_verified, ck_feats, ck_fused):
            if pth.exists():
                pth.unlink()
        for d in (ck_sfm, ck_sfm_raw):
            if d.exists():
                shutil.rmtree(d)
        for stale_p in (out / "depth_bundle.npz", out / "dense_cloud_utm.ply"):
            if stale_p.exists():
                stale_p.unlink()

    # ── checkpoint staleness guard (image set AND pipeline version) ─────────
    cur_names = sorted(r["name"] for r in recs)
    if force_sparse:
        _invalidate("force_sparse=True")
    elif resume:
        prev_names, prev_ver = None, None
        if ck_manifest.exists():
            try:
                man = json.loads(ck_manifest.read_text())
                prev_names = man.get("names")
                prev_ver = man.get("pipeline_version")
            except Exception:
                prev_names, prev_ver = None, None
        if prev_names is None and ck_db.exists():
            prev_names = _db_image_names(ck_db)   # legacy ckpt, no manifest

        have_any_ckpt = ck_db.exists() or ck_sfm.exists() or ck_sfm_raw.exists()

        if prev_names is not None and prev_names != cur_names:
            added = sorted(set(cur_names) - set(prev_names))
            removed = sorted(set(prev_names) - set(cur_names))
            _invalidate(f"checkpoint image set changed (prev {len(prev_names)} -> "
                        f"now {len(cur_names)}; +{len(added)} -{len(removed)})")
        elif have_any_ckpt and (prev_ver is None or prev_ver < PIPELINE_VERSION):
            _invalidate(
                f"checkpoints built by pipeline_version "
                f"{prev_ver if prev_ver is not None else '<11 (unstamped)'} "
                f"< {PIPELINE_VERSION}; their DB holds UNDEFINED/isotropic "
                f"priors and their sfm/ holds domed poses")
        elif prev_names is None and ck_db.exists():
            log("! could not read checkpoint manifest/DB names - cannot verify "
                "the checkpointed image set matches; proceeding with resume")

    # ── forced dense rebuild ────────────────────────────────────────────────
    # Drops ONLY the dense-derived artefacts. Sparse checkpoints survive, so
    # mapping + prior-BA are still resumed and the cost is dense alone. NOTE:
    # this is NOT enough after changing any prior setting - use force_sparse.
    if force_dense:
        if ck_fused.exists():
            n_old = _ply_point_count(ck_fused)
            ck_fused.unlink()
            log(f"force_dense -> dropped fused.ply checkpoint ({n_old:,} pts)")
        dense_ws = work / "dense"
        if dense_ws.exists():
            shutil.rmtree(dense_ws)
            log(f"force_dense -> wiped scratch dense workspace {dense_ws}")
        for stale_p in (out / "depth_bundle.npz", out / "dense_cloud_utm.ply",
                        out / "dsm.tif", out / "dtm.tif", out / "chm.tif"):
            if stale_p.exists():
                stale_p.unlink()
                log(f"force_dense -> dropped {stale_p.name}")

    # Three-level DB resume state (computed AFTER staleness invalidation)
    have_db_matched  = resume and ck_db.exists() and not ck_verified.exists()
    have_db_verified = resume and ck_db.exists() and ck_verified.exists()
    have_sfm_raw     = resume and (ck_sfm_raw / "cameras.bin").exists()
    have_sfm         = resume and (ck_sfm / "cameras.bin").exists()
    have_fused = resume and ck_fused.exists() and _ply_point_count(ck_fused) > 0
    model = None

    def _write_manifest():
        ck_manifest.write_text(json.dumps({
            "pipeline_version": PIPELINE_VERSION,
            "n": len(cur_names),
            "prior_std_xy": prior_std_xy,
            "prior_std_z": prior_std_z,
            "use_prior_position": bool(use_prior_position),
            "names": cur_names,
        }))

    # ── STAGE 2: features + matching (skip if any DB or later ckpt exists) ──
    if not (have_sfm or have_sfm_raw or have_db_verified or have_db_matched):
        feats_canon = _extract_features(
            recs, dev, tile_r, tile_c, kp_per_tile, clahe_clip,
            detection_threshold, nms_radius, fp16, log,
            cache_path=(ck_feats if cache_features and ck_feats.exists() else None))
        if cache_features and not ck_feats.exists():
            _save_feats(feats_canon, ck_feats)
            log(f"cached features -> {ck_feats}")

        focal_px = W0 * focal_mm / sensor_width_mm
        dewarp = None
        if camera_model == "DJI_DEWARP":
            dewarp = next((dw for dw in (_get_dewarp(r["path"]) for r in recs)
                           if dw is not None), None)
        cam_model_id, cam_params, cam_prior = _build_camera(
            camera_model, W0, H0, focal_px, dewarp, log)

        _build_db_and_match(
            DB, recs, feats_canon, pairs, utm_local, cov, name2id,
            W0, H0, cam_model_id, cam_params, cam_prior, min_matches, fp16, dev, log)

        _audit_db_priors(DB, log)

        shutil.copy(DB, ck_db)
        _write_manifest()
        log(f"matched DB checkpointed -> {ck_db}")
        have_db_matched = True   # fall through to verify below

    # ── STAGE 3: verify_matches (skip only if verified flag present) ─────────
    if not (have_sfm or have_sfm_raw or have_db_verified):
        if have_db_matched and not DB.exists():
            shutil.copy(ck_db, DB)
            log(f"resumed matched DB <- {ck_db}")

        PT = work / "pairs.txt"
        PT.write_text("\n".join(f"{a} {b}" for a, b in pairs))
        log(f"verify_matches on {len(pairs)} pairs ...")
        pycolmap.verify_matches(str(DB), str(PT))
        log("verify_matches ok")

        shutil.copy(DB, ck_db)
        ck_verified.touch()
        _write_manifest()
        status("verify_matches", checkpoint=str(ck_db))
        have_db_verified = True

    elif have_db_verified and not (have_sfm or have_sfm_raw):
        shutil.copy(ck_db, DB)
        log(f"resumed verified DB <- {ck_db}")

    # ── STAGE 4: incremental mapping + prior-BA ──────────────────────────────
    doming_pre = doming_post = None
    if have_sfm:
        model = pycolmap.Reconstruction(str(ck_sfm))
        log(f"resumed refined sparse model <- {ck_sfm} "
            f"({model.num_reg_images()} imgs, {model.num_points3D()} pts)")
        # Materialise the checkpoint into the scratch sparse dir, overwriting
        # anything already there. NEVER trust an existing SFM_DIR: if `work`
        # is reused, undistort_images would build the dense workspace from
        # another flight's poses.
        if SFM_DIR.exists():
            shutil.rmtree(SFM_DIR)
        SFM_DIR.mkdir(parents=True)
        model.write(SFM_DIR)
        if doming_report:
            doming_post = _doming_report(model, utm_local, log, "resumed sfm",
                                         doming_warn_m)
    else:
        if have_sfm_raw:
            model = pycolmap.Reconstruction(str(ck_sfm_raw))
            log(f"resumed raw model <- {ck_sfm_raw}; re-running prior-BA")
        else:
            model = _incremental_map(
                DB, images, SFM_RAW, min_matches,
                eff_refine_focal, ba_refine_principal_point,
                eff_refine_extra, log, origin_of=origin_of,
                use_prior_position=use_prior_position,
                use_robust_prior_loss=use_robust_prior_loss,
                prior_loss_scale=prior_loss_scale)
            if SFM_RAW.exists():
                _copytree(SFM_RAW, ck_sfm_raw)
            status("incremental_mapping",
                   registered=f"{model.num_reg_images()}/{len(name2id)}",
                   points=model.num_points3D())

        if doming_report:
            doming_pre = _doming_report(model, utm_local, log, "pre prior-BA",
                                        doming_warn_m)

        _prior_bundle_adjust(
            model, name2id, utm_local, cov, log,
            refine_focal=prior_ba_refine_focal,
            refine_pp=prior_ba_refine_pp,
            refine_extra=prior_ba_refine_extra,
            use_robust_prior_loss=use_robust_prior_loss,
            prior_loss_scale=prior_loss_scale,
            strict=prior_ba_strict)

        if doming_report:
            doming_post = _doming_report(model, utm_local, log, "post prior-BA",
                                         doming_warn_m)
            if doming_pre and doming_post:
                b0, b1 = doming_pre["bowl_m"], doming_post["bowl_m"]
                log(f"doming: bowl {b0:+.3f} m -> {b1:+.3f} m "
                    f"({abs(b1) - abs(b0):+.3f} m change)")
                if abs(b1) > abs(b0) + 0.01:
                    log("  ! prior-BA made doming WORSE. Check that "
                        "prior_std_z is not tighter than the true RTK vertical "
                        "sigma, and that obliques are present.")

        if SFM_DIR.exists():
            shutil.rmtree(SFM_DIR)
        SFM_DIR.mkdir(parents=True)
        model.write(SFM_DIR)
        _copytree(SFM_DIR, ck_sfm)
        _write_manifest()
        status("sparse_model", checkpoint=str(ck_sfm),
               bowl_m=(doming_post or {}).get("bowl_m"))

    # ── prior coverage + per-folder registration (ALL resume paths) ─────────
    reg_names = [model.images[i].name for i in model.reg_image_ids()]
    without_prior = [n for n in reg_names if n not in utm_local]
    log(f"RTK prior coverage: {len(reg_names) - len(without_prior)}/{len(reg_names)} "
        f"registered frames have priors")
    if without_prior:
        log(f"  ! registered WITHOUT prior: {without_prior[:8]}"
            + (" ..." if len(without_prior) > 8 else ""))
    if origin_of:
        reg_by = Counter(origin_of.get(n, "src") for n in reg_names)
        tot_by = Counter(origin_of.values())
        log(f"registered by folder: src {reg_by.get('src', 0)}/{tot_by.get('src', 0)}  "
            f"src2 {reg_by.get('src2', 0)}/{tot_by.get('src2', 0)}")
        if reg_by.get("src2", 0) < tot_by.get("src2", 0):
            log("  ! not all src2 frames registered - check the cross-flight "
                "pairing / submodel warnings above")

    # ── STAGE 5: georef ──────────────────────────────────────────────────────
    reg_ids = model.reg_image_ids()
    src_c, dst_c, matched_names = [], [], []
    for i in reg_ids:
        nm = model.images[i].name
        if nm in utm_local:
            src_c.append(_center(model.images[i]))
            dst_c.append(utm_local[nm])
            matched_names.append(nm)
    n_reg, n_match = len(reg_ids), len(matched_names)
    log(f"georef: {n_match}/{n_reg} registered images have an RTK prior match")

    if n_reg == 0:
        raise RuntimeError(
            "georef failed: model has 0 registered images - incremental_mapping "
            "registered nothing for this flight. Check the mapping stage log "
            "above for submodel sizes.")
    if n_match == 0:
        sample_model = sorted({model.images[i].name for i in reg_ids})[:5]
        sample_prior = sorted(utm_local.keys())[:5]
        raise RuntimeError(
            f"georef failed: 0/{n_reg} registered names match any RTK prior. "
            f"model names e.g. {sample_model}  vs  prior names e.g. {sample_prior}. "
            f"Likely a resumed sfm checkpoint built under a different src/src2 "
            f"configuration than this run - delete ck_sfm/ck_sfm_raw under "
            f"{ckpt} to force a clean remap, or confirm src2 usage matches.")
    if n_match < 5:
        log(f"  ! only {n_match} matched images - umeyama poorly constrained")

    src_c, dst_c = np.array(src_c), np.array(dst_c)
    s, Rg, tg = _umeyama(src_c, dst_c)
    resid = np.linalg.norm((s * (Rg @ src_c.T).T + tg) - dst_c, axis=1)
    log(f"georef: scale {s:.4f}  median {np.median(resid):.3f} m  max {resid.max():.3f} m")

    def colmap_to_utm(X):
        return (s * (Rg @ np.asarray(X).reshape(-1, 3).T).T + tg) + georef_origin

    # ── STAGE 6: save ortho inputs ───────────────────────────────────────────
    _save_ortho_inputs(out / "ortho_inputs.npz", model, s, Rg, tg, georef_origin,
                       utm_epsg, images, colmap_to_utm, log)

    # ── STAGE 7: sparse views + cleaned cloud ────────────────────────────────
    try:
        model.export_PLY(str(work / "sparse.ply"))
        shutil.copy(work / "sparse.ply", out / "sparse.ply")
    except Exception:
        pass
    if make_plots:
        _sparse_views(model, colmap_to_utm, iqr_factor, out, work, log)

    products = {"out_dir": str(out),
                "pipeline_version": PIPELINE_VERSION,
                "georef_median_resid_m": float(np.median(resid))}
    if doming_post:
        products.update({"doming_bowl_m": doming_post["bowl_m"],
                         "doming_r2": doming_post["r2"],
                         "doming_z_rms_m": doming_post["z_rms"]})

    # ── STAGE 8: dense (CUDA) -> fused.ply ──────────────────────────────────
    fused = None
    if do_dense:
        if have_fused:
            fused = ck_fused
            log(f"resumed dense cloud <- {ck_fused}")
        else:
            fused = _dense(work, SFM_DIR, images, dense_max_image_size, log,
                           resume=resume, geom_consistency=dense_geom_consistency,
                           cache_size_gb=6.0,
                           fusion_max_image_size=800,
                           fusion_min_num_pixels=8)

            _save_depth_bundle(work / "dense", out, log)

            if fused.exists():
                shutil.copy(fused, ck_fused)
                status("dense", checkpoint=str(ck_fused))

    # ── STAGE 9: DSM / DTM / CHM GeoTIFFs ───────────────────────────────────
    if fused and Path(fused).exists():
        chm_stats = _rasters(fused, colmap_to_utm, utm_epsg, grid, chm_max,
                             dtm_window_m, work, out, make_plots, log)
        products.update(chm_stats)
        _save_dense_utm(fused, colmap_to_utm, out / "dense_cloud_utm.ply", log)
        status("rasters", **chm_stats)
    else:
        log("dense skipped -> no DSM/DTM/CHM (sparse products still written)")

    products.update({
        "sparse_ply": str(out / "sparse.ply"),
        "ortho_inputs": str(out / "ortho_inputs.npz"),
        "dsm": str(out / "dsm.tif"),
        "dtm": str(out / "dtm.tif"),
        "chm": str(out / "chm.tif"),
    })
    status("complete")
    log(f"DONE in {time.time() - t_start:.0f}s -> {out}")
    return products


# ═══════════════════════════════════════════════════════════════════════════
# stage implementations
# ═══════════════════════════════════════════════════════════════════════════
def _report_rtk_quality(recs, prior_std_xy, prior_std_z, log, sample=None):
    """Compare the configured prior sigma against the RTK engine's own XMP
    sigma. Purely informational -- but if the fleet sigma is much larger than
    prior_std_z you are asking BA to satisfy noise, which can push error into
    focal length instead of removing the bowl."""
    try:
        rs = recs if sample is None else recs[:sample]
        sx, sz, n_fixed, n_flag = [], [], 0, 0
        for r in rs:
            q = _get_rtk_std(r["path"])
            if q["std_lon"] is not None and q["std_lat"] is not None:
                sx.append(max(q["std_lon"], q["std_lat"]))
            if q["std_hgt"] is not None:
                sz.append(q["std_hgt"])
            if q["flag"] is not None:
                n_flag += 1
                if int(q["flag"]) == 50:
                    n_fixed += 1
        if not sz and not sx:
            log("RTK XMP sigma: not present in these images (using configured "
                "prior sigma as-is)")
            return
        msg = "RTK XMP sigma:"
        if sx:
            msg += f"  xy median {np.median(sx):.3f} m"
        if sz:
            msg += f"  z median {np.median(sz):.3f} m"
        if n_flag:
            msg += f"  |  RTK-fixed {n_fixed}/{n_flag}"
        log(msg)
        if sz and np.median(sz) > prior_std_z * 2.5:
            log(f"  ! configured prior_std_z={prior_std_z:.3f} m is much tighter "
                f"than the reported vertical sigma ({np.median(sz):.3f} m). "
                f"This is intentional for doming suppression (the bowl is "
                f"correlated, the noise is not) but if post-BA Z rms stays high, "
                f"loosen prior_std_z rather than forcing BA to chase noise.")
        if n_flag and n_fixed < n_flag:
            log(f"  ! {n_flag - n_fixed} frame(s) were NOT RTK-fixed; their "
                f"true position error is far above the configured sigma. The "
                f"robust loss on prior residuals is what protects the block "
                f"from these.")
    except Exception as e:
        log(f"RTK quality report skipped: {type(e).__name__}: {e}")


def _audit_db_priors(DB, log):
    """Verify every image row has a pose_prior row, and report the frame."""
    try:
        con = sqlite3.connect(str(DB)); cur = con.cursor()
        n_img = cur.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        n_pri = cur.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0]
        missing = cur.execute(
            "SELECT name FROM images WHERE image_id NOT IN "
            "(SELECT image_id FROM pose_priors) LIMIT 8").fetchall()
        cs_rows = cur.execute(
            "SELECT DISTINCT coordinate_system FROM pose_priors").fetchall()
        con.close()
        cs = sorted(r[0] for r in cs_rows)
        log(f"DB prior audit: {n_pri}/{n_img} images have a pose_prior row  "
            f"[coordinate_system: {', '.join(_CS_NAME.get(c, str(c)) for c in cs)}]")
        if n_pri != n_img:
            log(f"  ! {n_img - n_pri} images WITHOUT a prior, e.g. "
                f"{[m[0] for m in missing]}")
        if CS_UNDEFINED in cs:
            log("  ! UNDEFINED coordinate_system present - prior-aware stages "
                "may ignore these rows (this is the v10 bug)")
    except Exception as e:
        log(f"DB prior audit skipped: {e}")


def _extract_features(recs, dev, tile_r, tile_c, kp_per_tile, clahe_clip,
                      detection_threshold, nms_radius, fp16, log, cache_path=None):
    """Heading-aware tiled+CLAHE SuperPoint. De-rotates each image to a common
    north-up frame so LightGlue can match cross-strip pairs (>45 deg apart)."""
    if cache_path is not None:
        log(f"loading cached features <- {cache_path}")
        return _load_feats(cache_path, dev)

    from lightglue import SuperPoint
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    extractor = SuperPoint(max_num_keypoints=kp_per_tile,
                           detection_threshold=detection_threshold,
                           nms_radius=nms_radius).eval().to(dev)
    use_amp = fp16 and dev.type == "cuda"

    def extract_canon(path, heading_deg):
        bgr = cv2.imread(str(path)); h, w = bgr.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), -heading_deg, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += nw / 2 - cx; M[1, 2] += nh / 2 - cy
        rot = cv2.warpAffine(bgr, M, (nw, nh))
        Minv = cv2.invertAffineTransform(M)

        th, tw = nh // tile_r, nw // tile_c
        kps_r, descs = [], []
        for ri in range(tile_r):
            for ci in range(tile_c):
                r0, r1 = ri * th, (ri + 1) * th if ri < tile_r - 1 else nh
                c0, c1 = ci * tw, (ci + 1) * tw if ci < tile_c - 1 else nw
                tile = rot[r0:r1, c0:c1]
                if tile.size == 0:
                    continue
                enh = clahe.apply(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY))
                t = torch.from_numpy(enh).float().div(255.)[None, None].to(dev)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                     enabled=use_amp):
                    f = extractor.extract(t)
                kp = f["keypoints"][0].float().cpu().numpy() + [c0, r0]
                if len(kp) == 0:
                    continue
                kps_r.append(kp.astype(np.float32))
                descs.append(f["descriptors"][0].float())
        kp_rot = np.concatenate(kps_r, 0)
        desc = torch.cat(descs, 0)[None]
        ones = np.ones((len(kp_rot), 1), np.float32)
        kp_orig = (np.hstack([kp_rot, ones]) @ Minv.T).astype(np.float32)
        return kp_orig, kp_rot, desc

    feats = {}
    log(f"tiled+CLAHE SuperPoint, de-rotated ({tile_r}x{tile_c} tiles, "
        f"{kp_per_tile}/tile, fp16={use_amp}) ...")
    for r in recs:
        kp_o, kp_r, d = extract_canon(r["path"], r["heading"] or 0.0)
        feats[r["name"]] = dict(kp=kp_o, kp_rot=kp_r, desc=d, n=len(kp_o))
    log(f"features done - avg {int(np.mean([feats[n]['n'] for n in feats]))} kp/img")
    return feats


def _build_db_and_match(DB, recs, feats, pairs, utm_local, cov, name2id,
                        W0, H0, cam_model_id, cam_params, cam_prior,
                        min_matches, fp16, dev, log):
    """v11: writes coordinate_system=CARTESIAN (was UNDEFINED) so prior-aware
    stages accept the rows, and stores the anisotropic covariance built by
    _prior_cov instead of an isotropic prior_std^2 * I."""
    from lightglue import LightGlue
    from lightglue.utils import rbd

    if DB.exists():
        DB.unlink()

    cov = np.asarray(cov, np.float64)
    sx, sy, sz = np.sqrt(np.diag(cov))
    log(f"pose_priors: coordinate_system=CARTESIAN({CS_CARTESIAN})  "
        f"sigma=({sx:.3f}, {sy:.3f}, {sz:.3f}) m")

    con = sqlite3.connect(str(DB)); cur = con.cursor(); cur.executescript(_DB_SCHEMA)
    cur.execute("INSERT INTO cameras VALUES (?,?,?,?,?,?)",
                (1, int(cam_model_id), W0, H0,
                 _blob(np.asarray(cam_params, np.float64)), int(cam_prior)))
    n_prior_rows = 0
    for nm, i in name2id.items():
        kp = feats[nm]["kp"].astype(np.float32)
        cur.execute("INSERT INTO images (image_id,name,camera_id) VALUES (?,?,?)",
                    (i, nm, 1))
        cur.execute("INSERT INTO keypoints VALUES (?,?,?,?)",
                    (i, kp.shape[0], 2, _blob(kp)))
        if nm in utm_local:
            cur.execute("INSERT INTO pose_priors VALUES (?,?,?,?)",
                        (i, _blob(np.asarray(utm_local[nm], np.float64)),
                         CS_CARTESIAN, _blob(cov)))
            n_prior_rows += 1
    con.commit()
    log(f"DB built: {len(name2id)} images, {n_prior_rows} pose_priors")

    matcher = LightGlue(features="superpoint").eval().to(dev)
    use_amp = fp16 and dev.type == "cuda"
    n_ok = 0
    log(f"matching {len(pairs)} pairs (canonical frame, fp16={use_amp}) ...")
    for a, b in pairs:
        fa, fb = feats[a], feats[b]
        sa = torch.tensor([[fa["kp_rot"][:, 0].max() + 1,
                            fa["kp_rot"][:, 1].max() + 1]],
                          dtype=torch.float32, device=dev)
        sb = torch.tensor([[fb["kp_rot"][:, 0].max() + 1,
                            fb["kp_rot"][:, 1].max() + 1]],
                          dtype=torch.float32, device=dev)
        d0 = {"keypoints": torch.from_numpy(fa["kp_rot"])[None].to(dev),
              "descriptors": fa["desc"].to(dev), "image_size": sa}
        d1 = {"keypoints": torch.from_numpy(fb["kp_rot"])[None].to(dev),
              "descriptors": fb["desc"].to(dev), "image_size": sb}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             enabled=use_amp):
            o = rbd(matcher({"image0": d0, "image1": d1}))
        m = o["matches"].cpu().numpy()
        if len(m) < min_matches:
            continue
        cur.execute("INSERT INTO matches VALUES (?,?,?,?)",
                    (_pidf(name2id[a], name2id[b]), m.shape[0], 2,
                     _blob(m.astype(np.uint32))))
        n_ok += 1
    con.commit(); con.close()
    log(f"matches written: {n_ok}/{len(pairs)}")


def _incremental_map(DB, images, SFM_RAW, min_matches,
                     refine_focal, refine_pp, refine_extra, log, origin_of=None,
                     use_prior_position=True, use_robust_prior_loss=True,
                     prior_loss_scale=0.30):
    """v11: sets use_prior_position so the DB pose_priors actually constrain
    the BAs run inside the incremental loop. Without this the priors are inert
    during mapping and a bare-soil block domes freely."""
    if SFM_RAW.exists():
        shutil.rmtree(SFM_RAW)
    SFM_RAW.mkdir(parents=True)

    opts = pycolmap.IncrementalPipelineOptions()
    mp = getattr(opts, "mapper", None)
    _set(mp, "init_min_tri_angle", 4.0)
    _set(mp, "abs_pose_min_inlier_ratio", 0.10)
    _set(mp, "abs_pose_min_num_inliers", 15)
    _set(opts, "min_num_matches", min_matches)
    for tgt in (opts, mp):
        _set(tgt, "ba_refine_focal_length", refine_focal)
        _set(tgt, "ba_refine_principal_point", refine_pp)
        _set(tgt, "ba_refine_extra_params", refine_extra)

    # ── RTK priors inside the mapper ──────────────────────────────────────
    seen, hit = {}, False
    for tgt, lbl in ((opts, "opts"), (mp, "mapper")):
        hit |= _set_logged(tgt, "use_prior_position", use_prior_position,
                           seen, lbl)
        _set_logged(tgt, "use_robust_loss_on_prior_position",
                    use_robust_prior_loss, seen, lbl)
        _set_logged(tgt, "prior_position_loss_scale", prior_loss_scale, seen, lbl)

    if seen:
        log("mapper prior options set: "
            + ", ".join(f"{k}={v}" for k, v in seen.items()))
    if use_prior_position and not hit:
        log("  ! use_prior_position NOT FOUND on IncrementalPipelineOptions in "
            "this pycolmap build - priors will NOT constrain mapping. Doming "
            "cannot be fixed at this stage; prior-BA is the only remaining "
            "constraint. Consider upgrading pycolmap.")

    log(f"incremental_mapping  (focal={refine_focal} pp={refine_pp} "
        f"extra={refine_extra} priors={bool(use_prior_position and hit)}) ...")
    maps = pycolmap.incremental_mapping(str(DB), str(images), str(SFM_RAW),
                                        options=opts)
    it = list(maps.values() if hasattr(maps, "values") else maps)
    it_sorted = sorted(it, key=lambda m: m.num_reg_images(), reverse=True)
    model = it_sorted[0]
    log(f"submodels (reg images): {[m.num_reg_images() for m in it_sorted]}")
    if len(it_sorted) > 1:
        dropped = sum(m.num_reg_images() for m in it_sorted[1:])
        log(f"  ! {len(it_sorted)} DISCONNECTED submodels - keeping largest "
            f"({model.num_reg_images()} imgs), DROPPING {dropped} imgs in "
            f"{len(it_sorted) - 1} smaller submodel(s).")
        if origin_of:
            for k, m in enumerate(it_sorted):
                by = Counter(origin_of.get(m.images[i].name, "src")
                             for i in m.reg_image_ids())
                log(f"    submodel #{k}: src {by.get('src', 0)} / "
                    f"src2 {by.get('src2', 0)}")
            log("    -> if a submodel is mostly src2, the two flights did NOT "
                "cross-match. Raise min_overlap, lower min_matches, or confirm "
                "the flights overlap on the ground.")
    log(f"registered {model.num_reg_images()} images, {model.num_points3D()} points "
        f"(submodels {len(it_sorted)})")
    return model


def _prior_bundle_adjust(model, name2id, utm_local, cov, log,
                         refine_focal=True, refine_pp=False, refine_extra=True,
                         use_robust_prior_loss=True, prior_loss_scale=0.30,
                         strict=True):
    """Fit the reconstruction to the RTK priors.

    v11 differences from v10:
      * coordinate_system set to CARTESIAN on every PosePrior
      * anisotropic covariance (Z tighter than XY)
      * focal length + extra params MAY be refined. This is the actual
        anti-doming mechanism: a bowl is encoded in the focal/distortion
        solution, so freezing intrinsics (v10) leaves BA able only to drag
        poses onto the priors while the curvature stays in the structure.
        Principal point stays fixed - weakly observable on bare soil.
      * failures raise instead of being swallowed into one log line.
    """
    cov = np.asarray(cov, np.float64)
    cs = _cartesian_cs()
    try:
        priors = {}
        for nm, iid in name2id.items():
            if nm in utm_local and iid in model.images:
                pp = pycolmap.PosePrior()
                pp.position = np.asarray(utm_local[nm], np.float64)
                if hasattr(pp, "coordinate_system"):
                    pp.coordinate_system = cs
                if hasattr(pp, "position_covariance"):
                    pp.position_covariance = cov
                priors[iid] = pp

        n_reg = model.num_reg_images()
        missing = [model.images[i].name for i in model.reg_image_ids()
                   if i not in priors]
        sx, sy, sz = np.sqrt(np.diag(cov))
        log(f"prior-BA: {len(priors)} priors built, {n_reg} registered images, "
            f"sigma=({sx:.3f}, {sy:.3f}, {sz:.3f}) m")
        if missing:
            log(f"  ! {len(missing)} registered images lack a prior: "
                f"{missing[:8]}" + (" ..." if len(missing) > 8 else ""))
        if not priors:
            raise RuntimeError("no usable pose priors - nothing to constrain")

        ba = pycolmap.BundleAdjustmentOptions()
        _set(ba, "refine_focal_length", refine_focal)
        _set(ba, "refine_principal_point", refine_pp)
        _set(ba, "refine_extra_params", refine_extra)
        log(f"prior-BA intrinsics: focal={refine_focal} pp={refine_pp} "
            f"extra={refine_extra}")

        pp_opts = (pycolmap.PosePriorBundleAdjustmentOptions()
                   if hasattr(pycolmap, "PosePriorBundleAdjustmentOptions") else None)
        seen = {}
        _set_logged(pp_opts, "use_robust_loss_on_prior_position",
                    use_robust_prior_loss, seen, "prior_opts")
        _set_logged(pp_opts, "prior_position_loss_scale",
                    prior_loss_scale, seen, "prior_opts")
        if seen:
            log("prior-BA options: " + ", ".join(f"{k}={v}" for k, v in seen.items()))

        cfg = pycolmap.BundleAdjustmentConfig()
        for iid in model.reg_image_ids():
            cfg.add_image(iid)
        pycolmap.create_pose_prior_bundle_adjuster(
            options=ba, prior_options=pp_opts, config=cfg,
            pose_priors=priors, reconstruction=model).solve()
        log("prior-BA ok")
    except Exception as e:
        log(f"prior-BA FAILED - {type(e).__name__}: {e}")
        traceback.print_exc()
        if strict:
            raise RuntimeError(
                "prior-BA failed and prior_ba_strict=True. v10 swallowed this "
                "silently, so an RTK-unconstrained reconstruction looked "
                "identical to a constrained one. Pass prior_ba_strict=False to "
                "restore the old skip-and-continue behaviour."
            ) from e
        log("  (continuing WITHOUT prior-BA - geometry is UNCONSTRAINED by RTK)")


def _ply_point_count(path):
    """0 if unreadable/missing - used to distinguish a real fused cloud from
    an empty header-only PLY (which .exists()/.st_size>0 alone can't catch)."""
    try:
        import plyfile
        return len(plyfile.PlyData.read(str(path))["vertex"].data)
    except Exception:
        return 0


def _start_ram_monitor(log, interval=15):
    """Background thread logging system RAM every `interval` seconds.
    Call .set() on the returned Event to stop it."""
    import threading
    try:
        import psutil
    except ImportError:
        class _Noop:
            def set(self):
                pass
        log("  [ram] psutil not installed - monitor disabled")
        return _Noop()
    stop = threading.Event()

    def _loop():
        while not stop.is_set():
            vm = psutil.virtual_memory()
            log(f"  [ram] used {vm.used/1e9:.2f}GB / {vm.total/1e9:.2f}GB "
                f"({vm.percent:.0f}%)  avail {vm.available/1e9:.2f}GB")
            stop.wait(interval)
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return stop


def _dense(work, SFM_DIR, images, max_image_size, log, resume=True,
           geom_consistency=False, input_type=None, cache_size_gb=2.0,
           fusion_max_image_size=None, fusion_min_num_pixels=None):
    DENSE           = Path(work) / "dense"
    undistort_done  = DENSE / ".undistort_done"
    pms_done        = DENSE / ".pms_done"
    stamp           = DENSE / ".source.json"
    fused           = DENSE / "fused.ply"

    # A workspace may only be resumed by the run that built it. Without this a
    # shared `work` makes every subsequent flight skip undistort/PMS/fusion and
    # inherit the PREVIOUS flight's cloud.
    want = {"images": str(Path(images).resolve()),
            "sparse": str(Path(SFM_DIR).resolve()),
            "max_image_size": int(max_image_size),
            "geom_consistency": bool(geom_consistency),
            "pipeline_version": PIPELINE_VERSION}

    stale = False
    if DENSE.exists():
        try:
            got = json.loads(stamp.read_text())
        except Exception:
            got = None
        if got != want:
            stale = True
            diff = "no stamp (legacy workspace)" if got is None else ", ".join(
                f"{k}: {got.get(k)!r} -> {want[k]!r}"
                for k in want if got.get(k) != want[k])
            log(f"dense: workspace belongs to another run -> rebuilding ({diff})")

    fresh_start = stale or not (resume and DENSE.exists() and undistort_done.exists())

    if fresh_start:
        if DENSE.exists():
            shutil.rmtree(DENSE)
        DENSE.mkdir(parents=True)
        log("dense: undistort ...")
        pycolmap.undistort_images(DENSE, SFM_DIR, images)
        stamp.write_text(json.dumps(want, indent=2))
        undistort_done.touch()
    else:
        log("dense: undistort - workspace found, skipping ok")

    if resume and pms_done.exists():
        log("dense: patch_match_stereo - already complete, skipping ok")
    else:
        log(f"dense: patch_match_stereo (CUDA, geom_consistency={geom_consistency}, "
            f"cache_size={cache_size_gb}GB) ...")
        pm_opts = pycolmap.PatchMatchOptions(max_image_size=max_image_size)
        if hasattr(pm_opts, "geom_consistency"):
            pm_opts.geom_consistency = geom_consistency
        pm_opts.cache_size = cache_size_gb
        pycolmap.patch_match_stereo(DENSE, options=pm_opts)
        pms_done.touch()

    itype = input_type or ("geometric" if geom_consistency else "photometric")

    if resume and fused.exists() and _ply_point_count(fused) > 0:
        n_pts = _ply_point_count(fused)
        log(f"dense: stereo_fusion - fused cloud found, skipping ok "
            f"({n_pts:,} pts, {fused.stat().st_size/1e6:.1f} MB)")
    else:
        dmd = DENSE / "stereo" / "depth_maps"
        n_photo = len(list(dmd.glob("*.photometric.bin"))) if dmd.exists() else 0
        n_geom  = len(list(dmd.glob("*.geometric.bin")))  if dmd.exists() else 0
        fusion_opts = pycolmap.StereoFusionOptions()
        fusion_opts.cache_size = cache_size_gb
        if fusion_max_image_size is not None:
            fusion_opts.max_image_size = fusion_max_image_size
        if fusion_min_num_pixels is not None:
            fusion_opts.min_num_pixels = fusion_min_num_pixels
        log(f"dense: stereo_fusion (input_type={itype}, cache_size={cache_size_gb}GB, "
            f"max_image_size={fusion_opts.max_image_size}, "
            f"min_num_pixels={fusion_opts.min_num_pixels}) ... "
            f"[{n_photo} photometric.bin, {n_geom} geometric.bin on disk]")
        ram_stop = _start_ram_monitor(log)
        try:
            pycolmap.stereo_fusion(fused, DENSE, input_type=itype, options=fusion_opts)
        finally:
            ram_stop.set()

        n_pts = _ply_point_count(fused)
        if n_pts == 0:
            raise RuntimeError(
                f"stereo_fusion produced 0 points (input_type={itype}, "
                f"{n_photo} .photometric.bin / {n_geom} .geometric.bin on disk).")
        log(f"stereo_fusion produced {n_pts:,} points")

    log(f"dense cloud: {fused}  exists={fused.exists()}")
    return fused


def _save_ortho_inputs(path, model, s, Rg, tg, origin, utm_epsg, images,
                       colmap_to_utm, log):
    cam = model.cameras[list(model.cameras)[0]]
    fx, fy, cx, cy = [float(x) for x in cam.params[:4]]
    reg = list(model.reg_image_ids())
    names = [model.images[i].name for i in reg]
    Rs = np.array([_get_pose(model.images[i]).rotation.matrix() for i in reg], np.float64)
    ts = np.array([np.asarray(_get_pose(model.images[i]).translation, float)
                   for i in reg], np.float64)
    cloud_utm = colmap_to_utm(np.array([p.xyz for p in model.points3D.values()]))
    np.savez_compressed(
        path, names=np.array(names), Rs=Rs, ts=ts,
        fx=fx, fy=fy, cx=cx, cy=cy,
        s=np.float64(s), Rg=Rg, tg=tg, origin=origin,
        epsg=np.int64(utm_epsg), images_dir=str(images),
        cloud_utm=cloud_utm.astype(np.float32))
    log(f"saved {path.name}  ({path.stat().st_size/1e6:.1f} MB)")
    # NB: images_dir points at the (ephemeral) merged work folder when src2 is
    # used; if the ortho compositor runs in a later session, re-create the merge
    # or repoint images_dir, because /content/proj is wiped between sessions.


def _write_ply_binary(pts, out_path):
    import plyfile
    verts = np.empty(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    verts["x"] = pts[:, 0]; verts["y"] = pts[:, 1]; verts["z"] = pts[:, 2]
    el = plyfile.PlyElement.describe(verts, "vertex")
    plyfile.PlyData([el], text=False).write(str(out_path))


def _save_dense_utm(fused, colmap_to_utm, out_path, log):
    import plyfile
    vv = plyfile.PlyData.read(str(fused))
    pts = colmap_to_utm(np.column_stack([vv["vertex"]["x"], vv["vertex"]["y"],
                                         vv["vertex"]["z"]]))
    _write_ply_binary(pts, out_path)
    log(f"saved {out_path.name}  ({len(pts):,} pts, binary PLY)")


def _rasters(fused, colmap_to_utm, utm_epsg, grid, chm_max, dtm_window_m,
             work, out, make_plots, log):
    import plyfile, rasterio
    from rasterio.transform import from_origin
    from scipy.ndimage import grey_opening

    v = plyfile.PlyData.read(str(fused))["vertex"].data
    xyz = colmap_to_utm(np.column_stack([v["x"], v["y"], v["z"]]))
    X, Y, Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    log(f"cloud UTM E {X.min():.1f}-{X.max():.1f}  N {Y.min():.1f}-{Y.max():.1f}  "
        f"elev span {np.ptp(Z):.2f} m  ({len(xyz):,} pts)")

    xmin, ymax = X.min(), Y.max()
    nx = int((X.max() - xmin) / grid) + 1
    ny = int((ymax - Y.min()) / grid) + 1
    ix = np.clip(((X - xmin) / grid).astype(np.int64), 0, nx - 1)
    iy = np.clip(((ymax - Y) / grid).astype(np.int64), 0, ny - 1)

    flat = iy * nx + ix
    order = np.argsort(Z, kind="stable")
    dsm_flat = np.full(nx * ny, np.nan, np.float32)
    dsm_flat[flat[order]] = Z[order].astype(np.float32)
    dsm = dsm_flat.reshape(ny, nx)

    filled = np.where(np.isnan(dsm), np.nanmin(dsm), dsm)
    win = max(3, int(round(dtm_window_m / grid)))
    dtm = grey_opening(filled, size=(win, win)).astype(np.float32)
    chm = np.clip(dsm - dtm, 0, chm_max).astype(np.float32)

    transform = from_origin(xmin, ymax, grid, grid)

    def _save(arr, name):
        with rasterio.open(work / name, "w", driver="GTiff", height=ny, width=nx,
                           count=1, dtype="float32", crs=f"EPSG:{utm_epsg}",
                           transform=transform, nodata=float("nan")) as d:
            d.write(arr, 1)
        shutil.copy(work / name, out / name)

    _save(dsm, "dsm.tif"); _save(dtm, "dtm.tif"); _save(chm, "chm.tif")
    log(f"CHM range {np.nanmin(chm):.2f} -> {np.nanmax(chm):.2f} m")

    if make_plots:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 6))
        plt.imshow(chm, cmap="viridis", vmin=0, vmax=min(3, chm_max))
        plt.colorbar(label="canopy height (m)")
        plt.title("CHM = DSM - DTM (UTM)")
        plt.savefig(out / "chm_preview.jpg", dpi=140, bbox_inches="tight")
        plt.close()
    return {"chm_min_m": float(np.nanmin(chm)), "chm_max_m": float(np.nanmax(chm))}


# ── plotting ────────────────────────────────────────────────────────────────
def _plot_match_graph(recs, pairs, fp_w, fp_h, A, path, oblique_names=None):
    import matplotlib.pyplot as plt
    oblique_names = oblique_names or set()
    fig, ax = plt.subplots(figsize=(11, 9)); ax.set_facecolor("#111")
    look = {r["name"]: (r["ct"], r["at"]) for r in recs}
    for a, b in pairs:
        if a in oblique_names or b in oblique_names:
            color = "#ff00ff"
        else:
            color = "#00ff88" if abs(look[a][0] - look[b][0]) < fp_w * 0.4 else "#ff9500"
        ax.plot([look[a][0], look[b][0]], [look[a][1], look[b][1]],
                color=color, lw=0.6, alpha=0.5)
    for r in recs:
        is_obl = r["name"] in oblique_names
        ax.scatter(r["ct"], r["at"], s=70 if is_obl else 60,
                   c="#ff00ff" if is_obl else "red",
                   marker="D" if is_obl else "o", zorder=3)
        hr = math.radians((r["heading"] or 0.0) - math.degrees(A))
        L = min(fp_h, fp_w) * 0.3
        ax.annotate("", xy=(r["ct"] + L * math.sin(hr), r["at"] + L * math.cos(hr)),
                    xytext=(r["ct"], r["at"]),
                    arrowprops=dict(arrowstyle="->", color="cyan", lw=0.8), zorder=4)
        try:
            ax.text(r["ct"], r["at"], "  " + r["name"].split("_")[-2],
                    color="yellow", fontsize=6, zorder=5)
        except Exception:
            pass
    ax.set_xlabel("cross-flight (m)", color="w")
    ax.set_ylabel("along-flight (m)", color="w")
    ttl = (f"Match graph ({len(pairs)} pairs)  green=same-row orange=cross-row"
           + ("  magenta=oblique" if oblique_names else ""))
    ax.set_title(ttl, color="w")
    ax.set_aspect("equal"); ax.tick_params(colors="w"); fig.patch.set_facecolor("#111")
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="#111"); plt.close(fig)


def _sparse_views(model, colmap_to_utm, iqr_factor, out, work, log):
    import matplotlib.pyplot as plt
    P = colmap_to_utm(np.array([p.xyz for p in model.points3D.values()]))
    C = colmap_to_utm(np.array([_center(model.images[i]) for i in model.reg_image_ids()]))

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sc = ax[0].scatter(P[:, 0], P[:, 1], c=P[:, 2], s=2, cmap="viridis")
    ax[0].scatter(C[:, 0], C[:, 1], c="red", marker="^", s=80, edgecolor="k", zorder=5)
    ax[0].set_title(f"TOP (UTM E-N)  {len(C)} cams / {len(P)} pts")
    ax[0].set_xlabel("E (m)"); ax[0].set_ylabel("N (m)"); ax[0].axis("equal")
    plt.colorbar(sc, ax=ax[0], label="elev (m)")
    ax[1].scatter(P[:, 0], P[:, 2], c=P[:, 2], s=2, cmap="viridis")
    ax[1].scatter(C[:, 0], C[:, 2], c="red", marker="^", s=80, edgecolor="k", zorder=5)
    ax[1].set_title("SIDE (E-elev)"); ax[1].set_xlabel("E (m)"); ax[1].set_ylabel("elev (m)")
    ax[1].axis("equal"); plt.tight_layout()
    fig.savefig(out / "sparse_views.jpg", dpi=140, bbox_inches="tight"); plt.close(fig)

    keep = np.ones(len(P), bool)
    for c in range(3):
        q1, q3 = np.percentile(P[:, c], [25, 75]); iqr = q3 - q1
        keep &= (P[:, c] >= q1 - iqr_factor * iqr) & (P[:, c] <= q3 + iqr_factor * iqr)
    Pc = P[keep]
    log(f"cleaned cloud {len(Pc)} pts (dropped {(~keep).sum()})")

    fig2, ax2 = plt.subplots(1, 2, figsize=(15, 6))
    sc2 = ax2[0].scatter(Pc[:, 0], Pc[:, 1], c=Pc[:, 2], s=2, cmap="viridis")
    ax2[0].scatter(C[:, 0], C[:, 1], c="red", marker="^", s=80, edgecolor="k", zorder=5)
    ax2[0].set_title(f"TOP cleaned  {len(Pc)} pts")
    ax2[0].set_xlabel("E (m)"); ax2[0].set_ylabel("N (m)"); ax2[0].axis("equal")
    plt.colorbar(sc2, ax=ax2[0], label="elev (m)")
    ax2[1].scatter(Pc[:, 0], Pc[:, 2], c=Pc[:, 2], s=2, cmap="viridis")
    ax2[1].scatter(C[:, 0], C[:, 2], c="red", marker="^", s=80, edgecolor="k", zorder=5)
    ax2[1].set_title("SIDE cleaned"); ax2[1].set_xlabel("E (m)")
    ax2[1].set_ylabel("elev (m)")
    ax2[1].axis("equal"); plt.tight_layout()
    fig2.savefig(out / "sparse_views_cleaned.jpg", dpi=140, bbox_inches="tight")
    plt.close(fig2)

    with open(out / "sparse_cleaned.ply", "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(Pc)}\n"
                "property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, Pc, fmt="%.3f")


# ── feature cache ────────────────────────────────────────────────────────────
def _save_feats(feats, path):
    blob = {nm: {"kp": d["kp"], "kp_rot": d["kp_rot"],
                 "desc": d["desc"].squeeze(0).cpu().numpy().astype(np.float16),
                 "n": d["n"]} for nm, d in feats.items()}
    with open(path, "wb") as fh:
        pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_feats(path, dev):
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    return {nm: {"kp": d["kp"], "kp_rot": d["kp_rot"],
                 "desc": torch.from_numpy(d["desc"].astype(np.float32))[None].to(dev),
                 "n": d["n"]} for nm, d in blob.items()}


def _copytree(srcdir, dstdir):
    dstdir = Path(dstdir)
    if dstdir.exists():
        shutil.rmtree(dstdir)
    shutil.copytree(srcdir, dstdir)


# ═══════════════════════════════════════════════════════════════════════════
# depth-map checkpoint to Drive
#   Saves pipeline_output/depth_bundle.npz:
#     names  : image basenames
#     K      : Nx4 (fx, fy, cx, cy) at the SAVED depth-map resolution
#     shapes : Nx2 (h, w) of saved maps
#     d0000. : float16 depth arrays (0 = invalid), model-frame units
#   ~200-300 MB for ~150 views at dm_save_scale=0.5 - vs ~1.8 GB raw .bin.
#   float16 at ~11 m depth ~ 1 cm precision (<< depth_tol), fine.
#   REQUIRES dense_geom_consistency=True: only *.geometric.bin is read.
# ═══════════════════════════════════════════════════════════════════════════
def _read_colmap_array(path):
    with open(path, 'rb') as f:
        vals, cur = [], b''
        while len(vals) < 3:
            ch = f.read(1)
            if ch == b'&':
                vals.append(int(cur)); cur = b''
            else:
                cur += ch
        w, h, c = vals
        data = np.fromfile(f, np.float32, w * h * c)
    return data.reshape(h, w, c).squeeze()


def _save_depth_bundle(dense_dir, out, log, dm_save_scale=0.5):
    """Checkpoint geometric depth maps + undistorted PINHOLE intrinsics."""
    try:
        dense_dir = Path(dense_dir)
        dmd = dense_dir / "stereo" / "depth_maps"
        if not dmd.exists():
            log("depth bundle skipped: no stereo/depth_maps"); return
        rec = pycolmap.Reconstruction(str(dense_dir / "sparse"))
        names, Ks, shapes, arrays = [], [], [], {}
        for img in rec.images.values():
            f = dmd / f"{img.name}.geometric.bin"
            if not f.exists():
                continue
            cam = rec.cameras[img.camera_id]
            p   = np.asarray(cam.params, np.float64)      # PINHOLE fx fy cx cy
            dm  = _read_colmap_array(f)
            sx  = dm.shape[1] / cam.width                 # max_image_size scale
            sy  = dm.shape[0] / cam.height
            if dm_save_scale != 1.0:
                dm = cv2.resize(dm, None, fx=dm_save_scale, fy=dm_save_scale,
                                interpolation=cv2.INTER_NEAREST)
                sx *= dm_save_scale; sy *= dm_save_scale
            i = len(names)
            names.append(Path(img.name).name)
            Ks.append([p[0]*sx, p[1]*sy, p[2]*sx, p[3]*sy])
            shapes.append(dm.shape)
            arrays[f"d{i:04d}"] = dm.astype(np.float16)
        if not names:
            log("depth bundle skipped: no geometric.bin files "
                "(dense_geom_consistency must be True)")
            return
        bp = Path(out) / "depth_bundle.npz"
        np.savez_compressed(bp, names=np.array(names),
                            K=np.array(Ks, np.float64),
                            shapes=np.array(shapes, np.int32), **arrays)
        log(f"saved {bp.name}  ({len(names)} maps, "
            f"{bp.stat().st_size/1e6:.0f} MB)")
    except Exception as e:
        log(f"depth bundle skipped: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════
def diagnose_doming(src, min_cams=8, verbose=True):
    """Quantify the radial bowl on an EXISTING sfm/ - no remapping, seconds.

    Reads DB priors (local UTM metres), Umeyama-aligns the model to them, and
    fits dz against radial distance from the block centre. A high R^2 IS
    doming; bowl_m is its amplitude across the block.

    Also reports the prior coordinate_system and sigma, which identifies
    v10-era checkpoints (UNDEFINED, isotropic 0.10) at a glance.
    """
    src = Path(src)
    ckpt = src / "pipeline_output" / "_checkpoints"
    sfm, db = ckpt / "sfm", ckpt / "database.db"
    if not (sfm / "cameras.bin").exists():
        print(f"[FAIL ] no sfm/ under {ckpt}"); return None
    if not db.exists():
        print(f"[FAIL ] no database.db under {ckpt}"); return None

    con = sqlite3.connect(str(db))
    rows = con.execute(
        "SELECT i.name, p.position, p.coordinate_system, p.position_covariance "
        "FROM pose_priors p JOIN images i ON i.image_id = p.image_id").fetchall()
    con.close()
    if not rows:
        print("[FAIL ] pose_priors table is empty"); return None

    ver = None
    man = ckpt / "image_manifest.json"
    if man.exists():
        try:
            ver = json.loads(man.read_text()).get("pipeline_version")
        except Exception:
            pass

    cs_set = {r[2] for r in rows}
    if verbose:
        print(f"version: pipeline_version="
              f"{ver if ver is not None else '<11 (unstamped)'}")
        print(f"priors : {len(rows)}   coordinate_system="
              f"{', '.join(_CS_NAME.get(c, str(c)) for c in cs_set)}")
        if cs_set == {CS_UNDEFINED}:
            print("  ! UNDEFINED - v10 checkpoint; priors may have been ignored "
                  "by prior-aware stages. Re-run with force_sparse=True.")

    sig = []
    for _, _, _, cb in rows:
        if cb and len(cb) >= 72:
            C = np.frombuffer(cb[:72], np.float64).reshape(3, 3)
            sig.append(np.sqrt(np.clip(np.diag(C), 0, None)))
    if sig and verbose:
        S = np.array(sig)
        print(f"sigma  : xy~{np.median(S[:, 0]):.4f} m  z~{np.median(S[:, 2]):.4f} m")

    priors = {r[0]: np.frombuffer(r[1][:24], np.float64) for r in rows if r[1]}
    model = pycolmap.Reconstruction(str(sfm))
    Sc, Dc = [], []
    for i in model.reg_image_ids():
        nm = model.images[i].name
        if nm in priors:
            Sc.append(_center(model.images[i])); Dc.append(priors[nm])
    if len(Sc) < min_cams:
        print(f"[FAIL ] only {len(Sc)} cams matched a prior"); return None

    Sc, Dc = np.array(Sc), np.array(Dc)
    s, R, t = _umeyama(Sc, Dc)
    P = (s * (R @ Sc.T).T + t)
    d = _fit_doming(P, Dc)
    if d is None:
        print("[FAIL ] doming fit not possible"); return None
    d["scale"] = float(s)
    d["pipeline_version"] = ver

    if verbose:
        print(f"cams   : {d['n']}   umeyama scale {s:.5f}")
        print(f"resid  : XY rms {d['xy_rms']:.3f} m   Z rms {d['z_rms']:.3f} m   "
              f"Z max |{d['z_max']:.3f}| m")
        print(f"\nDOMING FIT   dz = {d['a']:+.3e}*r^2 {d['b']:+.3e}*r {d['c']:+.3f}")
        print(f"  R^2          {d['r2']:.3f}   (radial explains this much of dz)")
        print(f"  bowl height  {d['bowl_m']:+.3f} m across r_max={d['r_max']:.1f} m")
        if d["r2"] > 0.4 and abs(d["bowl_m"]) > 0.05:
            print("  -> SIGNIFICANT DOMING. Radial signature is the giveaway; a "
                  "random dz field would not fit r^2 this well.")
            est = abs(d["bowl_m"]) * math.tan(math.radians(25.0))
            print(f"  -> expected ortho lateral smear at ~25 deg view angle: "
                  f"~{est:.2f} m per view, up to ~{2*est:.2f} m between views on "
                  f"opposite sides of the bowl (feature doubling).")
        elif abs(d["bowl_m"]) > 0.05:
            print("  -> vertical error present but NOT radially structured - "
                  "look for strip offsets / a tilt, not doming.")
        else:
            print("  -> no meaningful bowl at the camera level.")
    return d


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE CALL — recommended production configuration
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    FLIGHT = Path("/content/drive/MyDrive/Colab Notebooks/DJI/Zwiebel/"
                  "DJI_202605111043_008_Zwiebel2026")

    # 0) measure the bowl on the existing model (seconds, changes nothing)
    diagnose_doming(FLIGHT)

    # 1) rebuild with locked intrinsics + active RTK priors.
    #    force_sparse=True is REQUIRED after any change to camera_model or
    #    prior settings: force_dense alone preserves exactly the checkpoints
    #    that must change.
    products = run_pipeline(
        FLIGHT,
        force_sparse=True,
        camera_model="DJI_DEWARP",          # lock to factory calibration
        ba_refine_focal_length=False,       # locked: no per-flight focal drift
        ba_refine_principal_point=False,    # NEVER free this — see module docstring
        ba_refine_extra_params=False,       # locked: distortion is factory
        prior_std_xy=0.05,
        prior_std_z=0.035,                  # matches measured RTK vertical sigma
        use_prior_position=True,
        prior_ba_refine_focal=True,         # small RTK-anchored correction only
        prior_ba_refine_extra=True,
        prior_ba_refine_pp=False,
        prior_ba_strict=True,
        dense_geom_consistency=True,        # required for depth_bundle.npz
        fp16=True,
        do_dense=True,
        resume=True,
    )
    print(products)

    # 2) confirm the bowl collapsed
    diagnose_doming(FLIGHT)
