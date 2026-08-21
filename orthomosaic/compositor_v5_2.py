# ═══════════════════════════════════════════════════════════════════════════
# TRUE ORTHOPHOTO v5.2 — BEST-VIEW compositing for canopy-cover analysis
#
#   v5.2: filter oblique frames out of the ortho/DSM texturing stage. DJI's
#   end-of-flight oblique pass is deliberately kept in BA (anti-doming) but
#   was never removed again downstream — composite()'s "most nadir" weight
#   is purely image-space distance-from-center, so an oblique frame can
#   still win pixels outright wherever nadir coverage is thin, producing
#   raked/foreshortened texture patches in an otherwise-nadir ortho.
#   Nadir angle is computed straight from the registered pose (Rs) rotated
#   into UTM via Rg — no EXIF needed, no dependency on odm_light_cuda's
#   _get_pitch(). Oblique frames are excluded from BOTH the ortho texture
#   loop and the depth-map DSM loop (both iterate over `poses`), but the
#   BA-refined camera positions they helped produce are still used as-is.
#
#   v5.1: dense-cloud XY outliers were inflating the ortho bounding box
#   (IQR-trim dense_pts, hard clamp vs camera footprint) + tiled the PREVIEW
#   pass like FULL already was + preview_grid/full_grid now default to a
#   multiple of the native GSD instead of fixed absolute values, so they
#   scale correctly with flight altitude.
#
#   v5 (base): winner-take-all best-view compositing with a narrow top-2 seam
#   feather replaces weighted-average blending, which mixed misaligned views
#   of thin leaves into semi-transparent ghosts. DSM source priority:
#     depth_bundle.npz -> live dense workspace -> dense UTM PLY -> sparse cloud.
#   img_scale defaults to 1.0 (full-res) -- half-res source images smear
#   sub-pixel thin leaves BEFORE compositing and undercount canopy cover.
#   No radiometric/gain adjustment: pixel colour is left untouched so
#   vegetation indices computed downstream stay valid.
# ═══════════════════════════════════════════════════════════════════════════
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import cv2
import rasterio
import time
from rasterio.transform import from_origin
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt


def install_dependencies(quiet: bool = True):
    """Install non-preinstalled packages. Idempotent. Call once per Colab
    session before run_ortho()/run_ortho_batch() (mirrors
    reconstruction/odm_light_cuda_v11.install_dependencies)."""
    q = ["-q"] if quiet else []
    have = lambda mod: importlib.util.find_spec(mod) is not None
    need = [m for m in ("rasterio", "plyfile") if not have(m)]
    if need:
        subprocess.run([sys.executable, "-m", "pip", "install", *q, *need],
                       check=False)


def mount_drive(path="/content/drive"):
    """Mount Google Drive in Colab (no-op outside Colab)."""
    try:
        from google.colab import drive
        drive.mount(path)
    except Exception as e:
        print(f"drive mount skipped ({e})")


def run_ortho(
    npz_path:     Path,
    src2:         Path  = None,
    dense_dir:    Path  = None,
    depth_bundle: Path  = None,
    dense_ply:    Path  = None,
    preview_grid: float = None,     # None => auto = native_gsd * preview_scale
    preview_scale: float = 50.0,    # preview px spacing as multiple of native GSD
    full_grid:    float = None,     # None => auto = native_gsd * full_scale
    full_scale:   float = 1.0,      # full-res px spacing as multiple of native GSD
    tile_px:      int   = 1024,
    iqr_factor:   float = 3.0,
    footprint_margin: float = 30.0, # hard bounds clamp vs camera span
    max_nadir_deg: float = 20.0,    # exclude frames tilted more than this from ortho/DSM
    mask_dist:    float = 1.5,
    foot_margin:  float = 2.0,
    img_scale:    float = 1.0,
    dsm_grid:     float = 0.05,
    dsm_smooth:   float = 0.10,     # do NOT raise -- re-flattens canopy, reintroduces ghosting
    blend_pow:    float = 6.0,
    feather_lo:   float = 0.90,
    depth_below:  float = 3.0,
    cam_clear:    float = 1.0,
    cache_images: bool  = True,
    cache_budget_gb: float = 6.0,
    depth_tol:    float = 0.15,
    dm_stride:    int   = 3,
    dm_scale:     float = 1.0,
):
    npz_path = Path(npz_path)
    OUT      = npz_path.parent

    # ── load npz ─────────────────────────────────────────────────────────────
    d = np.load(npz_path, allow_pickle=True)
    names = d['names']
    Rs    = d['Rs'];   ts = d['ts']
    fx_raw, fy_raw = float(d['fx']), float(d['fy'])
    cx_raw, cy_raw = float(d['cx']), float(d['cy'])
    s  = float(d['s']);   Rg = d['Rg'];  tg = d['tg'];  origin = d['origin']
    EPSG   = int(d['epsg'])
    IMAGES = Path(str(d['images_dir']))
    cloud  = d['cloud_utm']
    print(f"loaded {len(names)} cameras · EPSG:{EPSG}")

    fx = fx_raw * img_scale;  fy = fy_raw * img_scale
    cx = cx_raw * img_scale;  cy = cy_raw * img_scale
    r_max = float(np.hypot(cx, cy))

    def utm_to_col(P):
        return ((P - tg - origin) / s) @ Rg

    def col_to_utm(P):
        return (s * (P @ Rg.T)) + tg + origin

    centers_col = np.einsum('kij,kj->ki', np.transpose(Rs, (0, 2, 1)), -ts)
    centers_utm = col_to_utm(centers_col)
    min_cam_z   = float(centers_utm[:, 2].min())

    # ── nadir angle per camera, straight from the registered pose ───────────
    # COLMAP convention: x_cam = R @ x_world + t, camera looks down +Z_cam.
    # Forward axis in local 'col' frame = row 2 of R. Rotate into UTM via Rg
    # (rotation only -- this is a direction, not a point) and compare to
    # straight-down [0,0,-1]. 0 deg = nadir, 90 deg = horizontal.
    fwd_col   = Rs[:, 2, :]
    fwd_utm   = fwd_col @ Rg.T
    fwd_utm  /= (np.linalg.norm(fwd_utm, axis=1, keepdims=True) + 1e-12)
    nadir_all = np.degrees(np.arccos(np.clip(-fwd_utm[:, 2], -1.0, 1.0)))
    _bins = [0, 5, 10, 20, 30, 45, 90]
    _cnt  = [int(((nadir_all >= _bins[i]) & (nadir_all < _bins[i+1])).sum())
             for i in range(len(_bins) - 1)]
    print("  nadir angle histogram: " +
          "  ".join(f"{_bins[i]}-{_bins[i+1]} deg:{_cnt[i]}"
                     for i in range(len(_bins) - 1)))
    # Expect a clean bimodal split: a tight cluster near 0-5 deg (survey grid)
    # and a separate cluster around 30-45 deg (DJI end-of-flight oblique
    # pass). If the gap sits somewhere other than max_nadir_deg=20, adjust the
    # threshold to fall in the gap. A continuous (non-bimodal) spread points
    # to gimbal instability rather than deliberate obliques.

    # ── Z gate + sparse clean (bounds + fallback DSM + ground_z) ─────────────
    Zc        = cloud[:, 2]
    z_ground0 = float(np.median(Zc))
    z_hi      = min_cam_z - cam_clear
    z_lo      = z_ground0 - depth_below
    print(f"\nZ gate: [{z_lo:.1f}, {z_hi:.1f}] m")
    gate  = (Zc >= z_lo) & (Zc <= z_hi)
    cloud = cloud[gate]
    if len(cloud) == 0:
        raise RuntimeError("Z gate removed all sparse points.")
    keep = np.ones(len(cloud), bool)
    for c in range(3):
        q1, q3 = np.percentile(cloud[:, c], [25, 75])
        iqr     = q3 - q1
        keep   &= (cloud[:, c] >= q1 - iqr_factor * iqr) & \
                  (cloud[:, c] <= q3 + iqr_factor * iqr)
    cloud = cloud[keep]
    X, Y, Z  = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    ground_z = float(np.median(Z))
    print(f"sparse cloud: kept {len(cloud)} pts, Z span {np.ptp(Z):.2f} m")

    # ── native GSD & auto grid resolution ────────────────────────────────────
    # GSD scales with AGL. A fixed preview/full grid either wastes time at low
    # altitude (oversampled, no extra real detail) or crawls at high altitude
    # (huge canvas for the same effective resolution). Default both to a
    # multiple of the actual native ground sample distance instead.
    agl_all    = centers_utm[:, 2] - ground_z
    native_gsd = float(np.median(agl_all[agl_all > 0])) / fx_raw   # m/px, pre img_scale
    if preview_grid is None:
        preview_grid = native_gsd * preview_scale
    if full_grid is None:
        full_grid = native_gsd * full_scale
    print(f"  native GSD ~= {native_gsd*100:.2f} cm/px  ->  "
          f"preview {preview_grid*100:.1f} cm/px (1:{preview_scale:.0f})  ·  "
          f"full {full_grid*100:.2f} cm/px ({full_scale:.1f}x native)")

    # ── DEPTH SOURCE 1: depth_bundle.npz ─────────────────────────────────────
    # Priority chain: depth_bundle.npz (per-view geometric depth, persists
    # across ephemeral Colab sessions) -> live dense workspace (same data,
    # this session only) -> dense UTM PLY (no per-view gating) -> sparse cloud
    # (fallback, most ghost-prone).
    depth_info = {}
    bp = Path(depth_bundle) if depth_bundle is not None else OUT / "depth_bundle.npz"
    if bp.exists():
        b = np.load(bp, allow_pickle=True)
        bn = [str(x) for x in b['names']];  K = b['K']
        for i, nm in enumerate(bn):
            dm = b[f"d{i:04d}"].astype(np.float32)
            kf = K[i].astype(np.float64)
            if dm_scale != 1.0:
                dm = cv2.resize(dm, None, fx=dm_scale, fy=dm_scale,
                                interpolation=cv2.INTER_NEAREST)
                kf = kf * dm_scale
            depth_info[nm] = (dm, kf[0], kf[1], kf[2], kf[3])
        print(f"depth source: bundle {bp.name} ({len(depth_info)} maps)")

    # ── DEPTH SOURCE 2: live dense workspace ─────────────────────────────────
    if not depth_info:
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

        cands = ([Path(dense_dir)] if dense_dir is not None else []) + \
                [Path('/content/proj/dense'), npz_path.parent / 'dense',
                 npz_path.parent.parent / 'work' / 'dense']
        DENSE = next((p for p in cands
                      if (p / 'stereo' / 'depth_maps').exists()), None)
        if DENSE is not None:
            import pycolmap
            rec = pycolmap.Reconstruction(str(DENSE / 'sparse'))
            dmd = DENSE / 'stereo' / 'depth_maps'
            for img in rec.images.values():
                f = dmd / f"{img.name}.geometric.bin"
                if not f.exists():
                    continue
                cam = rec.cameras[img.camera_id]
                p   = np.asarray(cam.params, np.float64)
                dm  = _read_colmap_array(f)
                sx  = dm.shape[1] / cam.width
                sy  = dm.shape[0] / cam.height
                if dm_scale != 1.0:
                    dm = cv2.resize(dm, None, fx=dm_scale, fy=dm_scale,
                                    interpolation=cv2.INTER_NEAREST)
                    sx *= dm_scale; sy *= dm_scale
                depth_info[Path(img.name).name] = (
                    dm, p[0]*sx, p[1]*sy, p[2]*sx, p[3]*sy)
            print(f"depth source: live workspace {DENSE} ({len(depth_info)} maps)")

    # ── DEPTH SOURCE 3: dense UTM PLY ────────────────────────────────────────
    dense_pts = None
    if not depth_info:
        pp = Path(dense_ply) if dense_ply is not None else None
        if pp is None:
            plys = sorted(OUT.glob("*.ply"), key=lambda f: f.stat().st_size,
                          reverse=True)
            pp = plys[0] if plys else None
        if pp is not None and pp.exists():
            import plyfile
            v  = plyfile.PlyData.read(str(pp))["vertex"].data
            dense_pts = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
            g = (dense_pts[:, 2] >= z_lo) & (dense_pts[:, 2] <= z_hi)
            dense_pts = dense_pts[g]
            n0 = len(dense_pts)
            keep_xy = np.ones(n0, bool)
            for c in (0, 1):
                q1, q3 = np.percentile(dense_pts[:, c], [25, 75])
                iqr = q3 - q1
                keep_xy &= (dense_pts[:, c] >= q1 - iqr_factor * iqr) & \
                           (dense_pts[:, c] <= q3 + iqr_factor * iqr)
            dense_pts = dense_pts[keep_xy]
            print(f"depth source: dense PLY {pp.name} "
                  f"({len(dense_pts):,}/{n0:,} pts after XY trim) - "
                  f"dense DSM ok, per-view gating off")
        else:
            print("depth source: NONE - sparse DSM only (ghost-prone).")

    # ── image resolver ───────────────────────────────────────────────────────
    search_dirs, seen = [], set()
    for cand in [IMAGES, npz_path.parent.parent,
                 Path(src2) if src2 is not None else None]:
        if cand is not None and str(cand) not in seen:
            search_dirs.append(cand); seen.add(str(cand))
    EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    name_index = {}
    for sd in search_dirs:
        if sd.exists():
            for f in sd.rglob('*'):
                if f.suffix.lower() in EXT and f.name not in name_index:
                    name_index[f.name] = f
    print(f"\n  image search dirs: {[str(x) for x in search_dirs]}")
    print(f"  indexed {len(name_index)} image files")

    def resolve(nm):
        p = IMAGES / str(nm)
        if p.exists():
            return p
        return name_index.get(Path(str(nm)).name)

    # ── bounds ───────────────────────────────────────────────────────────────
    BX = dense_pts[:, 0] if dense_pts is not None else X
    BY = dense_pts[:, 1] if dense_pts is not None else Y
    PAD  = 0.5
    xmin, xmax = BX.min() - PAD, BX.max() + PAD
    ymin, ymax = BY.min() - PAD, BY.max() + PAD

    cam_xmin, cam_xmax = centers_utm[:, 0].min(), centers_utm[:, 0].max()
    cam_ymin, cam_ymax = centers_utm[:, 1].min(), centers_utm[:, 1].max()
    cxmin, cxmax = cam_xmin - footprint_margin, cam_xmax + footprint_margin
    cymin, cymax = cam_ymin - footprint_margin, cam_ymax + footprint_margin
    if xmin < cxmin or xmax > cxmax or ymin < cymin or ymax > cymax:
        print(f"  ! bounds {xmax-xmin:.1f}x{ymax-ymin:.1f} m exceed camera "
              f"footprint +{footprint_margin:.0f} m margin "
              f"({cxmax-cxmin:.1f}x{cymax-cymin:.1f} m) - clamping "
              f"(residual outliers in cloud)")
        xmin, xmax = max(xmin, cxmin), min(xmax, cxmax)
        ymin, ymax = max(ymin, cymin), min(ymax, cymax)

    print(f"  ortho bounds: {xmax-xmin:.1f} x {ymax-ymin:.1f} m "
          f"(camera span {np.ptp(centers_utm[:,0]):.1f} x "
          f"{np.ptp(centers_utm[:,1]):.1f} m)")

    # ── pose table (nadir filter applied here) ───────────────────────────────
    poses = {}
    Ws = Hs = None
    n_missing = n_oblique = 0
    for k, nm in enumerate(names):
        if nadir_all[k] > max_nadir_deg:
            n_oblique += 1
            continue
        p = resolve(nm)
        if p is None:
            n_missing += 1
            continue
        cE, cN, cZ = centers_utm[k]
        agl    = max(cZ - ground_z, 1.0)
        foot_r = agl * (r_max / fx) + foot_margin
        if Ws is None:
            ex = cv2.imread(str(p))
            if ex is not None:
                Hf, Wf = ex.shape[:2]
                Ws = int(Wf * img_scale); Hs = int(Hf * img_scale)
                print(f"  image native {Wf}x{Hf} -> sampled {Ws}x{Hs}  "
                      f"GSD ~= {agl/fx_raw*100:.2f} cm/px")
        dmi = depth_info.get(Path(str(nm)).name)
        poses[k] = (Rs[k], ts[k], float(cE), float(cN), foot_r, str(p), dmi)
    n_dm = sum(1 for v in poses.values() if v[6] is not None)
    print(f"  found {len(poses)} camera paths, {n_dm} with depth maps"
          + (f"  ({n_missing} unresolved - check src2!)" if n_missing else "")
          + (f"  ({n_oblique} excluded as oblique >{max_nadir_deg:.0f} deg "
             f"off-nadir - kept in BA, not in ortho/DSM)" if n_oblique else ""))
    if not poses:
        raise FileNotFoundError("No images resolved (or all excluded as "
                                 "oblique - check max_nadir_deg).")

    # ── image cache (auto-disable if over budget) ────────────────────────────
    _cache = {}
    est_gb = len(poses) * Ws * Hs * 3 / 1e9
    if cache_images and est_gb > cache_budget_gb:
        cache_images = False
        print(f"  image cache DISABLED (~{est_gb:.1f} GB > {cache_budget_gb} GB "
              f"budget) -> lazy reads (slower; try img_scale=0.75 to re-enable)")
    elif cache_images:
        print(f"  image cache ~{est_gb:.1f} GB")

    def get_img(path):
        if cache_images and path in _cache:
            return _cache[path]
        bgr = cv2.imread(path)
        if bgr is None:
            return None
        if img_scale != 1.0:
            bgr = cv2.resize(bgr, (Ws, Hs), interpolation=cv2.INTER_AREA)
        if cache_images:
            _cache[path] = bgr
        return bgr

    # ── DSM build (source priority) ──────────────────────────────────────────
    owd = max(int((xmax - xmin) / dsm_grid), 1)
    ohd = max(int((ymax - ymin) / dsm_grid), 1)
    dsm_acc = np.full((ohd, owd), -np.inf, np.float32)
    hit     = np.zeros((ohd, owd), bool)

    def _splat(E, N, Zv):
        ok = (E >= xmin) & (E < xmax) & (N > ymin) & (N <= ymax) \
             & (Zv >= z_lo) & (Zv <= z_hi)
        if not ok.any():
            return
        ix = np.clip(((E[ok] - xmin) / dsm_grid).astype(int), 0, owd - 1)
        iy = np.clip(((ymax - N[ok]) / dsm_grid).astype(int), 0, ohd - 1)
        np.maximum.at(dsm_acc, (iy, ix), Zv[ok].astype(np.float32))
        hit[iy, ix] = True

    t0 = time.time()
    if n_dm > 0:
        print(f"\nDSM from {n_dm} depth maps @ {dsm_grid*100:.0f} cm/px ...")
        for k, (R, t, cE, cN, fr, ip, dmi) in poses.items():
            if dmi is None:
                continue
            dm, dfx, dfy, dcx, dcy = dmi
            vv, uu = np.mgrid[0:dm.shape[0]:dm_stride, 0:dm.shape[1]:dm_stride]
            dep = dm[::dm_stride, ::dm_stride].ravel()
            val = dep > 0
            if not val.any():
                continue
            u = uu.ravel()[val].astype(np.float64)
            v = vv.ravel()[val].astype(np.float64)
            dep = dep[val].astype(np.float64)
            pc  = np.column_stack([(u - dcx) / dfx * dep,
                                   (v - dcy) / dfy * dep, dep])
            P   = col_to_utm((pc - t) @ R)
            _splat(P[:, 0], P[:, 1], P[:, 2])
    elif dense_pts is not None:
        print(f"\nDSM from dense PLY @ {dsm_grid*100:.0f} cm/px ...")
        _splat(dense_pts[:, 0], dense_pts[:, 1], dense_pts[:, 2])
    else:
        print(f"\nDSM from sparse cloud @ {dsm_grid*100:.0f} cm/px ...")
        _splat(X, Y, Z)

    dsm_c = dsm_acc.copy()
    dsm_c[~hit] = np.nan
    dist, idx = distance_transform_edt(~hit, return_distances=True,
                                       return_indices=True)
    dsm_c = dsm_c[tuple(idx)]
    if dsm_smooth > 0:
        sig = max(dsm_smooth / dsm_grid, 1.0)
        dsm_c = cv2.GaussianBlur(dsm_c, (0, 0), sig)
    mask_c = dist <= (mask_dist / dsm_grid)
    print(f"  DSM {owd}x{ohd}  {100*hit.mean():.0f}% cells hit  "
          f"relief {np.nanmax(dsm_c)-np.nanmin(dsm_c):.2f} m  ({time.time()-t0:.0f}s)")

    def sample_dsm_mask(GX, GY):
        colm = ((GX - xmin) / dsm_grid - 0.5).astype(np.float32)
        rowm = ((ymax - GY) / dsm_grid - 0.5).astype(np.float32)
        z = cv2.remap(dsm_c, colm, rowm, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
        m = cv2.remap(mask_c.astype(np.float32), colm, rowm, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0) >= 0.5
        return z, m

    # ── composite: WINNER-TAKE-ALL + narrow top-2 seam feather ───────────────
    #   Weighted-average blend mixes misaligned views of the same thin leaf
    #   into a semi-transparent ghost (each camera projects an above-DSM leaf
    #   to a different XY). Fix: each output pixel takes the single most-
    #   nadir, depth-consistent view; a narrow top-2 feather is applied ONLY
    #   where the 2nd view is >= feather_lo as nadir as the best (i.e. right
    #   at seams), so leaf interiors stay razor-sharp.
    tol_model = depth_tol / s
    EPS = 1e-6

    def composite(GX, GY, GZ_tile):
        h, w = GX.shape
        gxr = GX.ravel(); gyr = GY.ravel()
        P   = utm_to_col(np.column_stack([gxr, gyr, GZ_tile.ravel()]))
        w1 = np.zeros((h, w), np.float32); c1 = np.zeros((h, w, 3), np.float32)
        w2 = np.zeros((h, w), np.float32); c2 = np.zeros((h, w, 3), np.float32)
        wu = np.zeros((h, w), np.float32); cu = np.zeros((h, w, 3), np.float32)

        for k, (R, t, cE, cN, foot_r, img_path, dmi) in poses.items():
            in_foot = np.hypot(gxr - cE, gyr - cN) < foot_r
            if not in_foot.any():
                continue
            bgr = get_img(img_path)
            if bgr is None:
                continue
            pc   = P @ R.T + t
            zraw = pc[:, 2]
            zc   = zraw.copy(); valid = zc > 1e-6; zc[~valid] = 1.0
            u    = fx * pc[:, 0] / zc + cx
            v    = fy * pc[:, 1] / zc + cy
            inb  = (valid & (u >= 0) & (u < Ws) & (v >= 0) & (v < Hs) & in_foot)
            wt   = np.clip(1.0 - np.hypot(u - cx, v - cy) / r_max,
                           0, 1).astype(np.float32) ** blend_pow
            wt[~inb] = 0.0
            smp = cv2.remap(bgr, u.reshape(h, w).astype(np.float32),
                            v.reshape(h, w).astype(np.float32), cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            wgu = wt.reshape(h, w)

            nb = wgu > wu
            cu[nb] = smp[nb]; wu[nb] = wgu[nb]

            if dmi is not None:
                dm, dfx, dfy, dcx, dcy = dmi
                ud = (dfx * pc[:, 0] / zc + dcx).reshape(h, w).astype(np.float32)
                vd = (dfy * pc[:, 1] / zc + dcy).reshape(h, w).astype(np.float32)
                dval = cv2.remap(dm, ud, vd, cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                ok = (dval > 0) & (np.abs(dval - zraw.reshape(h, w)) <= tol_model)
                wg = np.where(ok, wgu, 0.0).astype(np.float32)
            else:
                wg = wgu

            is1 = wg > w1
            w2[is1] = w1[is1]; c2[is1] = c1[is1]
            w1[is1] = wg[is1]; c1[is1] = smp[is1]
            is2 = (~is1) & (wg > w2)
            w2[is2] = wg[is2]; c2[is2] = smp[is2]

        ratio = w2 / np.maximum(w1, EPS)
        fw    = np.clip((ratio - feather_lo) / max(1.0 - feather_lo, EPS), 0, 1)
        a2    = (0.5 * fw)[..., None]
        gated = c1 * (1.0 - a2) + c2 * a2

        have_g = w1 > 0
        out = cu.copy()
        out[have_g] = gated[have_g]
        painted = (w1 > 0) | (wu > 0)
        fb      = (~have_g) & (wu > 0)
        return np.clip(out, 0, 255).astype(np.uint8), painted, fb

    # ── (1) PREVIEW — tiled, RAM bounded regardless of canvas size ──────────
    print(f"\nPREVIEW @ {preview_grid*100:.0f} cm/px ..."); t0 = time.time()
    owp = max(int((xmax - xmin) / preview_grid), 1)
    ohp = max(int((ymax - ymin) / preview_grid), 1)
    prev     = np.zeros((ohp, owp, 3), np.uint8)
    cov_all  = np.zeros((ohp, owp), bool)
    fb_all   = np.zeros((ohp, owp), bool)
    mask_all = np.zeros((ohp, owp), bool)
    pntx = (owp + tile_px - 1) // tile_px
    pnty = (ohp + tile_px - 1) // tile_px
    for ty in range(pnty):
        for tx in range(pntx):
            c0 = tx * tile_px;  r0 = ty * tile_px
            tw = min(tile_px, owp - c0); th = min(tile_px, ohp - r0)
            gx = xmin + (c0 + np.arange(tw) + 0.5) * preview_grid
            gy = ymax - (r0 + np.arange(th) + 0.5) * preview_grid
            GXt, GYt = np.meshgrid(gx, gy)
            GZt, mtile = sample_dsm_mask(GXt, GYt)
            ptile, covt, fbt = composite(GXt, GYt, GZt)
            prev[r0:r0+th, c0:c0+tw]     = ptile
            cov_all[r0:r0+th, c0:c0+tw]  = covt
            fb_all[r0:r0+th, c0:c0+tw]   = fbt
            mask_all[r0:r0+th, c0:c0+tw] = mtile
    prev[~mask_all] = 0
    fbtxt = (f"  fallback(ungated) {100*(fb_all & mask_all).mean():.1f}%"
             if n_dm > 0 else "")
    print(f"  {owp}x{ohp}  coverage {100*(cov_all & mask_all).mean():.0f}%"
          f"{fbtxt}  ({time.time()-t0:.0f}s)")
    plt.figure(figsize=(11, 11 * ohp / owp))
    plt.imshow(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB))
    plt.axis('off'); plt.title(f"PREVIEW {preview_grid*100:.0f} cm/px (best-view)")
    plt.show()

    # ── (2) FULL RES tiled GeoTIFF ───────────────────────────────────────────
    print(f"\nFULL @ {full_grid*100:.3f} cm/px, tiles {tile_px}px ..."); t0 = time.time()
    owf = int((xmax - xmin) / full_grid)
    ohf = int((ymax - ymin) / full_grid)
    print(f"  output canvas {owf}x{ohf} ({owf*ohf/1e6:.0f} MP)")
    transform = from_origin(xmin, ymax, full_grid, full_grid)
    tif       = OUT / "orthophoto.tif"
    ntx = (owf + tile_px - 1) // tile_px
    nty = (ohf + tile_px - 1) // tile_px
    print(f"  {ntx*nty} tiles ({ntx}x{nty})")

    with rasterio.open(tif, "w",
                       driver="GTiff", height=ohf, width=owf, count=4,
                       dtype="uint8", crs=f"EPSG:{EPSG}", transform=transform,
                       tiled=True, blockxsize=512, blockysize=512,
                       compress="deflate") as dst:
        dst.colorinterp = [rasterio.enums.ColorInterp.red,
                           rasterio.enums.ColorInterp.green,
                           rasterio.enums.ColorInterp.blue,
                           rasterio.enums.ColorInterp.alpha]
        k = 0
        for ty in range(nty):
            for tx in range(ntx):
                c0 = tx * tile_px;   r0 = ty * tile_px
                tw = min(tile_px, owf - c0)
                th = min(tile_px, ohf - r0)
                gx = xmin + (c0 + np.arange(tw) + 0.5) * full_grid
                gy = ymax - (r0 + np.arange(th) + 0.5) * full_grid
                GX, GY = np.meshgrid(gx, gy)
                GZ, mtile = sample_dsm_mask(GX, GY)
                tile, covg, _ = composite(GX, GY, GZ)
                alpha = (mtile & covg).astype(np.uint8) * 255
                tile[alpha == 0] = 0
                win = Window(c0, r0, tw, th)
                dst.write(tile[:, :, 2], 1, window=win)
                dst.write(tile[:, :, 1], 2, window=win)
                dst.write(tile[:, :, 0], 3, window=win)
                dst.write(alpha,          4, window=win)
                k += 1
                print(f"  tile {k}/{ntx*nty}", end="\r")
    print(f"\n  done ({time.time()-t0:.0f}s) -> {tif} "
          f"({tif.stat().st_size/1e6:.0f} MB)")
    print(f"  open in QGIS -- EPSG:{EPSG}, outside-field transparent")
    return tif
