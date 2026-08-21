# ═══════════════════════════════════════════════════════════════════════════
# BATCH WRAPPER — walk <DJI_ROOT>/<Crop>/<DJI_YYYYMMDDHHMMSS_XXX_desc>/
#   -> run_pipeline(...). Skips flights whose pipeline_output/ already looks
#   complete, so the whole tree can be re-run safely until every flight is
#   ready for orthomosaic.batch.run_ortho_batch().
#
#   Defaults to the locked-intrinsic (camera_model="DJI_DEWARP") production
#   configuration documented in reconstruction/odm_light_cuda_v11.py — this
#   is what removes doming on new flights; there is no separate "repair the
#   archive" step to run first. If a flight was already processed with an
#   older/self-calibrating config and needs redoing, pass force=True (or
#   force_sparse=True through **extra_kwargs) for that flight.
# ═══════════════════════════════════════════════════════════════════════════
from pathlib import Path
import json
import time
import traceback

from common.drive import mount_drive
from common.paths import DJI_ROOT, IMG_EXT, checkpoints_dir, ortho_inputs_path
from reconstruction.odm_light_cuda_v11 import run_pipeline, diagnose_doming

# Locked-intrinsic production preset (see odm_light_cuda_v11 module docstring,
# "LOCKED-INTRINSIC OPERATING MODE"). Works for both bare-soil and full-canopy
# scenes — the doming mechanism this fixes is about camera geometry, not
# ground cover, so there is no separate preset per crop stage.
LOCKED_INTRINSICS = dict(
    camera_model="DJI_DEWARP",
    ba_refine_focal_length=False,
    ba_refine_principal_point=False,   # NEVER free this — see module docstring
    ba_refine_extra_params=False,
    use_prior_position=True,
    prior_std_xy=0.05,
    prior_std_z=0.035,                 # matches measured RTK vertical sigma
    prior_ba_refine_focal=True,
    prior_ba_refine_extra=True,
    prior_ba_refine_pp=False,
    prior_ba_strict=True,
    dense_geom_consistency=True,       # required for depth_bundle.npz
    fp16=True,
)


def _ply_vertex_count(ply: Path) -> int:
    """Read the binary/ASCII PLY header and return the vertex count (0 if
    unreadable)."""
    try:
        with open(ply, "rb") as f:
            for _ in range(64):
                line = f.readline()
                if not line:
                    break
                s = line.decode("ascii", "ignore").strip()
                if s.startswith("element vertex"):
                    return int(s.split()[-1])
                if s == "end_header":
                    break
    except Exception:
        pass
    return 0


def _has_images(d: Path) -> bool:
    for p in d.iterdir():
        if p.is_file() and p.suffix in IMG_EXT:
            return True
    return False


def find_flight_dirs(root: Path):
    """<root>/<Crop>/DJI_* plus any DJI_* sitting directly under <root>,
    deduplicated by resolved path (guards against the same flight being
    reachable twice via a Drive shortcut)."""
    root = Path(root)
    out, seen = [], set()
    for crop in sorted(p for p in root.iterdir() if p.is_dir()):
        if crop.name.startswith((".", "_")):
            continue
        if crop.name.startswith("DJI_"):            # flight directly under root
            candidates = [crop]
        else:
            candidates = [p for p in sorted(crop.iterdir())
                         if p.is_dir() and p.name.startswith("DJI_")]
        for fl in candidates:
            rp = fl.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            out.append(fl)
    return out


def pipeline_status(src: Path, *, do_dense=True, require_depth_bundle=False,
                     min_dense_points=1000):
    """Returns (is_complete, reason). 'Complete' == everything
    orthomosaic.run_ortho() needs."""
    src = Path(src)
    out = src / "pipeline_output"
    ckpt = checkpoints_dir(src)

    if not out.is_dir():
        return False, "no pipeline_output/"

    npz = ortho_inputs_path(src)
    if not npz.is_file() or npz.stat().st_size < 1024:
        return False, "missing/empty ortho_inputs.npz"

    sfm = ckpt / "sfm"
    if not sfm.is_dir() or not any(sfm.iterdir()):
        return False, "missing sfm/ (refined sparse model)"

    if do_dense:
        ply = ckpt / "fused.ply"
        if not ply.is_file():
            return False, "missing fused.ply"
        n = _ply_vertex_count(ply)
        if n < min_dense_points:
            return False, f"fused.ply has {n} points (< {min_dense_points})"

    if require_depth_bundle:
        db = out / "depth_bundle.npz"
        if not db.is_file() or db.stat().st_size < 1024:
            return False, "missing/empty depth_bundle.npz"

    return True, "complete"


def preflight_dewarp(root=DJI_ROOT, crops=None):
    """Report which NEW flights are missing DewarpData in their XMP — those
    would silently fall back to self-calibrating OPENCV, the exact
    configuration that caused the archive-wide doming this preset fixes.
    Seconds, changes nothing. Worth running before a big batch."""
    from reconstruction.odm_light_cuda_v11 import _get_dewarp
    root = Path(root)
    flights = find_flight_dirs(root)
    if crops is not None:
        crops_lower = {c.lower() for c in crops}
        flights = [f for f in flights if f.parent.name.lower() in crops_lower]
    ok, bad = [], []
    for src in flights:
        jpgs = sorted(p for p in src.iterdir() if p.suffix in IMG_EXT)[:5]
        dw = next((_get_dewarp(p) for p in jpgs if _get_dewarp(p) is not None), None)
        key = f"{src.parent.name}/{src.name}"
        (ok if dw is not None else bad).append(key)
    print(f"preflight_dewarp: {len(ok)} ok, {len(bad)} missing DewarpData")
    for k in bad:
        print(f"  ! missing DewarpData: {k}")
    return ok, bad


def run_pipeline_batch(
    root=DJI_ROOT,
    *,
    crops=None,                 # e.g. ["Ribelmais", "Bohne"]; None = all
    only=None,                  # substring filter on flight folder name
    force=False,                # re-run even if outputs look complete
    dry_run=False,              # list what would run, execute nothing
    require_depth_bundle=False, # treat depth_bundle.npz as mandatory
    min_dense_points=1000,
    # ── run_pipeline parameters — locked-intrinsic preset by default ───────
    **overrides,                # override/extend LOCKED_INTRINSICS, e.g. work=...
):
    root = Path(root)
    flights = find_flight_dirs(root)
    if crops is not None:
        crops_lower = {c.lower() for c in crops}
        flights = [f for f in flights if f.parent.name.lower() in crops_lower]
    if only:
        flights = [f for f in flights if only.lower() in f.name.lower()]

    cfg = dict(LOCKED_INTRINSICS)
    cfg.update(overrides)
    cfg.setdefault("do_dense", True)
    if require_depth_bundle:
        cfg["dense_geom_consistency"] = True

    print(f"Root : {root}")
    print(f"Found: {len(flights)} flight folder(s)\n")

    results, todo = {}, []
    for src in flights:
        key = f"{src.parent.name}/{src.name}"
        if not _has_images(src):
            print(f"[skip ] {key:<60s} no images in folder")
            results[key] = {"status": "no_images"}
            continue
        ok, why = pipeline_status(
            src, do_dense=cfg["do_dense"],
            require_depth_bundle=require_depth_bundle,
            min_dense_points=min_dense_points,
        )
        if ok and not force:
            print(f"[done ] {key:<60s} {why}")
            results[key] = {"status": "skipped", "reason": why}
            continue
        print(f"[queue] {key:<60s} {'forced' if force else why}")
        todo.append((key, src, why))

    print(f"\n{len(todo)} folder(s) to process.")
    if dry_run or not todo:
        return results

    if force:
        cfg.setdefault("force_sparse", True)

    for i, (key, src, why) in enumerate(todo, 1):
        print("\n" + "=" * 78)
        print(f"[{i}/{len(todo)}] {key}   ({why})")
        print("=" * 78)
        t0 = time.time()
        try:
            products = run_pipeline(src, **cfg)
            ok, why2 = pipeline_status(
                src, do_dense=cfg["do_dense"],
                require_depth_bundle=require_depth_bundle,
                min_dense_points=min_dense_points,
            )
            d = diagnose_doming(src, verbose=False)
            results[key] = {
                "status": "ok" if ok else "incomplete",
                "reason": why2,
                "minutes": round((time.time() - t0) / 60, 1),
                "z_rms_m": d["z_rms"] if d else None,
                "products": products,
            }
            print(f"\n-> {key}: {'OK' if ok else 'INCOMPLETE - ' + why2} "
                  f"({results[key]['minutes']} min"
                  f"{', z_rms ' + format(d['z_rms'], '.3f') + ' m' if d else ''})")
        except Exception as e:
            results[key] = {
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
                "minutes": round((time.time() - t0) / 60, 1),
            }
            print(f"\n-> {key}: FAILED - {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 78)
    print("BATCH SUMMARY")
    print("=" * 78)
    for k, v in results.items():
        extra = v.get("reason") or v.get("error") or ""
        print(f"  {v['status']:<10s} {k:<60s} {extra}")
    (root / "_batch_status.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "products"}
         for k, v in results.items()}, indent=2))
    print(f"\nStatus written to {root / '_batch_status.json'}")
    return results


if __name__ == "__main__":
    mount_drive()

    # 0) DewarpData present everywhere? (seconds, changes nothing)
    preflight_dewarp(DJI_ROOT)

    # 1) inspect what is missing without running anything
    run_pipeline_batch(DJI_ROOT, dry_run=True)

    # 2) actually process everything still missing — locked-intrinsic preset
    # results = run_pipeline_batch(DJI_ROOT, require_depth_bundle=True)

    # 3) scope to specific crops or a date substring
    # results = run_pipeline_batch(DJI_ROOT, crops=["Mais", "Bohne"])
    # results = run_pipeline_batch(DJI_ROOT, only="202605")

    # 4) after the batch: anything with a high vertical residual worth a
    #    second look? (no separate rebuild step — just force=True it)
    # for k, v in results.items():
    #     if v.get("z_rms_m") is not None and v["z_rms_m"] > 0.15:
    #         print(k, v["z_rms_m"])
    # results2 = run_pipeline_batch(DJI_ROOT, only="<flight-name>", force=True)
