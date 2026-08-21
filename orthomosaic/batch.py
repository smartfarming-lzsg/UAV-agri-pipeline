"""Batch wrapper: run_ortho() over every flight with an ortho_inputs.npz.

Reuses batch.run_pipeline_batch.find_flight_dirs() for tree discovery so this
and the reconstruction batch wrapper agree on what counts as a flight folder.
"""
from pathlib import Path

from batch.run_pipeline_batch import find_flight_dirs
from common.paths import DJI_ROOT, ortho_inputs_path, orthophoto_path
from .compositor_v5_2 import run_ortho


def run_ortho_batch(root=DJI_ROOT, *, crops=None, only=None, force: bool = False,
                    **ortho_kwargs):
    """Run run_ortho() over every flight under root that has
    pipeline_output/ortho_inputs.npz. Skips flights that already have
    orthophoto.tif unless force=True.

    Recommended call:
        run_ortho_batch(DJI_ROOT, img_scale=1.0, dsm_smooth=0.10,
                        blend_pow=6.0, feather_lo=0.90, max_nadir_deg=20.0,
                        force=False, full_grid=None, tile_px=1024)
    """
    root = Path(root)
    flights = find_flight_dirs(root)
    if crops is not None:
        crops_lower = {c.lower() for c in crops}
        flights = [f for f in flights if f.parent.name.lower() in crops_lower]
    if only:
        flights = [f for f in flights if only.lower() in f.name.lower()]

    results = {}
    for flight in flights:
        key = f"{flight.parent.name}/{flight.name}"
        npz_path = ortho_inputs_path(flight)
        if not npz_path.is_file():
            print(f"[skip ] {key:<60s} no ortho_inputs.npz — run reconstruction first")
            continue
        tif = orthophoto_path(flight)
        if tif.exists() and not force:
            print(f"[done ] {key:<60s} orthophoto.tif exists")
            results[key] = tif
            continue
        print(f"\n{'='*70}\n{key}\n{'='*70}")
        try:
            results[key] = run_ortho(npz_path, **ortho_kwargs)
        except Exception as e:
            print(f"  ! FAILED: {type(e).__name__}: {e}")
            results[key] = None
    return results
