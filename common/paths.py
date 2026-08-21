"""Shared path conventions.

Flight folders live at:
    <DJI_ROOT>/<Crop>/DJI_YYYYMMDDHHMMSS_XXX_description/

Outputs always derive from Path(src) relative to the flight folder itself —
never hardcoded — so they land as siblings to existing directories (e.g.
plot_analysis_summary/ as a sibling of pipeline_output/, one level up at the
field ROOT).
"""
from pathlib import Path

DJI_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/DJI")

IMG_EXT = {".jpg", ".jpeg", ".tif", ".tiff", ".JPG", ".JPEG", ".TIF", ".TIFF"}

SUB = "pipeline_output"


def pipeline_output_dir(src) -> Path:
    """<src>/pipeline_output/"""
    return Path(src) / SUB


def checkpoints_dir(src) -> Path:
    """<src>/pipeline_output/_checkpoints/"""
    return pipeline_output_dir(src) / "_checkpoints"


def plot_analysis_dir(root) -> Path:
    """<root>/plot_analysis_summary/  (sibling of each flight's
    pipeline_output/, one level up at the field ROOT — not per-flight)."""
    return Path(root) / "plot_analysis_summary"


def ortho_inputs_path(src) -> Path:
    """<src>/pipeline_output/ortho_inputs.npz"""
    return pipeline_output_dir(src) / "ortho_inputs.npz"


def depth_bundle_path(src) -> Path:
    """<src>/pipeline_output/depth_bundle.npz"""
    return pipeline_output_dir(src) / "depth_bundle.npz"


def orthophoto_path(src) -> Path:
    """<src>/pipeline_output/orthophoto.tif"""
    return pipeline_output_dir(src) / "orthophoto.tif"
