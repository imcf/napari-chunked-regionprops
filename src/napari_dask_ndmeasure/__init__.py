"""napari-dask-ndmeasure — out-of-core regionprops for huge labelled images.

Measures Labels layers directly from their backing dask/zarr arrays via
``dask_image.ndmeasure``, so a hundred-thousand-object OME-ZARR volume never
needs to be pulled into RAM the way ``skimage.measure.regionprops`` would.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from ._measure import available_stats, measure_labels

try:
    __version__ = _pkg_version("napari-dask-ndmeasure")
except PackageNotFoundError:  # not installed (e.g. running from a checkout)
    __version__ = "0+unknown"

__all__ = ["measure_labels", "available_stats"]
