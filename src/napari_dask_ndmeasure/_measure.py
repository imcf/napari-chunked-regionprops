"""Out-of-core, chunk-parallel regionprops-style measurements.

Wraps ``dask_image.ndmeasure`` so a labelled OME-ZARR volume with hundreds of
thousands of objects can be measured directly from disk/dask — unlike
``skimage.measure.regionprops``, which needs the full labelled + intensity
array in RAM.
"""

from __future__ import annotations

from typing import Sequence

import dask.array as da
import numpy as np
import pandas as pd

#: stat name -> dask_image.ndmeasure function name, and whether it needs the
#: real intensity image (False = geometric only, computed on a dummy image).
_STATS: dict[str, tuple[str, bool]] = {
    "area": ("area", False),
    "centroid": ("center_of_mass", False),
    "mean_intensity": ("mean", True),
    "std_intensity": ("standard_deviation", True),
    "min_intensity": ("minimum", True),
    "max_intensity": ("maximum", True),
    "weighted_centroid": ("center_of_mass", True),
}

DEFAULT_STATS = ("area", "centroid", "mean_intensity", "std_intensity")


def available_stats() -> tuple[str, ...]:
    """Stat names accepted by ``measure_labels``'s ``stats=`` argument."""
    return tuple(_STATS)


def measure_labels(
    image: da.Array,
    labels: da.Array,
    *,
    stats: Sequence[str] = DEFAULT_STATS,
    scale: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Measure every non-background object in *labels* against *image*.

    Parameters
    ----------
    image : da.Array
        Intensity image, same shape as *labels*.
    labels : da.Array
        Integer label array (0 = background).
    stats : sequence of str, optional
        Which measurements to compute — see :func:`available_stats`.
        Default: area, centroid, mean/std intensity.
    scale : sequence of float or None, optional
        Physical size per axis (e.g. from the napari layer's ``scale``).
        When given, ``area`` is reported in physical units (voxel count ×
        voxel volume) and centroid columns get a ``_um`` twin alongside the
        pixel-coordinate one.

    Returns
    -------
    pandas.DataFrame
        One row per label, indexed by label id.

    Examples
    --------
    >>> import dask.array as da
    >>> import numpy as np
    >>> img = da.from_array(np.arange(16).reshape(4, 4).astype("float32"))
    >>> lab = da.from_array(
    ...     np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]])
    ... )
    >>> table = measure_labels(img, lab, stats=("area", "mean_intensity"))
    >>> table.loc[1, "area"]
    4
    """
    if image.shape != labels.shape:
        raise ValueError(
            f"shape mismatch: image={image.shape} labels={labels.shape}"
        )
    unknown = set(stats) - set(_STATS)
    if unknown:
        raise ValueError(
            f"unknown stats {sorted(unknown)}; choose from {available_stats()}"
        )

    import dask_image.ndmeasure as ndm

    ids = da.unique(labels[labels > 0]).compute()
    if ids.size == 0:
        return pd.DataFrame(index=pd.Index([], name="label"))

    dummy = da.ones_like(image, dtype="float32")
    columns: dict[str, np.ndarray] = {}
    for stat in stats:
        fn_name, needs_intensity = _STATS[stat]
        fn = getattr(ndm, fn_name)
        src = image if needs_intensity else dummy
        result = fn(src, labels, ids).compute()
        if result.ndim == 1:
            columns[stat] = result
        else:
            # center_of_mass: one column per spatial axis
            axis_names = "zyx"[-result.shape[1] :]
            for i, ax in enumerate(axis_names):
                columns[f"{stat}_{ax}"] = result[:, i]

    table = pd.DataFrame(columns, index=pd.Index(ids, name="label"))

    if scale is not None:
        scale = np.asarray(scale, dtype="float64")
        if "area" in table.columns:
            table["area_um3" if len(scale) == 3 else "area_um2"] = table[
                "area"
            ] * np.prod(scale)
        for prefix in ("centroid", "weighted_centroid"):
            cols = [c for c in table.columns if c.startswith(f"{prefix}_")]
            for c, s in zip(cols, scale):
                table[f"{c}_um"] = table[c] * s

    return table
