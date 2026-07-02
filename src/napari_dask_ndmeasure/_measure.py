"""Out-of-core, chunk-parallel regionprops-style measurements.

Wraps ``dask_image.ndmeasure`` so a labelled OME-ZARR volume with hundreds of
thousands of objects can be measured directly from disk/dask — unlike
``skimage.measure.regionprops``, which needs the full labelled + intensity
array in RAM.
"""

from __future__ import annotations

from typing import Any, Generator, Sequence

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

# Target ~128 MB dask chunks (dask's own default) when an input isn't already
# chunked sensibly. Without this, a plain numpy array (or a dask array
# wrapped without chunks=) becomes ONE giant chunk, and dask_image.ndmeasure
# then has to hold the whole thing in RAM during .compute() — silently
# defeating the entire point of this plugin on exactly the huge images it's
# meant for.
_CHUNK_TARGET = "auto"


def available_stats() -> tuple[str, ...]:
    """Return the stat names accepted by ``stats=`` on measurement functions.

    Returns
    -------
    tuple of str
        Every valid entry for :func:`measure_labels`'s and
        :func:`iter_measure_labels`'s ``stats`` argument.

    Examples
    --------
    >>> available_stats()
    ('area', 'centroid', 'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 'weighted_centroid')
    """
    return tuple(_STATS)


def _ensure_chunked(image: Any, labels: Any) -> tuple[da.Array, da.Array]:
    """Coerce *image*/*labels* to dask arrays with matching, bounded chunks.

    Handles plain numpy arrays and dask arrays that happen to be a single
    giant chunk (both common when a napari layer isn't a chunked OME-ZARR
    pyramid) — rechunking to ``_CHUNK_TARGET`` keeps memory bounded
    regardless of how the layer's data was originally stored.

    Parameters
    ----------
    image : array-like
        Intensity image — a numpy array, or a dask array with any chunking
        (including a single chunk covering the whole array).
    labels : array-like
        Integer label array, same shape as *image*.

    Returns
    -------
    tuple of (da.Array, da.Array)
        ``(image, labels)`` as dask arrays sharing an identical, bounded
        chunk grid (each chunk targets ``_CHUNK_TARGET`` bytes).

    Examples
    --------
    >>> import numpy as np
    >>> image, labels = _ensure_chunked(
    ...     np.zeros((4, 4), dtype="float32"), np.zeros((4, 4), dtype="int32")
    ... )
    >>> image.chunks == labels.chunks
    True
    """
    labels = da.asarray(labels).rechunk(_CHUNK_TARGET)
    image = da.asarray(image).rechunk(labels.chunks)
    return image, labels


def iter_measure_labels(
    image: da.Array,
    labels: da.Array,
    *,
    stats: Sequence[str] = DEFAULT_STATS,
    scale: Sequence[float] | None = None,
) -> Generator[tuple[int, int, str], None, pd.DataFrame]:
    """Generator version of :func:`measure_labels` — yields progress.

    Identical contract to :func:`measure_labels`, except it yields a
    progress tuple after each stat finishes, and returns (rather than just
    computes) the final table. This is the shape napari's
    ``napari.qt.threading.thread_worker`` expects for a progress-reporting
    background task: drive it with ``result = yield from
    iter_measure_labels(...)`` inside a ``@thread_worker``-decorated
    generator function, and its ``.yielded``/``.returned`` Qt signals fire
    with exactly these values.

    Parameters
    ----------
    image : da.Array
        Intensity image, same shape as *labels*. Rechunked automatically if
        it isn't already sensibly chunked (see :func:`_ensure_chunked`).
    labels : da.Array
        Integer label array (0 = background).
    stats : sequence of str, optional
        Which measurements to compute — see :func:`available_stats`.
        Default: area, centroid, mean/std intensity.
    scale : sequence of float or None, optional
        Physical size per axis. See :func:`measure_labels`.

    Yields
    ------
    tuple of (int, int, str)
        ``(done, total, stat_name)`` — ``done``/``total`` are 1-indexed
        (``done`` reaches ``total`` on the last yield), ``stat_name`` is the
        stat that just finished.

    Returns
    -------
    pandas.DataFrame
        The final measurement table — identical to what
        :func:`measure_labels` returns for the same arguments. Retrieve it
        via the generator's ``StopIteration.value``, or transparently via
        ``yield from`` (see above).

    Examples
    --------
    >>> import dask.array as da
    >>> import numpy as np
    >>> img = da.from_array(np.arange(16).reshape(4, 4).astype("float32"))
    >>> lab = da.from_array(
    ...     np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]])
    ... )
    >>> gen = iter_measure_labels(img, lab, stats=("area", "mean_intensity"))
    >>> next(gen)
    (1, 2, 'area')
    >>> next(gen)
    (2, 2, 'mean_intensity')
    >>> try:
    ...     next(gen)
    ... except StopIteration as stop:
    ...     table = stop.value
    >>> int(table.loc[1, "area"])
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

    image, labels = _ensure_chunked(image, labels)

    ids = da.unique(labels[labels > 0]).compute()
    if ids.size == 0:
        return pd.DataFrame(index=pd.Index([], name="label"))

    dummy = da.ones_like(image, dtype="float32")
    columns: dict[str, np.ndarray] = {}
    total = len(stats)
    for done, stat in enumerate(stats, start=1):
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
        yield done, total, stat

    table = pd.DataFrame(columns, index=pd.Index(ids, name="label"))

    if scale is not None:
        scale_arr = np.asarray(scale, dtype="float64")
        if "area" in table.columns:
            table["area_um3" if len(scale_arr) == 3 else "area_um2"] = table[
                "area"
            ] * np.prod(scale_arr)
        for prefix in ("centroid", "weighted_centroid"):
            cols = [c for c in table.columns if c.startswith(f"{prefix}_")]
            for c, s in zip(cols, scale_arr):
                table[f"{c}_um"] = table[c] * s

    return table


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
        Intensity image, same shape as *labels*. Rechunked automatically if
        it isn't already sensibly chunked (see :func:`_ensure_chunked`).
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
    >>> int(table.loc[1, "area"])
    4
    """
    gen = iter_measure_labels(image, labels, stats=stats, scale=scale)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value
