import dask.array as da
import numpy as np
import pytest

from napari_dask_ndmeasure._measure import (
    available_stats,
    iter_measure_labels,
    measure_labels,
)


def _synthetic():
    img = np.arange(16, dtype="float32").reshape(4, 4)
    lab = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
        ],
        dtype="int32",
    )
    return da.from_array(img, chunks=(2, 2)), da.from_array(lab, chunks=(2, 2))


def test_measure_labels_default_stats():
    img, lab = _synthetic()
    table = measure_labels(img, lab)

    assert list(table.index) == [1, 2]
    assert table.loc[1, "area"] == 4
    assert table.loc[2, "area"] == 4
    # label 1 = pixels 0,1,4,5 -> mean 2.5; label 2 = pixels 10,11,14,15 -> mean 12.5
    assert table.loc[1, "mean_intensity"] == pytest.approx(2.5)
    assert table.loc[2, "mean_intensity"] == pytest.approx(12.5)
    # geometric centroid of label 1 (rows 0-1, cols 0-1) is (0.5, 0.5)
    assert table.loc[1, "centroid_y"] == pytest.approx(0.5)
    assert table.loc[1, "centroid_x"] == pytest.approx(0.5)


def test_measure_labels_only_requested_stats():
    img, lab = _synthetic()
    table = measure_labels(img, lab, stats=("area",))
    assert list(table.columns) == ["area"]


def test_measure_labels_scale_adds_um_columns():
    img, lab = _synthetic()
    table = measure_labels(
        img, lab, stats=("area", "centroid"), scale=(0.5, 0.2)
    )
    assert table.loc[1, "area_um2"] == pytest.approx(4 * 0.5 * 0.2)
    assert table.loc[1, "centroid_y_um"] == pytest.approx(0.5 * 0.5)
    assert table.loc[1, "centroid_x_um"] == pytest.approx(0.5 * 0.2)


def test_measure_labels_shape_mismatch_raises():
    img = da.zeros((4, 4))
    lab = da.zeros((4, 5))
    with pytest.raises(ValueError, match="shape mismatch"):
        measure_labels(img, lab)


def test_measure_labels_unknown_stat_raises():
    img, lab = _synthetic()
    with pytest.raises(ValueError, match="unknown stats"):
        measure_labels(img, lab, stats=("bogus",))


def test_measure_labels_no_objects_returns_empty():
    img = da.zeros((4, 4), dtype="float32")
    lab = da.zeros((4, 4), dtype="int32")
    table = measure_labels(img, lab)
    assert table.empty


def test_available_stats_matches_measure_labels_options():
    img, lab = _synthetic()
    # every advertised stat should actually be usable
    table = measure_labels(img, lab, stats=available_stats())
    assert not table.empty


def test_measure_labels_rechunks_a_single_giant_chunk():
    # A bare numpy array, wrapped naively, becomes ONE chunk covering the
    # whole array -> dask_image.ndmeasure would then need the whole thing in
    # RAM. measure_labels must rechunk internally regardless.
    img = da.asarray(np.arange(400 * 400, dtype="float32").reshape(400, 400))
    lab = da.asarray(np.ones((400, 400), dtype="int32"))
    assert img.numblocks == (1, 1)  # sanity: reproduces the bug precondition

    table = measure_labels(img, lab, stats=("area",))
    assert table.loc[1, "area"] == 400 * 400


def _drain(gen):
    """Collect every yielded value from *gen*, plus its final return value."""
    progress = []
    try:
        while True:
            progress.append(next(gen))
    except StopIteration as stop:
        return progress, stop.value


def test_iter_measure_labels_yields_progress_then_returns_table():
    img, lab = _synthetic()
    gen = iter_measure_labels(img, lab, stats=("area", "mean_intensity"))
    progress, table = _drain(gen)

    assert [p[:2] for p in progress] == [(1, 2), (2, 2)]
    assert [p[2] for p in progress] == ["area", "mean_intensity"]
    assert table.loc[1, "area"] == 4


def test_iter_measure_labels_return_value_matches_measure_labels():
    img, lab = _synthetic()
    _, table_from_iter = _drain(iter_measure_labels(img, lab))
    table_from_wrapper = measure_labels(img, lab)
    assert table_from_iter.equals(table_from_wrapper)
