import dask.array as da
import numpy as np
import pytest

from napari_dask_ndmeasure._measure import available_stats, measure_labels


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
    table = measure_labels(img, lab, stats=("area", "centroid"), scale=(0.5, 0.2))
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
