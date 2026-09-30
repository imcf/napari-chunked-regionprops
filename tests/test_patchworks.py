"""patchworks object tables: loaded, related across layers, used as ids."""

import types

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from napari_chunked_regionprops import _patchworks  # noqa: E402
from napari_chunked_regionprops._widget import (  # noqa: E402
    MeasureWidget,
    _ids_hint,
)


def _write(store, name, lab, columns):
    """A label group with a patchworks-style object table."""
    root = zarr.open_group(str(store), mode="a")
    group = root.require_group(f"labels/{name}")
    group.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
    group.attrs["n_objects"] = int(len(columns["label"]))
    arr = group.create_array("0", shape=lab.shape, dtype=lab.dtype)
    arr[...] = lab
    table = group.create_group("table")
    for col, values in columns.items():
        values = np.asarray(values)
        table.create_array(col, shape=values.shape, dtype=values.dtype)[...] = (
            values
        )
    table.attrs["patchworks_table"] = {
        "labels": {
            "created": None,
            "n_objects": int(len(columns["label"])),
            "shape": list(lab.shape),
        },
        "columns": list(columns),
    }
    return f"{store}/labels/{name}"


@pytest.fixture
def store(tmp_path):
    """Two cells (ids 3 and 7: not sequential), a long thin cilium in cell 7
    and a nucleus in each cell."""
    cells = np.zeros((12, 40), "int32")
    cells[:, :18] = 3
    cells[:, 20:] = 7
    nuclei = np.zeros_like(cells)
    nuclei[4:8, 4:10] = 1
    nuclei[4:8, 26:32] = 2
    cilia = np.zeros_like(cells)
    cilia[2, 22:38] = 5  # 16 voxels long, 1 wide
    s = tmp_path / "s.zarr"
    _write(
        s,
        "cells",
        cells,
        {
            "label": [3, 7],
            "area_voxels": [216, 240],
            "centroid_y": [5.5, 5.5],
            "centroid_x": [8.5, 29.5],
            "bbox_min_y": [0, 0],
            "bbox_min_x": [0, 20],
            "bbox_max_y": [11, 11],
            "bbox_max_x": [17, 39],
            "cov_yy": [11.9, 11.9],
        },
    )
    _write(
        s,
        "nuclei",
        nuclei,
        {
            "label": [1, 2],
            "area_voxels": [24, 24],
            "centroid_y": [5.5, 5.5],
            "centroid_x": [6.5, 28.5],
            "cells_id": [3, 7],
        },
    )
    _write(
        s,
        "cilia",
        cilia,
        {
            "label": [5],
            "area_voxels": [16],
            "centroid_y": [2.0],
            "centroid_x": [29.5],
            "bbox_min_y": [2],
            "bbox_min_x": [22],
            "bbox_max_y": [2],
            "bbox_max_x": [37],
            "cells_id": [7],
        },
    )
    return s, {"cells": cells, "nuclei": nuclei, "cilia": cilia}


def _add(viewer, store, name, arrays, *, tag=True):
    return viewer.add_labels(
        arrays[name],
        name=name,
        metadata={"patchworks_labels": f"{store}/labels/{name}"} if tag else {},
    )


def test_table_found_from_metadata_or_source_path(store):
    s, _ = store
    tagged = types.SimpleNamespace(
        name="cells",
        metadata={"patchworks_labels": f"{s}/labels/cells"},
        source=None,
    )
    assert _patchworks.label_group(tagged) == f"{s}/labels/cells"
    # An OME-Zarr reader gives the store as the source path
    opened = types.SimpleNamespace(
        name="cells", metadata={}, source=types.SimpleNamespace(path=str(s))
    )
    assert _patchworks.label_group(opened) == f"{s}/labels/cells"
    none = types.SimpleNamespace(name="x", metadata={}, source=None)
    assert _patchworks.label_group(none) is None


def test_stale_table_is_refused(store):
    s, _ = store
    group = f"{s}/labels/cells"
    assert _patchworks.is_current(group)
    zarr.open_group(group, mode="r+").attrs["n_objects"] = 3  # re-segmented
    assert not _patchworks.is_current(group)
    with pytest.raises(ValueError, match="other labels"):
        _patchworks.read_table(group, corrected=False)
    assert _patchworks.raw_ids(group) is None


def test_widget_loads_the_table_without_measuring(make_napari_viewer, store):
    s, arrays = store
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((12, 40), "float32"), name="image")
    _add(viewer, s, "cells", arrays)
    widget = MeasureWidget(viewer)

    assert widget._table is not None
    assert list(widget._table.index) == [3, 7]
    assert "cov_yy" not in widget._table  # internal columns stay hidden
    assert "no measuring needed" in widget.status_label.text()
    assert "area_voxels" in viewer.layers["cells"].features


def test_measuring_uses_the_table_ids_and_keeps_its_columns(
    qtbot, make_napari_viewer, monkeypatch, store, tmp_path
):
    s, arrays = store
    viewer = make_napari_viewer()
    viewer.add_image(np.ones((12, 40), "float32"), name="image")
    layer = _add(viewer, s, "nuclei", arrays)
    np.testing.assert_array_equal(_ids_hint(layer, 0), [1, 2])

    widget = MeasureWidget(viewer)
    widget._save_dir = tmp_path

    def _boom(*a, **k):
        raise AssertionError("no scan for ids when the table lists them")

    monkeypatch.setattr("napari_chunked_regionprops._measure.da.unique", _boom)
    widget._table = None
    widget._on_measure_clicked()
    qtbot.waitUntil(
        lambda: widget._table is not None and "mean_intensity" in widget._table,
        timeout=5000,
    )
    assert widget._table.loc[1, "area_voxels"] == 24
    assert widget._table.loc[2, "cells_id"] == 7  # joined from the table


def test_selecting_highlights_parents_and_children(make_napari_viewer, store):
    s, arrays = store
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((12, 40), "float32"), name="image")
    for name in ("cells", "nuclei", "cilia"):
        _add(viewer, s, name, arrays)
    widget = MeasureWidget(viewer)

    # A cell: its nucleus and cilium light up in their own layers
    widget.labels_combo.setCurrentText("cells")
    related = widget.related_objects(viewer.layers["cells"], {7})
    assert related == {"nuclei": ("child", {2}), "cilia": ("child", {5})}
    row = list(widget._table.index).index(7)
    widget.results_table.item(row, 0).setSelected(True)
    nuclei = viewer.layers["nuclei"]
    rendered = nuclei.colormap.map(np.array([1, 2]))
    assert tuple(rendered[1]) == (1.0, 0.0, 1.0, 1.0)  # its nucleus
    assert rendered[0][3] < 0.5  # the other one, dimmed

    widget._on_clear_selection_clicked()
    assert "nuclei" not in widget._related_originals
    assert type(nuclei.colormap).__name__ != "DirectLabelColormap" or (
        nuclei.colormap.map(np.array([1]))[0][3] > 0.5
    )

    # A cilium: its cell lights up
    widget.labels_combo.setCurrentText("cilia")
    assert widget.related_objects(viewer.layers["cilia"], {5}) == {
        "cells": ("parent", {7})
    }


def test_camera_frames_a_thin_object_by_its_length(make_napari_viewer, store):
    s, arrays = store
    viewer = make_napari_viewer()
    layer = _add(viewer, s, "cilia", arrays)
    widget = MeasureWidget(viewer)
    widget._center_camera_on_label(layer, 5)
    canvas = min(viewer.window._qt_viewer.canvas.size)
    # 16 voxels long: framed on its length, not on 16 ** 0.5 = 4 voxels
    assert viewer.camera.zoom == pytest.approx(canvas / (16 * 1.5))


def test_corrected_view_with_patchworks(store):
    """With patchworks installed, the table shown is the reviewed one."""
    patchworks = pytest.importorskip("patchworks")
    s, _ = store
    try:
        rv = patchworks.Review(str(s))
        rv.decide("nuclei", 1, "wrong")
    except Exception as exc:  # pragma: no cover - patchworks versions
        pytest.skip(f"patchworks review unavailable: {exc}")
    table = _patchworks.read_table(f"{s}/labels/nuclei")
    assert list(table.index) == [2]
    raw = _patchworks.read_table(f"{s}/labels/nuclei", corrected=False)
    assert list(raw.index) == [1, 2]


def test_label_zarr_without_a_table_is_measured_as_before(
    qtbot, make_napari_viewer, tmp_path
):
    """A label zarr with no object table -- tagged by patchworks' viewer
    or not -- is simply measured from its voxels, ids scanned."""
    lab = np.zeros((6, 8), "int32")
    lab[1:3, 1:3] = 4
    lab[4:6, 5:8] = 9
    root = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
    group = root.require_group("labels/cells")
    group.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
    arr = group.create_array("0", shape=lab.shape, dtype=lab.dtype)
    arr[...] = lab

    viewer = make_napari_viewer()
    viewer.add_image(np.ones((6, 8), "float32"), name="image")
    import dask.array as da

    level1 = group.create_array("1", shape=(3, 4), dtype=lab.dtype)
    level1[...] = lab[::2, ::2]
    layer = viewer.add_labels(
        [da.from_zarr(arr), da.from_zarr(level1)],
        name="cells",
        multiscale=True,
        metadata={"patchworks_labels": f"{tmp_path}/s.zarr/labels/cells"},
    )
    assert _patchworks.label_group(layer) is None
    widget = MeasureWidget(viewer)
    widget._save_dir = tmp_path
    assert widget._table is None  # nothing to load: no table
    widget._on_measure_clicked()
    qtbot.waitUntil(lambda: widget._table is not None, timeout=5000)
    assert list(widget._table.index) == [4, 9]
    assert widget._table.loc[9, "area_voxels"] == 6
    assert "objects measured" in widget.status_label.text()
