import dask.array as da
import numpy as np

from napari_dask_ndmeasure._widget import MeasureWidget


def _add_layers(viewer):
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
    viewer.add_image(img, name="image", scale=(0.5, 0.2))
    viewer.add_labels(lab, name="labels", scale=(0.5, 0.2))


def test_widget_populates_layer_choices(make_napari_viewer):
    viewer = make_napari_viewer()
    _add_layers(viewer)
    widget = MeasureWidget(viewer)

    assert [
        widget.image_combo.itemText(i)
        for i in range(widget.image_combo.count())
    ] == ["image"]
    assert [
        widget.labels_combo.itemText(i)
        for i in range(widget.labels_combo.count())
    ] == ["labels"]


def test_widget_measure_end_to_end(qtbot, make_napari_viewer):
    viewer = make_napari_viewer()
    _add_layers(viewer)
    widget = MeasureWidget(viewer)

    widget._on_measure_clicked()
    qtbot.waitUntil(lambda: widget._table is not None, timeout=5000)

    assert widget._table is not None
    assert list(widget._table.index) == [1, 2]
    assert widget.results_table.rowCount() == 2
    assert widget.save_btn.isEnabled()

    labels_layer = viewer.layers["labels"]
    assert "area" in labels_layer.features.columns


def test_widget_level_range_updates_for_multiscale(make_napari_viewer):
    viewer = make_napari_viewer()
    lab0 = da.zeros((8, 8), dtype="int32", chunks=(4, 4))
    lab1 = da.zeros((4, 4), dtype="int32", chunks=(4, 4))
    viewer.add_labels([lab0, lab1], name="pyramid_labels", multiscale=True)
    viewer.add_image(np.zeros((4, 4), dtype="float32"), name="image")

    widget = MeasureWidget(viewer)
    widget.labels_combo.setCurrentText("pyramid_labels")
    assert widget.level_spin.maximum() == 1
