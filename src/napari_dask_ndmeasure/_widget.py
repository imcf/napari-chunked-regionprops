"""Dock widget: pick an Image + Labels layer, measure, browse/export the table."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ._measure import DEFAULT_STATS, available_stats, measure_labels

if TYPE_CHECKING:
    import napari


def _level_data(layer, level: int):
    """Return one dask array from a (possibly multiscale) layer's data."""
    data = layer.data
    return data[level] if layer.multiscale else data


class MeasureWidget(QWidget):
    """Measure a Labels layer against an Image layer via dask-image.

    Works directly on the layers' backing dask arrays — never pulls a
    hundred-thousand-object volume into RAM the way
    ``skimage.measure.regionprops`` would.
    """

    def __init__(self, napari_viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = napari_viewer
        self._table = None  # last computed pandas.DataFrame

        layout = QVBoxLayout()
        self.setLayout(layout)

        layers_box = QGroupBox("Layers")
        layers_layout = QVBoxLayout()
        layers_box.setLayout(layers_layout)

        row = QHBoxLayout()
        row.addWidget(QLabel("Image:"))
        self.image_combo = QComboBox()
        row.addWidget(self.image_combo)
        layers_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Labels:"))
        self.labels_combo = QComboBox()
        row.addWidget(self.labels_combo)
        layers_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Pyramid level:"))
        self.level_spin = QSpinBox()
        self.level_spin.setMinimum(0)
        self.level_spin.setToolTip(
            "Multiscale layers only: 0 = full resolution. Coarser levels are "
            "faster but less precise — a quick preview before a full run."
        )
        row.addWidget(self.level_spin)
        layers_layout.addLayout(row)

        layout.addWidget(layers_box)

        stats_box = QGroupBox("Measurements")
        stats_layout = QVBoxLayout()
        stats_box.setLayout(stats_layout)
        self.stats_list = QListWidget()
        for name in available_stats():
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in DEFAULT_STATS else Qt.Unchecked)
            self.stats_list.addItem(item)
        stats_layout.addWidget(self.stats_list)
        layout.addWidget(stats_box)

        self.measure_btn = QPushButton("Measure")
        self.measure_btn.clicked.connect(self._on_measure_clicked)
        layout.addWidget(self.measure_btn)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.results_table = QTableWidget()
        layout.addWidget(self.results_table)

        self.save_btn = QPushButton("Save CSV…")
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)

        self._viewer.layers.events.inserted.connect(self._refresh_layer_choices)
        self._viewer.layers.events.removed.connect(self._refresh_layer_choices)
        self.labels_combo.currentIndexChanged.connect(self._update_level_range)
        self._refresh_layer_choices()

    def _update_level_range(self):
        _, labels_layer = self._selected_layers()
        if labels_layer is not None and labels_layer.multiscale:
            self.level_spin.setMaximum(len(labels_layer.data) - 1)
        else:
            self.level_spin.setMaximum(0)

    def _refresh_layer_choices(self, event=None):
        from napari.layers import Image, Labels

        prev_image = self.image_combo.currentText()
        prev_labels = self.labels_combo.currentText()
        self.image_combo.clear()
        self.labels_combo.clear()
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                self.image_combo.addItem(layer.name)
            elif isinstance(layer, Labels):
                self.labels_combo.addItem(layer.name)
        _restore(self.image_combo, prev_image)
        _restore(self.labels_combo, prev_labels)
        self._update_level_range()

    def _selected_layers(self):
        image_name = self.image_combo.currentText()
        labels_name = self.labels_combo.currentText()
        if not image_name or not labels_name:
            return None, None
        return self._viewer.layers[image_name], self._viewer.layers[labels_name]

    def _selected_stats(self) -> tuple[str, ...]:
        stats = []
        for i in range(self.stats_list.count()):
            item = self.stats_list.item(i)
            if item.checkState() == Qt.Checked:
                stats.append(item.text())
        return tuple(stats)

    def _on_measure_clicked(self):
        from napari.qt.threading import thread_worker

        image_layer, labels_layer = self._selected_layers()
        if image_layer is None or labels_layer is None:
            QMessageBox.warning(self, "Missing layers", "Pick an Image and a Labels layer.")
            return
        stats = self._selected_stats()
        if not stats:
            QMessageBox.warning(self, "No measurements selected", "Check at least one.")
            return

        level = self.level_spin.value()
        image_data = _level_data(image_layer, level)
        labels_data = _level_data(labels_layer, level)
        scale = tuple(labels_layer.scale[-labels_data.ndim :])

        self.measure_btn.setEnabled(False)
        self.status_label.setText("Measuring…")

        @thread_worker
        def _run():
            return measure_labels(image_data, labels_data, stats=stats, scale=scale)

        worker = _run()
        worker.returned.connect(self._on_measured)
        worker.errored.connect(self._on_measure_error)
        worker.start()

    def _on_measure_error(self, exc: Exception):
        self.measure_btn.setEnabled(True)
        self.status_label.setText("")
        QMessageBox.critical(self, "Measurement failed", str(exc))

    def _on_measured(self, table):
        self.measure_btn.setEnabled(True)
        self._table = table
        self.status_label.setText(f"{len(table)} objects measured.")
        self._populate_table(table)
        self.save_btn.setEnabled(not table.empty)

        _, labels_layer = self._selected_layers()
        if labels_layer is not None and not table.empty:
            # feeds napari's built-in per-label feature display/coloring.
            features = table.reset_index()
            features["label"] = features["label"].astype(int)
            labels_layer.features = features

    def _populate_table(self, table):
        self.results_table.clear()
        self.results_table.setRowCount(len(table))
        columns = ["label", *table.columns]
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(columns)
        for row, (label_id, values) in enumerate(table.iterrows()):
            self.results_table.setItem(row, 0, QTableWidgetItem(str(label_id)))
            for col, value in enumerate(values, start=1):
                self.results_table.setItem(row, col, QTableWidgetItem(f"{value:.4g}"))

    def _on_save_clicked(self):
        if self._table is None or self._table.empty:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save measurements", "measurements.csv", "CSV (*.csv)")
        if path:
            self._table.to_csv(path)


def _restore(combo: QComboBox, previous: str) -> None:
    idx = combo.findText(previous)
    if idx >= 0:
        combo.setCurrentIndex(idx)
