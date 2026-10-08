"""Object tables written by patchworks, next to the labels they describe.

patchworks (https://github.com/imcf/patchworks) measures every object of a
label image once, on the cluster, and stores the result inside the label
group, one zarr array per column::

    image.zarr/labels/cilia_labels/
        0/ 1/ 2/ ...    the label pyramid
        table/          label, area_voxels, centroid_z, ..., cyto_labels_id

Column names follow this plugin's own (``label`` index, ``area_voxels``,
``centroid_<axis>``, ``area_um3``, ``centroid_<axis>_um``), so such a table
loads here as if measured here -- without reading any voxels -- and adds
what this plugin cannot know: which object of another label image each one
belongs to (``<parent>_id``), bounding boxes, and, from patchworks' review,
shape and position columns and the reviewer's corrections.

Nothing here needs patchworks itself: the table is read with zarr. When
patchworks *is* installed, its corrected view is used instead (review
decisions applied, children counted, shape and position derived).
"""

from __future__ import annotations

import logging
import re
import zipfile
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

#: Layer metadata key patchworks' viewer sets to the layer's label group.
METADATA_KEY = "patchworks_labels"
_TABLE_KEY = "patchworks_table"
_PROVENANCE_KEY = "patchworks"


def label_group(layer) -> str | None:
    """The patchworks label group a Labels layer shows, if it has a table.

    From the metadata patchworks' viewer sets, or else from the layer's
    source path when an OME-Zarr reader opened it (the store itself, or its
    ``labels/<layer name>`` group).
    """
    candidates = []
    meta = (getattr(layer, "metadata", None) or {}).get(METADATA_KEY)
    if meta:
        candidates.append(str(meta))
    source = getattr(getattr(layer, "source", None), "path", None)
    if source:
        path = str(source).rstrip("/")
        candidates += [path, f"{path}/labels/{layer.name}"]
    for group in candidates:
        if has_table(group):
            return group
    return None


def has_table(group: str) -> bool:
    try:
        return _TABLE_KEY in _open(f"{group}/table").attrs
    except Exception:  # noqa: BLE001 - any unreadable store: no table
        return False


#: A ``.zip`` path component: the archive name, then the end or a separator.
_ZIP_PART = re.compile(r"^(.*?\.zip)(?=$|[/\\])(.*)$", re.IGNORECASE)


def _open(path: str):
    """Open a group read-only, also *inside* a zipped store.

    patchworks bundles a store as one ``.zip`` holding ``<name>.zarr/...``,
    and its viewer points a layer at ``bundle.zip/labels/<name>``: a path
    through a file, which a plain ``zarr.open_group`` cannot follow.
    """
    import zarr

    m = _ZIP_PART.match(str(path))
    if m is None:
        return zarr.open_group(path, mode="r")
    archive, inner = m.group(1), m.group(2).replace("\\", "/").strip("/")
    with zipfile.ZipFile(archive) as zf:
        tops = {n.split("/", 1)[0] for n in zf.namelist() if "/" in n}
    # One top-level folder: the bundled store. Otherwise the zip *is* it.
    parts = [tops.pop()] if len(tops) == 1 else []
    prefix = "/".join(parts + ([inner] if inner else []))
    return zarr.open_group(
        zarr.storage.ZipStore(archive, mode="r"), path=prefix, mode="r"
    )


def _fingerprint(group) -> dict[str, Any]:
    """What identifies one label image: patchworks' own fingerprint."""
    attrs = dict(group.attrs)
    multiscales = attrs.get("multiscales") or (attrs.get("ome") or {}).get(
        "multiscales"
    )
    try:
        path = multiscales[0]["datasets"][0]["path"]
    except (KeyError, IndexError, TypeError):
        path = "0"
    prov = attrs.get(_PROVENANCE_KEY) or {}
    return {
        "created": prov.get("created"),
        "n_objects": attrs.get("n_objects"),
        "shape": list(group[path].shape),
    }


def is_current(group: str) -> bool:
    """Whether the table still describes the labels next to it.

    A table left behind by a re-segmentation must not be shown against the
    new labels -- the ids would point at different objects.
    """
    try:
        meta = dict(_open(f"{group}/table").attrs[_TABLE_KEY])
        return meta.get("labels") == _fingerprint(_open(group))
    except Exception:  # noqa: BLE001 - unreadable: treated as stale
        return False


def raw_ids(group: str) -> np.ndarray | None:
    """Every object id present in the label image, from its table: the
    exact id set, so measuring needs no scan of the volume for it."""
    if not is_current(group):
        return None
    try:
        return np.asarray(_open(f"{group}/table")["label"][...], dtype="int64")
    except Exception:  # noqa: BLE001 - no usable ids: scan the volume
        return None


def _split(group: str) -> tuple[str, str]:
    store, sep, name = str(group).rstrip("/").rpartition("/labels/")
    if not sep:
        raise ValueError(f"{group} is not a <store>/labels/<name> group")
    return store, name


def read_table(group: str, *, corrected: bool = True) -> pd.DataFrame:
    """The table of *group* as a DataFrame indexed by ``label``.

    With *corrected* and patchworks installed, the review's corrected view
    (rejected objects dropped, joined ones combined, counts, shape and
    position added); otherwise the table as measured. Columns of internal
    use only (the second moments, ``cov_*``) are left out.

    Raises
    ------
    ValueError
        If the table was computed from different labels than those stored.
    """
    import pandas as pd

    if not is_current(group):
        raise ValueError(
            f"the table in {group} was computed from other labels than the "
            "ones stored there; recompute it (patchworks tables <store>)"
        )
    df = None
    if corrected:
        try:
            from patchworks import Review
        except ImportError:
            Review = None
        if Review is not None:
            store, name = _split(group)
            try:
                df = Review(store).effective(name)
            except Exception:
                logger.warning(
                    "patchworks could not build the corrected view of %s; "
                    "showing the table as measured",
                    group,
                    exc_info=True,
                )
    if df is None:
        table = _open(f"{group}/table")
        listed = list(table.attrs[_TABLE_KEY]["columns"])
        names = listed + sorted(
            k for k in table.array_keys() if k not in listed
        )
        df = pd.DataFrame({n: table[n][...] for n in names}).set_index("label")
    df = df.drop(columns=[c for c in df.columns if c.startswith("cov_")])
    df.index.name = "label"
    return df


def parent_columns(table: pd.DataFrame, layer_names) -> dict[str, str]:
    """``{parent layer name: column}`` for this table's parent-id columns
    that name a Labels layer in the viewer (``cyto_labels_id`` ->
    ``cyto_labels``)."""
    out = {}
    for col in table.columns:
        if col.endswith("_id") and col[: -len("_id")] in layer_names:
            out[col[: -len("_id")]] = col
    return out
