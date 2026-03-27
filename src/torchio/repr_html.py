"""HTML table representations for Jupyter notebooks."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

import humanize

if TYPE_CHECKING:
    from .data.image import Image
    from .data.subject import Subject

# Minimal inline CSS so the tables look decent in any Jupyter theme.
_STYLE = """\
<style scoped>
.tio-table {
  border-collapse: collapse;
  font-family: monospace;
  font-size: 11px;
  margin: 4px 0;
}
.tio-table th, .tio-table td {
  text-align: left;
  padding: 3px 10px;
  border: 1px solid #ccc;
}
.tio-table th {
  font-weight: bold;
}
.tio-header {
  font-family: sans-serif;
  font-size: 14px;
  font-weight: bold;
  margin-bottom: 4px;
}
.tio-section {
  font-family: sans-serif;
  font-size: 13px;
  font-weight: bold;
  margin-top: 8px;
  margin-bottom: 2px;
}
</style>"""


def _pluralize(word: str, n: int) -> str:
    if n == 1:
        return f"1 {word}"
    return f"{n} {word}s" if not word.endswith("x") else f"{n} {word}es"


def _row(key: str, value: str) -> str:
    return f"<tr><th>{escape(key)}</th><td>{escape(value)}</td></tr>"


def image_to_html(image: Image) -> str:
    """Build an HTML table for an Image."""
    cls_name = type(image).__name__
    rows: list[str] = []

    rows.append(_row("Type", cls_name))

    if image.is_loaded:
        rows.append(_row("Shape", str(image.shape)))
        rows.append(
            _row("Spacing", "({})".format(", ".join(f"{s:.2f}" for s in image.spacing)))
        )
        rows.append(_row("Orientation", "".join(image.orientation) + "+"))
        rows.append(_row("dtype", str(image.data.dtype).replace("torch.", "")))
        rows.append(_row("Memory", humanize.naturalsize(image.memory, binary=True)))
    elif image.path is not None:
        rows.append(_row("Path", str(image.path)))
        # Shape/spacing can be read lazily from header
        try:
            rows.append(_row("Shape", str(image.shape)))
            sp = "({})".format(", ".join(f"{s:.2f}" for s in image.spacing))
            rows.append(_row("Spacing", sp))
            rows.append(_row("Orientation", "".join(image.orientation) + "+"))
        except Exception:
            pass

    # Annotations
    for name, pts in image.points.items():
        rows.append(_row(f"Points '{name}'", _pluralize("point", pts.num_points)))

    for name, boxes in image.bounding_boxes.items():
        rows.append(_row(f"BBoxes '{name}'", _pluralize("box", boxes.num_boxes)))

    table = f'{_STYLE}\n<table class="tio-table">\n' + "\n".join(rows) + "\n</table>"
    return table


def subject_to_html(subject: Subject) -> str:
    """Build an HTML table for a Subject."""
    parts: list[str] = [_STYLE]

    # Images table
    if subject.images:
        parts.append('<div class="tio-section">Images</div>')
        header = (
            "<tr><th>Name</th><th>Type</th><th>Shape</th>"
            "<th>Spacing</th><th>Orientation</th></tr>"
        )
        rows: list[str] = [header]
        for name, image in subject.images.items():
            img_type = type(image).__name__
            try:
                shape = str(image.shape)
                sp = "({})".format(", ".join(f"{s:.2f}" for s in image.spacing))
                orient = "".join(image.orientation) + "+"
            except Exception:
                shape = sp = orient = "?"
            rows.append(
                f"<tr><td>{escape(name)}</td><td>{escape(img_type)}</td>"
                f"<td>{escape(shape)}</td><td>{escape(sp)}</td>"
                f"<td>{escape(orient)}</td></tr>"
            )
        parts.append('<table class="tio-table">\n' + "\n".join(rows) + "\n</table>")

    # Points table
    if subject.points:
        parts.append('<div class="tio-section">Points</div>')
        rows = ["<tr><th>Name</th><th>Count</th><th>Axes</th></tr>"]
        for name, pts in subject.points.items():
            rows.append(
                f"<tr><td>{escape(name)}</td>"
                f"<td>{_pluralize('point', pts.num_points)}</td>"
                f"<td>{escape(pts.axes)}</td></tr>"
            )
        parts.append('<table class="tio-table">\n' + "\n".join(rows) + "\n</table>")

    # BoundingBoxes table
    if subject.bounding_boxes:
        parts.append('<div class="tio-section">Bounding Boxes</div>')
        rows = ["<tr><th>Name</th><th>Count</th><th>Format</th></tr>"]
        for name, boxes in subject.bounding_boxes.items():
            fmt = f"{boxes.format.axes} ({boxes.format.representation.value})"
            rows.append(
                f"<tr><td>{escape(name)}</td>"
                f"<td>{_pluralize('box', boxes.num_boxes)}</td>"
                f"<td>{escape(fmt)}</td></tr>"
            )
        parts.append('<table class="tio-table">\n' + "\n".join(rows) + "\n</table>")

    # Metadata table
    if subject.metadata:
        parts.append('<div class="tio-section">Metadata</div>')
        rows = ["<tr><th>Key</th><th>Value</th></tr>"]
        for key, value in subject.metadata.items():
            rows.append(_row(key, str(value)))
        parts.append('<table class="tio-table">\n' + "\n".join(rows) + "\n</table>")

    return "\n".join(parts)
