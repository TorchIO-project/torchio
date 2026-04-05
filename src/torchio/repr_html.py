"""HTML table representations for Jupyter notebooks."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING
from typing import Any

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
    """Build an HTML representation for an Image with an embedded plot."""
    cls_name = type(image).__name__
    rows: list[str] = []

    rows.append(_row("Type", cls_name))

    try:
        sp = "({})".format(", ".join(f"{s:.2f}" for s in image.spacing))
        ori = "({})".format(", ".join(f"{o:.2f}" for o in image.origin))
        angles = "({})".format(
            ", ".join(f"{a:.1f}°" for a in image.affine.euler_angles)
        )
        dt = str(image.dtype).replace("torch.", "")
        rows.append(_row("Channels", str(image.num_channels)))
        rows.append(_row("Spatial shape", str(image.spatial_shape)))
        rows.append(_row("Spacing", f"{sp} mm"))
        rows.append(_row("Origin", f"{ori} mm"))
        rows.append(_row("Orientation", "".join(image.orientation) + "+"))
        rows.append(_row("Euler angles", angles))
        rows.append(_row("dtype", dt))
        rows.append(_row("Memory", humanize.naturalsize(image.memory, binary=True)))
    except Exception:
        if image.path is not None:
            rows.append(_row("Path", str(image.path)))

    for name, pts in image.points.items():
        rows.append(_row(f"Points '{name}'", _pluralize("point", pts.num_points)))
    for name, boxes in image.bounding_boxes.items():
        rows.append(_row(f"BBoxes '{name}'", _pluralize("box", boxes.num_boxes)))

    table = f'{_STYLE}\n<table class="tio-table">\n' + "\n".join(rows) + "\n</table>"

    if plot_html := _try_plot_base64(image):
        return f"{plot_html}\n{table}"
    return table


def _render_fig_base64(render_fn: Any) -> str | None:
    """Call *render_fn* and return a base64 <img> tag.

    Renders via the Agg canvas without changing the global matplotlib
    backend, so interactive plotting still works afterwards.
    """
    import base64
    import io

    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = render_fn()
        if fig is None:
            return None
        FigureCanvasAgg(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return f'<img src="data:image/png;base64,{b64}" />'
    except Exception:
        return None


def _try_plot_base64(image: Image) -> str | None:
    """Render a 3-slice plot as an inline base64 ``<img>`` tag."""
    try:
        from .visualization import plot_image
    except ImportError:
        return None

    return _render_fig_base64(
        lambda: plot_image(image, show=False, figsize=(9, 3)),
    )


def subject_to_html(subject: Subject) -> str:
    """Build an HTML table for a Subject."""
    parts: list[str] = [_STYLE]

    if subject.images:
        parts.append(_images_table_html(subject))
    if subject.points:
        parts.append(_points_table_html(subject))
    if subject.bounding_boxes:
        parts.append(_bboxes_table_html(subject))
    if subject.metadata:
        parts.append(_metadata_table_html(subject))

    if plot_html := _try_subject_plot_base64(subject):
        parts.append(plot_html)

    return "\n".join(parts)


def _images_table_html(subject: Subject) -> str:
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
    return (
        '<div class="tio-section">Images</div>\n'
        '<table class="tio-table">\n' + "\n".join(rows) + "\n</table>"
    )


def _points_table_html(subject: Subject) -> str:
    rows = ["<tr><th>Name</th><th>Count</th><th>Axes</th></tr>"]
    for name, pts in subject.points.items():
        rows.append(
            f"<tr><td>{escape(name)}</td>"
            f"<td>{_pluralize('point', pts.num_points)}</td>"
            f"<td>{escape(pts.axes)}</td></tr>"
        )
    return (
        '<div class="tio-section">Points</div>\n'
        '<table class="tio-table">\n' + "\n".join(rows) + "\n</table>"
    )


def _bboxes_table_html(subject: Subject) -> str:
    rows = ["<tr><th>Name</th><th>Count</th><th>Format</th></tr>"]
    for name, boxes in subject.bounding_boxes.items():
        fmt = f"{boxes.format.axes} ({boxes.format.representation.value})"
        rows.append(
            f"<tr><td>{escape(name)}</td>"
            f"<td>{_pluralize('box', boxes.num_boxes)}</td>"
            f"<td>{escape(fmt)}</td></tr>"
        )
    return (
        '<div class="tio-section">Bounding Boxes</div>\n'
        '<table class="tio-table">\n' + "\n".join(rows) + "\n</table>"
    )


def _metadata_table_html(subject: Subject) -> str:
    rows = ["<tr><th>Key</th><th>Value</th></tr>"]
    for key, value in subject.metadata.items():
        rows.append(_row(key, str(value)))
    return (
        '<div class="tio-section">Metadata</div>\n'
        '<table class="tio-table">\n' + "\n".join(rows) + "\n</table>"
    )


def _try_subject_plot_base64(subject: Subject) -> str | None:
    """Render a subject grid plot as an inline base64 ``<img>`` tag."""
    try:
        from .visualization import plot_subject
    except ImportError:
        return None

    num_images = len(subject.images)
    return _render_fig_base64(
        lambda: plot_subject(
            subject,
            show=False,
            figsize=(12, 3 * num_images),
        ),
    )
