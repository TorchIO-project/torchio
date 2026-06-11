#!/usr/bin/env python3
"""Pre-build script: generate markdown pages from sphinx-gallery example scripts.

Reads Python scripts from ``docs/examples/``, executes them, captures stdout
and matplotlib figures, and writes one ``.md`` page per example plus an
``index.md`` gallery page with thumbnail cards.

Generated files go into ``docs/examples/`` alongside the sources.
Run before ``zensical build``; use ``git checkout docs/examples/*.md`` to revert.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOCS_DIR = _REPO_ROOT / "docs"
_EXAMPLES_DIR = _DOCS_DIR / "examples"
_IMAGES_DIR = _DOCS_DIR / "images" / "gallery"

# Scripts to skip (animation / external deps)
_SKIP: set[str] = set()

# ---------------------------------------------------------------------------
# RST → Markdown helpers
# ---------------------------------------------------------------------------
_RST_ROLE_RE = re.compile(r":(?:func|class|meth|attr|mod|obj):`~?([^`]+)`")
_RST_REF_RE = re.compile(r":ref:`([^`]+)`")
_RST_LINK_RE = re.compile(r"`([^<]+)<([^>]+)>`_")
_RST_HEADING_RE = re.compile(r'^(.+)\n([=\-~^"]+)$', re.MULTILINE)
_RST_DOUBLE_BACKTICK_RE = re.compile(r"``([^`]+)``")


def _rst_to_md(text: str) -> str:
    """Best-effort RST inline markup → Markdown."""

    # Convert RST heading underlines to markdown headings
    def _heading_repl(m: re.Match) -> str:
        title = m.group(1).strip()
        char = m.group(2)[0]
        # = is h2, - is h3, ~ is h4
        level = {"=": "##", "-": "##", "~": "###", "^": "####"}.get(char, "##")
        return f"{level} {title}"

    text = _RST_HEADING_RE.sub(_heading_repl, text)
    text = _RST_ROLE_RE.sub(r"`\1`", text)
    text = _RST_REF_RE.sub(r"\1", text)
    text = _RST_LINK_RE.sub(r"[\1](\2)", text)
    text = _RST_DOUBLE_BACKTICK_RE.sub(r"`\1`", text)
    return text


# ---------------------------------------------------------------------------
# Script parser
# ---------------------------------------------------------------------------


def _parse_docstring(source: str) -> tuple[str, str, str]:
    """Return (title, description, remaining_source)."""
    # Match the module docstring
    m = re.match(r'^("""|\'\'\')(.*?)\1', source, re.DOTALL)
    if not m:
        return ("Untitled", "", source)
    doc = m.group(2).strip()
    rest = source[m.end() :].lstrip("\n")
    lines = doc.split("\n")
    title = lines[0].strip()
    # Skip the underline (===)
    desc_lines: list[str] = []
    i = 1
    while i < len(lines) and re.match(r"^[=\-~]+$", lines[i].strip()):
        i += 1
    desc_lines = [line.strip() for line in lines[i:]]
    description = _rst_to_md("\n".join(desc_lines).strip())
    return title, description, rest


def _parse_blocks(source: str) -> list[dict]:
    """Split source into text and code blocks.

    Sphinx-gallery uses ``# %%`` to delimit sections. Comment-only lines
    starting with ``#`` (after a section marker) become narrative text blocks.
    Everything else is a code block.
    """
    blocks: list[dict] = []
    # Split on # %% markers
    sections = re.split(r"^# %%\s*$", source, flags=re.MULTILINE)

    for section in sections:
        section = section.strip("\n")
        if not section:
            continue
        _parse_section(section, blocks)

    return blocks


def _parse_section(section: str, blocks: list[dict]) -> None:
    """Parse a single section into text/code blocks.

    Only unindented comment lines (column 0) at the *start* of a section or
    after a previous text block are treated as narrative.  Indented comments
    (inside class/function bodies) stay as code.
    """
    lines = section.split("\n")
    current_text: list[str] = []
    current_code: list[str] = []
    in_code = False  # once we see a code line, stay in code mode

    def flush_code() -> None:
        if current_code:
            blocks.append({"type": "code", "content": "\n".join(current_code)})
            current_code.clear()

    def flush_text() -> None:
        if current_text:
            text = _rst_to_md("\n".join(current_text))
            blocks.append({"type": "text", "content": text})
            current_text.clear()

    for line in lines:
        is_toplevel_comment = not in_code and (line.startswith("# ") or line == "#")
        if is_toplevel_comment:
            flush_code()
            text_line = line[2:] if line.startswith("# ") else ""
            current_text.append(text_line)
        else:
            flush_text()
            current_code.append(line)
            if line.strip():
                in_code = True

    flush_text()
    if current_code:
        code = "\n".join(current_code)
        if code.strip():
            blocks.append({"type": "code", "content": code})


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def _execute_example(
    script_path: Path,
) -> tuple[list[dict], list[Path]]:
    """Execute the script, collecting stdout and figures per code block.

    Returns a list of block dicts (with 'stdout' and 'figures' added to code
    blocks) and a flat list of all figure paths generated.
    """
    title, description, source = _parse_docstring(script_path.read_text())
    blocks = _parse_blocks(source)
    all_figures: list[Path] = []

    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    stem = script_path.stem
    fig_counter = 0

    # Build combined namespace for execution
    namespace: dict = {"__name__": "__main__"}

    # Change cwd to examples dir so relative paths (e.g., GIF) work
    original_cwd = os.getcwd()
    os.chdir(script_path.parent)

    try:
        for block in blocks:
            if block["type"] != "code":
                continue

            code = block["content"]
            # Remove plt.show() calls — we capture figures ourselves
            code_exec = re.sub(
                r"^\s*plt\.show\(\)\s*$",
                "",
                code,
                flags=re.MULTILINE,
            )

            plt.close("all")
            stdout_capture = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout_capture):
                    exec(code_exec, namespace)
            except Exception as exc:
                print(
                    f"  WARNING ({stem}): {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                block["stdout"] = f"{type(exc).__name__}: {exc}"
                block["figures"] = []
                continue

            block["stdout"] = stdout_capture.getvalue()

            # Save any animations as GIFs
            from matplotlib.animation import FuncAnimation as _FuncAnimation

            anim_paths: list[Path] = []
            for name, obj in list(namespace.items()):
                if isinstance(obj, _FuncAnimation):
                    fig_counter += 1
                    fname = f"{stem}_{fig_counter:03d}.gif"
                    fpath = _IMAGES_DIR / fname
                    try:
                        obj.save(str(fpath), writer="pillow")
                        anim_paths.append(fpath)
                        all_figures.append(fpath)
                        print(f"  Animation {fpath}", file=sys.stderr)
                    except Exception as exc:
                        print(
                            f"  WARNING ({stem}): animation save failed: {exc}",
                            file=sys.stderr,
                        )
                    # Remove from namespace to avoid re-saving
                    namespace[name] = None

            # Save any open static figures (skip if animation already captured)
            figs = [plt.figure(n) for n in plt.get_fignums()]
            fig_paths: list[Path] = []
            if not anim_paths:
                for fig in figs:
                    fig_counter += 1
                    fname = f"{stem}_{fig_counter:03d}.png"
                    fpath = _IMAGES_DIR / fname
                    fig.savefig(fpath, bbox_inches="tight", dpi=150)
                    fig_paths.append(fpath)
                    all_figures.append(fpath)
            block["figures"] = anim_paths + fig_paths
            plt.close("all")
    finally:
        os.chdir(original_cwd)

    return blocks, all_figures, title, description


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------


def _generate_page(
    script_path: Path,
    blocks: list[dict],
    title: str,
    description: str,
) -> str:
    """Generate the markdown content for a single example page."""
    lines: list[str] = []
    lines.append(f"# {title}\n")
    if description:
        lines.append(f"{description}\n")

    for block in blocks:
        if block["type"] == "text":
            lines.append(block["content"])
            lines.append("")
        elif block["type"] == "code":
            code = block["content"].rstrip()
            # Remove plt.show() for display
            code_display = re.sub(
                r"\n?\s*plt\.show\(\)\s*$",
                "",
                code,
            ).rstrip()
            if code_display.strip():
                lines.append(f"```python\n{code_display}\n```\n")

            stdout = block.get("stdout", "").strip()
            if stdout:
                lines.append(f"```text\n{stdout}\n```\n")

            for fig_path in block.get("figures", []):
                rel = os.path.relpath(
                    str(fig_path.relative_to(_DOCS_DIR)),
                    str(script_path.stem),  # page will be at examples/<stem>.md
                ).replace(os.sep, "/")
                # Relative from docs/examples/ to docs/images/gallery/
                rel = os.path.relpath(
                    str(fig_path),
                    str(_EXAMPLES_DIR),
                ).replace(os.sep, "/")
                lines.append(f"![{title}]({rel})\n")

    # Link to source
    gh_url = (
        f"https://github.com/TorchIO-project/torchio/blob/main/"
        f"docs/examples/{script_path.name}"
    )
    lines.append(f"\n---\n\n[View source on GitHub]({gh_url})\n")

    return "\n".join(lines)


def _generate_index(
    examples: list[dict],
) -> str:
    """Generate the gallery index page with a card grid."""
    lines: list[str] = []
    lines.append("# Examples Gallery\n")
    lines.append(
        "Below is a gallery of examples covering several features of TorchIO.\n",
    )

    # Use Material grid cards
    lines.append('<div class="grid cards" markdown>\n')

    for ex in examples:
        thumb = ex.get("thumbnail")
        title = ex["title"]
        stem = ex["stem"]
        desc = ex.get("description", "")
        # First line of description
        short_desc = desc.split("\n")[0] if desc else ""

        if thumb:
            thumb_rel = os.path.relpath(
                str(thumb),
                str(_EXAMPLES_DIR),
            ).replace(os.sep, "/")
            lines.append(f"-   [![{title}]({thumb_rel})]({stem}.md)\n")
            lines.append(f"    **[{title}]({stem}.md)**\n")
        else:
            lines.append(f"-   **[{title}]({stem}.md)**\n")

        if short_desc:
            lines.append(f"    {short_desc}\n")

    lines.append("</div>\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _make_thumbnail(figures: list[Path]) -> Path | None:
    """Create a thumbnail from the first figure, returning its path."""
    if not figures:
        return None
    first = figures[0]
    thumb_path = first.with_name(first.stem + "_thumb" + first.suffix)
    from PIL import Image as PILImage

    try:
        img = PILImage.open(first)
        if getattr(img, "n_frames", 1) > 1:
            thumb_path = thumb_path.with_suffix(".png")
            img.seek(0)
            img_copy = img.copy()
            img_copy.thumbnail((400, 300))
            img_copy.save(thumb_path)
        else:
            img.thumbnail((400, 300))
            img.save(thumb_path)
        print(f"  Thumbnail {thumb_path}", file=sys.stderr)
        return thumb_path
    except Exception:
        return first


def _load_cached_example(script: Path, hash_marker: str) -> dict | None:
    """Return cached example metadata if the markdown is up to date."""
    md_path = _EXAMPLES_DIR / f"{script.stem}.md"
    if not (md_path.exists() and hash_marker in md_path.read_text()):
        return None
    print("  Cached (hash match)", file=sys.stderr)
    title, description, _ = _parse_docstring(script.read_text())
    thumbs = sorted(_IMAGES_DIR.glob(f"{script.stem}_*_thumb.png"))
    return {
        "stem": script.stem,
        "title": title,
        "description": description,
        "thumbnail": thumbs[0] if thumbs else None,
    }


def _process_script(script: Path) -> dict:
    """Process a single example script into a metadata dict."""
    print(f"Processing {script.name}...", file=sys.stderr)

    source_hash = hashlib.md5(script.read_bytes()).hexdigest()[:12]
    hash_marker = f"<!-- hash:{source_hash} -->"

    cached = _load_cached_example(script, hash_marker)
    if cached is not None:
        return cached

    blocks, figures, title, description = _execute_example(script)
    page = _generate_page(script, blocks, title, description)
    page = f"{hash_marker}\n{page}"
    md_path = _EXAMPLES_DIR / f"{script.stem}.md"
    md_path.write_text(page)
    print(f"  Generated {md_path}", file=sys.stderr)

    return {
        "stem": script.stem,
        "title": title,
        "description": description,
        "thumbnail": _make_thumbnail(figures),
    }


def main() -> None:
    scripts = sorted(_EXAMPLES_DIR.glob("plot_*.py"))
    scripts = [s for s in scripts if s.name not in _SKIP]

    if not scripts:
        print("No example scripts found", file=sys.stderr)
        return

    examples = [_process_script(script) for script in scripts]

    index_content = _generate_index(examples)
    index_path = _EXAMPLES_DIR / "index.md"
    index_path.write_text(index_content)
    print(f"Generated {index_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
