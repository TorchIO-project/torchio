"""Python-Markdown extension that executes ``python plot`` code blocks.

Finds fenced code blocks marked with ``python plot`` and replaces them with
the rendered matplotlib figure.  Works as a Markdown **Preprocessor** so it
runs before the fenced-code extension converts the blocks to HTML.

Register in *zensical.toml* (or *mkdocs.yml*) as a markdown extension::

    [project.markdown_extensions.plot_directive]

The extension must be importable, so ensure the directory containing this
file is on ``PYTHONPATH`` (or install it as a package).
"""

from __future__ import annotations

import hashlib
import os
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
from markdown import Extension  # noqa: E402
from markdown.preprocessors import Preprocessor  # noqa: E402

# Directory where generated plot images are stored (relative to site_dir).
_PLOT_DIR = Path('docs') / 'images' / 'plots'

# Match ```python plot ... ``` blocks, allowing leading whitespace.
_FENCE_RE = re.compile(
    r'^(\s*)(`{3,})python\s+plot\s*$\n(.*?)^\1\2\s*$',
    re.MULTILINE | re.DOTALL,
)


class PlotPreprocessor(Preprocessor):
    """Replace ``python plot`` fenced blocks with generated images."""

    def run(self, lines: list[str]) -> list[str]:
        text = '\n'.join(lines)
        if 'python plot' not in text:
            return lines

        _PLOT_DIR.mkdir(parents=True, exist_ok=True)

        def _replace(match: re.Match) -> str:
            indent = match.group(1)
            code = textwrap.dedent(match.group(3))
            digest = hashlib.md5(code.encode()).hexdigest()[:12]  # noqa: S324
            fname = f'plot_{digest}.png'
            fpath = _PLOT_DIR / fname

            if not fpath.exists():
                plt.close('all')
                try:
                    exec(code, {'__name__': '__main__'})  # noqa: S102
                except Exception as exc:
                    return (
                        f'{indent}!!! warning "Plot generation failed"\n'
                        f'{indent}    {type(exc).__name__}: {exc}\n'
                    )
                fig = plt.gcf()
                fig.savefig(fpath, bbox_inches='tight', dpi=150)
                plt.close('all')

            # Use a path relative to the docs/ directory
            rel = os.path.relpath(fpath, 'docs')
            return f'{indent}![plot]({rel})\n'

        text = _FENCE_RE.sub(_replace, text)
        return text.split('\n')


class PlotDirectiveExtension(Extension):
    """Markdown extension entry point."""

    def extendMarkdown(self, md):  # noqa: N802
        md.preprocessors.register(PlotPreprocessor(md), 'plot_directive', 30)


def makeExtension(**kwargs):  # noqa: N802
    """Entry point for Python-Markdown."""
    return PlotDirectiveExtension(**kwargs)
