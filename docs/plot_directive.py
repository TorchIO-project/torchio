#!/usr/bin/env python3
"""Pre-process markdown files: execute ``python plot`` fenced code blocks.

Run this script before building the docs. It finds all markdown files under
docs/ with ```python plot blocks, executes them, saves the matplotlib figure,
and replaces the block with an image + collapsible source code.

The original markdown files are modified in place. Use git to revert changes.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402

_DOCS_DIR = Path('docs')
_PLOT_DIR = _DOCS_DIR / 'images' / 'plots'

_FENCE_RE = re.compile(
    r'^(\s*)(`{3,})python\s+plot\s*$\n(.*?)^\1\2\s*$',
    re.MULTILINE | re.DOTALL,
)


def process_file(md_path: Path) -> bool:
    """Process a single markdown file. Returns True if modified."""
    text = md_path.read_text()
    if 'python plot' not in text:
        return False

    page_dir = str(md_path.parent.relative_to(_DOCS_DIR))
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
                msg = f'{type(exc).__name__}: {exc}'
                print(f'  WARNING: Plot failed: {msg}', file=sys.stderr)
                return (
                    f'{indent}!!! warning "Plot generation failed"\n{indent}    {msg}\n'
                )
            fig = plt.gcf()
            fig.savefig(fpath, bbox_inches='tight', dpi=150)
            plt.close('all')
            print(f'  Generated {fpath}', file=sys.stderr)
        else:
            print(f'  Cached    {fpath}', file=sys.stderr)

        img_rel = os.path.relpath(
            str(fpath.relative_to(_DOCS_DIR)),
            page_dir,
        ).replace(os.sep, '/')

        bt = '`' * 3
        indented_code = textwrap.indent(code.rstrip(), indent + '    ')
        return (
            f'{indent}![plot]({img_rel})\n'
            f'\n'
            f'{indent}??? note "Source code"\n'
            f'{indent}    {bt}python\n'
            f'{indented_code}\n'
            f'{indent}    {bt}\n'
        )

    new_text = _FENCE_RE.sub(_replace, text)
    if new_text != text:
        md_path.write_text(new_text)
        return True
    return False


def main() -> None:
    md_files = sorted(_DOCS_DIR.rglob('*.md'))
    modified = 0
    for md_path in md_files:
        print(f'Processing {md_path}', file=sys.stderr)
        if process_file(md_path):
            modified += 1
    print(f'Modified {modified} file(s)', file=sys.stderr)


if __name__ == '__main__':
    main()
