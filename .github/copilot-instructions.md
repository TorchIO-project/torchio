We use `uv` for package managing and `just` to run tasks.
Install uv with `curl -LsSf https://astral.sh/uv/install.sh | sh`, then install this package with `uv sync --all-extras --all-groups`.
Install `just` with `uv tool install rust-just`.
If you want to run something, use the `uv run` prefix (e.g. `uv run python [args]` or `uv run pytest [args]`.
Before pushing your changes, run all the `tox` tasks, pre-commit hooks with `uv run pre-commit --all-files` and build the docs with `just build-docs`.
In pull requests, make one commit per review comment.
When making changes in code, make sure to follow the code style and conventions used in the project.
