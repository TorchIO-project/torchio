default:
    @just --list

clean:
    rm -rf .mypy_cache
    rm -rf .pytest_cache
    rm -rf .tox
    rm -rf .venv
    rm -rf dist
    rm -rf **/__pycache__
    rm -rf src/*.egg-info
    rm -f .coverage
    rm -f coverage.*

@install_uv:
	if ! command -v uv >/dev/null 2>&1; then \
		echo "uv is not installed. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

setup: install_uv
    uv sync --all-extras --all-groups
    uv run pre-commit install

bump part="patch":
    uv run bump-my-version bump {{part}} --verbose

bump-dry part="patch":
    uv run bump-my-version bump {{part}} --dry-run --verbose --allow-dirty

bump-python:
    #!/usr/bin/env -S uv run --script
    # /// script
    # dependencies = [
    #     "packaging",
    # ]
    # ///
    from pathlib import Path
    from packaging.version import Version
    python_version_path = Path(".python-version")
    old_version_string = python_version_path.read_text().strip()
    old_version = Version(old_version_string)
    new_version_string = f"3.{old_version.minor + 1}"
    python_version_path.write_text(new_version_string + "\n")

    docs_config_path = Path(".readthedocs.yml")
    docs_config = docs_config_path.read_text()
    docs_config = docs_config.replace(
        f'python: "{old_version_string}"',
        f'python: "{new_version_string}"',
    )
    docs_config_path.write_text(docs_config)

    tests_workflow_path = Path(".github/workflows/tests.yml")
    tests_workflow = tests_workflow_path.read_text()
    tests_workflow = tests_workflow.replace(
        f'"{old_version_string}"]',
        f'"{old_version_string}", "{new_version_string}"]',
    )
    tests_workflow = tests_workflow.replace(
        f"matrix.python == '{old_version_string}'",
        f"matrix.python == '{new_version_string}'",
    )
    tests_workflow_path.write_text(tests_workflow)

    pyproject_path = Path("pyproject.toml")
    pyproject = pyproject_path.read_text()
    old_str = f'    "Programming Language :: Python :: {old_version_string}",\n'
    new_str = f'    "Programming Language :: Python :: {new_version_string}",\n'
    pyproject = pyproject.replace(
        old_str,
        old_str + new_str,
    )
    pyproject_path.write_text(pyproject)

push:
    git push && git push --tags

quality_cmd := "uv run --group quality"

types:
    {{quality_cmd}} -- tox -e types

lint:
    {{quality_cmd}} -- ruff check

format:
    {{quality_cmd}} -- ruff format --diff

test:
    uv run --group test -- tox -e pytest

docs_cmd := "uv run --group doc --directory docs"

[positional-arguments]
build-docs *args='':
    {{docs_cmd}} -- sphinx-build -M html source build

[positional-arguments]
serve-docs *args='':
    {{docs_cmd}} -- sphinx-autobuild source build "$@"
