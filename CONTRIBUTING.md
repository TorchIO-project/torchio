# Contributing to TorchIO

Contributions are welcome and greatly appreciated.
Every little bit helps, and credit will always be given.

TorchIO development happens on the `main` branch.
Maintenance fixes for TorchIO v1 belong on the `v1` branch.

## Types of contributions

### Report bugs

Report bugs [on GitHub](https://github.com/TorchIO-project/torchio/issues/new?assignees=&labels=bug&template=bug_report.md&title=).

When reporting a bug, include:

- Your TorchIO version.
- Any local setup details that might help with troubleshooting.
- Detailed steps to reproduce the bug.
- The full traceback, when there is one.

You can print your local setup details with:

```shell
uv run https://raw.githubusercontent.com/TorchIO-project/torchio/refs/heads/main/print_system.py
```

### Fix bugs

Look through the [GitHub issues](https://github.com/TorchIO-project/torchio/issues) for bugs.
Issues tagged with `bug` and `help wanted` are open to whoever wants to implement them.

### Implement features

Look through the [GitHub issues](https://github.com/TorchIO-project/torchio/issues) for feature requests.
Issues tagged with `enhancement` and `help wanted` are open to whoever wants to implement them.

### Write documentation

TorchIO can always use more documentation, whether in the official docs, docstrings, tutorials, examples, blog posts, or articles.

Docs are built with [Zensical](https://zensical.org/).
Follow the [Diataxis](https://diataxis.fr/) framework when adding or reorganizing documentation.

### Submit feedback

The best way to send feedback is to [file an issue](https://github.com/TorchIO-project/torchio/issues).

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible so it is easier to implement.
- Remember that this is a volunteer-driven project and that contributions are welcome.

## Get started

### 1. Create or find an issue

It is good practice to discuss proposed changes before opening a pull request, because the feature might already be implemented or planned.

### 2. Fork the repository

[Create a fork](https://github.com/TorchIO-project/torchio/fork) of the
`TorchIO-project/torchio` repository on GitHub.

### 3. Clone your fork locally

```bash
git clone git@github.com:your_github_username_here/torchio.git
cd torchio
```

### 4. Install the development environment

TorchIO uses [uv](https://docs.astral.sh/uv/) for Python environments and [mise](https://mise.jdx.dev/) for task automation and tool pinning.

Install mise, then trust the repository configuration and run the setup task:

```bash
mise trust
mise run setup
```

The setup task installs pinned tools, syncs all dependency groups, and installs the pre-commit hooks through [prek](https://github.com/j178/prek).

If you do not use mise, the equivalent core setup is:

```bash
uv sync --all-groups
uvx prek install --install-hooks
```

### 5. Create a branch

Create a branch from `main` for TorchIO v2 changes. If your work addresses an
issue, start the branch name with the issue number:

```bash
git checkout -b 55-name-of-your-bugfix-or-feature
```

### 6. Make your changes

Follow the existing project style:

- Use type annotations for new code.
- Prefer existing helpers and patterns over new one-off utilities.
- Add Google-style docstrings to new public classes, functions, and methods.
- Include usage examples in new public docstrings when practical.
- Add or update tests for behavior changes.
- Update documentation when user-facing behavior changes.

For TorchIO v2 transforms, do not add new `Random*` transform names.
Use the v2 transform names such as `Affine`, `Flip`, and `Noise`.

### 7. Run checks

Run the smallest check that covers your change before opening a pull request.

For unit tests:

```bash
mise run test
```

To run a subset of tests:

```bash
uv run tox -e test -- tests/data/test_image.py
```

For linting, formatting, and type checking:

```bash
mise run quality
```

You can also run individual checks:

```bash
mise run lint
mise run format-check
mise run types
```

To run the pre-commit hooks on all files:

```bash
mise run prek
```

### 8. Check documentation

If you changed documentation, examples, or docstrings, build the docs and test the code snippets:

```bash
mise run docs:build
mise run docs:test
```

To serve the docs locally while editing:

```bash
mise run docs:serve
```

### 9. Commit and push

Stage only the files that belong to your change, then commit and push:

```bash
git add path/to/changed_files
git commit -m "Fix nasty bug"
git push origin 55-name-of-your-bugfix-or-feature
```

Write clear commit messages.
These posts have useful guidance:

- [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)
- [Write Joyous Git Commit Messages](https://medium.com/@joshuatauberer/write-joyous-git-commit-messages-2f98891114c4)

### 10. Submit a pull request

Open a pull request on GitHub.
Fill in the template, link the related issue, and mark checklist items that apply to your change.

## Tips

Add a fork as a remote with:

```bash
mise run add-remote your_github_username_here
```

Run all tox environments with:

```bash
mise run tox
```
