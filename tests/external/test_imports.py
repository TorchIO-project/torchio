"""Tests for external.imports module."""

from unittest.mock import patch

import pytest

from torchio.external.imports import _check_executable
from torchio.external.imports import _check_module


class TestCheckModule:
    """Tests for _check_module function."""

    def test_missing_module_raises_import_error(self):
        """_check_module raises ImportError when module is not found."""
        with pytest.raises(ImportError, match='torchio_nonexistent_pkg'):
            _check_module(
                module='torchio_nonexistent_pkg',
                extra='test',
            )

    def test_missing_module_uses_package_name(self):
        """ImportError message uses package name when provided."""
        with pytest.raises(ImportError, match='my-custom-package'):
            _check_module(
                module='torchio_nonexistent_pkg',
                extra='test',
                package='my-custom-package',
            )

    def test_existing_module_passes(self):
        """_check_module does not raise for installed modules."""
        _check_module(module='torch', extra='test')


class TestCheckExecutable:
    """Tests for _check_executable function."""

    def test_missing_executable_raises(self):
        """Missing executable raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match='nonexistent_binary_xyz'):
            _check_executable('nonexistent_binary_xyz')

    def test_existing_executable_passes(self):
        """Existing executable does not raise."""
        _check_executable('python3')

    def test_which_returns_none(self):
        """FileNotFoundError raised when which() returns None."""
        with patch('torchio.external.imports.which', return_value=None):
            with pytest.raises(FileNotFoundError, match='ffmpeg'):
                _check_executable('ffmpeg')


class TestGetHelpers:
    """Tests for get_* import helpers."""

    def test_get_pandas_missing(self):
        """get_pandas raises ImportError when pandas not installed."""
        from torchio.external.imports import get_pandas

        with patch('torchio.external.imports.find_spec', return_value=None):
            with pytest.raises(ImportError, match='pandas'):
                get_pandas()

    def test_get_colorcet_missing(self):
        """get_colorcet raises ImportError when not installed."""
        from torchio.external.imports import get_colorcet

        with patch('torchio.external.imports.find_spec', return_value=None):
            with pytest.raises(ImportError, match='colorcet'):
                get_colorcet()

    def test_get_ffmpeg_missing_module(self):
        """get_ffmpeg raises ImportError when ffmpeg module not found."""
        from torchio.external.imports import get_ffmpeg

        with patch('torchio.external.imports.find_spec', return_value=None):
            with pytest.raises(ImportError, match='ffmpeg-python'):
                get_ffmpeg()
