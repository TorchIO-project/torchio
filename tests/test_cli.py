"""Tests for the torchio CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "torchio.cli", *args],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def nii_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4)), path)
    return path


class TestInfo:
    def test_prints_metadata(self, nii_path: Path) -> None:
        result = _run("info", str(nii_path))
        assert result.returncode == 0
        assert "spatial:" in result.stdout
        assert "spacing:" in result.stdout
        assert "orientation:" in result.stdout


class TestConvert:
    def test_convert_nii_to_nii(self, nii_path: Path, tmp_path: Path) -> None:
        output = tmp_path / "out.nii"
        result = _run("convert", str(nii_path), str(output))
        assert result.returncode == 0
        assert output.exists()

    def test_convert_nonexistent(self, tmp_path: Path) -> None:
        result = _run("convert", "nonexistent.nii", str(tmp_path / "out.nii"))
        assert result.returncode != 0


class TestTransform:
    def test_apply_noise(self, nii_path: Path, tmp_path: Path) -> None:
        output = tmp_path / "noisy.nii.gz"
        result = _run(
            "transform",
            str(nii_path),
            str(output),
            "Noise",
            "std=0.1",
        )
        assert result.returncode == 0
        assert output.exists()

    def test_unknown_transform(self, nii_path: Path, tmp_path: Path) -> None:
        output = tmp_path / "out.nii.gz"
        result = _run("transform", str(nii_path), str(output), "FakeTransform")
        assert result.returncode != 0
        assert "Unknown transform" in result.stderr


class TestCacheDir:
    def test_prints_path(self) -> None:
        result = _run("cache", "dir")
        assert result.returncode == 0
        assert "torchio" in result.stdout.strip()


class TestPlot:
    def test_plot_to_file(self, nii_path: Path, tmp_path: Path) -> None:
        output = tmp_path / "plot.png"
        result = _run("plot", str(nii_path), "--output", str(output))
        assert result.returncode == 0
        assert output.exists()
        assert output.stat().st_size > 0
