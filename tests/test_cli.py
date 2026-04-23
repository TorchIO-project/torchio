"""Tests for the torchio CLI.

Commands are invoked in-process via their dataclass ``.run()`` method
rather than through ``subprocess``, avoiding the ~1s per-test overhead
of spawning a new Python interpreter and re-importing PyTorch.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from torchio.cli import Animate
from torchio.cli import Cache
from torchio.cli import Convert
from torchio.cli import Dir
from torchio.cli import Info
from torchio.cli import Plot
from torchio.cli import Transform


@pytest.fixture
def nii_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4)), path)
    return path


class TestInfo:
    def test_prints_metadata(
        self,
        nii_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        Info(path=nii_path).run()
        captured = capsys.readouterr()
        assert "spatial:" in captured.out
        assert "spacing:" in captured.out
        assert "orientation:" in captured.out


class TestConvert:
    def test_convert_nii_to_nii(self, nii_path: Path, tmp_path: Path) -> None:
        output = tmp_path / "out.nii"
        Convert(input=nii_path, output=output).run()
        assert output.exists()

    def test_convert_nonexistent(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Convert(
                input=Path("nonexistent.nii"),
                output=tmp_path / "out.nii",
            ).run()

    def test_preserves_dtype(self, tmp_path: Path) -> None:
        input_path = tmp_path / "in.nii.gz"
        data = np.zeros((4, 5, 6), dtype=np.int16)
        nib.save(nib.Nifti1Image(data, np.eye(4)), input_path)
        output = tmp_path / "out.nii.gz"
        Convert(input=input_path, output=output).run()
        loaded = nib.load(output)
        assert loaded.header.get_data_dtype() == np.int16

    def test_no_stdout(
        self,
        nii_path: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        output = tmp_path / "out.nii.gz"
        Convert(input=nii_path, output=output).run()
        captured = capsys.readouterr()
        assert captured.out == ""


class TestTransform:
    def test_apply_noise(self, nii_path: Path, tmp_path: Path) -> None:
        output = tmp_path / "noisy.nii.gz"
        Transform(
            input=nii_path,
            output=output,
            name="Noise",
            args=["std=0.1"],
        ).run()
        assert output.exists()

    def test_unknown_transform(self, nii_path: Path, tmp_path: Path) -> None:
        output = tmp_path / "out.nii.gz"
        with pytest.raises(SystemExit):
            Transform(
                input=nii_path,
                output=output,
                name="FakeTransform",
                args=[],
            ).run()


class TestCacheDir:
    def test_prints_path(self, capsys: pytest.CaptureFixture[str]) -> None:
        Cache(command=Dir()).run()
        captured = capsys.readouterr()
        assert "torchio" in captured.out.strip()


class TestPlot:
    def test_plot_to_file(self, nii_path: Path, tmp_path: Path) -> None:
        output = tmp_path / "plot.png"
        Plot(path=nii_path, output=output).run()
        assert output.exists()
        assert output.stat().st_size > 0


class TestAnimate:
    def test_animate_gif(self, nii_path: Path, tmp_path: Path) -> None:
        output = tmp_path / "anim.gif"
        Animate(path=nii_path, output=output, seconds=1.0, direction="I").run()
        assert output.exists()
        assert output.stat().st_size > 0

    def test_animate_unsupported_format(
        self,
        nii_path: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "bad.avi"
        with pytest.raises(SystemExit):
            Animate(path=nii_path, output=output).run()
