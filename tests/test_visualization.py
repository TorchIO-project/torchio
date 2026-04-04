"""Tests for visualization and Image repr."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from matplotlib.figure import Figure

import torchio as tio
from torchio.data.affine import Affine

matplotlib.use("Agg")


class TestEulerAngles:
    def test_identity_gives_zeros(self) -> None:
        a = Affine()
        angles = a.euler_angles
        assert all(abs(v) < 1e-6 for v in angles)

    def test_rotation_around_z(self) -> None:
        theta = np.radians(15)
        m = np.eye(4)
        m[0, 0] = np.cos(theta)
        m[0, 1] = -np.sin(theta)
        m[1, 0] = np.sin(theta)
        m[1, 1] = np.cos(theta)
        a = Affine(m)
        x, y, z = a.euler_angles
        assert abs(z - 15.0) < 0.1
        assert abs(x) < 0.1
        assert abs(y) < 0.1

    def test_rotation_around_x(self) -> None:
        theta = np.radians(30)
        m = np.eye(4)
        m[1, 1] = np.cos(theta)
        m[1, 2] = -np.sin(theta)
        m[2, 1] = np.sin(theta)
        m[2, 2] = np.cos(theta)
        a = Affine(m)
        x, _y, _z = a.euler_angles
        assert abs(x - 30.0) < 0.1


class TestImageRepr:
    def test_multiline_format(self) -> None:
        img = tio.ScalarImage.from_tensor(torch.rand(1, 10, 20, 30))
        r = repr(img)
        assert "ScalarImage(\n" in r
        assert "channels:" in r
        assert "spatial:" in r
        assert "spacing:" in r
        assert "orientation:" in r
        assert "angles:" in r
        assert "dtype:" in r
        assert "memory:" in r

    def test_lazy_shows_backend(self, tmp_path) -> None:
        path = tmp_path / "test.nii"
        nib.save(nib.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4)), path)
        img = tio.ScalarImage(path)
        r = repr(img)
        assert "lazy" in r
        assert "NIfTI" in r

    def test_loaded_shows_loaded(self) -> None:
        img = tio.ScalarImage.from_tensor(torch.rand(1, 10, 10, 10))
        r = repr(img)
        assert "in memory" in r

    def test_origin_shown(self) -> None:
        affine = Affine.from_spacing((1, 1, 1), origin=(10.0, 20.0, 30.0))
        img = tio.ScalarImage.from_tensor(
            torch.rand(1, 5, 5, 5),
            affine=affine,
        )
        r = repr(img)
        assert "10.00" in r
        assert "origin:" in r


class TestPlotImage:
    def test_returns_figure(self) -> None:
        img = tio.ScalarImage.from_tensor(torch.rand(1, 10, 20, 30))
        fig = img.plot(show=False)
        assert isinstance(fig, Figure)

    def test_custom_indices(self) -> None:
        img = tio.ScalarImage.from_tensor(torch.rand(1, 10, 20, 30))
        fig = img.plot(indices=(5, 10, 15), show=False)
        axes = fig.axes
        assert len(axes) == 3
        # Titles show the slice index for the sliced axis
        titles = [ax.get_title() for ax in axes]
        assert any("5" in t for t in titles)
        assert any("10" in t for t in titles)
        assert any("15" in t for t in titles)

    def test_views_are_sagittal_coronal_axial(self) -> None:
        img = tio.ScalarImage.from_tensor(torch.rand(1, 10, 20, 30))
        fig = img.plot(show=False)
        titles = [ax.get_title() for ax in fig.axes]
        assert "Sagittal" in titles[0]
        assert "Coronal" in titles[1]
        assert "Axial" in titles[2]

    def test_orientation_labels_show_tensor_axis(self) -> None:
        img = tio.ScalarImage.from_tensor(torch.rand(1, 10, 10, 10))
        fig = img.plot(show=False)
        ax = fig.axes[0]
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        assert "↔" in xlabel
        assert "↔" in ylabel
        # Should contain tensor axis names (I, J, or K)
        all_labels = xlabel + ylabel
        assert any(c in all_labels for c in ("I", "J", "K"))

    def test_save_to_file(self, tmp_path) -> None:
        img = tio.ScalarImage.from_tensor(torch.rand(1, 10, 10, 10))
        out = tmp_path / "test.png"
        img.plot(output_path=out, show=False)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_external_axes(self) -> None:
        fig, axes = plt.subplots(1, 3)
        img = tio.ScalarImage.from_tensor(torch.rand(1, 10, 10, 10))
        result = img.plot(axes=axes, show=False)
        assert result is fig

    def test_label_map_uses_nearest(self) -> None:
        label = tio.LabelMap.from_tensor(torch.randint(0, 3, (1, 10, 10, 10)))
        fig = label.plot(show=False)
        ax = fig.axes[0]
        im = ax.images[0]
        assert im.get_interpolation() == "none"

    def test_voxels_mode(self) -> None:
        img = tio.ScalarImage.from_tensor(torch.rand(1, 10, 10, 10))
        fig = img.plot(show=False, voxels=True)
        assert isinstance(fig, Figure)

    def test_consistent_views_across_orientations(self) -> None:
        """Sagittal/Coronal/Axial order is the same for RAS and LPS."""
        img_ras = tio.ScalarImage.from_tensor(torch.rand(1, 10, 20, 30))
        fig_ras = img_ras.plot(show=False)

        ornt = nib.orientations.axcodes2ornt(("L", "P", "S"))
        affine = tio.Affine(nib.orientations.inv_ornt_aff(ornt, (10, 20, 30)))
        img_lps = tio.ScalarImage.from_tensor(
            torch.rand(1, 10, 20, 30),
            affine=affine,
        )
        fig_lps = img_lps.plot(show=False)

        titles_ras = [ax.get_title().split("[")[0].strip() for ax in fig_ras.axes]
        titles_lps = [ax.get_title().split("[")[0].strip() for ax in fig_lps.axes]
        assert titles_ras == titles_lps == ["Sagittal", "Coronal", "Axial"]


class TestReprHtml:
    def test_contains_table(self) -> None:
        img = tio.ScalarImage.from_tensor(torch.rand(1, 10, 10, 10))
        html = img._repr_html_()
        assert "tio-table" in html
        assert "Channels" in html
        assert "Spatial shape" in html
        assert "Euler angles" in html

    def test_contains_plot(self) -> None:
        img = tio.ScalarImage.from_tensor(torch.rand(1, 10, 10, 10))
        html = img._repr_html_()
        assert "data:image/png;base64" in html
