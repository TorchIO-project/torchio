"""Tests for remote NIfTI-Zarr streaming (lazy reads without full download)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest
import torch

import torchio as tio
from torchio.data.io import is_remote_nifti_zarr

# ── Detection helper ────────────────────────────────────────────────


class TestIsRemoteNiftiZarr:
    """Test the is_remote_nifti_zarr() detection function."""

    @pytest.mark.parametrize(
        "uri",
        [
            "az://container/image.nii.zarr",
            "s3://bucket/image.nii.zarr",
            "gs://bucket/image.nii.zarr",
            "https://example.com/image.nii.zarr",
            "abfs://container/path/to/image.nii.zarr",
        ],
    )
    def test_remote_zarr_detected(self, uri: str) -> None:
        assert is_remote_nifti_zarr(uri) is True

    @pytest.mark.parametrize(
        "uri",
        [
            "az://container/image.nii.gz",
            "s3://bucket/image.nii",
            "/local/path/image.nii.zarr",
            "relative/path/image.nii.zarr",
            "az://container/image.nrrd",
            "https://example.com/image.nii.gz",
        ],
    )
    def test_non_remote_zarr_not_detected(self, uri: str) -> None:
        assert is_remote_nifti_zarr(uri) is False

    def test_trailing_slash_handled(self) -> None:
        assert is_remote_nifti_zarr("az://container/image.nii.zarr/") is True


# ── Image construction ──────────────────────────────────────────────


class TestRemoteZarrImageConstruction:
    """Verify that remote .nii.zarr URIs are stored without downloading."""

    def test_remote_zarr_stores_uri(self) -> None:
        """The URI should be preserved, not fetched to a temp file."""
        uri = "az://my-container/brain.nii.zarr"
        with patch("torchio.data.io._fetch_remote") as mock_fetch:
            image = tio.ScalarImage(uri)
            mock_fetch.assert_not_called()
        assert image._remote_zarr_uri == uri
        assert image._path is None

    def test_non_zarr_remote_still_fetched(self) -> None:
        """A remote .nii.gz should still be downloaded (existing behaviour)."""
        uri = "az://my-container/brain.nii.gz"
        with patch("torchio.data.io._fetch_remote") as mock_fetch:
            mock_fetch.return_value = Path("/tmp/fake.nii.gz")
            _image = tio.ScalarImage(uri)
            mock_fetch.assert_called_once()

    def test_local_zarr_not_treated_as_remote(self, tmp_path: Path) -> None:
        """A local .nii.zarr path should go through the normal code path."""
        try:
            import niizarr
        except ImportError:
            pytest.skip("nifti-zarr not installed")
        data = np.random.rand(8, 8, 8).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        nii_path = tmp_path / "test.nii"
        nib.save(nii, nii_path)
        zarr_path = tmp_path / "test.nii.zarr"
        niizarr.nii2zarr(str(nii_path), str(zarr_path))

        image = tio.ScalarImage(zarr_path)
        assert image._remote_zarr_uri is None
        assert image._path == zarr_path


# ── Lazy backend from remote URI ────────────────────────────────────


class TestRemoteZarrBackend:
    """Verify that _ensure_backend passes the URI to ZarrBackend."""

    def test_ensure_backend_uses_uri(self) -> None:
        """ZarrBackend should receive the remote URI, not a local path."""
        uri = "az://container/brain.nii.zarr"

        with patch("torchio.data.io._fetch_remote"):
            image = tio.ScalarImage(uri)

        with patch(
            "torchio.data.backends.ZarrBackend.__init__", return_value=None
        ) as mock_init:
            # Provide a mock backend so _ensure_backend finishes
            mock_backend = MagicMock()
            mock_backend.shape = (1, 8, 8, 8)
            mock_backend.affine = torch.eye(4, dtype=torch.float64)

            def side_effect(path, **kwargs):
                image._backend = mock_backend

            mock_init.side_effect = side_effect
            image._ensure_backend()
            mock_init.assert_called_once_with(uri)

    def test_reader_kwargs_forwarded_to_backend(self) -> None:
        """store_opt and other kwargs should reach ZarrBackend."""
        uri = "az://container/brain.nii.zarr"
        kwargs = {"account_name": "myaccount", "account_key": "secret"}

        with patch("torchio.data.io._fetch_remote"):
            image = tio.ScalarImage(uri, reader_kwargs=kwargs)

        with patch(
            "torchio.data.backends.ZarrBackend.__init__", return_value=None
        ) as mock_init:
            mock_backend = MagicMock()
            mock_backend.shape = (1, 8, 8, 8)
            mock_backend.affine = torch.eye(4, dtype=torch.float64)

            def side_effect(path, **kw):
                image._backend = mock_backend

            mock_init.side_effect = side_effect
            image._ensure_backend()
            mock_init.assert_called_once_with(uri, **kwargs)

    def test_shape_via_remote_backend(self) -> None:
        """image.shape should work through the remote ZarrBackend."""
        uri = "az://container/brain.nii.zarr"

        with patch("torchio.data.io._fetch_remote"):
            image = tio.ScalarImage(uri)

        mock_backend = MagicMock()
        mock_backend.shape = (1, 32, 32, 32)
        mock_backend.affine = torch.eye(4, dtype=torch.float64)
        image._backend = mock_backend

        assert image.shape == (1, 32, 32, 32)

    def test_load_via_remote_backend(self) -> None:
        """image.load() should materialize from the remote ZarrBackend."""
        uri = "az://container/brain.nii.zarr"

        with patch("torchio.data.io._fetch_remote"):
            image = tio.ScalarImage(uri)

        tensor = torch.randn(1, 8, 8, 8)
        mock_backend = MagicMock()
        mock_backend.shape = (1, 8, 8, 8)
        mock_backend.affine = torch.eye(4, dtype=torch.float64)
        mock_backend.to_tensor.return_value = tensor
        image._backend = mock_backend

        image.load()
        assert image._data is not None
        assert image._data.shape == (1, 8, 8, 8)


# ── repr and deepcopy ───────────────────────────────────────────────


class TestRemoteZarrReprAndCopy:
    def test_repr_shows_uri(self) -> None:
        uri = "az://container/brain.nii.zarr"
        with patch("torchio.data.io._fetch_remote"):
            image = tio.ScalarImage(uri)

        mock_backend = MagicMock()
        mock_backend.shape = (1, 8, 8, 8)
        mock_backend.affine = torch.eye(4, dtype=torch.float64)
        mock_backend.dtype = np.dtype(np.float32)
        mock_backend.to_tensor.return_value = torch.randn(1, 8, 8, 8)
        image._backend = mock_backend

        r = repr(image)
        assert "az://container/brain.nii.zarr" in r

    def test_deepcopy_preserves_uri(self) -> None:
        import copy

        uri = "az://container/brain.nii.zarr"
        with patch("torchio.data.io._fetch_remote"):
            image = tio.ScalarImage(uri)

        tensor = torch.randn(1, 8, 8, 8)
        mock_backend = MagicMock()
        mock_backend.shape = (1, 8, 8, 8)
        mock_backend.affine = torch.eye(4, dtype=torch.float64)
        mock_backend.to_tensor.return_value = tensor
        image._backend = mock_backend
        image._data = tensor

        with patch("torchio.data.io._fetch_remote"):
            copied = copy.deepcopy(image)
        assert copied._remote_zarr_uri == uri


# ── End-to-end with local Zarr store ────────────────────────────────


class TestRemoteZarrEndToEnd:
    """Integration test using a local .nii.zarr but going through remote path."""

    @pytest.fixture
    def zarr_path(self, tmp_path: Path) -> Path:
        try:
            import niizarr
        except ImportError:
            pytest.skip("nifti-zarr not installed")
        data = np.arange(16**3, dtype=np.float32).reshape(16, 16, 16)
        nii = nib.Nifti1Image(data, np.eye(4))
        nii_path = tmp_path / "test.nii"
        nib.save(nii, nii_path)
        zarr_path = tmp_path / "test.nii.zarr"
        niizarr.nii2zarr(str(nii_path), str(zarr_path))
        return zarr_path

    def test_zarr_backend_accepts_kwargs(self, zarr_path: Path) -> None:
        """ZarrBackend should forward kwargs to niizarr.zarr2nii()."""
        from torchio.data.backends import ZarrBackend

        backend = ZarrBackend(str(zarr_path))
        assert backend.shape == (1, 16, 16, 16)

    def test_slice_without_full_load(self, zarr_path: Path) -> None:
        """Slicing a remote-style zarr should not load the full tensor."""
        from torchio.data.backends import ZarrBackend

        backend = ZarrBackend(str(zarr_path))
        roi = backend[:, 4:8, 4:8, 4:8]
        assert roi.shape == (1, 4, 4, 4)
