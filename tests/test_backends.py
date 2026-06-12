"""Tests for lazy image data backends."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from torchio import ScalarImage
from torchio.data.backends import NibabelBackend
from torchio.data.backends import TensorBackend


class TestNibabelBackend:
    @pytest.fixture
    def nifti_path(self, tmp_path: Path) -> Path:
        data = np.random.randn(10, 12, 14).astype(np.float32)
        nii = nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)
        return path

    @pytest.fixture
    def multichannel_nifti_path(self, tmp_path: Path) -> Path:
        data = np.random.randn(10, 12, 14, 3).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "multi.nii.gz"
        nib.save(nii, path)
        return path

    def test_shape_3d(self, nifti_path: Path):
        nii = nib.load(nifti_path)
        backend = NibabelBackend(nii)
        assert backend.shape == (1, 10, 12, 14)

    def test_shape_4d(self, multichannel_nifti_path: Path):
        nii = nib.load(multichannel_nifti_path)
        backend = NibabelBackend(nii)
        assert backend.shape == (3, 10, 12, 14)

    def test_affine(self, nifti_path: Path):
        nii = nib.load(nifti_path)
        backend = NibabelBackend(nii)
        np.testing.assert_array_equal(
            backend.affine,
            np.diag([2.0, 2.0, 2.0, 1.0]),
        )

    def test_to_tensor_3d(self, nifti_path: Path):
        nii = nib.load(nifti_path)
        backend = NibabelBackend(nii)
        tensor = backend.to_tensor()
        assert tensor.shape == (1, 10, 12, 14)
        assert tensor.dtype == torch.float32

    def test_to_tensor_4d(self, multichannel_nifti_path: Path):
        nii = nib.load(multichannel_nifti_path)
        backend = NibabelBackend(nii)
        tensor = backend.to_tensor()
        assert tensor.shape == (3, 10, 12, 14)

    @pytest.mark.parametrize(
        ("np_dtype", "torch_dtype"),
        [
            (np.int16, torch.int16),
            (np.int32, torch.int32),
            (np.uint8, torch.uint8),
            (np.uint16, torch.int32),  # upcast: torch has no uint16
            (np.float32, torch.float32),
            (np.float64, torch.float64),
        ],
    )
    def test_to_tensor_preserves_dtype(
        self,
        tmp_path: Path,
        np_dtype: type,
        torch_dtype: torch.dtype,
    ) -> None:
        data = np.zeros((10, 12, 14), dtype=np_dtype)
        path = tmp_path / f"dtype_{np.dtype(np_dtype).name}.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)
        backend = NibabelBackend(nib.load(path))
        tensor = backend.to_tensor()
        assert tensor.dtype == torch_dtype

    def test_getitem_3d(self, nifti_path: Path):
        nii = nib.load(nifti_path)
        backend = NibabelBackend(nii)
        # Spatial slice in (I, J, K) space
        sliced = backend[:, 2:5, 3:7, 4:8]
        assert sliced.shape == (1, 3, 4, 4)

    def test_does_not_load_full_data_for_shape(self, nifti_path: Path):
        nii = nib.load(nifti_path)
        backend = NibabelBackend(nii)
        _ = backend.shape
        # dataobj should still be a proxy, not a loaded array
        assert not isinstance(nii.dataobj, np.ndarray)

    def test_invalid_ndim_raises(self, tmp_path: Path):
        data = np.random.randn(10, 12, 14, 3, 2).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "bad.nii.gz"
        nib.save(nii, path)
        nii = nib.load(path)
        with pytest.raises(ValueError, match="3D or 4D"):
            NibabelBackend(nii)


class TestImageWithBackends:
    def test_from_tensor_uses_tensor_backend(self):
        image = ScalarImage(torch.randn(1, 10, 10, 10))
        assert isinstance(image._backend, TensorBackend)

    def test_nifti_uses_nibabel_backend(self, tmp_path: Path):
        data = np.random.randn(10, 10, 10).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)

        image = ScalarImage(path)
        assert not image.is_loaded
        _ = image.data  # trigger load
        assert isinstance(image._backend, NibabelBackend)

    def test_shape_without_loading_uses_backend(self, tmp_path: Path):
        data = np.random.randn(10, 12, 14).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)

        image = ScalarImage(path)
        assert image.shape == (1, 10, 12, 14)
        # Backend should be created but data not materialized
        assert image._backend is not None
        assert image._data is None

    def test_dataobj_returns_backend(self, tmp_path: Path):
        data = np.random.randn(10, 10, 10).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)

        image = ScalarImage(path)
        dataobj = image.dataobj
        assert isinstance(dataobj, NibabelBackend)

    def test_dataobj_from_tensor(self):
        image = ScalarImage(torch.randn(1, 10, 10, 10))
        dataobj = image.dataobj
        assert isinstance(dataobj, TensorBackend)

    def test_data_caches_tensor(self, tmp_path: Path):
        data = np.random.randn(10, 10, 10).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)

        image = ScalarImage(path)
        tensor1 = image.data
        tensor2 = image.data
        assert tensor1 is tensor2  # same object, cached

    def test_lazy_slice_via_dataobj(self, tmp_path: Path):
        data = np.random.randn(10, 12, 14).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)

        image = ScalarImage(path)
        backend = image.dataobj
        sliced = backend[:, 2:5, 3:7, 4:8]
        assert sliced.shape == (1, 3, 4, 4)
        assert image._data is None  # full tensor never materialized


class TestZarrBackend:
    """Tests for ZarrBackend using nifti-zarr."""

    @pytest.fixture(scope="class")
    def zarr_path(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        try:
            import niizarr
        except ImportError:
            pytest.skip("nifti-zarr not installed")
        tmp_path = tmp_path_factory.mktemp("zarr")
        data = np.random.rand(16, 16, 16).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        nii_path = tmp_path / "test.nii"
        nib.save(nii, nii_path)
        zarr_path = tmp_path / "test.nii.zarr"
        niizarr.nii2zarr(str(nii_path), str(zarr_path))
        return zarr_path

    def test_zarr_image_shape(self, zarr_path: Path):
        image = ScalarImage(zarr_path)
        assert image.shape == (1, 16, 16, 16)

    def test_zarr_lazy_load(self, zarr_path: Path):
        from torchio.data.backends import ZarrBackend

        image = ScalarImage(zarr_path)
        backend = image.dataobj
        assert isinstance(backend, (NibabelBackend, ZarrBackend))
        assert image._data is None

    def test_zarr_slice(self, zarr_path: Path):
        image = ScalarImage(zarr_path)
        backend = image.dataobj
        sliced = backend[:, 2:12, 2:12, 2:12]
        assert sliced.shape == (1, 10, 10, 10)
        assert image._data is None

    def test_zarr_materialize(self, zarr_path: Path):
        image = ScalarImage(zarr_path)
        tensor = image.data
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 16, 16, 16)

    def test_read_nifti_zarr(self, zarr_path: Path):
        from torchio.data.io import read_nifti_zarr

        tensor, affine = read_nifti_zarr(zarr_path)
        assert tensor.shape == (1, 16, 16, 16)
        assert affine.shape == (4, 4)


class TestBackendCoherence:
    """Regression tests: `_backend` must stay coherent with `data`.

    See assessment flaws on stale backend state after `set_data()` and
    `to()`, and strategy section "Fix correctness issues".
    """

    @pytest.fixture
    def nifti_path(self, tmp_path: Path) -> Path:
        data = np.random.randn(10, 12, 14).astype(np.float32)
        nii = nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))
        path = tmp_path / "coherence.nii.gz"
        nib.save(nii, path)
        return path

    def test_set_data_refreshes_backend_shape(self) -> None:
        image = ScalarImage(torch.randn(1, 4, 4, 4))
        image.set_data(torch.randn(2, 6, 6, 6))
        assert image.dataobj.shape == (2, 6, 6, 6)
        assert tuple(image.dataobj.to_tensor().shape) == (2, 6, 6, 6)

    def test_set_data_refreshes_backend_values_tensor_source(self) -> None:
        image = ScalarImage(torch.zeros(1, 4, 4, 4))
        new = torch.ones(1, 4, 4, 4)
        image.set_data(new)
        assert torch.equal(image.dataobj.to_tensor(), new)

    def test_set_data_refreshes_backend_path_source(self, nifti_path: Path) -> None:
        image = ScalarImage(nifti_path)
        new = torch.full((1, 10, 12, 14), 7.0)
        image.set_data(new)
        # dataobj must reflect the in-memory tensor, not the on-disk data.
        assert torch.equal(image.dataobj.to_tensor(), new)

    def test_set_data_on_empty_image_defaults_to_identity(self) -> None:
        # Created empty then filled: no affine source, so default to identity
        # without crashing, and keep dataobj consistent.
        image = ScalarImage()
        image.set_data(torch.zeros(1, 2, 3, 4))
        assert image.dataobj.shape == (1, 2, 3, 4)
        np.testing.assert_allclose(image.affine.numpy(), np.eye(4))
        np.testing.assert_allclose(
            image.affine.numpy(), np.asarray(image.dataobj.affine)
        )

    def test_set_data_preserves_disk_affine(self, nifti_path: Path) -> None:
        image = ScalarImage(nifti_path)
        image.set_data(torch.full((1, 10, 12, 14), 7.0))
        disk = np.diag([2.0, 2.0, 2.0, 1.0])
        np.testing.assert_allclose(image.affine.numpy(), disk)
        np.testing.assert_allclose(np.asarray(image.dataobj.affine), disk)

    def test_to_refreshes_backend_dtype(self) -> None:
        image = ScalarImage(torch.randn(1, 4, 4, 4).float())
        image.to(torch.float64)
        assert image.dataobj.dtype == np.dtype("float64")
        assert image.dataobj.to_tensor().dtype == torch.float64

    def test_to_refreshes_backend_path_source(self, nifti_path: Path) -> None:
        image = ScalarImage(nifti_path)
        image.to(torch.float64)
        assert image.dataobj.dtype == np.dtype("float64")
        assert image.dataobj.to_tensor().dtype == torch.float64


class TestAffineOverride:
    """Regression tests: an overridden affine must be reported consistently.

    See assessment flaw that `image.affine` and `image.dataobj.affine` can
    disagree when the affine is overridden in the constructor.
    """

    @pytest.fixture
    def nifti_path(self, tmp_path: Path) -> Path:
        data = np.random.randn(10, 12, 14).astype(np.float32)
        nii = nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))
        path = tmp_path / "override.nii.gz"
        nib.save(nii, path)
        return path

    def test_override_matches_dataobj_nifti(self, nifti_path: Path) -> None:
        custom = np.diag([3.0, 4.0, 5.0, 1.0])
        image = ScalarImage(nifti_path, affine=custom)
        np.testing.assert_allclose(image.affine.numpy(), custom)
        np.testing.assert_allclose(np.asarray(image.dataobj.affine), custom)

    def test_no_override_uses_disk_affine(self, nifti_path: Path) -> None:
        image = ScalarImage(nifti_path)
        disk = np.diag([2.0, 2.0, 2.0, 1.0])
        np.testing.assert_allclose(image.affine.numpy(), disk)
        np.testing.assert_allclose(np.asarray(image.dataobj.affine), disk)

    def test_override_matches_dataobj_tensor(self) -> None:
        custom = np.diag([3.0, 4.0, 5.0, 1.0])
        image = ScalarImage(torch.randn(1, 4, 4, 4), affine=custom)
        np.testing.assert_allclose(image.affine.numpy(), custom)
        np.testing.assert_allclose(np.asarray(image.dataobj.affine), custom)


class TestVectorNifti5D:
    """Lazy slicing of 5D vector NIfTI as written by SimpleITK: (I, J, K, 1, C)."""

    @pytest.fixture
    def path_5d(self, tmp_path: Path) -> Path:
        data = np.random.randn(8, 9, 10, 1, 3).astype(np.float32)
        path = tmp_path / "vector.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)
        return path

    def test_shape(self, path_5d: Path) -> None:
        image = ScalarImage(path_5d)
        assert image.shape == (3, 8, 9, 10)
        assert image._data is None

    def test_lazy_spatial_slice(self, path_5d: Path) -> None:
        image = ScalarImage(path_5d)
        ref = ScalarImage(path_5d).data
        sliced = image.dataobj[:, 1:4, 2:5, 0:6]
        assert image._data is None  # full tensor never materialized
        assert isinstance(sliced, torch.Tensor)
        assert tuple(sliced.shape) == (3, 3, 3, 6)
        assert torch.allclose(sliced.float(), ref[:, 1:4, 2:5, 0:6].float())

    def test_lazy_channel_slice_preserves_dim(self, path_5d: Path) -> None:
        image = ScalarImage(path_5d)
        ref = ScalarImage(path_5d).data
        sliced = image.dataobj[1]
        assert tuple(sliced.shape) == (1, 8, 9, 10)
        assert torch.allclose(sliced.float(), ref[1:2].float())


class TestBackendSlicingContract:
    """The backend slicing contract: always 4D `(C, I, J, K)` tensors.

    Direct `dataobj[...]` access must match `ref[normalize_index(index)]` for
    every backend, where `ref` is the materialized `(C, I, J, K)` tensor.
    """

    @pytest.fixture(params=["tensor", "nifti_3d", "nifti_4d", "nifti_5d"])
    def image_and_ref(
        self,
        request: pytest.FixtureRequest,
        tmp_path: Path,
    ) -> tuple[ScalarImage, torch.Tensor]:
        kind = request.param
        if kind == "tensor":
            data = torch.randn(3, 8, 9, 10)
            return ScalarImage(data), data.clone()
        if kind == "nifti_3d":
            arr = np.random.randn(8, 9, 10).astype(np.float32)
        elif kind == "nifti_4d":
            arr = np.random.randn(8, 9, 10, 3).astype(np.float32)
        else:  # nifti_5d
            arr = np.random.randn(8, 9, 10, 1, 3).astype(np.float32)
        path = tmp_path / f"{kind}.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.eye(4)), path)
        return ScalarImage(path), ScalarImage(path).data

    @pytest.mark.parametrize(
        "index",
        [
            slice(None),
            0,
            -1,
            slice(0, 1),
            (slice(None), slice(1, 4)),
            (slice(None), slice(1, 4), slice(2, 5), slice(0, 6)),
            ...,
            (..., slice(0, 5)),
            (slice(None), slice(-4, None)),
            (0, 1, 2, 3),
        ],
    )
    def test_matches_reference(
        self,
        image_and_ref: tuple[ScalarImage, torch.Tensor],
        index,
    ) -> None:
        from torchio.data.backends import normalize_index

        image, ref = image_and_ref
        result = image.dataobj[index]
        expected = ref[normalize_index(index)]
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 4
        assert tuple(result.shape) == tuple(expected.shape)
        assert torch.allclose(result.float(), expected.float())

    def test_multichannel_selection(self, tmp_path: Path) -> None:
        arr = np.random.randn(8, 9, 10, 3).astype(np.float32)
        path = tmp_path / "multi.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.eye(4)), path)
        image = ScalarImage(path)
        ref = ScalarImage(path).data
        sliced = image.dataobj[0:2, 1:4]
        assert tuple(sliced.shape) == (2, 3, 9, 10)
        assert torch.allclose(sliced.float(), ref[0:2, 1:4].float())


class TestSlicingReturnTypes:
    """`__getitem__` returns tensors, preserving device/dtype where possible."""

    def test_tensor_backend_preserves_dtype(self) -> None:
        data = torch.randn(2, 4, 5, 6, dtype=torch.float64)
        image = ScalarImage(data)
        sliced = image.dataobj[:, 1:3]
        assert isinstance(sliced, torch.Tensor)
        assert sliced.dtype == torch.float64
        assert sliced.device == data.device

    def test_nibabel_backend_returns_tensor_lazily(self, tmp_path: Path) -> None:
        arr = np.random.randn(8, 9, 10).astype(np.float32)
        path = tmp_path / "lazy.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.eye(4)), path)
        image = ScalarImage(path)
        sliced = image.dataobj[:, 2:5]
        assert isinstance(sliced, torch.Tensor)
        assert image._data is None  # still lazy

    def test_too_many_indices_raises(self) -> None:
        image = ScalarImage(torch.randn(1, 4, 4, 4))
        with pytest.raises(IndexError, match="Too many indices"):
            _ = image.dataobj[0, 0, 0, 0, 0]


class TestBackendResolver:
    """Backend selection is delegated to the registry-based resolver."""

    def test_resolve_nifti_path(self, tmp_path: Path) -> None:
        from torchio.data.backends import BackendRequest
        from torchio.data.backends import resolve_backend

        arr = np.random.randn(8, 9, 10).astype(np.float32)
        path = tmp_path / "r.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.eye(4)), path)
        backend = resolve_backend(BackendRequest(path=path))
        assert isinstance(backend, NibabelBackend)

    def test_resolve_non_nifti_returns_none(self, tmp_path: Path) -> None:
        from torchio.data.backends import BackendRequest
        from torchio.data.backends import resolve_backend

        path = tmp_path / "image.png"
        backend = resolve_backend(BackendRequest(path=path))
        assert backend is None

    def test_resolve_applies_affine_override(self, tmp_path: Path) -> None:
        from torchio.data.backends import BackendRequest
        from torchio.data.backends import resolve_backend

        arr = np.random.randn(8, 9, 10).astype(np.float32)
        path = tmp_path / "r.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.diag([2.0, 2.0, 2.0, 1.0])), path)
        custom = torch.diag(torch.tensor([3.0, 4.0, 5.0, 1.0], dtype=torch.float64))
        backend = resolve_backend(BackendRequest(path=path, affine=custom))
        assert backend is not None
        np.testing.assert_allclose(np.asarray(backend.affine), custom.numpy())

    def test_image_delegates_to_resolver(self, tmp_path: Path, monkeypatch) -> None:
        import torchio.data.image as image_module

        arr = np.random.randn(8, 9, 10).astype(np.float32)
        path = tmp_path / "r.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.eye(4)), path)

        calls: list = []
        real_resolve = image_module.resolve_backend

        def spy(request):
            calls.append(request)
            return real_resolve(request)

        monkeypatch.setattr(image_module, "resolve_backend", spy)
        image = ScalarImage(path)
        _ = image.shape  # triggers backend resolution
        assert len(calls) == 1
        assert calls[0].path == path


class TestBackendRegistration:
    """Controlled extensibility: register custom backends and lazy readers."""

    def test_register_and_resolve_custom_backend(self, tmp_path: Path) -> None:
        from torchio.data import register_backend
        from torchio.data import resolve_backend
        from torchio.data import unregister_backend
        from torchio.data.backends import BackendRequest

        marker = tmp_path / "custom.fake"
        marker.touch()

        def matcher(request: BackendRequest) -> bool:
            return request.path is not None and request.path.suffix == ".fake"

        def factory(request: BackendRequest):
            return TensorBackend(torch.zeros(1, 2, 3, 4))

        register_backend("fake", matcher, factory)
        try:
            backend = resolve_backend(BackendRequest(path=marker))
            assert isinstance(backend, TensorBackend)
            assert backend.shape == (1, 2, 3, 4)
        finally:
            unregister_backend("fake")
        assert resolve_backend(BackendRequest(path=marker)) is None

    def test_custom_backend_takes_priority_over_builtin(self, tmp_path: Path) -> None:
        from torchio.data import register_backend
        from torchio.data import resolve_backend
        from torchio.data import unregister_backend
        from torchio.data.backends import BackendRequest

        arr = np.random.randn(8, 9, 10).astype(np.float32)
        path = tmp_path / "p.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.eye(4)), path)

        sentinel = TensorBackend(torch.ones(1, 1, 1, 1))

        register_backend(
            "override-nifti",
            lambda req: req.path is not None and req.path.name.endswith(".nii.gz"),
            lambda req: sentinel,
        )
        try:
            backend = resolve_backend(BackendRequest(path=path))
            assert backend is sentinel
        finally:
            unregister_backend("override-nifti")

    def test_image_uses_registered_backend(self, tmp_path: Path) -> None:
        from torchio.data import register_backend
        from torchio.data import unregister_backend

        marker = tmp_path / "image.fake"
        marker.touch()
        data = torch.arange(2 * 4 * 5 * 6, dtype=torch.float32).reshape(2, 4, 5, 6)

        register_backend(
            "fake-image",
            lambda req: req.path is not None and req.path.suffix == ".fake",
            lambda req: TensorBackend(data, affine=torch.eye(4, dtype=torch.float64)),
        )
        try:
            image = ScalarImage(marker)
            assert image.shape == (2, 4, 5, 6)
            assert image._data is None  # shape read lazily via the backend
            assert torch.equal(image.data, data)
        finally:
            unregister_backend("fake-image")


class _LazyNiftiReader:
    """A custom reader that is also a LazyReader (builds a NibabelBackend)."""

    def __call__(self, path: Path, **kwargs):  # simple-reader fallback
        backend = self.create_backend(_request_for(path, kwargs))
        return backend.to_tensor(), backend.affine.numpy()

    def create_backend(self, request):
        nii = nib.load(request.path, **dict(request.reader_kwargs))
        return NibabelBackend(nii, affine=request.affine)


def _request_for(path, kwargs):
    from torchio.data.backends import BackendRequest

    return BackendRequest(path=path, reader_kwargs=kwargs)


class TestLazyCustomReader:
    """A custom reader exposing `create_backend` enables lazy access."""

    @pytest.fixture
    def nifti_path(self, tmp_path: Path) -> Path:
        arr = np.random.randn(8, 9, 10).astype(np.float32)
        path = tmp_path / "lazy_reader.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.diag([2.0, 3.0, 4.0, 1.0])), path)
        return path

    def test_shape_is_lazy(self, nifti_path: Path) -> None:
        image = ScalarImage(nifti_path, reader=_LazyNiftiReader())
        assert image.shape == (1, 8, 9, 10)
        assert image._data is None  # not materialized

    def test_dtype_and_affine_lazy(self, nifti_path: Path) -> None:
        image = ScalarImage(nifti_path, reader=_LazyNiftiReader())
        assert image.dtype == np.dtype("float32")
        np.testing.assert_allclose(image.affine.numpy(), np.diag([2.0, 3.0, 4.0, 1.0]))
        assert image._data is None

    def test_lazy_slice(self, nifti_path: Path) -> None:
        image = ScalarImage(nifti_path, reader=_LazyNiftiReader())
        sliced = image.dataobj[:, 1:4, 2:5, 0:6]
        assert tuple(sliced.shape) == (1, 3, 3, 6)
        assert image._data is None

    def test_materialization_still_works(self, nifti_path: Path) -> None:
        image = ScalarImage(nifti_path, reader=_LazyNiftiReader())
        assert tuple(image.data.shape) == (1, 8, 9, 10)


class TestSimpleCustomReaderUnchanged:
    """Simple `(tensor, affine)` readers keep loading eagerly, as before."""

    def test_simple_reader_loads_eagerly(self, tmp_path: Path) -> None:
        calls: list = []

        def reader(path: Path, **kwargs):
            calls.append(path)
            return torch.ones(1, 3, 4, 5), np.eye(4)

        path = tmp_path / "anything.mha"
        image = ScalarImage(path, reader=reader)
        assert image.shape == (1, 3, 4, 5)
        assert calls  # reader was invoked (full load), no lazy backend
        assert image._data is not None


class TestNormalizeIndex:
    """Error handling of the shared index-normalization helper."""

    def test_invalid_top_level_type(self) -> None:
        from torchio.data.backends import normalize_index

        with pytest.raises(TypeError, match="not understood"):
            normalize_index("foo")  # type: ignore[arg-type]

    def test_invalid_element_in_tuple(self) -> None:
        from torchio.data.backends import normalize_index

        with pytest.raises(TypeError, match="not understood"):
            normalize_index((0, "bad"))  # type: ignore[arg-type]

    def test_multiple_ellipsis(self) -> None:
        from torchio.data.backends import normalize_index

        with pytest.raises(IndexError, match="one ellipsis"):
            normalize_index((..., 0, ...))

    def test_too_many_indices(self) -> None:
        from torchio.data.backends import normalize_index

        with pytest.raises(IndexError, match="Too many indices"):
            normalize_index((0, 0, 0, 0, 0))

    def test_negative_one_keeps_last(self) -> None:
        from torchio.data.backends import normalize_index

        assert normalize_index(-1)[0] == slice(-1, None)
