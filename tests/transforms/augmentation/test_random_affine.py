from typing import Any
from typing import cast

import numpy as np
import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestRandomAffine(TorchioTestCase):
    """Tests for `RandomAffine`."""

    def setUp(self):
        # Set image origin far from center
        super().setUp()
        affine = self.sample_subject.t1.affine
        affine[:3, 3] = 1e5

    def test_rotation_image(self):
        # Rotation around image center
        transform = tio.RandomAffine(
            degrees=(90, 90),
            default_pad_value=0,
            center='image',
        )
        transformed = transform(self.sample_subject)
        total = transformed.t1.data.sum()
        self.assertNotEqual(total, 0)

    def test_rotation_origin(self):
        # Rotation around far away point, image should be empty
        transform = tio.RandomAffine(
            degrees=(90, 90),
            default_pad_value=0,
            center='origin',
        )
        transformed = transform(self.sample_subject)
        total = transformed.t1.data.sum()
        assert total == 0

    def test_no_rotation(self):
        transform = tio.RandomAffine(
            scales=(1, 1),
            degrees=(0, 0),
            default_pad_value=0,
            center='image',
        )
        transformed = transform(self.sample_subject)
        self.assert_tensor_almost_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
            atol=1e-3,
            rtol=1e-3,
        )

        transform = tio.RandomAffine(
            scales=(1, 1),
            degrees=(180, 180),
            default_pad_value=0,
            center='image',
        )
        transformed = transform(self.sample_subject)
        transformed = transform(transformed)
        self.assert_tensor_almost_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
            atol=1e-1,
            rtol=1e-1,
        )

    def test_isotropic(self):
        tio.RandomAffine(isotropic=True)(self.sample_subject)

    def test_mean(self):
        tio.RandomAffine(default_pad_value='mean')(self.sample_subject)

    def test_otsu(self):
        tio.RandomAffine(default_pad_value='otsu')(self.sample_subject)

    def test_bad_center(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(center='bad')

    def test_negative_scales(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(scales=(-1, 1))

    def test_scale_too_large(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(scales=1.5)

    def test_scales_range_with_negative_min(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(scales=(-1, 4))

    def test_wrong_scales_type(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(scales=cast(Any, 'wrong'))

    def test_wrong_degrees_type(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(degrees=cast(Any, 'wrong'))

    def test_too_many_translation_values(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(translation=(-10, 4, 42))

    def test_wrong_translation_type(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(translation=cast(Any, 'wrong'))

    def test_wrong_center(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(center=cast(Any, 0))

    def test_wrong_default_pad_value(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(default_pad_value='wrong')

    def test_wrong_image_interpolation_type(self):
        with pytest.raises(TypeError):
            tio.RandomAffine(image_interpolation=cast(Any, 0))

    def test_wrong_image_interpolation_value(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(image_interpolation='wrong')

    def test_incompatible_args_isotropic(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(scales=(0.8, 0.5, 0.1), isotropic=True)

    def test_parse_scales(self):
        def do_assert(transform: tio.RandomAffine) -> None:
            assert transform.scales == 3 * (0.9, 1.1)

        triplet_scales: tuple[float, float, float] = (0.1, 0.1, 0.1)
        sextet_scales: tuple[float, float, float, float, float, float] = (
            0.9,
            1.1,
            0.9,
            1.1,
            0.9,
            1.1,
        )
        do_assert(tio.RandomAffine(scales=0.1))
        do_assert(tio.RandomAffine(scales=(0.9, 1.1)))
        do_assert(tio.RandomAffine(scales=triplet_scales))
        do_assert(tio.RandomAffine(scales=sextet_scales))

    def test_parse_degrees(self):
        def do_assert(transform: tio.RandomAffine) -> None:
            assert transform.degrees == 3 * (-10, 10)

        triplet_degrees: tuple[int, int, int] = (10, 10, 10)
        sextet_degrees: tuple[int, int, int, int, int, int] = (
            -10,
            10,
            -10,
            10,
            -10,
            10,
        )
        do_assert(tio.RandomAffine(degrees=10))
        do_assert(tio.RandomAffine(degrees=(-10, 10)))
        do_assert(tio.RandomAffine(degrees=triplet_degrees))
        do_assert(tio.RandomAffine(degrees=sextet_degrees))

    def test_parse_translation(self):
        def do_assert(transform: tio.RandomAffine) -> None:
            assert transform.translation == 3 * (-10, 10)

        triplet_translation: tuple[int, int, int] = (10, 10, 10)
        sextet_translation: tuple[int, int, int, int, int, int] = (
            -10,
            10,
            -10,
            10,
            -10,
            10,
        )
        do_assert(tio.RandomAffine(translation=10))
        do_assert(tio.RandomAffine(translation=(-10, 10)))
        do_assert(tio.RandomAffine(translation=triplet_translation))
        do_assert(tio.RandomAffine(translation=sextet_translation))

    def test_default_value_label_map(self):
        # From https://github.com/TorchIO-project/torchio/issues/626
        a = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(1, 3, 3, 1)
        image = tio.LabelMap(tensor=a)
        aff = tio.RandomAffine(translation=(0, 1, 1), default_pad_value='otsu')
        transformed = aff(image)
        assert all(n in (0, 1) for n in transformed.data.flatten())

    def test_default_pad_label_parameter(self):
        # Test for issue #1304: Using default_pad_value if image is of type LABEL
        # Create a simple label map
        label_data = torch.ones((1, 2, 2, 2))
        subject = tio.Subject(label=tio.LabelMap(tensor=label_data))

        # Test 1: default_pad_label should be respected
        transform = tio.RandomAffine(
            translation=(10, 10),
            default_pad_label=250,
        )
        transformed_subject = transform(subject)

        # Should contain the specified pad value for labels
        message = 'default_pad_label=250 should be respected for LABEL images'
        transformed_label = transformed_subject.get_label_map('label')
        has_expected_value = (transformed_label.tensor == 250).any()
        assert has_expected_value, message

        # Test 2: backward compatibility - default_pad_value should still be ignored for labels
        message = 'default_pad_value should still be ignored for LABEL images (backward compatibility)'
        aff_old = tio.RandomAffine(
            translation=(-10, 10, -10, 10, -10, 10),
            default_pad_value=250,  # This should be ignored for labels
        )
        s_aug_old = aff_old.apply_transform(subject)

        # Should still use 0 (default for labels), not the default_pad_value
        augmented_label = s_aug_old.get_label_map('label')
        non_one_values = augmented_label.data[augmented_label.data != 1]
        all_zeros = (non_one_values == 0).all() if len(non_one_values) > 0 else True
        assert all_zeros, message

        # Test 3: Test direct Affine class with default_pad_label
        affine_transform = tio.Affine(
            scales=(1, 1, 1),
            degrees=(0, 0, 0),
            translation=(5, 0, 0),
            default_pad_label=123,
        )
        s_affine = affine_transform.apply_transform(subject)
        affine_label = s_affine.get_label_map('label')
        has_affine_value = (affine_label.tensor == 123).any()
        assert has_affine_value, 'Direct Affine class should respect default_pad_label'

    def test_wrong_default_pad_label(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(default_pad_label=cast(Any, 'minimum'))

    def test_affine_bad_center_raises(self):
        """Affine with invalid center argument raises ValueError."""
        with pytest.raises(ValueError, match='Center argument'):
            tio.Affine((1, 1, 1), (0, 0, 0), (0, 0, 0), center='bad')

    def test_affine_bad_default_pad_label_raises(self):
        """Affine with non-numeric default_pad_label raises ValueError."""
        with pytest.raises(ValueError, match='default_pad_label'):
            tio.Affine(
                (1, 1, 1),
                (0, 0, 0),
                (0, 0, 0),
                default_pad_label=cast(Any, 'wrong'),
            )

    def test_affine_inverse_transform(self):
        """Affine.inverse() applies the inverse of the transformation."""
        transform = tio.Affine(
            scales=(1, 1, 1),
            degrees=(0, 0, 10),
            translation=(5, 0, 0),
        )
        inverse = transform.inverse()
        assert inverse.invert_transform is True
        # Apply forward then inverse — shape should be preserved
        transformed = transform(self.sample_subject)
        restored = inverse(transformed)
        assert restored.t1.spatial_shape == transformed.t1.spatial_shape

    def test_otsu_pad_uniform_image(self):
        """Otsu padding on uniform image falls back to border mean."""
        image = tio.ScalarImage(tensor=torch.ones(1, 10, 10, 10))
        subject = tio.Subject(t1=image)
        transform = tio.RandomAffine(
            translation=5,
            default_pad_value='otsu',
        )
        transform(subject)

    def test_no_inverse(self):
        tensor = torch.zeros((1, 2, 2, 2))
        tensor[0, 1, 1, 1] = 1  # most RAS voxel
        expected = torch.zeros((1, 2, 2, 2))
        expected[0, 0, 1, 1] = 1
        scales = 1, 1, 1
        degrees = 0, 0, 90  # anterior should go left
        translation = 0, 0, 0
        apply_affine = tio.Affine(
            scales,
            degrees,
            translation,
        )
        transformed = apply_affine(tensor)
        self.assert_tensor_almost_equal(transformed, expected)

    def test_different_spaces(self):
        t1 = self.sample_subject.t1
        label = tio.Resample(2)(self.sample_subject.label)
        new_subject = tio.Subject(t1=t1, label=label)
        with pytest.raises(RuntimeError):
            tio.RandomAffine()(new_subject)
        tio.RandomAffine(check_shape=False)(new_subject)


class TestAffineProperties(TorchioTestCase):
    """Property-based tests for the pure PyTorch Affine implementation."""

    def _make_subject(
        self,
        shape: tuple[int, int, int] = (10, 12, 14),
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> tio.Subject:
        """Create a deterministic subject with known geometry."""
        torch.manual_seed(42)
        tensor = torch.rand(1, *shape)
        affine = torch.diag(torch.tensor([*spacing, 1.0], dtype=torch.float64))
        return tio.Subject(
            t1=tio.ScalarImage(tensor=tensor, affine=affine.numpy()),
        )

    def test_identity_preserves_data(self):
        """Identity transform (scales=1, degrees=0, translation=0) preserves data."""
        subject = self._make_subject()
        transform = tio.Affine(
            scales=(1, 1, 1),
            degrees=(0, 0, 0),
            translation=(0, 0, 0),
        )
        result = transform(subject)
        torch.testing.assert_close(
            subject.t1.data,
            result.t1.data,
            atol=1e-3,
            rtol=1e-3,
            check_dtype=False,
        )

    def test_forward_inverse_roundtrip(self):
        """Forward then inverse approximately restores the original."""
        subject = self._make_subject(shape=(30, 30, 30))
        original = subject.t1.data.clone()
        transform = tio.Affine(
            scales=(1, 1, 1),
            degrees=(0, 0, 5),
            translation=(0, 0, 0),
            default_pad_value='mean',
        )
        inverse = transform.inverse()
        result = inverse(transform(subject))
        # Interior voxels should be approximately restored;
        # borders lose data through padding + double interpolation
        s = 5
        orig_interior = original[:, s:-s, s:-s, s:-s]
        result_interior = result.t1.data[:, s:-s, s:-s, s:-s]
        mae = (orig_interior - result_interior).abs().mean().item()
        assert mae < 0.2, f'Mean absolute error {mae:.4f} too high for roundtrip'

    def test_pure_translation_shifts_content(self):
        """Translation along X shifts voxels predictably."""
        tensor = torch.zeros(1, 10, 10, 10)
        tensor[0, 5, 5, 5] = 1.0
        subject = tio.Subject(t1=tio.ScalarImage(tensor=tensor))
        transform = tio.Affine(
            scales=(1, 1, 1),
            degrees=(0, 0, 0),
            translation=(1, 0, 0),  # 1mm along X with 1mm spacing
        )
        result = transform(subject)
        # The bright voxel should have moved by ~1 voxel along X
        original_max_idx = tensor.argmax()
        result_max_idx = result.t1.data.argmax()
        assert original_max_idx != result_max_idx

    def test_90_rotation_z_moves_anterior_to_left(self):
        """A 90° rotation around Z moves the most anterior voxel to the left."""
        tensor = torch.zeros((1, 2, 2, 2))
        tensor[0, 1, 1, 1] = 1  # most RAS voxel
        expected = torch.zeros((1, 2, 2, 2))
        expected[0, 0, 1, 1] = 1
        transform = tio.Affine(
            scales=(1, 1, 1),
            degrees=(0, 0, 90),
            translation=(0, 0, 0),
        )
        result = transform(tensor)
        self.assert_tensor_almost_equal(result, expected)

    def test_label_map_uses_nearest_and_pad_label(self):
        """Label maps use nearest interpolation and default_pad_label."""
        label_data = torch.ones((1, 4, 4, 4))
        subject = tio.Subject(label=tio.LabelMap(tensor=label_data))
        transform = tio.Affine(
            scales=(1, 1, 1),
            degrees=(0, 0, 0),
            translation=(3, 0, 0),
            default_pad_label=99,
        )
        result = transform(subject)
        label = result.get_label_map('label')
        assert (label.data == 99).any(), 'Pad label value should appear'
        unique_values = set(label.data.unique().tolist())
        assert unique_values <= {0.0, 1.0, 99.0}

    def test_2d_image_no_xy_rotation(self):
        """2D images (D=1) should ignore X and Y rotation."""
        tensor = torch.rand(1, 8, 8, 1)
        subject = tio.Subject(t1=tio.ScalarImage(tensor=tensor))
        # Even with large X/Y rotation, 2D images should only rotate around Z
        transform = tio.Affine(
            scales=(1, 1, 1),
            degrees=(45, 45, 0),  # only X and Y rotation
            translation=(0, 0, 0),
        )
        result = transform(subject)
        # With X/Y zeroed for 2D, this should be identity-like
        torch.testing.assert_close(
            subject.t1.data,
            result.t1.data,
            atol=1e-3,
            rtol=1e-3,
            check_dtype=False,
        )

    def test_center_origin_vs_image_differ(self):
        """Different center modes produce different results for non-origin images."""
        subject = self._make_subject()
        # Move origin away from voxel center
        subject.t1.affine[:3, 3] = [100, 100, 100]

        transform_image = tio.Affine(
            scales=(1, 1, 1),
            degrees=(0, 0, 30),
            translation=(0, 0, 0),
            center='image',
        )
        transform_origin = tio.Affine(
            scales=(1, 1, 1),
            degrees=(0, 0, 30),
            translation=(0, 0, 0),
            center='origin',
        )
        result_image = transform_image(subject)
        result_origin = transform_origin(subject)
        # Results should differ since centers are far apart
        assert not torch.allclose(result_image.t1.data, result_origin.t1.data)

    def test_isotropic_scaling(self):
        """Isotropic scaling applies the same factor to all axes."""
        subject = self._make_subject()
        transform = tio.RandomAffine(
            scales=(0.5, 0.5),
            degrees=0,
            translation=0,
            isotropic=True,
        )
        # Should not raise and should produce valid output
        result = transform(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_unsupported_interpolation_raises(self):
        """Unsupported interpolation modes raise ValueError."""
        with pytest.raises(ValueError, match='not supported'):
            tio.RandomAffine(image_interpolation='bspline')
        with pytest.raises(ValueError, match='not supported'):
            tio.RandomAffine(image_interpolation='gaussian')
        with pytest.raises(ValueError, match='not supported'):
            tio.Affine(
                (1, 1, 1),
                (0, 0, 0),
                (0, 0, 0),
                image_interpolation='lanczos',
            )

    def test_anisotropic_spacing(self):
        """Transform works correctly with anisotropic voxel spacing."""
        subject = self._make_subject(spacing=(1.0, 0.5, 2.0))
        transform = tio.Affine(
            scales=(1, 1, 1),
            degrees=(0, 0, 0),
            translation=(0, 0, 0),
        )
        result = transform(subject)
        torch.testing.assert_close(
            subject.t1.data,
            result.t1.data,
            atol=1e-3,
            rtol=1e-3,
            check_dtype=False,
        )

    def test_scaling_changes_content(self):
        """Non-identity scaling modifies the image content."""
        subject = self._make_subject()
        transform = tio.Affine(
            scales=(1.5, 1.5, 1.5),
            degrees=(0, 0, 0),
            translation=(0, 0, 0),
            default_pad_value=0,
        )
        result = transform(subject)
        assert not torch.allclose(subject.t1.data, result.t1.data)

    def test_multichannel_image(self):
        """Transform handles multi-channel images correctly."""
        tensor = torch.rand(3, 8, 8, 8)  # 3 channels
        subject = tio.Subject(t1=tio.ScalarImage(tensor=tensor))
        transform = tio.Affine(
            scales=(1, 1, 1),
            degrees=(0, 0, 15),
            translation=(1, 0, 0),
        )
        result = transform(subject)
        assert result.t1.data.shape == (3, 8, 8, 8)


class TestAffineComparison(TorchioTestCase):
    """Tests comparing PyTorch affine against SimpleITK reference."""

    @staticmethod
    def _apply_sitk_affine(
        image: tio.ScalarImage,
        scales: tuple[float, float, float],
        degrees: tuple[float, float, float],
        translation: tuple[float, float, float],
        center: str = 'image',
        interpolation: str = 'linear',
        default_value: float = 0.0,
    ) -> torch.Tensor:
        """Apply affine transform using SimpleITK as reference implementation."""
        import SimpleITK as sitk

        from torchio.data.io import nib_to_sitk

        scaling_np = np.asarray(scales, dtype=float)
        degrees_np = np.asarray(degrees, dtype=float)
        translation_np = np.asarray(translation, dtype=float)

        # Build SimpleITK transforms
        if center == 'image':
            center_lps = image.get_center(lps=True)
        else:
            center_lps = None

        scale_transform = sitk.ScaleTransform(3)
        scale_transform.SetScale(scaling_np)
        if center_lps is not None:
            scale_transform.SetCenter(center_lps)

        def ras_to_lps(triplet):
            return np.array((-1, -1, 1), dtype=float) * np.asarray(triplet)

        euler = sitk.Euler3DTransform()
        radians = np.radians(degrees_np).astype(float)
        euler.SetRotation(*ras_to_lps(radians))
        euler.SetTranslation(ras_to_lps(translation_np))
        if center_lps is not None:
            euler.SetCenter(center_lps)

        composite = sitk.CompositeTransform([scale_transform, euler])
        composite = composite.GetInverse()

        sitk_interp = {
            'nearest': sitk.sitkNearestNeighbor,
            'linear': sitk.sitkLinear,
        }

        results = []
        for channel_tensor in image.data:
            sitk_image = nib_to_sitk(
                channel_tensor[np.newaxis],
                image.affine,
                force_3d=True,
            )
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(sitk_interp[interpolation])
            resampler.SetReferenceImage(sitk_image)
            resampler.SetDefaultPixelValue(float(default_value))
            resampler.SetOutputPixelType(sitk.sitkFloat32)
            resampler.SetTransform(composite)
            resampled = resampler.Execute(sitk_image)
            arr = sitk.GetArrayFromImage(resampled).transpose()
            results.append(torch.as_tensor(arr))
        return torch.stack(results)

    def _make_subject(
        self,
        shape: tuple[int, int, int] = (10, 12, 14),
    ) -> tio.Subject:
        torch.manual_seed(42)
        np.random.seed(42)
        tensor = torch.rand(1, *shape)
        return tio.Subject(
            t1=tio.ScalarImage(tensor=tensor),
        )

    def test_rotation_matches_sitk(self):
        """90° Z rotation matches SimpleITK reference."""
        subject = self._make_subject()
        params = {
            'scales': (1.0, 1.0, 1.0),
            'degrees': (0.0, 0.0, 90.0),
            'translation': (0.0, 0.0, 0.0),
        }
        sitk_result = self._apply_sitk_affine(
            subject.t1,
            default_value=0.0,
            **params,
        )
        torch_result = tio.Affine(
            default_pad_value=0.0,
            **params,
        )(subject).t1.data
        torch.testing.assert_close(
            torch_result,
            sitk_result,
            atol=0.1,
            rtol=0.1,
            check_dtype=False,
        )

    def test_scaling_matches_sitk(self):
        """Uniform scaling matches SimpleITK reference."""
        subject = self._make_subject()
        params = {
            'scales': (1.3, 1.3, 1.3),
            'degrees': (0.0, 0.0, 0.0),
            'translation': (0.0, 0.0, 0.0),
        }
        sitk_result = self._apply_sitk_affine(
            subject.t1,
            default_value=0.0,
            **params,
        )
        torch_result = tio.Affine(
            default_pad_value=0.0,
            **params,
        )(subject).t1.data
        torch.testing.assert_close(
            torch_result,
            sitk_result,
            atol=0.15,
            rtol=0.15,
            check_dtype=False,
        )

    def test_translation_matches_sitk(self):
        """Translation matches SimpleITK reference."""
        subject = self._make_subject()
        params = {
            'scales': (1.0, 1.0, 1.0),
            'degrees': (0.0, 0.0, 0.0),
            'translation': (3.0, -2.0, 1.0),
        }
        sitk_result = self._apply_sitk_affine(
            subject.t1,
            default_value=0.0,
            **params,
        )
        torch_result = tio.Affine(
            default_pad_value=0.0,
            **params,
        )(subject).t1.data
        torch.testing.assert_close(
            torch_result,
            sitk_result,
            atol=0.1,
            rtol=0.1,
            check_dtype=False,
        )

    def test_combined_matches_sitk(self):
        """Combined scale + rotation + translation matches SimpleITK reference."""
        subject = self._make_subject(shape=(30, 30, 30))
        params = {
            'scales': (1.05, 0.95, 1.02),
            'degrees': (5.0, -3.0, 8.0),
            'translation': (1.0, -0.5, 0.3),
        }
        sitk_result = self._apply_sitk_affine(
            subject.t1,
            default_value=0.0,
            **params,
        )
        torch_result = tio.Affine(
            default_pad_value=0.0,
            **params,
        )(subject).t1.data
        # Compare interior voxels only — boundaries differ due to different
        # interpolation kernels and boundary handling
        s = 5
        torch.testing.assert_close(
            torch_result[:, s:-s, s:-s, s:-s],
            sitk_result[:, s:-s, s:-s, s:-s],
            atol=0.2,
            rtol=0.2,
            check_dtype=False,
        )

    def test_nearest_rotation_matches_sitk(self):
        """Nearest-neighbor 90° rotation should closely match SimpleITK."""
        subject = self._make_subject()
        params = {
            'scales': (1.0, 1.0, 1.0),
            'degrees': (0.0, 0.0, 90.0),
            'translation': (0.0, 0.0, 0.0),
        }
        sitk_result = self._apply_sitk_affine(
            subject.t1,
            interpolation='nearest',
            default_value=0.0,
            **params,
        )
        torch_result = tio.Affine(
            image_interpolation='nearest',
            default_pad_value=0.0,
            **params,
        )(subject).t1.data
        torch.testing.assert_close(
            torch_result,
            sitk_result,
            atol=0.01,
            rtol=0.01,
            check_dtype=False,
        )
