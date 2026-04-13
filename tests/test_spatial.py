"""Tests for spatial transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import torchio as tio
from torchio import AffineMatrix
from torchio.transforms import Affine as AffineTransform
from torchio.transforms.spatial import _border_mean
from torchio.transforms.spatial import _build_sampling_grid
from torchio.transforms.spatial import _check_folding
from torchio.transforms.spatial import _check_shared_space
from torchio.transforms.spatial import _compute_channel_pad_value
from torchio.transforms.spatial import _is_spacing_list
from torchio.transforms.spatial import _is_spacing_tuple
from torchio.transforms.spatial import _is_target_space_tuple
from torchio.transforms.spatial import _normalize_parameter_value
from torchio.transforms.spatial import _otsu_threshold
from torchio.transforms.spatial import _parse_center
from torchio.transforms.spatial import _parse_control_points
from torchio.transforms.spatial import _parse_default_pad_value
from torchio.transforms.spatial import _parse_interpolation
from torchio.transforms.spatial import _parse_locked_borders
from torchio.transforms.spatial import _parse_num_control_points
from torchio.transforms.spatial import _parse_spacing
from torchio.transforms.spatial import _parse_target_space_tuple
from torchio.transforms.spatial import _prepare_fill_value
from torchio.transforms.spatial import _to_nonnegative_parameter_range
from torchio.transforms.spatial import _to_positive_range
from torchio.transforms.spatial import _validate_isotropic


def _make_subject(
    shape: tuple[int, int, int] = (11, 11, 11),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tio.Subject:
    data = torch.arange(
        shape[0] * shape[1] * shape[2],
        dtype=torch.float32,
    ).reshape(1, *shape)
    label = torch.zeros(1, *shape, dtype=torch.float32)
    label[
        0,
        shape[0] // 4 : shape[0] // 4 + 2,
        shape[1] // 4 : shape[1] // 4 + 2,
        shape[2] // 4 : shape[2] // 4 + 2,
    ] = 1
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    return tio.Subject(
        t1=tio.ScalarImage(data, affine=affine),
        seg=tio.LabelMap(label, affine=affine),
    )


class TestSpatial:
    def test_identity(self) -> None:
        subject = _make_subject()
        transformed = tio.Spatial()(subject)

        torch.testing.assert_close(transformed.t1.data, subject.t1.data)
        torch.testing.assert_close(transformed.seg.data, subject.seg.data)
        np.testing.assert_allclose(
            transformed.t1.affine.numpy(),
            subject.t1.affine.numpy(),
        )

    def test_affine_first_changes_result(self) -> None:
        subject = _make_subject()
        control_points = torch.zeros(5, 5, 5, 3)
        control_points[2, 2, 2, 0] = 2.0

        kwargs = {
            "scales": 1.0,
            "degrees": (0.0, 0.0, 45.0),
            "translation": 0.0,
            "control_points": control_points,
            "default_pad_value": 0.0,
            "default_pad_label": 0.0,
        }

        first = tio.Spatial(affine_first=True, **kwargs)(subject)
        second = tio.Spatial(affine_first=False, **kwargs)(subject)

        assert not torch.allclose(first.t1.data, second.t1.data)

    def test_2d_suppresses_out_of_plane(self) -> None:
        data = torch.rand(1, 8, 8, 1)
        subject = tio.Subject(t1=tio.ScalarImage(data))
        transformed = AffineTransform(
            scales=1.0,
            degrees=(0.0, 0.0, 10.0),
            translation=0.0,
            center="image",
            default_pad_value=0.0,
        )(subject)
        assert transformed.t1.spatial_shape[-1] == 1


class TestResample:
    def test_spacing_target_changes_shape_and_affine(self) -> None:
        subject = _make_subject(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
        transformed = tio.Resample(2)(subject)

        assert transformed.t1.spatial_shape == (6, 6, 6)
        assert transformed.seg.spatial_shape == (6, 6, 6)
        np.testing.assert_allclose(transformed.t1.spacing, (2.0, 2.0, 2.0))
        np.testing.assert_allclose(
            transformed.t1.affine.numpy(),
            transformed.seg.affine.numpy(),
        )

    def test_named_image_target_uses_reference_space(self) -> None:
        reference = tio.ScalarImage(
            torch.ones(1, 6, 6, 6),
            affine=np.diag([2.0, 2.0, 2.0, 1.0]),
        )
        moving = tio.ScalarImage(
            torch.ones(1, 12, 12, 12),
            affine=np.diag([1.0, 1.0, 1.0, 1.0]),
        )
        subject = tio.Subject(t1=reference, t2=moving)

        transformed = tio.Resample("t1")(subject)

        assert transformed.t2.spatial_shape == transformed.t1.spatial_shape
        np.testing.assert_allclose(
            transformed.t2.affine.numpy(),
            transformed.t1.affine.numpy(),
        )

    def test_inverse_restores_geometry(self) -> None:
        subject = _make_subject(shape=(12, 12, 12))
        transformed = tio.Resample(2)(subject)

        restored = transformed.apply_inverse_transform()

        assert restored.t1.spatial_shape == subject.t1.spatial_shape
        np.testing.assert_allclose(
            restored.t1.affine.numpy(),
            subject.t1.affine.numpy(),
        )

    def test_target_image_object(self) -> None:
        subject = _make_subject(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
        reference = tio.ScalarImage(
            torch.ones(1, 6, 6, 6),
            affine=np.diag([2.0, 2.0, 2.0, 1.0]),
        )
        transformed = tio.Resample(target=reference)(subject)
        assert transformed.t1.spatial_shape == (6, 6, 6)

    def test_target_tuple_spacing(self) -> None:
        subject = _make_subject(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
        transformed = tio.Resample(target=(2.0, 2.0, 2.0))(subject)
        assert transformed.t1.spatial_shape == (6, 6, 6)

    def test_target_shape_affine_pair(self) -> None:
        subject = _make_subject(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
        target_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        transformed = tio.Resample(target=((6, 6, 6), target_affine))(subject)
        assert transformed.t1.spatial_shape == (6, 6, 6)

    def test_target_file_path(self, tmp_path: Path) -> None:
        import nibabel as nib

        ref_data = np.zeros((6, 6, 6), dtype=np.float32)
        ref_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        nib.save(nib.Nifti1Image(ref_data, ref_affine), tmp_path / "ref.nii")
        subject = _make_subject(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
        transformed = tio.Resample(target=str(tmp_path / "ref.nii"))(subject)
        assert transformed.t1.spatial_shape == (6, 6, 6)

    def test_target_ndarray_spacing(self) -> None:
        subject = _make_subject(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
        transformed = tio.Resample(target=np.array([2.0, 2.0, 2.0]))(subject)
        assert transformed.t1.spatial_shape == (6, 6, 6)

    def test_antialias_smooths_before_downsample(self) -> None:
        subject = _make_subject(shape=(20, 20, 20), spacing=(0.5, 0.5, 0.5))
        no_aa = tio.Resample(2)(subject)
        with_aa = tio.Resample(2, antialias=True)(subject)
        assert with_aa.t1.spatial_shape == no_aa.t1.spatial_shape
        # Antialiased result should be smoother (lower high-freq energy)
        assert not torch.allclose(with_aa.t1.data, no_aa.t1.data)

    def test_antialias_skips_label_maps(self) -> None:
        subject = _make_subject(shape=(20, 20, 20), spacing=(0.5, 0.5, 0.5))
        transformed = tio.Resample(2, antialias=True)(subject)
        unique = set(transformed.seg.data.unique().tolist())
        assert unique <= {0.0, 1.0}

    def test_antialias_noop_on_upsample(self) -> None:
        subject = _make_subject(shape=(6, 6, 6), spacing=(2.0, 2.0, 2.0))
        no_aa = tio.Resample(1)(subject)
        with_aa = tio.Resample(1, antialias=True)(subject)
        torch.testing.assert_close(with_aa.t1.data, no_aa.t1.data)


class TestAffine:
    def test_transform_changes_data(self) -> None:
        subject = _make_subject()
        transform = AffineTransform(
            scales=1.0,
            degrees=(0.0, 0.0, 90.0),
            translation=0.0,
            center="image",
            default_pad_value=0.0,
            default_pad_label=0.0,
        )

        transformed = transform(subject)

        assert not torch.allclose(transformed.t1.data, subject.t1.data)
        np.testing.assert_allclose(
            transformed.t1.affine.numpy(),
            subject.t1.affine.numpy(),
        )

    def test_inverse_restores_geometry(self) -> None:
        subject = _make_subject()
        transform = AffineTransform(
            scales=(1.1, 0.9, 1.0),
            degrees=(0.0, 0.0, 20.0),
            translation=(1.0, -2.0, 0.5),
            center="image",
            default_pad_value=0.0,
            default_pad_label=0.0,
        )

        restored = transform(subject).apply_inverse_transform()

        assert restored.t1.spatial_shape == subject.t1.spatial_shape
        np.testing.assert_allclose(
            restored.t1.affine.numpy(),
            subject.t1.affine.numpy(),
        )

    def test_isotropic_scales(self) -> None:
        subject = _make_subject()
        transform = AffineTransform(
            scales=(0.9, 1.1),
            isotropic=True,
            degrees=0.0,
            translation=0.0,
            default_pad_value=0.0,
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape

    def test_center_origin(self) -> None:
        subject = _make_subject()
        transform = AffineTransform(
            scales=1.0,
            degrees=(0.0, 0.0, 10.0),
            translation=0.0,
            center="origin",
            default_pad_value=0.0,
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape

    def test_choice_degrees(self) -> None:
        subject = _make_subject()
        transform = AffineTransform(
            scales=1.0,
            degrees=tio.Choice([-90, 0, 90, 180]),
            translation=0.0,
            default_pad_value=0.0,
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape

    def test_per_axis_mixed_specs(self) -> None:
        subject = _make_subject()
        transform = AffineTransform(
            scales=1.0,
            degrees=(0, 0, tio.Choice([-90, 0, 90])),
            translation=0.0,
            default_pad_value=0.0,
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape

    def test_distribution_parameter(self) -> None:
        from torch.distributions import Normal

        subject = _make_subject()
        transform = AffineTransform(
            scales=1.0,
            degrees=Normal(0.0, 5.0),
            translation=0.0,
            default_pad_value=0.0,
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape


class TestElasticDeformation:
    def test_accepts_tensor_control_points(self) -> None:
        subject = _make_subject()
        control_points = torch.zeros(5, 5, 5, 3)
        control_points[2, 2, 2, 0] = 2.0

        transformed = tio.ElasticDeformation(control_points=control_points)(subject)

        assert not torch.allclose(transformed.t1.data, subject.t1.data)

    def test_label_interpolation_preserves_label_values(self) -> None:
        subject = _make_subject()
        transformed = AffineTransform(
            scales=(1.1, 1.0, 1.0),
            degrees=(0.0, 0.0, 15.0),
            translation=0.0,
            center="image",
            default_pad_value=0.0,
            default_pad_label=0.0,
        )(subject)

        unique = set(transformed.seg.data.unique().tolist())
        assert unique <= {0.0, 1.0}

    def test_sampled_max_displacement(self) -> None:
        subject = _make_subject()
        transformed = tio.ElasticDeformation(
            max_displacement=2.0,
            num_control_points=5,
            locked_borders=1,
        )(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape

    def test_inverse_with_elastic(self) -> None:
        subject = _make_subject()
        control_points = torch.zeros(5, 5, 5, 3)
        control_points[2, 2, 2, 0] = 1.0
        transformed = tio.ElasticDeformation(
            control_points=control_points,
        )(subject)
        restored = transformed.apply_inverse_transform()
        assert restored.t1.spatial_shape == subject.t1.spatial_shape

    def test_folding_warning(self) -> None:
        with pytest.warns(RuntimeWarning, match="folding"):
            _check_folding(
                control_points=np.ones((5, 5, 5, 3)),
                max_displacement=(100.0, 100.0, 100.0),
                shape=(10, 10, 10),
                spacing=np.array([1.0, 1.0, 1.0]),
            )


class TestPadValue:
    def test_pad_value_mean(self) -> None:
        subject = _make_subject()
        transform = tio.Spatial(
            degrees=(0.0, 0.0, 30.0),
            scales=1.0,
            translation=0.0,
            default_pad_value="mean",
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape

    def test_pad_value_otsu(self) -> None:
        subject = _make_subject()
        transform = tio.Spatial(
            degrees=(0.0, 0.0, 30.0),
            scales=1.0,
            translation=0.0,
            default_pad_value="otsu",
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape

    def test_pad_value_numeric_nonzero(self) -> None:
        subject = _make_subject()
        transform = tio.Spatial(
            degrees=(0.0, 0.0, 30.0),
            scales=1.0,
            translation=0.0,
            default_pad_value=42.0,
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape

    def test_compute_channel_pad_minimum(self) -> None:
        t = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3)
        assert _compute_channel_pad_value(t, "minimum") == 0.0

    def test_compute_channel_pad_mean(self) -> None:
        t = torch.ones(3, 3, 3)
        val = _compute_channel_pad_value(t, "mean")
        assert abs(val - 1.0) < 1e-5

    def test_compute_channel_pad_otsu(self) -> None:
        t = torch.ones(3, 3, 3)
        val = _compute_channel_pad_value(t, "otsu")
        assert isinstance(val, float)


class TestBorderMeanAndOtsu:
    def test_border_mean_no_otsu(self) -> None:
        t = torch.ones(5, 5, 5) * 3.0
        assert abs(_border_mean(t, filter_otsu=False) - 3.0) < 1e-5

    def test_border_mean_with_otsu(self) -> None:
        t = torch.ones(5, 5, 5)
        result = _border_mean(t, filter_otsu=True)
        assert isinstance(result, float)

    def test_otsu_threshold_empty(self) -> None:
        assert _otsu_threshold(torch.tensor([])) == 0.0

    def test_otsu_threshold_basic(self) -> None:
        values = torch.tensor([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
        threshold = _otsu_threshold(values)
        assert 0.0 <= threshold <= 10.0


class TestValidation:
    def test_locked_borders_invalid(self) -> None:
        with pytest.raises(ValueError, match="locked_borders"):
            _parse_locked_borders(5)

    def test_locked_borders_2_with_4_control_points(self) -> None:
        with pytest.raises(ValueError, match="identity elastic field"):
            tio.Spatial(
                num_control_points=4,
                locked_borders=2,
            )

    def test_invalid_default_pad_label(self) -> None:
        with pytest.raises(TypeError, match="default_pad_label"):
            tio.Spatial(default_pad_label="bad")  # type: ignore[arg-type]

    def test_negative_scales(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            _to_positive_range(-1.0)

    def test_negative_max_displacement(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _to_nonnegative_parameter_range(-1.0)

    def test_isotropic_with_per_axis(self) -> None:
        with pytest.raises(ValueError, match="isotropic"):
            _validate_isotropic((1.0, 1.0, 1.0), isotropic=True)

    def test_parse_num_control_points_too_small(self) -> None:
        with pytest.raises(ValueError, match="greater than 3"):
            _parse_num_control_points(3)

    def test_parse_control_points_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="n_i, n_j, n_k, 3"):
            _parse_control_points(torch.zeros(5, 5, 2))

    def test_parse_control_points_axis_too_small(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            _parse_control_points(torch.zeros(3, 5, 5, 3))

    def test_parse_interpolation_invalid(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            _parse_interpolation("cubic")  # type: ignore[arg-type]

    def test_parse_interpolation_not_string(self) -> None:
        with pytest.raises(TypeError, match="string"):
            _parse_interpolation(42)  # type: ignore[arg-type]

    def test_parse_default_pad_value_invalid_string(self) -> None:
        with pytest.raises(ValueError, match="minimum"):
            _parse_default_pad_value("bad")  # type: ignore[arg-type]

    def test_parse_center_invalid(self) -> None:
        with pytest.raises(ValueError, match="center"):
            _parse_center("middle")  # type: ignore[arg-type]

    def test_parse_spacing_negative(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _parse_spacing(-1.0)

    def test_parse_spacing_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="3 values"):
            _parse_spacing((1.0, 2.0))

    def test_parse_spacing_ndarray_wrong_size(self) -> None:
        with pytest.raises(ValueError, match="3 values"):
            _parse_spacing(np.array([1.0, 2.0]))

    def test_parse_spacing_tuple_3(self) -> None:
        result = _parse_spacing((1.0, 2.0, 3.0))
        assert result == (1.0, 2.0, 3.0)

    def test_parse_spacing_ndarray(self) -> None:
        result = _parse_spacing(np.array([1.0, 2.0, 3.0]))
        assert result == (1.0, 2.0, 3.0)

    def test_target_not_understood(self) -> None:
        subject = _make_subject()
        with pytest.raises(ValueError, match="not understood"):
            tio.Resample(target=object())(subject)  # type: ignore[arg-type]

    def test_target_unknown_string(self) -> None:
        subject = _make_subject()
        with pytest.raises(ValueError, match="Unknown target"):
            tio.Resample(target="nonexistent_image")(subject)

    def test_shared_space_shape_mismatch(self) -> None:
        from torchio.data.batch import ImagesBatch

        batch_a = ImagesBatch(
            torch.rand(1, 1, 8, 8, 8),
            [AffineMatrix()],
        )
        batch_b = ImagesBatch(
            torch.rand(1, 1, 10, 10, 10),
            [AffineMatrix()],
        )
        with pytest.raises(RuntimeError, match="shape"):
            _check_shared_space(
                {"a": batch_a, "b": batch_b},
                (8, 8, 8),
                AffineMatrix(),
            )

    def test_shared_space_affine_mismatch(self) -> None:
        from torchio.data.batch import ImagesBatch

        batch_a = ImagesBatch(
            torch.rand(1, 1, 8, 8, 8),
            [AffineMatrix()],
        )
        batch_b = ImagesBatch(
            torch.rand(1, 1, 8, 8, 8),
            [AffineMatrix(np.diag([2.0, 2.0, 2.0, 1.0]))],
        )
        with pytest.raises(RuntimeError, match="same affine"):
            _check_shared_space(
                {"a": batch_a, "b": batch_b},
                (8, 8, 8),
                AffineMatrix(),
            )


class TestTypeGuards:
    def test_is_spacing_tuple(self) -> None:
        assert _is_spacing_tuple((1.0, 2.0, 3.0))
        assert not _is_spacing_tuple((1.0, 2.0))
        assert not _is_spacing_tuple([1.0, 2.0, 3.0])

    def test_is_spacing_list(self) -> None:
        assert _is_spacing_list([1.0, 2.0, 3.0])
        assert not _is_spacing_list((1.0, 2.0, 3.0))

    def test_is_target_space_tuple(self) -> None:
        assert _is_target_space_tuple(((6, 6, 6), np.eye(4)))
        assert not _is_target_space_tuple((1.0, 2.0, 3.0))


class TestEdgeCases:
    """Tests for defensive branches and edge cases."""

    def test_exclude_all_images(self) -> None:
        """make_params returns empty when all images are excluded (line 210)."""
        subject = _make_subject()
        transform = tio.Spatial(exclude=["t1", "seg"])
        from torchio.data.batch import SubjectsBatch

        batch = SubjectsBatch.from_subjects([subject])
        params = transform.make_params(batch)
        assert params == {"selected_images": []}

    def test_apply_transform_empty_selection(self) -> None:
        """apply_transform returns batch unchanged (line 278)."""
        subject = _make_subject()
        from torchio.data.batch import SubjectsBatch

        batch = SubjectsBatch.from_subjects([subject])
        transform = tio.Spatial()
        original_data = batch.images["t1"].data.clone()
        result = transform.apply_transform(batch, {"selected_images": []})
        torch.testing.assert_close(result.images["t1"].data, original_data)

    def test_inverse_missing_original_space(self) -> None:
        """inverse raises when original is None (lines 330-331)."""
        transform = tio.Spatial()
        params = {
            "affine_matrix": None,
            "control_points": None,
            "original": None,
            "affine_first": True,
            "image_interpolation": "linear",
            "label_interpolation": "nearest",
            "default_pad_value": "minimum",
            "default_pad_label": 0,
        }
        with pytest.raises(RuntimeError, match="original output space"):
            transform.inverse(params)

    def test_apply_spatial_empty_names(self) -> None:
        """_apply_spatial_to_batch returns early for empty names (line 556)."""
        from torchio.transforms.spatial import _apply_spatial_to_batch

        subject = _make_subject()
        from torchio.data.batch import SubjectsBatch

        batch = SubjectsBatch.from_subjects([subject])
        original = batch.images["t1"].data.clone()
        _apply_spatial_to_batch(
            batch=batch,
            image_names=[],
            target_space=None,
            affine_matrix=None,
            control_points=None,
            max_displacement=None,
            affine_first=True,
            image_interpolation="linear",
            label_interpolation="nearest",
            antialias=False,
            default_pad_value="minimum",
            default_pad_label=0.0,
        )
        torch.testing.assert_close(batch.images["t1"].data, original)

    def test_build_grid_elastic_no_max_displacement(self) -> None:
        """_build_sampling_grid computes max_displacement from field (line 742)."""
        control_points = torch.zeros(5, 5, 5, 3)
        control_points[2, 2, 2, 0] = 0.5
        grid = _build_sampling_grid(
            input_shape=(11, 11, 11),
            input_affine=AffineMatrix(),
            output_shape=(11, 11, 11),
            output_affine=AffineMatrix(),
            affine_matrix=None,
            control_points=control_points,
            max_displacement=None,
            affine_first=True,
            device=torch.device("cpu"),
        )
        assert grid.shape == (1, 11, 11, 11, 3)

    def test_batch_fill_value_bad_type(self) -> None:
        """_batch_fill_value raises TypeError for non-str non-number (lines 907-911)."""
        from torchio.data.batch import ImagesBatch
        from torchio.transforms.spatial import _batch_fill_value

        batch = ImagesBatch(
            torch.rand(1, 1, 4, 4, 4),
            [AffineMatrix()],
        )
        with pytest.raises(TypeError, match="string or number"):
            _batch_fill_value(
                batch,
                default_pad_value=object(),  # type: ignore[arg-type]
                default_pad_label=0.0,
            )

    def test_prepare_fill_value_multidim(self) -> None:
        """_prepare_fill_value returns a >1D tensor unchanged (line 945)."""
        fill = torch.ones(1, 2, 1, 1, 1)
        ref = torch.zeros(1, 2, 3, 3, 3)
        result = _prepare_fill_value(fill, ref)
        assert result is not None
        assert result.shape == (1, 2, 1, 1, 1)

    def test_compute_channel_pad_unknown(self) -> None:
        """_compute_channel_pad_value raises for unknown strategy (lines 959-960)."""
        with pytest.raises(ValueError, match="Unknown"):
            _compute_channel_pad_value(
                torch.ones(3, 3, 3),
                "bad_strategy",  # type: ignore[arg-type]
            )

    def test_parse_target_space_tuple_wrong_length(self) -> None:
        """_parse_target_space_tuple raises for non-3 shape (lines 1386-1387)."""
        with pytest.raises(ValueError, match="length 3"):
            _parse_target_space_tuple([6, 6], np.eye(4))

    def test_normalize_parameter_value_distribution(self) -> None:
        """_normalize_parameter_value returns Distribution unchanged (line 1548)."""
        from torch.distributions import Normal

        dist = Normal(0.0, 1.0)
        assert _normalize_parameter_value(dist) is dist


class TestExports:
    def test_root_exports_expose_transform_and_matrix(self) -> None:
        assert hasattr(tio, "Spatial")
        assert hasattr(tio, "Resample")
        assert hasattr(tio, "ElasticDeformation")
        assert tio.Affine is AffineTransform
        assert tio.AffineMatrix is AffineMatrix
