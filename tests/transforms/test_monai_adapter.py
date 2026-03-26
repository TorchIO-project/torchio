import warnings

import numpy as np
import pytest
import torch
from monai.transforms import MapTransform
from monai.transforms import NormalizeIntensity
from monai.transforms import NormalizeIntensityd
from monai.transforms import ScaleIntensity
from monai.transforms import ScaleIntensityd
from monai.transforms import SpatialCrop
from monai.transforms import SpatialCropd

import torchio as tio

from ..utils import TorchioTestCase


class TestMonaiAdapterDict(TorchioTestCase):
    """Tests for :class:`MonaiAdapter` with dictionary transforms."""

    def test_not_callable(self):
        with pytest.raises(TypeError, match='callable'):
            tio.MonaiAdapter('not a callable')

    def test_intensity_transform(self):
        """Verify that a MONAI intensity transform modifies the data."""
        original_data = self.sample_subject.t1.data.clone()
        transform = tio.MonaiAdapter(ScaleIntensityd(keys=['t1'], factor=2.0))
        transformed = transform(self.sample_subject)
        assert not torch.equal(transformed.t1.data, original_data)

    def test_multiple_keys(self):
        """Verify that a transform is applied to multiple keys."""
        original_t1 = self.sample_subject.t1.data.clone()
        original_t2 = self.sample_subject.t2.data.clone()
        transform = tio.MonaiAdapter(
            ScaleIntensityd(keys=['t1', 't2'], factor=2.0),
        )
        transformed = transform(self.sample_subject)
        assert not torch.equal(transformed.t1.data, original_t1)
        assert not torch.equal(transformed.t2.data, original_t2)

    def test_non_image_entries_preserved(self):
        """Non-image entries in the Subject should be preserved."""
        subject = tio.Subject(
            t1=tio.ScalarImage(tensor=torch.randn(1, 10, 10, 10)),
            age=42,
            name='test',
        )
        transform = tio.MonaiAdapter(ScaleIntensityd(keys=['t1'], factor=1.0))
        transformed = transform(subject)
        assert transformed['age'] == 42
        assert transformed['name'] == 'test'

    def test_affine_preserved_intensity(self):
        """Affine should be unchanged for intensity-only transforms."""
        original_affine = self.sample_subject.t1.affine.copy()
        transform = tio.MonaiAdapter(
            NormalizeIntensityd(keys=['t1']),
        )
        transformed = transform(self.sample_subject)
        np.testing.assert_array_equal(
            transformed.t1.affine,
            original_affine,
        )

    def test_affine_updated_spatial(self):
        """Cropping should shift the affine origin but keep the rotation/scale."""
        subject = tio.Subject(
            t1=tio.ScalarImage(tensor=torch.randn(1, 32, 32, 32)),
        )
        original_affine = subject.t1.affine.copy()
        transform = tio.MonaiAdapter(
            SpatialCropd(keys=['t1'], roi_start=[4, 4, 4], roi_end=[20, 20, 20]),
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == (16, 16, 16)
        new_affine = transformed.t1.affine
        # Rotation/scale (3×3) should be unchanged
        np.testing.assert_array_equal(new_affine[:3, :3], original_affine[:3, :3])
        # Translation (origin) should differ due to cropping offset
        assert not np.array_equal(new_affine[:3, 3], original_affine[:3, 3])

    def test_spatial_shape_changes(self):
        """Tensor shape should be updated by spatial transforms."""
        subject = tio.Subject(
            t1=tio.ScalarImage(tensor=torch.randn(1, 32, 32, 32)),
            seg=tio.LabelMap(tensor=torch.ones(1, 32, 32, 32)),
        )
        transform = tio.MonaiAdapter(
            SpatialCropd(
                keys=['t1', 'seg'],
                roi_start=[0, 0, 0],
                roi_end=[16, 16, 16],
            ),
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == (16, 16, 16)
        assert transformed.seg.spatial_shape == (16, 16, 16)

    def test_compose_integration(self):
        """MonaiAdapter should work inside tio.Compose."""
        pipeline = tio.Compose(
            [
                tio.MonaiAdapter(ScaleIntensityd(keys=['t1'], factor=2.0)),
                tio.RandomFlip(p=0),
            ]
        )
        original_data = self.sample_subject.t1.data.clone()
        transformed = pipeline(self.sample_subject)
        assert not torch.equal(transformed.t1.data, original_data)

    def test_compose_monai_between_torchio(self):
        """MONAI transforms should chain correctly with TorchIO transforms."""
        pipeline = tio.Compose(
            [
                tio.RandomFlip(p=0),
                tio.MonaiAdapter(ScaleIntensityd(keys=['t1'], factor=1.0)),
                tio.RandomFlip(p=0),
            ]
        )
        pipeline(self.sample_subject)

    def test_tensor_input(self):
        """Dict MonaiAdapter should work with 4D tensor input via DataParser."""
        tensor = torch.randn(1, 10, 10, 10)
        transform = tio.MonaiAdapter(
            ScaleIntensityd(keys=['default_image_name'], factor=2.0),
            include=['default_image_name'],
        )
        result = transform(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == tensor.shape

    def test_image_input(self):
        """Dict MonaiAdapter should work with Image input via DataParser."""
        image = tio.ScalarImage(tensor=torch.randn(1, 10, 10, 10))
        transform = tio.MonaiAdapter(
            ScaleIntensityd(keys=['default_image_name'], factor=2.0),
            include=['default_image_name'],
        )
        result = transform(image)
        assert isinstance(result, tio.ScalarImage)

    def test_probability(self):
        """The p parameter should control application probability."""
        transform = tio.MonaiAdapter(
            ScaleIntensityd(keys=['t1'], factor=100.0),
            p=0,
        )
        original_data = self.sample_subject.t1.data.clone()
        transformed = transform(self.sample_subject)
        assert torch.equal(transformed.t1.data, original_data)

    def test_label_not_modified_when_not_in_keys(self):
        """Images not in MONAI keys should not be modified."""
        original_label = self.sample_subject.label.data.clone()
        transform = tio.MonaiAdapter(
            ScaleIntensityd(keys=['t1'], factor=2.0),
        )
        transformed = transform(self.sample_subject)
        assert torch.equal(transformed.label.data, original_label)


class TestMonaiAdapterArray(TorchioTestCase):
    """Tests for :class:`MonaiAdapter` with array transforms."""

    def test_array_intensity_transform(self):
        """Array transform should modify all images."""
        original_t1 = self.sample_subject.t1.data.clone()
        original_t2 = self.sample_subject.t2.data.clone()
        transform = tio.MonaiAdapter(ScaleIntensity(factor=2.0))
        transformed = transform(self.sample_subject)
        assert not torch.equal(transformed.t1.data, original_t1)
        assert not torch.equal(transformed.t2.data, original_t2)

    def test_array_include(self):
        """Array transform should respect the include parameter."""
        original_t2 = self.sample_subject.t2.data.clone()
        original_label = self.sample_subject.label.data.clone()
        transform = tio.MonaiAdapter(
            ScaleIntensity(factor=2.0),
            include=['t1'],
        )
        transformed = transform(self.sample_subject)
        assert not torch.equal(
            transformed.t1.data,
            self.sample_subject.t1.data,
        )
        assert torch.equal(transformed.t2.data, original_t2)
        assert torch.equal(transformed.label.data, original_label)

    def test_array_exclude(self):
        """Array transform should respect the exclude parameter."""
        original_label = self.sample_subject.label.data.clone()
        transform = tio.MonaiAdapter(
            ScaleIntensity(factor=2.0),
            exclude=['label'],
        )
        transformed = transform(self.sample_subject)
        assert not torch.equal(
            transformed.t1.data,
            self.sample_subject.t1.data,
        )
        assert torch.equal(transformed.label.data, original_label)

    def test_array_affine_preserved_intensity(self):
        """Affine should be unchanged for array intensity transforms."""
        original_affine = self.sample_subject.t1.affine.copy()
        transform = tio.MonaiAdapter(NormalizeIntensity())
        transformed = transform(self.sample_subject)
        np.testing.assert_array_equal(
            transformed.t1.affine,
            original_affine,
        )

    def test_array_affine_updated_spatial(self):
        """Cropping should shift the affine origin but keep the rotation/scale."""
        subject = tio.Subject(
            t1=tio.ScalarImage(tensor=torch.randn(1, 32, 32, 32)),
        )
        original_affine = subject.t1.affine.copy()
        transform = tio.MonaiAdapter(
            SpatialCrop(roi_start=[4, 4, 4], roi_end=[20, 20, 20]),
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == (16, 16, 16)
        new_affine = transformed.t1.affine
        np.testing.assert_array_equal(new_affine[:3, :3], original_affine[:3, :3])
        assert not np.array_equal(new_affine[:3, 3], original_affine[:3, 3])

    def test_array_tensor_input(self):
        """Array MonaiAdapter should work with 4D tensor input."""
        tensor = torch.randn(1, 10, 10, 10)
        transform = tio.MonaiAdapter(ScaleIntensity(factor=2.0))
        result = transform(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == tensor.shape

    def test_array_image_input(self):
        """Array MonaiAdapter should work with Image input."""
        image = tio.ScalarImage(tensor=torch.randn(1, 10, 10, 10))
        transform = tio.MonaiAdapter(ScaleIntensity(factor=2.0))
        result = transform(image)
        assert isinstance(result, tio.ScalarImage)

    def test_array_compose_integration(self):
        """Array MonaiAdapter should work inside tio.Compose."""
        pipeline = tio.Compose(
            [
                tio.MonaiAdapter(NormalizeIntensity()),
                tio.RandomFlip(p=0),
            ]
        )
        pipeline(self.sample_subject)

    def test_array_probability(self):
        """The p parameter should work with array transforms."""
        transform = tio.MonaiAdapter(
            ScaleIntensity(factor=100.0),
            p=0,
        )
        original_data = self.sample_subject.t1.data.clone()
        transformed = transform(self.sample_subject)
        assert torch.equal(transformed.t1.data, original_data)


class TestMonaiAdapterEdgeCases(TorchioTestCase):
    """Tests for edge cases and robustness of :class:`MonaiAdapter`."""

    def test_history_not_recorded(self):
        """MonaiAdapter should not be added to subject history.

        MONAI transform objects are not serializable, so recording them
        in the history would cause failures during replay via
        ``Subject.get_applied_transforms()``.
        """
        transform = tio.MonaiAdapter(ScaleIntensityd(keys=['t1'], factor=2.0))
        transformed = transform(self.sample_subject)
        assert len(transformed.applied_transforms) == 0

    def test_random_array_warns_multi_image(self):
        """A randomizable array transform on a multi-image subject should warn.

        When a MONAI ``Randomizable`` array transform is applied per-image,
        each image receives different random parameters, breaking spatial
        alignment.
        """
        from monai.transforms import RandScaleIntensity

        transform = tio.MonaiAdapter(RandScaleIntensity(factors=0.5, prob=1.0))
        with pytest.warns(UserWarning, match='Randomizable'):
            transform(self.sample_subject)

    def test_random_array_no_warn_single_image(self):
        """No warning for a randomizable array transform on a single image."""
        from monai.transforms import RandScaleIntensity

        subject = tio.Subject(
            t1=tio.ScalarImage(tensor=torch.randn(1, 10, 10, 10)),
        )
        transform = tio.MonaiAdapter(RandScaleIntensity(factors=0.5, prob=1.0))
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            transform(subject)

    def test_new_dict_key_wrapped_as_image(self):
        """New MetaTensor keys from MONAI dict transforms should become Images.

        When a MONAI dictionary transform creates a new key whose value
        is a ``MetaTensor`` or ``torch.Tensor``, the adapter should wrap
        it as a ``ScalarImage`` so it is visible to downstream TorchIO
        processing.
        """
        from monai.transforms import CopyItemsd

        subject = tio.Subject(
            t1=tio.ScalarImage(tensor=torch.randn(1, 10, 10, 10)),
        )
        transform = tio.MonaiAdapter(
            CopyItemsd(keys=['t1'], names=['t1_copy']),
        )
        transformed = transform(subject)
        assert 't1_copy' in transformed
        assert isinstance(transformed['t1_copy'], tio.ScalarImage)
        # New keys should be accessible via attribute syntax
        assert isinstance(transformed.t1_copy, tio.ScalarImage)

    def test_multi_sample_transform_raises(self):
        """MONAI transforms returning list[dict] should raise a clear error."""

        class FakeMultiSampleTransform(MapTransform):
            """Simulates a MONAI dict transform returning multiple samples."""

            def __init__(self):
                super().__init__(keys=['t1'])

            def __call__(self, data):
                return [data, data]

        transform = tio.MonaiAdapter(FakeMultiSampleTransform())
        with pytest.raises(TypeError, match='single mapping'):
            transform(self.sample_subject)

    def test_new_non_image_tensor_kept_raw(self):
        """Non-image tensors (0D/1D) from MONAI should not become Images."""

        class FakeStatsTransform(MapTransform):
            """Adds a scalar stat key to the output dict."""

            def __init__(self):
                super().__init__(keys=['t1'])

            def __call__(self, data):
                data = dict(data)
                data['t1'] = data['t1']
                data['mean'] = torch.tensor(0.5)
                data['indices'] = torch.tensor([0, 1, 2])
                return data

        subject = tio.Subject(
            t1=tio.ScalarImage(tensor=torch.randn(1, 10, 10, 10)),
        )
        transform = tio.MonaiAdapter(FakeStatsTransform())
        transformed = transform(subject)
        assert isinstance(transformed['mean'], torch.Tensor)
        assert not isinstance(transformed['mean'], tio.Image)
        assert isinstance(transformed['indices'], torch.Tensor)
        assert not isinstance(transformed['indices'], tio.Image)

    def test_repr(self):
        """__repr__ includes the adapter name and wrapped transform."""
        transform = tio.MonaiAdapter(NormalizeIntensityd(keys=['t1']))
        result = repr(transform)
        assert 'MonaiAdapter' in result
        assert 'NormalizeIntensityd' in result

    def test_to_hydra_config_raises(self):
        """to_hydra_config raises NotImplementedError."""
        transform = tio.MonaiAdapter(NormalizeIntensityd(keys=['t1']))
        with pytest.raises(NotImplementedError, match='Hydra'):
            transform.to_hydra_config()

    def test_array_transform_non_tensor_result_raises(self):
        """Array transform returning non-tensor raises TypeError."""

        class BadArrayTransform:
            def __call__(self, x):
                return 'not a tensor'

        subject = tio.Subject(
            t1=tio.ScalarImage(tensor=torch.randn(1, 10, 10, 10)),
        )
        transform = tio.MonaiAdapter(BadArrayTransform())
        with pytest.raises(TypeError, match='Expected a torch.Tensor'):
            transform(subject)

    def test_dict_transform_non_tensor_for_image_raises(self):
        """Dict transform returning non-tensor for image key raises TypeError."""

        class BadDictTransform(MapTransform):
            def __init__(self):
                super().__init__(keys=['t1'])

            def __call__(self, data):
                data = dict(data)
                data['t1'] = 'not a tensor'
                return data

        subject = tio.Subject(
            t1=tio.ScalarImage(tensor=torch.randn(1, 10, 10, 10)),
        )
        transform = tio.MonaiAdapter(BadDictTransform())
        with pytest.raises(TypeError, match='Expected a torch.Tensor'):
            transform(subject)
