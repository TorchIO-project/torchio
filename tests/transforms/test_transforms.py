import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from nibabel.nifti1 import Nifti1Image

import torchio as tio
from torchio.data.io import nib_to_sitk
from torchio.transforms.data_parser import DataParser
from torchio.transforms.fourier import FourierTransform
from torchio.transforms.interpolation import get_sitk_interpolator

from ..utils import TorchioTestCase


def _parse_range_with_invalid_value(
    transform: tio.Transform,
    value: object,
) -> None:
    transform._parse_range(cast(Any, value), 'name')


class TestTransforms(TorchioTestCase):
    """Tests for all transforms."""

    def get_transform(
        self,
        channels: Sequence[str],
        is_3d: bool = True,
        labels: bool = True,
    ) -> tio.Compose:
        landmarks_dict: dict[str, str | Path | np.ndarray] = {
            channel: np.linspace(0, 100, 13) for channel in channels
        }
        disp = 1 if is_3d else (1, 1, 0.01)
        elastic = tio.RandomElasticDeformation(max_displacement=disp)
        affine_elastic = tio.RandomAffineElasticDeformation(
            elastic_kwargs={'max_displacement': disp}
        )
        cp_args = (9, 21, 30) if is_3d else (21, 30, 1)
        resize_args = (10, 20, 30) if is_3d else (10, 20, 1)
        flip_axes = axes_downsample = (0, 1, 2) if is_3d else (0, 1)
        swap_patch = (2, 3, 4) if is_3d else (3, 4, 1)
        pad_args = (1, 2, 3, 0, 5, 6) if is_3d else (0, 0, 3, 0, 5, 6)
        crop_args = (3, 2, 8, 0, 1, 4) if is_3d else (0, 0, 8, 0, 1, 4)
        remapping = {1: 2, 2: 1, 3: 20, 4: 25}
        one_of_transforms: dict[tio.Transform, float] = {
            tio.RandomAffine(): 3,
            elastic: 1,
        }
        transforms: list[tio.Transform] = [
            tio.CropOrPad(cp_args),
            tio.EnsureShapeMultiple(2, method='crop'),
            tio.Resize(resize_args),
            tio.ToCanonical(),
            tio.RandomAnisotropy(downsampling=(1.75, 2), axes=axes_downsample),
            tio.CopyAffine(channels[0]),
            tio.Resample((1, 1.1, 1.25)),
            tio.RandomFlip(axes=flip_axes, flip_probability=1),
            tio.RandomMotion(),
            tio.RandomGhosting(axes=(0, 1, 2)),
            tio.RandomSpike(),
            tio.RandomNoise(),
            tio.RandomBlur(),
            tio.RandomSwap(patch_size=swap_patch, num_iterations=5),
            tio.Lambda(lambda x: 2 * x, types_to_apply=tio.INTENSITY),
            tio.RandomBiasField(),
            tio.RescaleIntensity(out_min_max=(0, 1)),
            tio.ZNormalization(),
            tio.HistogramStandardization(landmarks_dict),
            elastic,
            tio.RandomAffine(),
            affine_elastic,
            tio.OneOf(one_of_transforms),
            tio.RemapLabels(remapping=remapping, masking_method='Left'),
            tio.RemoveLabels([1, 3]),
            tio.SequentialLabels(),
            tio.Pad(pad_args, padding_mode=3),
            tio.Crop(crop_args),
        ]
        if labels:
            transforms.append(tio.RandomLabelsToImage(label_key='label'))
        return tio.Compose(transforms)

    def test_transforms_dict(self):
        transform = tio.RandomNoise(include=('t1', 't2'))
        input_dict: dict[str, object] = {
            name: image.data
            for name, image in self.sample_subject.get_images_dict(
                intensity_only=False
            ).items()
        }
        transformed = transform(input_dict)
        assert isinstance(transformed, dict)

    def test_transforms_dict_no_keys(self):
        transform = tio.RandomNoise()
        input_dict: dict[str, object] = {
            name: image.data
            for name, image in self.sample_subject.get_images_dict(
                intensity_only=False
            ).items()
        }
        with pytest.raises(RuntimeError):
            transform(input_dict)

    def test_transforms_image(self):
        transform = self.get_transform(
            channels=('default_image_name',),
            labels=False,
        )
        transformed = transform(self.sample_subject.t1)
        assert isinstance(transformed, tio.ScalarImage)

    def test_transforms_tensor(self):
        tensor = torch.rand(2, 4, 5, 8)
        transform = self.get_transform(
            channels=('default_image_name',),
            labels=False,
        )
        transformed = transform(tensor)
        assert isinstance(transformed, torch.Tensor)

    def test_transforms_array(self):
        tensor = torch.rand(2, 4, 5, 8).numpy()
        transform = self.get_transform(
            channels=('default_image_name',),
            labels=False,
        )
        transformed = transform(tensor)
        assert isinstance(transformed, np.ndarray)

    def test_transforms_sitk(self):
        tensor = torch.rand(2, 4, 5, 8)
        affine = np.diag((-1, 2, -3, 1))
        image = nib_to_sitk(tensor, affine)
        transform = self.get_transform(
            channels=('default_image_name',),
            labels=False,
        )
        transformed = transform(image)
        assert isinstance(transformed, sitk.Image)

    def test_transforms_subject_3d(self):
        transform = self.get_transform(channels=('t1', 't2'), is_3d=True)
        transformed = transform(self.sample_subject)
        assert isinstance(transformed, tio.Subject)

    def test_transforms_subject_2d(self):
        transform = self.get_transform(channels=('t1', 't2'), is_3d=False)
        subject = self.make_2d(self.sample_subject)
        transformed = transform(subject)
        assert isinstance(transformed, tio.Subject)

    def test_transforms_subject_4d(self):
        composed = self.get_transform(channels=('t1', 't2'), is_3d=True)
        subject = self.make_multichannel(self.sample_subject)
        subject = self.flip_affine_x(subject)
        transformed = None
        for transform in composed.transforms:
            repr(transform)  # cover __repr__
            transformed = transform(subject)
            trsf_channels = len(transformed.t1.data)
            assert trsf_channels > 1, f'Lost channels in {transform.name}'
            exclude = (
                'RandomLabelsToImage',
                'RemapLabels',
                'RemoveLabels',
                'SequentialLabels',
                'CopyAffine',
            )
            if transform.name not in exclude:
                assert subject.shape[0] == transformed.shape[0], (
                    f'Different number of channels after {transform.name}'
                )
                self.assert_tensor_not_equal(
                    subject.t1.data[1],
                    transformed.t1.data[1],
                    msg=f'No changes after {transform.name}',
                )
            subject = transformed
        assert isinstance(transformed, tio.Subject)

    def test_transform_noop(self):
        transform = tio.RandomMotion(p=0)
        transformed = transform(self.sample_subject)
        assert transformed is self.sample_subject
        tensor = torch.rand(2, 4, 5, 8).numpy()
        transformed = transform(tensor)
        assert transformed is tensor

    def test_original_unchanged(self):
        subject = copy.deepcopy(self.sample_subject)
        composed = self.get_transform(channels=('t1', 't2'), is_3d=True)
        subject = self.flip_affine_x(subject)
        for transform in composed.transforms:
            original_data = copy.deepcopy(subject.t1.data)
            transform(subject)
            self.assert_tensor_equal(
                subject.t1.data,
                original_data,
                msg=f'Changes after {transform.name}',
            )

    def test_transforms_use_include(self):
        original_subject = copy.deepcopy(self.sample_subject)
        transform = tio.RandomNoise(include=['t1'])
        transformed = transform(self.sample_subject)

        self.assert_tensor_not_equal(
            original_subject.t1.data,
            transformed.t1.data,
            msg=f'Changes after {transform.name}',
        )

        self.assert_tensor_equal(
            original_subject.t2.data,
            transformed.t2.data,
            msg=f'Changes after {transform.name}',
        )

    def test_transforms_use_exclude(self):
        original_subject = copy.deepcopy(self.sample_subject)
        transform = tio.RandomNoise(exclude=['t2'])
        transformed = transform(self.sample_subject)

        self.assert_tensor_not_equal(
            original_subject.t1.data,
            transformed.t1.data,
            msg=f'Changes after {transform.name}',
        )

        self.assert_tensor_equal(
            original_subject.t2.data,
            transformed.t2.data,
            msg=f'Changes after {transform.name}',
        )

    def test_transforms_use_include_and_exclude(self):
        with pytest.raises(ValueError):
            tio.RandomNoise(include=['t2'], exclude=['t1'])

    def test_keys_deprecated(self):
        with pytest.warns(FutureWarning):
            tio.RandomNoise(keys=['t2'])

    def test_keep_original(self):
        subject = copy.deepcopy(self.sample_subject)
        old, new = 't1', 't1_original'
        transformed = tio.RandomAffine(keep={old: new})(subject)
        assert old in transformed
        assert new in transformed
        self.assert_tensor_equal(
            transformed[new].data,
            subject[old].data,
        )
        self.assert_tensor_not_equal(
            transformed[new].data,
            transformed[old].data,
        )


class TestTransform(TorchioTestCase):
    def test_abstract_transform(self):
        with pytest.raises(TypeError):
            tio.Transform()

    def test_arguments_are_not_dict(self):
        transform = tio.Noise(0, 1, 0)
        assert not transform.arguments_are_dict()

    def test_arguments_are_dict(self):
        transform = tio.Noise({'im': 0}, {'im': 1}, {'im': 0})
        assert transform.arguments_are_dict()

    def test_arguments_are_and_are_not_dict(self):
        transform = tio.Noise(0, {'im': 1}, {'im': 0})
        with pytest.raises(ValueError):
            transform.arguments_are_dict()

    def test_min_constraint(self):
        transform = tio.RandomNoise()
        assert transform._parse_range(3, 'name', min_constraint=0) == (0, 3)

    def test_bad_over_max(self):
        transform = tio.RandomNoise()
        with pytest.raises(ValueError):
            transform._parse_range(2, 'name', max_constraint=1)

    def test_bad_over_max_range(self):
        transform = tio.RandomNoise()
        with pytest.raises(ValueError):
            transform._parse_range((0, 2), 'name', max_constraint=1)

    def test_bad_type(self):
        transform = tio.RandomNoise()
        with pytest.raises(ValueError):
            transform._parse_range(2.5, 'name', type_constraint=int)

    def test_no_numbers(self):
        transform = tio.RandomNoise()
        with pytest.raises(ValueError):
            _parse_range_with_invalid_value(transform, 'j')

    def test_apply_transform_missing(self):
        class T(tio.Transform):
            pass

        with pytest.raises(TypeError):
            T()

    def test_non_invertible(self):
        transform = tio.RandomBlur()
        with pytest.raises(RuntimeError):
            transform.inverse()

    def test_batch_history(self):
        # https://github.com/TorchIO-project/torchio/discussions/743
        subject = self.sample_subject
        transform = tio.Compose(
            [
                tio.RandomAffine(),
                tio.CropOrPad(5),
                tio.OneHot(),
            ]
        )
        dataset = tio.SubjectsDataset([subject], transform=transform)
        loader = tio.SubjectsLoader(
            dataset,
            collate_fn=tio.utils.history_collate,
        )
        batch = tio.utils.get_first_item(loader)
        transformed: tio.Subject = tio.utils.get_subjects_from_batch(batch)[0]
        inverse = transformed.apply_inverse_transform()
        images1 = subject.get_images(intensity_only=False)
        images2 = inverse.get_images(intensity_only=False)
        for image1, image2 in zip(images1, images2, strict=True):
            assert image1.shape == image2.shape

    def test_bad_bounds_mask(self):
        transform = tio.ZNormalization(masking_method='test')
        with pytest.raises(ValueError):
            transform(self.sample_subject)

    def test_bounds_mask(self):
        transform = tio.ZNormalization()
        tensor = torch.rand((1, 2, 2, 2))
        with pytest.raises(ValueError):
            transform.get_mask_from_anatomical_label('test', tensor)

        def get_mask(label):
            mask = transform.get_mask_from_anatomical_label(label, tensor)
            return mask

        left = get_mask('Left')
        assert left[:, 0].sum() == 4 and left[:, 1].sum() == 0
        right = get_mask('Right')
        assert right[:, 1].sum() == 4 and right[:, 0].sum() == 0
        posterior = get_mask('Posterior')
        assert posterior[:, :, 0].sum() == 4 and posterior[:, :, 1].sum() == 0
        anterior = get_mask('Anterior')
        assert anterior[:, :, 1].sum() == 4 and anterior[:, :, 0].sum() == 0
        inferior = get_mask('Inferior')
        assert inferior[..., 0].sum() == 4 and inferior[..., 1].sum() == 0
        superior = get_mask('Superior')
        assert superior[..., 1].sum() == 4 and superior[..., 0].sum() == 0

        mask = transform.get_mask_from_bounds(3 * (0, 1), tensor)
        assert mask[0, 0, 0, 0] == 1
        assert mask.sum() == 1

    def test_label_keys(self):
        # Adapted from the issue in which the feature was requested:
        # https://github.com/TorchIO-project/torchio/issues/866#issue-1222255576
        size = 1, 10, 10, 10
        image = torch.rand(size)
        num_classes = 2  # excluding background
        label = torch.randint(num_classes + 1, size)

        data_dict: dict[str, object] = {'image': image, 'label': label}

        transform = tio.RandomAffine(
            include=['image', 'label'],
            label_keys=['label'],
        )
        transformed_dict = transform(data_dict)
        transformed_label = transformed_dict['label']
        assert isinstance(transformed_label, torch.Tensor)

        # If the image is indeed transformed as a label map, nearest neighbor
        # interpolation is used by default and therefore no intermediate values
        # can exist in the output
        num_unique_values = len(torch.unique(transformed_label))
        assert num_unique_values <= num_classes + 1

    def test_nibabel_input(self):
        image = self.sample_subject.t1
        image_nib = Nifti1Image(image.data[0].numpy(), image.affine)
        transformed = tio.RandomAffine()(image_nib)
        transformed.get_fdata()
        _ = transformed.affine

        image = self.subject_4d.t1
        tensor_5d = image.data[np.newaxis].permute(2, 3, 4, 0, 1)
        image_nib = Nifti1Image(tensor_5d.numpy(), image.affine)
        transformed = tio.RandomAffine()(image_nib)
        transformed.get_fdata()
        _ = transformed.affine

    def test_bad_shape(self):
        tensor = torch.rand(1, 2, 3)
        with pytest.raises(ValueError, match='must be a 4D tensor'):
            tio.RandomAffine()(tensor)

    def test_bad_keys_type(self):
        # From https://github.com/TorchIO-project/torchio/issues/923
        with self.assertRaises(ValueError):
            tio.RandomAffine(include='t1')

    def test_init_args(self):
        transform = tio.Compose([tio.RandomNoise()])
        base_args = transform._get_base_args()
        assert 'parse_input' not in base_args

        transform = tio.OneOf([tio.RandomNoise()])
        base_args = transform._get_base_args()
        assert 'parse_input' not in base_args

        transform = tio.RandomNoise()
        base_args = transform._get_base_args()
        assert all(
            arg in base_args
            for arg in [
                'copy',
                'include',
                'exclude',
                'keep',
                'parse_input',
                'label_keys',
            ]
        )

    def test_repr_inverted_transform(self):
        """Repr of an inverted transform should include 'invert=True'."""
        transform = tio.OneHot()
        inverted = transform.inverse()
        repr_str = repr(inverted)
        assert 'invert=True' in repr_str

    def test_repr_without_args_names(self):
        """Repr falls back to super().__repr__ when args_names is absent."""
        transform = tio.RandomFlip()
        delattr(transform, 'args_names')
        repr_str = repr(transform)
        assert 'RandomFlip' in repr_str

    def test_add_base_args_no_overwrite(self):
        """Existing keys should not be overwritten by default."""
        transform = tio.RandomFlip()
        args: dict[str, object] = {'copy': True, 'custom_key': 42}
        result = transform._add_base_args(args)
        assert result['copy'] is True
        assert result['custom_key'] == 42

    def test_add_base_args_with_overwrite(self):
        """With overwrite_on_existing=True, existing keys are replaced."""
        transform = tio.RandomFlip()
        args: dict[str, object] = {'copy': 'original'}
        result = transform._add_base_args(args, overwrite_on_existing=True)
        assert result['copy'] == transform.copy

    def test_parse_params_numpy_array(self):
        """Parsing a numpy array parameter supports np.ravel conversion."""
        transform = tio.RandomAffine()
        result = transform.parse_params(
            np.array([5.0]),
            around=0,
            name='test',
        )
        assert len(result) == 6

    def test_parse_params_bad_length_raises(self):
        """Sequence of length 4 should raise ValueError."""
        transform = tio.RandomAffine()
        with pytest.raises(ValueError, match='length 2, 3 or 6'):
            transform.parse_params(
                (1.0, 2.0, 3.0, 4.0),
                around=0,
                name='scales',
            )

    def test_parse_range_non_iterable_raises(self):
        """A non-number, non-iterable input should raise ValueError."""
        transform = tio.RandomAffine()
        with pytest.raises(ValueError, match='sequence of len 2'):
            transform._parse_range(cast(Any, object()), 'test_param')

    def test_parse_range_non_number_values_raises(self):
        """Sequence with non-numeric values should raise ValueError."""
        transform = tio.RandomAffine()
        with pytest.raises(ValueError, match='values must be numbers'):
            transform._parse_range(cast(Any, ('a', 'b')), 'test_param')

    def test_non_iterable_include_raises(self):
        """Passing a non-iterable as include should raise ValueError."""
        with pytest.raises(ValueError, match='must be a sequence of strings'):
            tio.Transform.parse_include_and_exclude_keys(
                include=cast(Any, 42),
                exclude=None,
                label_keys=None,
            )

    def test_parse_bounds_none(self):
        """None should return None."""
        assert tio.Transform.parse_bounds(None) is None

    def test_parse_bounds_negative_raises(self):
        """Negative bounds should raise ValueError."""
        with pytest.raises(ValueError, match='integers greater or equal to zero'):
            tio.Transform.parse_bounds(-1)

    def test_parse_bounds_float_raises(self):
        """Float bounds should raise TypeError (not iterable)."""
        with pytest.raises(TypeError):
            tio.Transform.parse_bounds(cast(Any, 1.5))

    def test_parse_bounds_bad_length_raises(self):
        """Bounds of length 2 or 4 should raise ValueError."""
        with pytest.raises(ValueError, match='3 or 6 integers'):
            tio.Transform.parse_bounds((1, 2))
        with pytest.raises(ValueError, match='3 or 6 integers'):
            tio.Transform.parse_bounds((1, 2, 3, 4))

    def test_masking_method_invalid_type_raises(self):
        """An invalid masking_method type should raise ValueError."""
        transform = tio.RandomNoise()
        tensor = torch.randn(1, 10, 10, 10)
        subject = self.sample_subject
        with pytest.raises(ValueError, match='Masking method must be one of'):
            transform.get_mask_from_masking_method(
                cast(Any, 3.14),
                subject,
                tensor,
            )

    def test_masking_method_int_bounds(self):
        """Integer masking_method should create a bounds-based mask."""
        transform = tio.RandomNoise()
        tensor = torch.randn(1, 10, 20, 30)
        mask = transform.get_mask_from_masking_method(
            2,
            self.sample_subject,
            tensor,
        )
        assert mask.shape == tensor.shape
        assert mask.dtype == torch.bool

    def test_masking_method_tuple_bounds(self):
        """Tuple masking_method should create a bounds-based mask."""
        transform = tio.RandomNoise()
        tensor = torch.randn(1, 10, 20, 30)
        mask = transform.get_mask_from_masking_method(
            (1, 1, 1),
            self.sample_subject,
            tensor,
        )
        assert mask.shape == tensor.shape
        assert mask.dtype == torch.bool

    def test_get_name_with_module(self):
        """_get_name_with_module returns 'module.ClassName' string."""
        transform = tio.RandomFlip()
        name = transform._get_name_with_module()
        assert 'RandomFlip' in name
        assert '.' in name

    def test_to_hydra_config(self):
        """to_hydra_config returns a dict with _target_ key."""
        transform = tio.RandomFlip(axes=(0, 1))
        config = transform.to_hydra_config()
        assert '_target_' in config
        assert 'RandomFlip' in config['_target_']

    def test_tuples_to_lists(self):
        """_tuples_to_lists recursively converts tuples inside dicts/lists."""
        data: dict[str, object] = {
            'a': (1, 2, 3),
            'b': [(4, 5), 6],
            'c': 'keep',
        }
        result = tio.Transform._tuples_to_lists(data)
        assert isinstance(result['a'], list)
        assert isinstance(result['b'], list)
        assert isinstance(result['b'][0], list)
        assert result['c'] == 'keep'


class TestFourierTransformFallback(TorchioTestCase):
    """Tests for the NumPy FFT fallback in FourierTransform."""

    def test_fourier_transform_numpy_fallback(self):
        """Ensure forward FFT works via NumPy when torch.fft raises."""
        tensor = torch.randn(4, 4, 4)
        with patch('torch.fft.fftn', side_effect=ModuleNotFoundError):
            result = FourierTransform.fourier_transform(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == tensor.shape

    def test_inv_fourier_transform_numpy_fallback(self):
        """Ensure inverse FFT works via NumPy when torch.fft raises."""
        tensor = torch.randn(4, 4, 4).to(torch.complex64)
        with patch('torch.fft.ifftshift', side_effect=ModuleNotFoundError):
            result = FourierTransform.inv_fourier_transform(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == tensor.shape

    def test_fourier_roundtrip(self):
        """Verify that forward then inverse FFT approximately recovers input."""
        tensor = torch.randn(4, 4, 4)
        freq = FourierTransform.fourier_transform(tensor)
        recovered = FourierTransform.inv_fourier_transform(freq)
        torch.testing.assert_close(
            recovered.real.float(),
            tensor,
            atol=1e-5,
            rtol=1e-5,
        )


class TestGetSitkInterpolator(TorchioTestCase):
    """Tests for interpolation type validation."""

    def test_non_string_interpolation_raises(self):
        """Passing a non-string to get_sitk_interpolator raises ValueError."""
        with pytest.raises(ValueError, match='Interpolation must be a string'):
            get_sitk_interpolator(cast(Any, 42))

    def test_non_string_list_interpolation_raises(self):
        """Passing a list to get_sitk_interpolator raises ValueError."""
        with pytest.raises(ValueError, match='Interpolation must be a string'):
            get_sitk_interpolator(cast(Any, ['linear']))


class TestIntensityTransform(TorchioTestCase):
    """Tests for IntensityTransform.get_parameter static method."""

    def test_get_parameter_from_dict(self):
        """Retrieve a per-image parameter from a mapping."""
        result = tio.transforms.IntensityTransform.get_parameter(
            {'image': 0.5},
            'image',
        )
        assert result == 0.5

    def test_get_parameter_scalar_passthrough(self):
        """A non-mapping value should be returned directly."""
        result = tio.transforms.IntensityTransform.get_parameter(0.7, 'image')
        assert result == 0.7


class TestDataParser(TorchioTestCase):
    """Tests for DataParser edge cases."""

    def test_unrecognized_input_type_raises(self):
        """Passing an unsupported type should raise ValueError."""
        parser = DataParser(data=cast(Any, 12345))
        with pytest.raises(ValueError, match='Input type not recognized'):
            parser.get_subject()

    def test_dict_with_non_tensor_image_value_raises(self):
        """Dict values selected as images must be tensors or arrays."""
        parser = DataParser(
            data={'image': 'not_a_tensor', 'other': 42},
            keys=['image'],
        )
        with pytest.raises(TypeError, match='tensors or arrays'):
            parser.get_subject()

    def test_dict_with_valid_tensor_value(self):
        """Dict with tensor values should parse correctly."""
        tensor = torch.randn(1, 10, 10, 10)
        parser = DataParser(
            data={'image': tensor, 'metadata': 'info'},
            keys=['image'],
        )
        subject = parser.get_subject()
        assert 'image' in subject
        assert 'metadata' in subject

    def test_dict_with_label_keys(self):
        """Dict with label_keys should create LabelMap images."""
        tensor = torch.randint(0, 3, (1, 10, 10, 10)).float()
        parser = DataParser(
            data={'seg': tensor},
            keys=['seg'],
            label_keys=['seg'],
        )
        subject = parser.get_subject()
        assert isinstance(subject['seg'], tio.LabelMap)


class TestComposeCoverage(TorchioTestCase):
    """Tests for Compose edge cases and utility methods."""

    def test_non_callable_raises(self):
        """Non-callable objects in Compose should raise TypeError."""
        with pytest.raises(TypeError, match='not callable'):
            tio.Compose([cast(Any, 'not_a_transform')])

    def test_len(self):
        """__len__ returns the number of transforms."""
        compose = tio.Compose([tio.RandomFlip(), tio.RandomNoise()])
        assert len(compose) == 2

    def test_getitem(self):
        """__getitem__ returns the transform at the given index."""
        flip = tio.RandomFlip()
        noise = tio.RandomNoise()
        compose = tio.Compose([flip, noise])
        assert compose[0] is flip
        assert compose[1] is noise

    def test_repr(self):
        """__repr__ contains the class name and transforms."""
        compose = tio.Compose([tio.RandomFlip()])
        repr_str = repr(compose)
        assert 'Compose' in repr_str

    def test_is_invertible_all(self):
        """is_invertible returns True when all transforms are invertible."""
        compose = tio.Compose([tio.OneHot()])
        assert compose.is_invertible()

    def test_inverse_no_invertible_warns(self):
        """Compose.inverse() warns when no transforms are invertible."""
        compose = tio.Compose([])
        with pytest.warns(RuntimeWarning, match='No invertible transforms'):
            compose.inverse()

    def test_inverse_skips_non_invertible_warns(self):
        """Compose.inverse() warns about non-invertible transforms."""
        compose = tio.Compose([tio.OneHot(), tio.RemoveLabels([1])])
        with pytest.warns(RuntimeWarning, match='Skipping'):
            inverted = compose.inverse()
        assert len(inverted) == 1

    def test_compose_to_hydra_config(self):
        """to_hydra_config returns a nested Hydra configuration dict."""
        compose = tio.Compose([tio.RandomFlip(), tio.RandomNoise()])
        config = compose.to_hydra_config()
        assert '_target_' in config
        assert 'transforms' in config
        assert len(config['transforms']) == 2
        assert all('_target_' in t for t in config['transforms'])
