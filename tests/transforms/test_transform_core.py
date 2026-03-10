import copy

import numpy as np
import pytest
import SimpleITK as sitk
import torch

import torchio as tio

from ..utils import TorchioTestCase


class AddValueTransform(tio.Transform):
    def __init__(self, value: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.invert_transform = False
        self.args_names = ['value']

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        delta = -self.value if self.invert_transform else self.value
        for image in subject.get_images(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        ):
            image.set_data(image.data + delta)
        return subject


class TupleArgsTransform(tio.Transform):
    def __init__(
        self,
        pairs: tuple[tuple[int, int], tuple[int, int]] = ((1, 2), (3, 4)),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pairs = pairs
        self.args_names = ['pairs']

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        return subject


class TestTransformCore(TorchioTestCase):
    def test_parse_params_accepts_arrays_and_tensors(self):
        transform = AddValueTransform()

        assert transform.parse_params(torch.tensor([[1, 2, 3]]), None, 'scales') == (
            0,
            1,
            0,
            2,
            0,
            3,
        )
        assert transform.parse_params(np.array([1, 2]), None, 'degrees') == (
            1,
            2,
            1,
            2,
            1,
            2,
        )
        assert transform.parse_params(np.array([1, 2, 3]), 10, 'translation') == (
            9,
            11,
            8,
            12,
            7,
            13,
        )

    def test_parse_params_without_ranges_preserves_flat_values(self):
        transform = AddValueTransform()

        params = transform.parse_params(
            torch.tensor([1, 2, 3, 4]),
            None,
            'degrees',
            make_ranges=False,
        )

        assert params == (1, 2, 3, 4)

    def test_parse_params_invalid_length_raises(self):
        transform = AddValueTransform()

        with pytest.raises(ValueError, match='length 2, 3 or 6, not 4'):
            transform.parse_params((1, 2, 3, 4), None, 'degrees')

    def test_parse_range_constraints_and_type_validation(self):
        transform = AddValueTransform()

        assert transform._parse_range(3, 'margin') == (-3, 3)
        assert transform._parse_range(
            (1, 3),
            'margin',
            min_constraint=0,
            max_constraint=5,
            type_constraint=int,
        ) == (1, 3)

        with pytest.raises(ValueError, match='equal or greater than the first'):
            transform._parse_range((3, 1), 'margin')
        with pytest.raises(ValueError, match='first value must be greater than 0'):
            transform._parse_range((-1, 1), 'margin', min_constraint=0)
        with pytest.raises(ValueError, match='must be of type'):
            transform._parse_range((1, 1.5), 'margin', type_constraint=int)

    def test_to_range_and_parse_probability(self):
        assert AddValueTransform.to_range(5, None) == (0.0, 5.0)
        assert AddValueTransform.to_range(5, 10) == (5.0, 15.0)
        assert AddValueTransform.parse_probability(0) == 0
        assert AddValueTransform.parse_probability(1) == 1

        with pytest.raises(
            ValueError, match='Probability must be a number in \\[0, 1\\]'
        ):
            AddValueTransform.parse_probability(-0.1)
        with pytest.raises(
            ValueError, match='Probability must be a number in \\[0, 1\\]'
        ):
            AddValueTransform.parse_probability('0.5')  # type: ignore[arg-type]

    def test_parse_interpolation_and_sitk_interpolator(self):
        transform = AddValueTransform()

        assert transform.parse_interpolation('Linear') == 'linear'
        assert transform.get_sitk_interpolator('linear') == sitk.sitkLinear

        with pytest.raises(TypeError, match='Interpolation must be a string'):
            transform.parse_interpolation(1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match='supported values'):
            transform.parse_interpolation('splinez')

    def test_parse_bounds_accepts_expected_formats(self):
        transform = AddValueTransform()

        assert transform.parse_bounds(None) is None
        assert transform.parse_bounds(2) == (2, 2, 2, 2, 2, 2)
        assert transform.parse_bounds((1, 2, 3)) == (1, 1, 2, 2, 3, 3)
        assert transform.parse_bounds([np.int64(1), 2, 3, 4, 5, 6]) == (
            1,
            2,
            3,
            4,
            5,
            6,
        )

        with pytest.raises(ValueError, match='greater or equal to zero'):
            transform.parse_bounds((1, -1, 0))
        with pytest.raises(ValueError, match='3 or 6 integers'):
            transform.parse_bounds((1, 2))
        with pytest.raises(ValueError, match='greater or equal to zero'):
            transform.parse_bounds((1, 2, 3.5))

    def test_mask_helpers_cover_none_callable_labels_and_bounds(self):
        transform = AddValueTransform()
        tensor = torch.arange(64, dtype=torch.float32).reshape(1, 4, 4, 4)
        label_tensor = (tensor.long() % 4).to(torch.uint8)
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=tensor),
            label=tio.LabelMap(tensor=label_tensor),
        )

        ones_mask = transform.get_mask_from_masking_method(None, subject, tensor)
        assert ones_mask.dtype == torch.bool
        assert torch.all(ones_mask)

        callable_mask = transform.get_mask_from_masking_method(
            lambda data: data > 10,
            subject,
            tensor,
        )
        self.assert_tensor_equal(callable_mask, tensor > 10)

        label_mask = transform.get_mask_from_masking_method('label', subject, tensor)
        self.assert_tensor_equal(label_mask, label_tensor.bool())

        selected_mask = transform.get_mask_from_masking_method(
            'label',
            subject,
            tensor,
            labels=[2, 3],
        )
        self.assert_tensor_equal(
            selected_mask,
            (label_tensor == 2) | (label_tensor == 3),
        )

        anatomical_mask = transform.get_mask_from_masking_method(
            'left',
            subject,
            tensor,
        )
        self.assert_tensor_equal(
            anatomical_mask,
            transform.get_mask_from_anatomical_label('Left', tensor),
        )

        int_bounds_mask = transform.get_mask_from_masking_method(1, subject, tensor)
        self.assert_tensor_equal(
            int_bounds_mask, transform.get_mask_from_bounds(1, tensor)
        )

        tuple_bounds_mask = transform.get_mask_from_masking_method(
            (1, 0, 1),
            subject,
            tensor,
        )
        self.assert_tensor_equal(
            tuple_bounds_mask,
            transform.get_mask_from_bounds((1, 0, 1), tensor),
        )

        with pytest.raises(ValueError, match='Masking method must be one of'):
            transform.get_mask_from_masking_method(object(), subject, tensor)

    def test_mean_and_ones_helpers_return_boolean_masks(self):
        tensor = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]])

        self.assert_tensor_equal(
            AddValueTransform.ones(tensor),
            torch.ones_like(tensor, dtype=torch.bool),
        )
        self.assert_tensor_equal(
            AddValueTransform.mean(tensor),
            tensor > tensor.float().mean(),
        )

    def test_copy_false_include_and_keep_preserve_original(self):
        subject = copy.deepcopy(self.sample_subject)
        original_t1 = subject.t1.data.clone()
        original_t2 = subject.t2.data.clone()

        transform = AddValueTransform(
            copy=False,
            include=['t1'],
            keep={'t1': 't1_original'},
        )
        transformed = transform(subject)

        assert transformed is subject
        self.assert_tensor_equal(transformed.t1.data, original_t1 + 1)
        self.assert_tensor_equal(transformed.t2.data, original_t2)
        self.assert_tensor_equal(transformed.t1_original.data, original_t1)

    def test_parse_input_false_returns_subject_without_history(self):
        subject = copy.deepcopy(self.sample_subject)
        original_t1 = subject.t1.data.clone()
        original_label = subject.label.data.clone()

        transform = AddValueTransform(
            copy=False,
            parse_input=False,
            exclude=['label'],
        )
        transformed = transform(subject)

        assert transformed is subject
        self.assert_tensor_equal(transformed.t1.data, original_t1 + 1)
        self.assert_tensor_equal(transformed.label.data, original_label)
        assert transformed.applied_transforms == []

    def test_use_seed_restores_global_rng_state(self):
        torch.manual_seed(42)
        expected_first = torch.rand(3)
        expected_second = torch.rand(3)

        torch.manual_seed(42)
        first = torch.rand(3)
        with AddValueTransform._use_seed(7):
            inside = torch.rand(3)
        second = torch.rand(3)

        torch.manual_seed(7)
        expected_inside = torch.rand(3)

        self.assert_tensor_equal(first, expected_first)
        self.assert_tensor_equal(inside, expected_inside)
        self.assert_tensor_equal(second, expected_second)

    def test_validate_keys_sequence_and_add_base_args(self):
        with pytest.raises(ValueError, match='"exclude" must be a sequence of strings'):
            AddValueTransform(exclude=3)
        with pytest.raises(
            ValueError, match='"label_keys" must be a sequence of strings'
        ):
            AddValueTransform(label_keys='label')

        transform = AddValueTransform(
            copy=False,
            include=['t1'],
            keep={'t1': 't1_original'},
        )
        arguments = {'copy': True, 'custom': 'value'}

        result = transform._add_base_args(arguments.copy())
        assert result['copy'] is True
        assert result['custom'] == 'value'
        assert result['include'] == ['t1']

        overwritten = transform._add_base_args(
            arguments.copy(),
            overwrite_on_existing=True,
        )
        assert overwritten['copy'] is False

    def test_inverse_repr_and_hydra_config_cover_helpers(self):
        transform = AddValueTransform(value=2)
        inverse = transform.inverse()

        assert inverse is not transform
        assert inverse.invert_transform is True
        assert transform.invert_transform is False
        assert 'invert=True' in repr(inverse)

        transformed = transform(self.sample_subject)
        restored = inverse(transformed)
        self.assert_tensor_equal(restored.t1.data, self.sample_subject.t1.data)

        config = TupleArgsTransform(
            include=('t1', 't2'),
            pairs=((1, 2), (3, 4)),
        ).to_hydra_config()
        assert config['_target_'].endswith('.TupleArgsTransform')
        assert config['include'] == ['t1', 't2']
        assert config['pairs'] == [[1, 2], [3, 4]]
