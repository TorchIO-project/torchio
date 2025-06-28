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
            tio.RandomAffine(scales='wrong')

    def test_wrong_degrees_type(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(degrees='wrong')

    def test_too_many_translation_values(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(translation=(-10, 4, 42))

    def test_wrong_translation_type(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(translation='wrong')

    def test_wrong_center(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(center=0)

    def test_wrong_default_pad_value(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(default_pad_value='wrong')

    def test_wrong_image_interpolation_type(self):
        with pytest.raises(TypeError):
            tio.RandomAffine(image_interpolation=0)

    def test_wrong_image_interpolation_value(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(image_interpolation='wrong')

    def test_incompatible_args_isotropic(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(scales=(0.8, 0.5, 0.1), isotropic=True)

    def test_parse_scales(self):
        def do_assert(transform):
            assert transform.scales == 3 * (0.9, 1.1)

        do_assert(tio.RandomAffine(scales=0.1))
        do_assert(tio.RandomAffine(scales=(0.9, 1.1)))
        do_assert(tio.RandomAffine(scales=3 * (0.1,)))
        do_assert(tio.RandomAffine(scales=3 * [0.9, 1.1]))

    def test_parse_degrees(self):
        def do_assert(transform):
            assert transform.degrees == 3 * (-10, 10)

        do_assert(tio.RandomAffine(degrees=10))
        do_assert(tio.RandomAffine(degrees=(-10, 10)))
        do_assert(tio.RandomAffine(degrees=3 * (10,)))
        do_assert(tio.RandomAffine(degrees=3 * [-10, 10]))

    def test_parse_translation(self):
        def do_assert(transform):
            assert transform.translation == 3 * (-10, 10)

        do_assert(tio.RandomAffine(translation=10))
        do_assert(tio.RandomAffine(translation=(-10, 10)))
        do_assert(tio.RandomAffine(translation=3 * (10,)))
        do_assert(tio.RandomAffine(translation=3 * [-10, 10]))

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
        has_expected_value = (transformed_subject['label'].tensor == 250).any()
        assert has_expected_value, message

        # Test 2: backward compatibility - default_pad_value should still be ignored for labels
        message = 'default_pad_value should still be ignored for LABEL images (backward compatibility)'
        aff_old = tio.RandomAffine(
            translation=(-10, 10, -10, 10, -10, 10),
            default_pad_value=250,  # This should be ignored for labels
        )
        s_aug_old = aff_old.apply_transform(subject)

        # Should still use 0 (default for labels), not the default_pad_value
        non_one_values = s_aug_old['label'].data[s_aug_old['label'].data != 1]
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
        has_affine_value = (s_affine['label'].tensor == 123).any()
        assert has_affine_value, 'Direct Affine class should respect default_pad_label'

    def test_wrong_default_pad_label(self):
        with pytest.raises(ValueError):
            tio.RandomAffine(default_pad_label='minimum')

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
