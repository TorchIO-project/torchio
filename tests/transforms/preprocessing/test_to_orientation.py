import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestToOrientation(TorchioTestCase):
    def test_invalid_orientation_length(self):
        with pytest.raises(ValueError, match='3-letter'):
            tio.ToOrientation('RA')  # Too short

    def test_invalid_orientation_characters(self):
        with pytest.raises(ValueError, match='three distinct characters'):
            tio.ToOrientation('XYZ')

    def test_missing_axis_direction(self):
        match = 'must include one character for each axis'
        with pytest.raises(ValueError, match=match):
            tio.ToOrientation('RAA')  # no S/I direction

    def test_no_change_if_already_correct(self):
        transform = tio.ToOrientation('RAS')
        subject = transform(self.sample_subject)
        self.assert_tensor_equal(subject.t1.data, self.sample_subject.t1.data)
        self.assert_tensor_equal(subject.t1.affine, self.sample_subject.t1.affine)

    def test_ras_to_las(self):
        # Step 1: Set initial orientation to RAS (default)
        ras_subject = self.sample_subject

        # Step 2: RAS -> LAS
        to_las = tio.ToOrientation('LAS')
        las_subject = to_las(ras_subject)

        self.assertEqual(las_subject.t1.orientation, ('L', 'A', 'S'))

        # Manually compute expected LAS affine
        expected_affine = ras_subject.t1.affine.copy()
        expected_affine[0, 0] = -ras_subject.t1.affine[0, 0]
        expected_affine[0, 3] = (
            ras_subject.t1.affine[0, 0] * (ras_subject.t1.spatial_shape[0] - 1)
            + ras_subject.t1.affine[0, 3]
        )

        # Check transformation validity
        flipped_data = torch.flip(ras_subject.t1.data, dims=[1])
        self.assert_tensor_almost_equal(
            las_subject.t1.data,
            flipped_data,
            check_stride=False,
        )
        self.assert_tensor_almost_equal(
            las_subject.t1.affine,
            expected_affine,
        )

    def test_ras_to_las_to_ras(self):
        # Step 1: Start with RAS orientation
        original_subject = self.sample_subject
        original_data = original_subject.t1.data.clone()
        original_affine = original_subject.t1.affine.copy()

        # Step 2: RAS -> LAS
        to_las = tio.ToOrientation('LAS')
        las_subject = to_las(original_subject)
        self.assertEqual(las_subject.t1.orientation, ('L', 'A', 'S'))

        # Step 3: LAS -> RAS
        to_ras = tio.ToOrientation('RAS')
        recovered_subject = to_ras(las_subject)
        self.assertEqual(recovered_subject.t1.orientation, ('R', 'A', 'S'))

        # Step 4: Check if data and affine are restored
        self.assert_tensor_almost_equal(
            recovered_subject.t1.data,
            original_data,
            check_stride=False,
        )
        self.assert_tensor_almost_equal(
            recovered_subject.t1.affine,
            original_affine,
        )
