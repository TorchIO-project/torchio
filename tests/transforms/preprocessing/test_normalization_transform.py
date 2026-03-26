import pytest

from torchio.transforms.preprocessing.intensity.normalization_transform import (
    NormalizationTransform,
)

from ...utils import TorchioTestCase


class TestNormalizationTransform(TorchioTestCase):
    """Tests for the abstract NormalizationTransform base class."""

    def test_apply_normalization_not_implemented(self):
        """Calling apply_transform on the base class raises NotImplementedError."""
        transform = NormalizationTransform()
        with pytest.raises(NotImplementedError):
            transform(self.sample_subject)
