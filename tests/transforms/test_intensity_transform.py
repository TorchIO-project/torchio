import torchio as tio

from ..utils import TorchioTestCase


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
