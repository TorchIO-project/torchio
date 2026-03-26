from typing import Any
from typing import cast

import pytest

from torchio.transforms.interpolation import get_sitk_interpolator

from ..utils import TorchioTestCase


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
