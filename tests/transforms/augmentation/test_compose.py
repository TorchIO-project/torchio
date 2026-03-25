from typing import Any
from typing import cast

import pytest

import torchio as tio

from ...utils import TorchioTestCase


class TestCompose(TorchioTestCase):
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
