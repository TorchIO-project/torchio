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

    # --- dict support ---

    def test_dict_input(self):
        """Compose accepts a dict mapping names to transforms."""
        flip = tio.RandomFlip()
        noise = tio.RandomNoise()
        compose = tio.Compose({'flip': flip, 'noise': noise})
        assert len(compose) == 2
        assert compose.transforms == [flip, noise]

    def test_dict_getitem_by_name(self):
        """Transforms can be accessed by name when created from a dict."""
        flip = tio.RandomFlip()
        noise = tio.RandomNoise()
        compose = tio.Compose({'flip': flip, 'noise': noise})
        assert compose['flip'] is flip
        assert compose['noise'] is noise

    def test_dict_getitem_by_index(self):
        """Transforms can still be accessed by index when created from a dict."""
        flip = tio.RandomFlip()
        noise = tio.RandomNoise()
        compose = tio.Compose({'flip': flip, 'noise': noise})
        assert compose[0] is flip
        assert compose[1] is noise

    def test_dict_getitem_invalid_name_raises(self):
        """Accessing a non-existent name raises KeyError."""
        compose = tio.Compose({'flip': tio.RandomFlip()})
        with pytest.raises(KeyError):
            compose['nonexistent']

    def test_list_getitem_by_name_raises(self):
        """Accessing by name when created from a list raises TypeError."""
        compose = tio.Compose([tio.RandomFlip()])
        with pytest.raises(TypeError, match='String indexing is not supported'):
            compose['flip']

    def test_dict_apply(self):
        """Compose from a dict applies transforms in order."""
        compose = tio.Compose({'flip': tio.RandomFlip(), 'noise': tio.RandomNoise()})
        subject = self.sample_subject
        transformed = compose(subject)
        assert isinstance(transformed, tio.Subject)

    def test_dict_non_callable_raises(self):
        """Non-callable values in a dict raise TypeError."""
        with pytest.raises(TypeError, match='not callable'):
            tio.Compose({'bad': cast(Any, 'not_a_transform')})

    def test_dict_non_string_key_raises(self):
        """Non-string keys in a dict raise TypeError."""
        with pytest.raises(TypeError, match='All keys.*must be strings'):
            tio.Compose({1: tio.RandomFlip()})
