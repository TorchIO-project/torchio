import pytest

import torchio as tio


class InvertibleIdentity(tio.Transform):
    def __init__(self, value):
        super().__init__(parse_input=False)
        self.value = value
        self.invert_transform = False
        self.args_names = ['value']

    def apply_transform(self, subject):
        return subject


class NonInvertibleIdentity(tio.Transform):
    def __init__(self, value):
        super().__init__(parse_input=False)
        self.value = value
        self.args_names = ['value']

    def apply_transform(self, subject):
        return subject


class AppendValue(tio.Transform):
    def __init__(self, value):
        super().__init__(parse_input=False)
        self.value = value
        self.args_names = ['value']

    def apply_transform(self, subject):
        return [*subject, self.value]


def test_compose_rejects_non_callable():
    with pytest.raises(TypeError, match='not callable'):
        tio.Compose([tio.RandomNoise(), 1])


def test_compose_list_interface():
    first = tio.RandomNoise()
    second = tio.RandomBlur()
    transform = tio.Compose([first, second])

    assert len(transform) == 2
    assert transform[0] is first
    assert transform[-1] is second


def test_compose_repr_includes_nested_transforms():
    transform = tio.Compose([tio.RandomNoise(), tio.RandomBlur()])

    representation = repr(transform)

    assert representation.startswith('Compose([')
    assert 'RandomNoise' in representation
    assert 'RandomBlur' in representation


def test_compose_applies_transforms_in_order():
    transform = tio.Compose([AppendValue('first'), AppendValue('second')])

    transformed = transform([])

    assert transformed == ['first', 'second']


def test_compose_is_invertible():
    invertible = tio.Compose(
        [InvertibleIdentity('first'), InvertibleIdentity('second')]
    )
    mixed = tio.Compose([InvertibleIdentity('first'), NonInvertibleIdentity('second')])

    assert invertible.is_invertible()
    assert not mixed.is_invertible()


def test_compose_inverse_skips_non_invertible_transforms():
    transform = tio.Compose(
        [
            InvertibleIdentity('first'),
            NonInvertibleIdentity('skip'),
            InvertibleIdentity('third'),
        ],
        copy=False,
    )

    with pytest.warns(RuntimeWarning) as recorded:
        inverse = transform.inverse()

    assert [str(w.message) for w in recorded] == [
        'Skipping NonInvertibleIdentity as it is not invertible',
    ]
    assert [t.value for t in inverse.transforms] == ['third', 'first']
    assert all(t.invert_transform for t in inverse.transforms)
    assert inverse.copy is False


def test_compose_inverse_warns_when_no_invertible_transforms():
    transform = tio.Compose([NonInvertibleIdentity('only')])

    with pytest.warns(RuntimeWarning) as recorded:
        inverse = transform.inverse()

    assert [str(w.message) for w in recorded] == [
        'Skipping NonInvertibleIdentity as it is not invertible',
        'No invertible transforms found',
    ]
    assert inverse.transforms == []


def test_compose_to_hydra_config():
    transform = tio.Compose([tio.Flip(axes=(0, 2))], copy=False)

    config = transform.to_hydra_config()

    assert config['_target_'] == 'torchio.transforms.augmentation.composition.Compose'
    assert config['copy'] is False
    assert config['include'] is None
    assert config['exclude'] is None
    assert config['transforms'][0]['_target_'].endswith('Flip')
    assert config['transforms'][0]['axes'] == [0, 2]
