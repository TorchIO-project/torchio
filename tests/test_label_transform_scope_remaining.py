"""Contour, KeepLargestComponent and SequentialLabels must respect include/exclude.

PR #1511 fixed OneHot, RemapLabels and RemoveLabels to use
``self._get_images(batch)`` instead of ``batch.images`` in their
``apply_transform`` methods.  Contour, KeepLargestComponent and
SequentialLabels (including its inverse) had the same bug: they
iterated over ``batch.images`` directly, ignoring the ``include``
and ``exclude`` kwargs inherited from Transform.
"""

from __future__ import annotations

import copy

import pytest
import torch

import torchio as tio


def _make_subject() -> tio.Subject:
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0]).reshape(1, 1, 2, 2)
    return tio.Subject(
        a=tio.LabelMap(labels.clone()),
        b=tio.LabelMap(labels.clone()),
        scan=tio.ScalarImage(labels + 10),
    )


def _make_input(
    subject: tio.Subject, as_batch: bool
) -> tio.Subject | tio.SubjectsBatch:
    if not as_batch:
        return subject
    other = copy.deepcopy(subject)
    for image in other.images.values():
        image.set_data(image.data.flip(-1))
    return tio.SubjectsBatch.from_subjects([subject, other])


@pytest.mark.parametrize("as_batch", [False, True], ids=["subject", "batch"])
@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(tio.Contour, id="contour"),
        pytest.param(tio.KeepLargestComponent, id="keep-largest"),
        pytest.param(tio.SequentialLabels, id="sequential"),
    ],
)
@pytest.mark.parametrize(
    ("selection", "selected"),
    [
        pytest.param({"include": ["a", "scan"]}, {"a"}, id="include"),
        pytest.param({"exclude": ["b"]}, {"a"}, id="exclude"),
    ],
)
def test_label_transform_respects_image_selection(
    as_batch: bool,
    factory: type,
    selection: dict[str, list[str]],
    selected: set[str],
) -> None:
    """Excluded label maps must not be modified."""
    data = _make_input(_make_subject(), as_batch)
    original = {name: image.data.clone() for name, image in data.images.items()}

    result = factory(**selection)(data)

    for name, before in original.items():
        if name not in selected:
            torch.testing.assert_close(
                result[name].data,
                before,
                msg=f"{factory.__name__} modified excluded image '{name}'",
            )


@pytest.mark.parametrize("as_batch", [False, True], ids=["subject", "batch"])
def test_sequential_labels_inverse_respects_selection(as_batch: bool) -> None:
    """The inverse of SequentialLabels must also respect include/exclude."""
    subject = _make_subject()
    # Give b labels {0, 5, 10} so the renumbering would be visible
    subject["b"].set_data(torch.tensor([0.0, 5.0, 10.0, 5.0]).reshape(1, 1, 2, 2))
    data = _make_input(subject, as_batch)
    original_b = data["b"].data.clone()

    transform = tio.SequentialLabels(exclude=["b"])
    transformed = transform(data)
    # b must be untouched in the forward pass
    torch.testing.assert_close(transformed["b"].data, original_b)

    restored = transformed.apply_inverse_transform()
    # b must still be untouched after the inverse pass
    torch.testing.assert_close(restored["b"].data, original_b)
