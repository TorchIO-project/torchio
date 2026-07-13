"""Tests for custom Image subclasses with transforms."""

from __future__ import annotations

import copy

import pytest
import torch

import torchio as tio


class HistoryScalarImage(tio.ScalarImage):
    """Custom Image that requires an extra ``history`` argument.

    This is the exact subclass from the #1391 reproduction snippet.
    """

    def __init__(self, tensor, affine, history, **kwargs):
        super().__init__(tensor=tensor, affine=affine, **kwargs)
        self.history = history

    def new_like(self, tensor, affine=None):
        return type(self)(
            tensor=tensor,
            affine=affine if affine is not None else self.affine,
            history=self.history,
            check_nans=self.check_nans,
            reader=self.reader,
        )


class MetadataLabelMap(tio.LabelMap):
    """Custom LabelMap with optional metadata."""

    def __init__(self, tensor, affine, labels_info=None, **kwargs):
        super().__init__(tensor=tensor, affine=affine, **kwargs)
        self.labels_info = labels_info or {}

    def new_like(self, tensor, affine=None):
        return type(self)(
            tensor=tensor,
            affine=affine if affine is not None else self.affine,
            labels_info=self.labels_info,
            check_nans=self.check_nans,
            reader=self.reader,
        )


@pytest.fixture()
def history_image():
    tensor = torch.rand(1, 10, 10, 10)
    affine = torch.eye(4)
    return HistoryScalarImage(tensor=tensor, affine=affine, history=['created'])


@pytest.fixture()
def history_subject(history_image):
    return tio.Subject(image=history_image)


class TestIssue1391Reproduction:
    """Exact reproduction of the snippet in issue #1391."""

    def test_crop_custom_subclass(self):
        img = HistoryScalarImage(
            torch.rand(1, 10, 10, 10),
            affine=torch.eye(4),
            history=[],
        )
        subject = tio.Subject(image=img)
        transform = tio.Crop(cropping=2)
        result = transform(subject)
        assert isinstance(result.image, HistoryScalarImage)
        assert result.image.shape == (1, 6, 6, 6)


class TestNewLike:
    """Tests for the Image.new_like() factory method."""

    def test_new_like_preserves_type(self, history_image):
        new = history_image.new_like(torch.rand(1, 5, 5, 5))
        assert isinstance(new, HistoryScalarImage)

    def test_new_like_preserves_custom_attribute(self, history_image):
        new = history_image.new_like(torch.rand(1, 5, 5, 5))
        assert new.history == ['created']

    def test_new_like_uses_new_tensor(self, history_image):
        new_tensor = torch.rand(1, 5, 5, 5)
        new = history_image.new_like(new_tensor)
        assert torch.equal(new.data, new_tensor)

    def test_new_like_uses_new_affine(self, history_image):
        new_affine = torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0]))
        new = history_image.new_like(torch.rand(1, 5, 5, 5), affine=new_affine)
        assert torch.allclose(
            torch.as_tensor(new.affine, dtype=torch.float32),
            new_affine,
        )

    def test_new_like_defaults_to_original_affine(self, history_image):
        new = history_image.new_like(torch.rand(1, 5, 5, 5))
        assert torch.allclose(
            torch.as_tensor(new.affine, dtype=torch.float32),
            torch.as_tensor(history_image.affine, dtype=torch.float32),
        )

    def test_new_like_standard_scalar_image(self):
        image = tio.ScalarImage(tensor=torch.rand(1, 8, 8, 8), affine=torch.eye(4))
        new = image.new_like(torch.rand(1, 4, 4, 4))
        assert isinstance(new, tio.ScalarImage)

    def test_new_like_standard_label_map(self):
        image = tio.LabelMap(tensor=torch.randint(0, 3, (1, 8, 8, 8)))
        new = image.new_like(torch.randint(0, 3, (1, 4, 4, 4)))
        assert isinstance(new, tio.LabelMap)

    def test_new_like_propagates_extra_dict_keys(self):
        image = tio.ScalarImage(
            tensor=torch.rand(1, 8, 8, 8),
            affine=torch.eye(4),
            age=30,
            site='hospital_a',
        )
        new = image.new_like(torch.rand(1, 4, 4, 4))
        assert new['age'] == 30
        assert new['site'] == 'hospital_a'

    def test_crop_preserves_extra_dict_keys(self):
        image = tio.ScalarImage(
            tensor=torch.rand(1, 10, 10, 10),
            affine=torch.eye(4),
            age=30,
        )
        subject = tio.Subject(image=image)
        result = tio.Crop(cropping=2)(subject)
        assert result.image['age'] == 30


class TestCropWithCustomSubclass:
    def test_crop_preserves_type_and_attribute(self, history_subject):
        result = tio.Crop(cropping=2)(history_subject)
        assert isinstance(result.image, HistoryScalarImage)
        assert result.image.history == ['created']
        assert result.image.shape == (1, 6, 6, 6)

    def test_crop_or_pad_preserves_type(self, history_subject):
        result = tio.CropOrPad(target_shape=(6, 6, 6))(history_subject)
        assert isinstance(result.image, HistoryScalarImage)
        assert result.image.history == ['created']

    def test_chained_crops_preserve_type(self, history_subject):
        transform = tio.Compose([tio.Crop(cropping=1), tio.Crop(cropping=1)])
        result = transform(history_subject)
        assert isinstance(result.image, HistoryScalarImage)
        assert result.image.history == ['created']
        assert result.image.shape == (1, 6, 6, 6)

    def test_crop_custom_label_map(self):
        tensor = torch.randint(0, 3, (1, 8, 8, 8))
        affine = torch.eye(4)
        labels_info = {0: 'bg', 1: 'tissue', 2: 'lesion'}
        label = MetadataLabelMap(tensor=tensor, affine=affine, labels_info=labels_info)
        subject = tio.Subject(seg=label)
        result = tio.Crop(cropping=1)(subject)
        assert isinstance(result.seg, MetadataLabelMap)
        assert result.seg.labels_info == labels_info
        assert result.seg.shape == (1, 6, 6, 6)


class TestToReferenceSpaceWithCustomSubclass:
    def test_from_tensor_preserves_type(self, history_image):
        embedding = torch.rand(1, 10, 10, 10)
        result = tio.ToReferenceSpace.from_tensor(embedding, history_image)
        assert isinstance(result, HistoryScalarImage)
        assert result.history == ['created']


class TestCopyWithCustomSubclass:
    def test_copy_preserves_type(self, history_image):
        copied = copy.copy(history_image)
        assert isinstance(copied, HistoryScalarImage)
        assert copied.history == ['created']

    def test_copy_preserves_data(self, history_image):
        copied = copy.copy(history_image)
        assert torch.equal(copied.data, history_image.data)

    def test_copy_standard_image(self):
        image = tio.ScalarImage(tensor=torch.rand(1, 8, 8, 8), affine=torch.eye(4))
        copied = copy.copy(image)
        assert isinstance(copied, tio.ScalarImage)
        assert torch.equal(copied.data, image.data)
