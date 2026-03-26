"""Tests for Subject."""

from __future__ import annotations

import copy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from einops import rearrange

from torchio import LabelMap
from torchio import ScalarImage
from torchio import Subject


class TestSubjectCreation:
    def test_create_with_kwargs(self):
        subject = Subject(
            t1=ScalarImage.from_tensor(torch.randn(1, 10, 10, 10)),
            seg=LabelMap.from_tensor(torch.randint(0, 5, (1, 10, 10, 10))),
        )
        assert len(subject.images()) == 2

    def test_create_from_unpacked_dict(self):
        data = {
            "t1": ScalarImage.from_tensor(torch.randn(1, 10, 10, 10)),
            "seg": LabelMap.from_tensor(torch.randint(0, 5, (1, 10, 10, 10))),
        }
        subject = Subject(**data)
        assert len(subject.images()) == 2

    def test_metadata_from_kwargs(self):
        subject = Subject(
            t1=ScalarImage.from_tensor(torch.randn(1, 10, 10, 10)),
            age=45,
            name="John",
        )
        assert subject.metadata["age"] == 45
        assert subject.metadata["name"] == "John"

    def test_empty_subject_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            Subject()

    def test_no_images_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            Subject(age=45)


class TestSubjectAccess:
    @pytest.fixture
    def subject(self) -> Subject:
        return Subject(
            t1=ScalarImage.from_tensor(torch.randn(1, 10, 10, 10)),
            seg=LabelMap.from_tensor(torch.randint(0, 5, (1, 10, 10, 10))),
            age=45,
        )

    def test_getattr_image(self, subject: Subject):
        assert isinstance(subject.t1, ScalarImage)
        assert isinstance(subject.seg, LabelMap)

    def test_getattr_metadata(self, subject: Subject):
        assert subject.age == 45

    def test_getitem(self, subject: Subject):
        assert isinstance(subject["t1"], ScalarImage)
        assert isinstance(subject["seg"], LabelMap)

    def test_getattr_missing_raises(self, subject: Subject):
        with pytest.raises(AttributeError):
            subject.nonexistent

    def test_getitem_missing_raises(self, subject: Subject):
        with pytest.raises(KeyError):
            subject["nonexistent"]

    def test_images_returns_only_images(self, subject: Subject):
        images = subject.images()
        assert len(images) == 2
        assert "t1" in images
        assert "seg" in images

    def test_metadata_access(self, subject: Subject):
        assert subject.metadata["age"] == 45

    def test_contains(self, subject: Subject):
        assert "t1" in subject
        assert "nonexistent" not in subject


class TestSubjectProperties:
    @pytest.fixture
    def subject(self) -> Subject:
        return Subject(
            t1=ScalarImage.from_tensor(torch.randn(1, 10, 20, 30)),
            seg=LabelMap.from_tensor(torch.randint(0, 5, (1, 10, 20, 30))),
        )

    def test_spatial_shape(self, subject: Subject):
        assert subject.spatial_shape == (10, 20, 30)

    def test_shape(self, subject: Subject):
        assert subject.shape == (1, 10, 20, 30)

    def test_spacing(self, subject: Subject):
        assert subject.spacing == (1.0, 1.0, 1.0)

    def test_inconsistent_shapes_raises(self):
        subject = Subject(
            t1=ScalarImage.from_tensor(torch.randn(1, 10, 10, 10)),
            t2=ScalarImage.from_tensor(torch.randn(1, 20, 20, 20)),
        )
        with pytest.raises(RuntimeError, match="Inconsistent"):
            subject.spatial_shape

    def test_inconsistent_spacing_raises(self):
        subject = Subject(
            t1=ScalarImage.from_tensor(
                torch.randn(1, 10, 10, 10),
                affine=np.diag([1.0, 1.0, 1.0, 1.0]),
            ),
            t2=ScalarImage.from_tensor(
                torch.randn(1, 10, 10, 10),
                affine=np.diag([2.0, 2.0, 2.0, 1.0]),
            ),
        )
        with pytest.raises(RuntimeError, match="Inconsistent"):
            subject.spacing

    def test_single_image_properties(self):
        subject = Subject(
            t1=ScalarImage.from_tensor(torch.randn(1, 10, 10, 10)),
        )
        assert subject.spatial_shape == (10, 10, 10)
        assert subject.spacing == (1.0, 1.0, 1.0)

    def test_applied_transforms_starts_empty(self, subject: Subject):
        assert subject.applied_transforms == []


class TestSubjectHistory:
    def test_add_transform(self):
        subject = Subject(
            t1=ScalarImage.from_tensor(torch.randn(1, 10, 10, 10)),
        )
        subject.applied_transforms.append(
            {"name": "Affine", "parameters": {"scales": (1.1, 1.1, 1.1)}}
        )
        assert len(subject.applied_transforms) == 1

    def test_clear_history(self):
        subject = Subject(
            t1=ScalarImage.from_tensor(torch.randn(1, 10, 10, 10)),
        )
        subject.applied_transforms.append(
            {"name": "Affine", "parameters": {"scales": (1.1, 1.1, 1.1)}}
        )
        subject.clear_history()
        assert len(subject.applied_transforms) == 0


class TestSubjectLoad:
    def test_load_all(self, tmp_path: Path):
        tensor = torch.randn(1, 10, 10, 10)
        array = rearrange(tensor.numpy(), "c i j k -> i j k c")
        nii = nib.Nifti1Image(array, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)

        subject = Subject(t1=ScalarImage(path))
        assert not subject.t1.is_loaded
        subject.load()
        assert subject.t1.is_loaded


class TestSubjectCopy:
    def test_copy(self):
        subject = Subject(
            t1=ScalarImage.from_tensor(torch.randn(1, 10, 10, 10)),
            age=45,
        )
        copied = copy.deepcopy(subject)
        assert isinstance(copied, Subject)
        assert isinstance(copied.t1, ScalarImage)
        assert copied.metadata["age"] == 45
        # Verify it's a deep copy
        copied.t1.set_data(torch.zeros(1, 10, 10, 10))
        assert not torch.equal(subject.t1.data, copied.t1.data)


class TestSubjectRepr:
    def test_repr(self):
        subject = Subject(
            t1=ScalarImage.from_tensor(torch.randn(1, 10, 10, 10)),
            seg=LabelMap.from_tensor(torch.randint(0, 5, (1, 10, 10, 10))),
        )
        r = repr(subject)
        assert "Subject" in r
        assert "t1" in r
        assert "seg" in r


class TestSubjectIteration:
    def test_iter_yields_image_keys(self):
        subject = Subject(
            t1=ScalarImage.from_tensor(torch.randn(1, 10, 10, 10)),
            seg=LabelMap.from_tensor(torch.randint(0, 5, (1, 10, 10, 10))),
            age=45,
        )
        keys = list(subject)
        assert "t1" in keys
        assert "seg" in keys
        assert "age" not in keys
