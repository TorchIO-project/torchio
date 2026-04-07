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
from torchio.data.bboxes import BoundingBoxes
from torchio.data.bboxes import BoundingBoxFormat
from torchio.data.points import Points


class TestSubjectCreation:
    def test_create_with_kwargs(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            seg=LabelMap(torch.randint(0, 5, (1, 10, 10, 10))),
        )
        assert len(subject.images) == 2

    def test_create_from_unpacked_dict(self):
        data = {
            "t1": ScalarImage(torch.randn(1, 10, 10, 10)),
            "seg": LabelMap(torch.randint(0, 5, (1, 10, 10, 10))),
        }
        subject = Subject(**data)
        assert len(subject.images) == 2

    def test_metadata_from_kwargs(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            age=45,
            name="John",
        )
        assert subject.metadata["age"] == 45
        assert subject.metadata["name"] == "John"

    def test_empty_subject_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            Subject()

    def test_metadata_only_subject(self):
        subject = Subject(age=45, name="patient_1")
        assert subject.age == 45
        assert subject.name == "patient_1"
        assert len(subject.images) == 0

    def test_points_only_subject(self):
        pts = Points(torch.randn(5, 3))
        subject = Subject(landmarks=pts)
        assert len(subject.points) == 1


class TestSubjectAccess:
    @pytest.fixture
    def subject(self) -> Subject:
        return Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            seg=LabelMap(torch.randint(0, 5, (1, 10, 10, 10))),
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
        images = subject.images
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
            t1=ScalarImage(torch.randn(1, 10, 20, 30)),
            seg=LabelMap(torch.randint(0, 5, (1, 10, 20, 30))),
        )

    def test_spatial_shape(self, subject: Subject):
        assert subject.spatial_shape == (10, 20, 30)

    def test_shape(self, subject: Subject):
        assert subject.shape == (1, 10, 20, 30)

    def test_spacing(self, subject: Subject):
        assert subject.spacing == (1.0, 1.0, 1.0)

    def test_inconsistent_shapes_raises(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            t2=ScalarImage(torch.randn(1, 20, 20, 20)),
        )
        with pytest.raises(RuntimeError, match="Inconsistent"):
            subject.spatial_shape

    def test_inconsistent_spacing_raises(self):
        subject = Subject(
            t1=ScalarImage(
                torch.randn(1, 10, 10, 10),
                affine=np.diag([1.0, 1.0, 1.0, 1.0]),
            ),
            t2=ScalarImage(
                torch.randn(1, 10, 10, 10),
                affine=np.diag([2.0, 2.0, 2.0, 1.0]),
            ),
        )
        with pytest.raises(RuntimeError, match="Inconsistent"):
            subject.spacing

    def test_single_image_properties(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
        )
        assert subject.spatial_shape == (10, 10, 10)
        assert subject.spacing == (1.0, 1.0, 1.0)

    def test_applied_transforms_starts_empty(self, subject: Subject):
        assert subject.applied_transforms == []


class TestSubjectHistory:
    def test_add_transform(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
        )
        subject.applied_transforms.append(
            {"name": "Affine", "parameters": {"scales": (1.1, 1.1, 1.1)}}
        )
        assert len(subject.applied_transforms) == 1

    def test_clear_history(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
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
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
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
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            seg=LabelMap(torch.randint(0, 5, (1, 10, 10, 10))),
        )
        r = repr(subject)
        assert "Subject" in r
        assert "t1" in r
        assert "seg" in r


class TestSubjectIteration:
    def test_iter_yields_image_keys(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            seg=LabelMap(torch.randint(0, 5, (1, 10, 10, 10))),
            age=45,
        )
        keys = list(subject)
        assert "t1" in keys
        assert "seg" in keys
        assert "age" not in keys

    def test_iter_yields_all_spatial_keys(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            landmarks=Points(torch.randn(5, 3)),
            tumors=BoundingBoxes(
                torch.randn(2, 6),
                format=BoundingBoxFormat.IJKIJK,
            ),
            age=45,
        )
        keys = list(subject)
        assert "t1" in keys
        assert "landmarks" in keys
        assert "tumors" in keys
        assert "age" not in keys


class TestSubjectWithPoints:
    def test_points_access(self):
        pts = Points(torch.randn(5, 3))
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            landmarks=pts,
        )
        assert subject.landmarks is pts
        assert subject["landmarks"] is pts

    def test_points_dict(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            lm1=Points(torch.randn(3, 3)),
            lm2=Points(torch.randn(7, 3)),
        )
        pts = subject.points
        assert len(pts) == 2
        assert "lm1" in pts
        assert "lm2" in pts

    def test_contains_points(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            landmarks=Points(torch.randn(5, 3)),
        )
        assert "landmarks" in subject

    def test_len_includes_points(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            landmarks=Points(torch.randn(5, 3)),
        )
        assert len(subject) == 2


class TestSubjectWithBoundingBoxes:
    def test_bboxes_access(self):
        boxes = BoundingBoxes(
            torch.randn(3, 6),
            format=BoundingBoxFormat.IJKIJK,
        )
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            tumors=boxes,
        )
        assert subject.tumors is boxes
        assert subject["tumors"] is boxes

    def test_bboxes_dict(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            tumors=BoundingBoxes(
                torch.randn(3, 6),
                format=BoundingBoxFormat.IJKIJK,
            ),
            organs=BoundingBoxes(
                torch.randn(5, 6),
                format=BoundingBoxFormat.IJKWHD,
            ),
        )
        bb = subject.bounding_boxes
        assert len(bb) == 2
        assert "tumors" in bb
        assert "organs" in bb

    def test_contains_bboxes(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            tumors=BoundingBoxes(
                torch.randn(1, 6),
                format=BoundingBoxFormat.IJKIJK,
            ),
        )
        assert "tumors" in subject

    def test_len_includes_bboxes(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            tumors=BoundingBoxes(
                torch.randn(1, 6),
                format=BoundingBoxFormat.IJKIJK,
            ),
        )
        assert len(subject) == 2


class TestSubjectMixed:
    def test_all_types(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            seg=LabelMap(torch.randint(0, 5, (1, 10, 10, 10))),
            landmarks=Points(torch.randn(5, 3)),
            tumors=BoundingBoxes(
                torch.randn(2, 6),
                format=BoundingBoxFormat.IJKIJK,
            ),
            age=45,
        )
        assert len(subject.images) == 2
        assert len(subject.points) == 1
        assert len(subject.bounding_boxes) == 1
        assert subject.metadata["age"] == 45
        assert len(subject) == 4  # 2 images + 1 points + 1 bboxes

    def test_repr_with_all_types(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            landmarks=Points(torch.randn(5, 3)),
            tumors=BoundingBoxes(
                torch.randn(2, 6),
                format=BoundingBoxFormat.IJKIJK,
            ),
        )
        r = repr(subject)
        assert "images" in r
        assert "points" in r
        assert "bboxes" in r


class TestSubjectSlicing:
    @pytest.fixture
    def subject(self) -> Subject:
        return Subject(
            t1=ScalarImage(torch.randn(1, 20, 30, 40)),
            seg=LabelMap(torch.randint(0, 3, (1, 20, 30, 40))),
            age=42,
        )

    def test_slice_single_dim(self, subject: Subject):
        cropped = subject[5:15]
        assert cropped.t1.spatial_shape == (10, 30, 40)
        assert cropped.seg.spatial_shape == (10, 30, 40)

    def test_slice_two_dims(self, subject: Subject):
        cropped = subject[5:15, 10:20]
        assert cropped.t1.spatial_shape == (10, 10, 40)
        assert cropped.seg.spatial_shape == (10, 10, 40)

    def test_slice_three_dims(self, subject: Subject):
        cropped = subject[2:12, 5:25, 10:30]
        assert cropped.t1.spatial_shape == (10, 20, 20)

    def test_slice_with_ellipsis(self, subject: Subject):
        cropped = subject[..., 10:30]
        assert cropped.t1.spatial_shape == (20, 30, 20)

    def test_slice_with_int(self, subject: Subject):
        cropped = subject[5]
        assert cropped.t1.spatial_shape == (1, 30, 40)

    def test_preserves_metadata(self, subject: Subject):
        cropped = subject[5:15]
        assert cropped.age == 42

    def test_preserves_channels(self):
        subject = Subject(
            rgb=ScalarImage(torch.randn(3, 20, 30, 40)),
        )
        cropped = subject[5:15]
        assert cropped.rgb.shape == (3, 10, 30, 40)

    def test_preserves_points(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 20, 30, 40)),
            landmarks=Points(torch.randn(5, 3)),
        )
        cropped = subject[5:15]
        assert len(cropped.points) == 1
        assert "landmarks" in cropped.points

    def test_preserves_bboxes(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 20, 30, 40)),
            tumors=BoundingBoxes(
                torch.tensor([[1, 2, 3, 4, 5, 6]]),
                format=BoundingBoxFormat.IJKIJK,
            ),
        )
        cropped = subject[5:15]
        assert len(cropped.bounding_boxes) == 1

    def test_preserves_transform_history(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 20, 30, 40)),
        )
        subject.applied_transforms.append({"name": "RandomFlip"})
        cropped = subject[5:15]
        assert len(cropped.applied_transforms) == 1

    def test_inconsistent_shapes_raises(self):
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 20, 30, 40)),
            t2=ScalarImage(torch.randn(1, 10, 30, 40)),
        )
        with pytest.raises(RuntimeError, match="Inconsistent"):
            subject[5:10]

    def test_string_key_still_works(self, subject: Subject):
        img = subject["t1"]
        assert isinstance(img, ScalarImage)

    def test_no_images_slice_raises(self):
        subject = Subject(landmarks=Points(torch.randn(5, 3)))
        with pytest.raises(RuntimeError, match="no images"):
            subject[5:15]

    def test_is_new_subject(self, subject: Subject):
        cropped = subject[5:15]
        assert isinstance(cropped, Subject)
        assert cropped is not subject
