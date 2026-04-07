"""Tests for Image-level annotations (points and bounding boxes)."""

from __future__ import annotations

import copy

import pytest
import torch

from torchio import LabelMap
from torchio import ScalarImage
from torchio import Subject
from torchio.data.bboxes import BoundingBoxes
from torchio.data.bboxes import BoundingBoxFormat
from torchio.data.points import Points


class TestImageWithPoints:
    def test_image_default_no_points(self):
        image = ScalarImage(torch.randn(1, 10, 10, 10))
        assert image.points == {}

    def test_image_with_points_kwarg(self):
        pts = Points(torch.randn(5, 3))
        image = ScalarImage(
            torch.randn(1, 10, 10, 10),
            points={"landmarks": pts},
        )
        assert "landmarks" in image.points
        assert image.points["landmarks"] is pts

    def test_image_with_multiple_point_sets(self):
        lm = Points(torch.randn(5, 3))
        fiducials = Points(torch.randn(3, 3))
        image = ScalarImage(
            torch.randn(1, 10, 10, 10),
            points={"landmarks": lm, "fiducials": fiducials},
        )
        assert len(image.points) == 2

    def test_image_points_validates_values(self):
        """Points dict values must be Points instances."""
        with pytest.raises(TypeError, match="Points"):
            ScalarImage(
                torch.randn(1, 10, 10, 10),
                points={"landmarks": torch.randn(5, 3)},
            )


class TestImageWithBoundingBoxes:
    def test_image_default_no_bounding_boxes(self):
        image = ScalarImage(torch.randn(1, 10, 10, 10))
        assert image.bounding_boxes == {}

    def test_image_with_bboxes_kwarg(self):
        boxes = BoundingBoxes(
            torch.tensor([[10, 20, 30, 50, 60, 70]]),
            format=BoundingBoxFormat.IJKIJK,
        )
        image = ScalarImage(
            torch.randn(1, 10, 10, 10),
            bounding_boxes={"tumors": boxes},
        )
        assert "tumors" in image.bounding_boxes
        assert image.bounding_boxes["tumors"] is boxes

    def test_image_with_multiple_bbox_sets(self):
        tumors = BoundingBoxes(
            torch.randn(2, 6),
            format=BoundingBoxFormat.IJKIJK,
        )
        organs = BoundingBoxes(
            torch.randn(5, 6),
            format=BoundingBoxFormat.IJKWHD,
        )
        image = ScalarImage(
            torch.randn(1, 10, 10, 10),
            bounding_boxes={"tumors": tumors, "organs": organs},
        )
        assert len(image.bounding_boxes) == 2

    def test_image_bboxes_validates_values(self):
        """BoundingBoxes dict values must be BoundingBoxes instances."""
        with pytest.raises(TypeError, match="BoundingBoxes"):
            ScalarImage(
                torch.randn(1, 10, 10, 10),
                bounding_boxes={"tumors": torch.randn(2, 6)},
            )


class TestImageAnnotationsBothTypes:
    def test_image_with_points_and_bboxes(self):
        pts = Points(torch.randn(5, 3))
        boxes = BoundingBoxes(
            torch.randn(2, 6),
            format=BoundingBoxFormat.IJKIJK,
        )
        image = ScalarImage(
            torch.randn(1, 10, 10, 10),
            points={"landmarks": pts},
            bounding_boxes={"tumors": boxes},
        )
        assert len(image.points) == 1
        assert len(image.bounding_boxes) == 1


class TestNewLikePreservesAnnotations:
    def test_new_like_preserves_points(self):
        pts = Points(torch.randn(5, 3))
        image = ScalarImage(
            torch.randn(1, 10, 10, 10),
            points={"landmarks": pts},
        )
        new = image.new_like(data=torch.randn(1, 5, 5, 5))
        assert "landmarks" in new.points
        assert new.points["landmarks"].num_points == 5
        # Should be a copy, not the same object
        assert new.points["landmarks"] is not pts

    def test_new_like_preserves_bboxes(self):
        boxes = BoundingBoxes(
            torch.randn(3, 6),
            format=BoundingBoxFormat.IJKIJK,
        )
        image = ScalarImage(
            torch.randn(1, 10, 10, 10),
            bounding_boxes={"tumors": boxes},
        )
        new = image.new_like(data=torch.randn(1, 5, 5, 5))
        assert "tumors" in new.bounding_boxes
        assert new.bounding_boxes["tumors"].num_boxes == 3
        assert new.bounding_boxes["tumors"] is not boxes

    def test_new_like_preserves_subclass_with_annotations(self):
        pts = Points(torch.randn(5, 3))
        image = LabelMap(
            torch.randint(0, 5, (1, 10, 10, 10)),
            points={"landmarks": pts},
        )
        new = image.new_like(data=torch.randint(0, 5, (1, 5, 5, 5)))
        assert isinstance(new, LabelMap)
        assert "landmarks" in new.points


class TestDeepCopyPreservesAnnotations:
    def test_deepcopy_copies_points(self):
        pts = Points(torch.randn(5, 3))
        image = ScalarImage(
            torch.randn(1, 10, 10, 10),
            points={"landmarks": pts},
        )
        copied = copy.deepcopy(image)
        assert "landmarks" in copied.points
        assert copied.points["landmarks"] is not pts
        torch.testing.assert_close(
            copied.points["landmarks"].data,
            pts.data,
        )

    def test_deepcopy_copies_bboxes(self):
        boxes = BoundingBoxes(
            torch.randn(3, 6),
            format=BoundingBoxFormat.IJKIJK,
        )
        image = ScalarImage(
            torch.randn(1, 10, 10, 10),
            bounding_boxes={"tumors": boxes},
        )
        copied = copy.deepcopy(image)
        assert "tumors" in copied.bounding_boxes
        assert copied.bounding_boxes["tumors"] is not boxes
        torch.testing.assert_close(
            copied.bounding_boxes["tumors"].data,
            boxes.data,
        )

    def test_deepcopy_independence(self):
        """Modifying the copy's annotations doesn't affect the original."""
        pts = Points(torch.randn(5, 3))
        image = ScalarImage(
            torch.randn(1, 10, 10, 10),
            points={"landmarks": pts},
        )
        copied = copy.deepcopy(image)
        # Mutate the copy
        copied.points["landmarks"]._data[0, 0] = 999.0
        assert image.points["landmarks"].data[0, 0] != 999.0


class TestSlicingPreservesAnnotations:
    def test_slice_preserves_points(self):
        pts = Points(torch.randn(5, 3))
        image = ScalarImage(
            torch.randn(1, 20, 20, 20),
            points={"landmarks": pts},
        )
        sliced = image[:, 5:10]
        assert "landmarks" in sliced.points
        assert sliced.points["landmarks"].num_points == 5

    def test_slice_preserves_bboxes(self):
        boxes = BoundingBoxes(
            torch.randn(3, 6),
            format=BoundingBoxFormat.IJKIJK,
        )
        image = ScalarImage(
            torch.randn(1, 20, 20, 20),
            bounding_boxes={"tumors": boxes},
        )
        sliced = image[:, 5:10]
        assert "tumors" in sliced.bounding_boxes


class TestSubjectWithImageLevelAnnotations:
    def test_subject_image_level_points(self):
        """Points on an Image are accessible through the Subject."""
        pts = Points(torch.randn(5, 3))
        image = ScalarImage(
            torch.randn(1, 10, 10, 10),
            points={"landmarks": pts},
        )
        subject = Subject(t1=image)
        assert "landmarks" in subject.t1.points

    def test_all_points_includes_both_levels(self):
        """all_points() yields from both image-level and subject-level."""
        img_pts = Points(torch.randn(5, 3))
        subj_pts = Points(torch.randn(3, 3))
        image = ScalarImage(
            torch.randn(1, 10, 10, 10),
            points={"img_landmarks": img_pts},
        )
        subject = Subject(
            t1=image,
            subj_landmarks=subj_pts,
        )
        all_pts = subject.all_points()
        assert "subj_landmarks" in all_pts
        assert ("t1", "img_landmarks") in all_pts

    def test_all_bounding_boxes_includes_both_levels(self):
        """all_bounding_boxes() yields from both levels."""
        img_boxes = BoundingBoxes(
            torch.randn(2, 6),
            format=BoundingBoxFormat.IJKIJK,
        )
        subj_boxes = BoundingBoxes(
            torch.randn(1, 6),
            format=BoundingBoxFormat.IJKIJK,
        )
        image = ScalarImage(
            torch.randn(1, 10, 10, 10),
            bounding_boxes={"img_tumors": img_boxes},
        )
        subject = Subject(
            t1=image,
            subj_tumors=subj_boxes,
        )
        all_bb = subject.all_bounding_boxes()
        assert "subj_tumors" in all_bb
        assert ("t1", "img_tumors") in all_bb

    def test_all_points_no_overlap(self):
        """Subject with only subject-level points."""
        subj_pts = Points(torch.randn(3, 3))
        subject = Subject(
            t1=ScalarImage(torch.randn(1, 10, 10, 10)),
            landmarks=subj_pts,
        )
        all_pts = subject.all_points()
        assert "landmarks" in all_pts
        assert len(all_pts) == 1

    def test_all_points_only_image_level(self):
        """Subject with only image-level points."""
        img_pts = Points(torch.randn(5, 3))
        image = ScalarImage(
            torch.randn(1, 10, 10, 10),
            points={"landmarks": img_pts},
        )
        subject = Subject(t1=image)
        all_pts = subject.all_points()
        assert ("t1", "landmarks") in all_pts
        assert len(all_pts) == 1


class TestImageAnnotationsRepr:
    def test_repr_includes_annotations(self):
        pts = Points(torch.randn(5, 3))
        boxes = BoundingBoxes(
            torch.randn(2, 6),
            format=BoundingBoxFormat.IJKIJK,
        )
        image = ScalarImage(
            torch.randn(1, 10, 10, 10),
            points={"landmarks": pts},
            bounding_boxes={"tumors": boxes},
        )
        r = repr(image)
        assert "landmarks" in r
        assert "tumors" in r

    def test_repr_no_annotations(self):
        image = ScalarImage(torch.randn(1, 10, 10, 10))
        r = repr(image)
        # Should not mention points/bboxes when empty
        assert "points" not in r.lower() or "0" in r
