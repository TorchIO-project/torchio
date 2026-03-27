"""Tests for _repr_html_ on Image and Subject."""

from __future__ import annotations

import torch

import torchio as tio
from torchio.data.bboxes import BoundingBoxes
from torchio.data.bboxes import BoundingBoxFormat
from torchio.data.points import Points


class TestImageReprHtml:
    def test_returns_html_string(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 16, 16, 16))
        html = image._repr_html_()
        assert isinstance(html, str)
        assert "<table" in html

    def test_contains_shape(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 8, 10, 12))
        html = image._repr_html_()
        assert "(1, 8, 10, 12)" in html

    def test_contains_spacing(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 16, 16, 16))
        html = image._repr_html_()
        assert "1.00" in html

    def test_contains_orientation(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 16, 16, 16))
        html = image._repr_html_()
        assert "RAS" in html

    def test_contains_class_name(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 16, 16, 16))
        html = image._repr_html_()
        assert "ScalarImage" in html

    def test_label_map_class_name(self) -> None:
        image = tio.LabelMap.from_tensor(torch.randint(0, 3, (1, 8, 8, 8)))
        html = image._repr_html_()
        assert "LabelMap" in html

    def test_contains_dtype(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 8, 8, 8))
        html = image._repr_html_()
        assert "float32" in html

    def test_shows_points(self) -> None:
        pts = Points(torch.tensor([[1.0, 2.0, 3.0]]))
        image = tio.ScalarImage.from_tensor(
            torch.rand(1, 16, 16, 16),
            points={"landmarks": pts},
        )
        html = image._repr_html_()
        assert "landmarks" in html
        assert "1 point" in html

    def test_shows_bounding_boxes(self) -> None:
        boxes = BoundingBoxes(
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
            format=BoundingBoxFormat.IJKIJK,
        )
        image = tio.ScalarImage.from_tensor(
            torch.rand(1, 16, 16, 16),
            bounding_boxes={"tumors": boxes},
        )
        html = image._repr_html_()
        assert "tumors" in html
        assert "1 box" in html

    def test_shows_memory(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 16, 16, 16))
        html = image._repr_html_()
        assert "Memory" in html

    def test_unloaded_shows_path(self, tmp_path) -> None:
        """Unloaded image shows path instead of detailed metadata."""
        import nibabel as nib
        import numpy as np

        path = tmp_path / "test.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((8, 8, 8)), np.eye(4)),
            path,
        )
        image = tio.ScalarImage(path)
        html = image._repr_html_()
        assert "test.nii.gz" in html


class TestSubjectReprHtml:
    def test_returns_html_string(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 16, 16, 16)),
        )
        html = subject._repr_html_()
        assert isinstance(html, str)
        assert "<table" in html

    def test_contains_image_names(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 16, 16, 16)),
            seg=tio.LabelMap.from_tensor(torch.randint(0, 3, (1, 16, 16, 16))),
        )
        html = subject._repr_html_()
        assert "t1" in html
        assert "seg" in html

    def test_contains_image_types(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 16, 16, 16)),
            seg=tio.LabelMap.from_tensor(torch.randint(0, 3, (1, 16, 16, 16))),
        )
        html = subject._repr_html_()
        assert "ScalarImage" in html
        assert "LabelMap" in html

    def test_shows_metadata(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 16, 16, 16)),
            age=42,
            diagnosis="healthy",
        )
        html = subject._repr_html_()
        assert "age" in html
        assert "42" in html
        assert "diagnosis" in html
        assert "healthy" in html

    def test_shows_points(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 16, 16, 16)),
            landmarks=Points(torch.rand(5, 3)),
        )
        html = subject._repr_html_()
        assert "landmarks" in html
        assert "5 points" in html

    def test_shows_bboxes(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 16, 16, 16)),
            tumors=BoundingBoxes(
                torch.rand(3, 6),
                format=BoundingBoxFormat.IJKIJK,
            ),
        )
        html = subject._repr_html_()
        assert "tumors" in html
        assert "3 boxes" in html

    def test_shows_shapes(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 8, 10, 12)),
        )
        html = subject._repr_html_()
        assert "(1, 8, 10, 12)" in html

    def test_metadata_only_subject(self) -> None:
        subject = tio.Subject(age=42, name="test")
        html = subject._repr_html_()
        assert "age" in html
        assert "42" in html
