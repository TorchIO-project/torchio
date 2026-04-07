"""Tests for the Pad transform and Crop/Pad invertibility."""

from __future__ import annotations

import torch

import torchio as tio


class TestPad:
    def test_pad_uniform(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 10, 10, 10))
        subject = tio.Subject(t1=image)
        result = tio.Pad(padding=5)(subject)
        assert result.t1.shape == (1, 20, 20, 20)

    def test_pad_per_axis(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 10, 10, 10))
        subject = tio.Subject(t1=image)
        result = tio.Pad(padding=(2, 4, 6))(subject)
        assert result.t1.shape == (1, 14, 18, 22)

    def test_pad_six_values(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 10, 10, 10))
        subject = tio.Subject(t1=image)
        result = tio.Pad(padding=(1, 2, 3, 4, 5, 6))(subject)
        assert result.t1.shape == (1, 13, 17, 21)

    def test_pad_constant_fill(self) -> None:
        tensor = torch.ones(1, 4, 4, 4)
        image = tio.ScalarImage(tensor)
        subject = tio.Subject(t1=image)
        result = tio.Pad(padding=1, fill=0)(subject)
        # Corners should be 0 (padded)
        assert result.t1.data[0, 0, 0, 0] == 0
        # Interior should be 1 (original)
        assert result.t1.data[0, 1, 1, 1] == 1

    def test_pad_reflect(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 4, 4, 4))
        subject = tio.Subject(t1=image)
        result = tio.Pad(padding=1, padding_mode="reflect")(subject)
        assert result.t1.shape == (1, 6, 6, 6)

    def test_pad_all_images(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 10, 10, 10)),
            seg=tio.LabelMap(torch.randint(0, 3, (1, 10, 10, 10))),
        )
        result = tio.Pad(padding=5)(subject)
        assert result.t1.shape == (1, 20, 20, 20)
        assert result.seg.shape == (1, 20, 20, 20)

    def test_pad_affine_updated(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 10, 10, 10))
        subject = tio.Subject(t1=image)
        original_origin = subject.t1.affine.origin
        result = tio.Pad(padding=(5, 0, 0, 0, 0, 0))(subject)
        # Origin should shift back along first axis
        assert result.t1.affine.origin[0] != original_origin[0]

    def test_pad_history(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 10, 10, 10))
        subject = tio.Subject(t1=image)
        result = tio.Pad(padding=5)(subject)
        assert len(result.applied_transforms) == 1
        assert result.applied_transforms[0].name == "Pad"

    def test_pad_accepts_tensor(self) -> None:
        tensor = torch.rand(1, 10, 10, 10)
        result = tio.Pad(padding=2)(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 14, 14, 14)

    def test_pad_batch(self) -> None:
        from torchio.data.batch import SubjectsBatch

        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 10, 10, 10)),
            )
            for _ in range(3)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        result = tio.Pad(padding=2)(batch)
        assert result.t1.data.shape == (3, 1, 14, 14, 14)


class TestCropPadInvertibility:
    def test_pad_invertible(self) -> None:
        assert tio.Pad(padding=5).invertible

    def test_crop_invertible(self) -> None:
        assert tio.Crop(cropping=5).invertible

    def test_pad_then_inverse_gives_original_shape(self) -> None:
        tensor = torch.rand(1, 10, 10, 10)
        subject = tio.Subject(t1=tio.ScalarImage(tensor.clone()))
        pad = tio.Pad(padding=5)
        padded = pad(subject)
        assert padded.t1.shape == (1, 20, 20, 20)
        restored = padded.apply_inverse_transform()
        assert restored.t1.shape == (1, 10, 10, 10)
        torch.testing.assert_close(restored.t1.data, tensor)

    def test_crop_then_inverse_gives_original_shape(self) -> None:
        tensor = torch.rand(1, 20, 20, 20)
        subject = tio.Subject(t1=tio.ScalarImage(tensor.clone()))
        crop = tio.Crop(cropping=5)
        cropped = crop(subject)
        assert cropped.t1.shape == (1, 10, 10, 10)
        restored = cropped.apply_inverse_transform()
        # Shape restored, but cropped data is lost (filled with 0)
        assert restored.t1.shape == (1, 20, 20, 20)

    def test_crop_pad_compose_inverse(self) -> None:
        tensor = torch.rand(1, 20, 20, 20)
        subject = tio.Subject(t1=tio.ScalarImage(tensor.clone()))
        pipeline = tio.Compose(
            [
                tio.Crop(cropping=2),
                tio.Pad(padding=3),
            ]
        )
        transformed = pipeline(subject)
        assert transformed.t1.shape == (1, 22, 22, 22)
        restored = transformed.apply_inverse_transform()
        assert restored.t1.shape == (1, 20, 20, 20)
