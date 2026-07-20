"""Tests for the Pad transform and Crop/Pad invertibility."""

from __future__ import annotations

import warnings

import pytest
import torch

import torchio as tio
from torchio.transforms.spatial._padding import pad_tensor


class TestPad:
    def test_pad_tensor_rejects_invalid_number_of_dimensions(self) -> None:
        with pytest.raises(ValueError, match="4D or 5D"):
            pad_tensor(
                torch.ones(2, 2, 2),
                (0, 0, 0, 0, 0, 0),
                "mean",
                0,
            )

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

    def test_invalid_padding_mode(self) -> None:
        with pytest.raises(ValueError, match="padding_mode"):
            tio.Pad(padding=1, padding_mode="maximum")

    @pytest.mark.parametrize(
        ("padding_mode", "expected"),
        [
            ("mean", 1.5),
            ("median", 1.5),
            ("minimum", 0),
        ],
    )
    def test_pad_statistic_mode(
        self,
        padding_mode: str,
        expected: float,
    ) -> None:
        tensor = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
        result = tio.Pad(
            padding=(0, 0, 0, 1, 0, 0),
            padding_mode=padding_mode,
        )(tensor)
        torch.testing.assert_close(
            result[0, 0, 2],
            torch.full((2,), expected, dtype=tensor.dtype),
        )

    @pytest.mark.parametrize("padding_mode", ["mean", "median", "minimum"])
    def test_pad_statistic_mode_per_batch_element(
        self,
        padding_mode: str,
    ) -> None:
        subjects = [
            tio.Subject(t1=tio.ScalarImage(torch.full((1, 2, 2, 2), value)))
            for value in (1.0, 3.0)
        ]
        batch = tio.SubjectsBatch.from_subjects(subjects)
        result = tio.Pad(padding=1, padding_mode=padding_mode)(batch)
        torch.testing.assert_close(
            result.t1.data[:, 0, 0, 0, 0],
            torch.tensor([1.0, 3.0]),
        )

    @pytest.mark.parametrize(
        ("padding_mode", "expected"),
        [
            ("mean", 0),
            ("median", 1),
        ],
    )
    def test_pad_statistic_mode_warns_for_integer_truncation(
        self,
        padding_mode: str,
        expected: int,
    ) -> None:
        tensor = torch.tensor([0, 1, 1, 1]).reshape(1, 1, 2, 2)
        with pytest.warns(RuntimeWarning, match="might be truncated"):
            result = tio.Pad(
                padding=(0, 0, 0, 1, 0, 0),
                padding_mode=padding_mode,
            )(tensor)
        assert result.dtype == tensor.dtype
        assert result[0, 0, 2, 0].item() == expected

    def test_pad_minimum_does_not_warn_for_integer_input(self) -> None:
        tensor = torch.tensor([0, 1, 1, 1]).reshape(1, 1, 2, 2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = tio.Pad(
                padding=(0, 0, 0, 1, 0, 0),
                padding_mode="minimum",
            )(tensor)
        assert result[0, 0, 2, 0].item() == 0
        assert not any(issubclass(item.category, RuntimeWarning) for item in caught)

    @pytest.mark.parametrize("padding_mode", ["mean", "median", "minimum"])
    def test_pad_statistic_mode_is_differentiable(
        self,
        padding_mode: str,
    ) -> None:
        tensor = torch.rand(1, 2, 2, 2, requires_grad=True)
        result = tio.Pad(
            padding=1,
            padding_mode=padding_mode,
            copy=False,
        )(tensor)
        result.sum().backward()
        assert tensor.grad is not None

    @pytest.mark.parametrize("padding_mode", ["mean", "median"])
    def test_pad_statistic_mode_preserves_float64_precision(
        self,
        padding_mode: str,
    ) -> None:
        values = torch.tensor([1.0, 1.0 + 2**-40], dtype=torch.float64)
        tensor = values.reshape(1, 1, 1, 2)
        result = tio.Pad(
            padding=(0, 0, 0, 1, 0, 0),
            padding_mode=padding_mode,
        )(tensor)
        expected = values.mean()
        torch.testing.assert_close(
            result[0, 0, 1, 0],
            expected,
            rtol=0,
            atol=0,
        )

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

    def test_crop_or_pad_inverse_respects_include_scope(self) -> None:
        a = torch.ones(1, 4, 4, 4)
        b = torch.ones(1, 4, 4, 4) * 2
        subject = tio.Subject(
            a=tio.ScalarImage(a.clone()),
            b=tio.ScalarImage(b.clone()),
        )

        transformed = tio.CropOrPad(6, include=["a"])(subject)
        restored = transformed.apply_inverse_transform()

        assert restored.a.shape == (1, 4, 4, 4)
        torch.testing.assert_close(restored.a.data, a)
        torch.testing.assert_close(restored.b.data, b)
