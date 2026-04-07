"""Tests for the Flip spatial transform."""

from __future__ import annotations

import pytest
import torch

import torchio as tio

HAS_MPS = torch.backends.mps.is_available()


class TestFlip:
    def test_flip_axis_0(self) -> None:
        tensor = torch.arange(8).reshape(1, 2, 2, 2).float()
        image = tio.ScalarImage(tensor)
        subject = tio.Subject(t1=image)
        result = tio.Flip(axes=0)(subject)
        expected = torch.flip(tensor, [1])
        torch.testing.assert_close(result.t1.data, expected)

    def test_flip_single_int_axis(self) -> None:
        """axes=0 should work the same as axes=(0,)."""
        tensor = torch.arange(8).reshape(1, 2, 2, 2).float()
        s1 = tio.Subject(t1=tio.ScalarImage(tensor.clone()))
        s2 = tio.Subject(t1=tio.ScalarImage(tensor.clone()))
        r1 = tio.Flip(axes=0)(s1)
        r2 = tio.Flip(axes=(0,))(s2)
        torch.testing.assert_close(r1.t1.data, r2.t1.data)

    def test_flip_multiple_axes(self) -> None:
        tensor = torch.arange(8).reshape(1, 2, 2, 2).float()
        image = tio.ScalarImage(tensor)
        subject = tio.Subject(t1=image)
        result = tio.Flip(axes=(0, 1))(subject)
        expected = torch.flip(tensor, [1, 2])
        torch.testing.assert_close(result.t1.data, expected)

    def test_flip_all_images(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
            seg=tio.LabelMap(torch.randint(0, 3, (1, 4, 4, 4))),
        )
        original_t1 = subject.t1.data.clone()
        original_seg = subject.seg.data.clone()
        result = tio.Flip(axes=2)(subject)
        assert not torch.equal(result.t1.data, original_t1)
        assert not torch.equal(result.seg.data, original_seg)

    def test_flip_is_self_inverse(self) -> None:
        tensor = torch.rand(1, 4, 5, 6)
        image = tio.ScalarImage(tensor.clone())
        subject = tio.Subject(t1=image)
        flip = tio.Flip(axes=(0, 1, 2))
        result = flip(flip(subject))
        torch.testing.assert_close(result.t1.data, tensor)

    def test_flip_with_probability_zero(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        original = subject.t1.data.clone()
        result = tio.Flip(axes=0, p=0.0)(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_flip_probability_per_axis(self) -> None:
        """flip_probability=0 should not flip any axis."""
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        original = subject.t1.data.clone()
        result = tio.Flip(axes=(0, 1, 2), flip_probability=0.0)(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_flip_probability_one(self) -> None:
        """flip_probability=1 should always flip all specified axes."""
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        original = subject.t1.data.clone()
        result = tio.Flip(axes=(0, 1, 2), flip_probability=1.0)(subject)
        expected = torch.flip(original, [1, 2, 3])
        torch.testing.assert_close(result.t1.data, expected)

    def test_flip_history_recorded(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        result = tio.Flip(axes=1)(subject)
        assert len(result.applied_transforms) == 1
        assert result.applied_transforms[0].name == "Flip"

    def test_flip_accepts_image(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 4, 4, 4))
        result = tio.Flip(axes=0)(image)
        assert isinstance(result, tio.Image)

    def test_flip_accepts_tensor(self) -> None:
        tensor = torch.rand(1, 4, 4, 4)
        result = tio.Flip(axes=0)(tensor)
        assert isinstance(result, torch.Tensor)

    def test_flip_in_compose(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        pipeline = tio.Compose([tio.Flip(axes=0), tio.Flip(axes=1)])
        result = pipeline(subject)
        assert len(result.applied_transforms) == 2

    def test_flip_differentiable(self) -> None:
        tensor = torch.rand(1, 4, 4, 4, requires_grad=True)
        result = tio.Flip(axes=0, copy=False)(tensor)
        loss = result.sum()
        loss.backward()
        assert tensor.grad is not None

    @pytest.mark.skipif(not HAS_MPS, reason="No MPS")
    def test_flip_on_mps(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        subject.to("mps")
        result = tio.Flip(axes=0)(subject)
        assert result.t1.device.type == "mps"

    def test_invalid_axis(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        with pytest.raises(ValueError, match="0, 1, or 2"):
            tio.Flip(axes=3)(subject)

    def test_string_axis(self) -> None:
        """Anatomical label 'Left' should resolve to an axis."""
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        original = subject.t1.data.clone()
        # Default affine is RAS, so 'Left'/'Right' = axis 0
        result = tio.Flip(axes="Left")(subject)
        expected = torch.flip(original, [1])
        torch.testing.assert_close(result.t1.data, expected)

    def test_string_axis_lr(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        result = tio.Flip(axes="LR")(subject)
        assert result.t1.shape == (1, 4, 4, 4)

    def test_invalid_string_axis(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        with pytest.raises(ValueError, match="Unknown anatomical"):
            tio.Flip(axes="X")(subject)

    def test_flip_invertible(self) -> None:
        assert tio.Flip(axes=0).invertible

    def test_flip_inverse_round_trip(self) -> None:
        """Subject.apply_inverse_transform round-trips a flip."""
        tensor = torch.rand(1, 4, 5, 6)
        subject = tio.Subject(t1=tio.ScalarImage(tensor.clone()))
        flip = tio.Flip(axes=(0, 1, 2))
        flipped = flip(subject)
        restored = flipped.apply_inverse_transform()
        torch.testing.assert_close(restored.t1.data, tensor)

    def test_compose_inverse(self) -> None:
        """Compose inverse via Subject.apply_inverse_transform."""
        tensor = torch.rand(1, 4, 5, 6)
        subject = tio.Subject(t1=tio.ScalarImage(tensor.clone()))
        pipeline = tio.Compose(
            [
                tio.Flip(axes=0),
                tio.Flip(axes=1),
            ]
        )
        transformed = pipeline(subject)
        assert len(transformed.applied_transforms) == 2
        restored = transformed.apply_inverse_transform()
        torch.testing.assert_close(restored.t1.data, tensor)

    def test_inverse_on_image_via_subject(self) -> None:
        """Inverse works by copying history to a new subject."""
        tensor = torch.rand(1, 4, 5, 6)
        subject = tio.Subject(t1=tio.ScalarImage(tensor.clone()))
        flipped = tio.Flip(axes=0)(subject)
        # Create a prediction subject and copy history
        pred_subject = tio.Subject(
            pred=tio.ScalarImage(flipped.t1.data.clone()),
        )
        pred_subject.applied_transforms = flipped.applied_transforms
        restored = pred_subject.apply_inverse_transform()
        torch.testing.assert_close(restored.pred.data, tensor)

    def test_inverse_skips_non_invertible(self) -> None:
        """Non-invertible transforms are skipped with a warning."""
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        pipeline = tio.Compose(
            [
                tio.Flip(axes=0),
                tio.Noise(std=0.1),  # not invertible
            ]
        )
        transformed = pipeline(subject)
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            restored = transformed.apply_inverse_transform()
            assert any("not invertible" in str(x.message) for x in w)
        assert restored.t1.shape == (1, 4, 4, 4)

    def test_ignore_intensity(self) -> None:
        """ignore_intensity=True skips intensity transforms silently."""
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        pipeline = tio.Compose(
            [
                tio.Flip(axes=0),
                tio.Noise(std=0.1),
            ]
        )
        transformed = pipeline(subject)
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            transformed.apply_inverse_transform(ignore_intensity=True)
            # No warning about Noise since intensity is ignored
            assert not any("Noise" in str(x.message) for x in w)
