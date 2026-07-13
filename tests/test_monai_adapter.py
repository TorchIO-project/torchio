"""Tests for MonaiAdapter transform."""

from __future__ import annotations

import pytest
import torch

import torchio as tio


def _monai_available() -> bool:
    try:
        import monai  # noqa: F401

        return True
    except ImportError:
        return False


HAS_MONAI = _monai_available()


@pytest.mark.skipif(not HAS_MONAI, reason="MONAI not installed")
class TestMonaiAdapterArray:
    def test_array_transform(self) -> None:
        from monai.transforms import NormalizeIntensity

        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 8, 8, 8) + 1),
        )
        adapter = tio.MonaiAdapter(NormalizeIntensity())
        result = adapter(subject)
        # NormalizeIntensity should zero-mean the data
        assert abs(result.t1.data.mean().item()) < 0.5

    def test_array_respects_include(self) -> None:
        from monai.transforms import NormalizeIntensity

        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 8, 8, 8) + 5),
            t2=tio.ScalarImage(torch.rand(1, 8, 8, 8) + 5),
        )
        original_t2 = subject.t2.data.clone()
        adapter = tio.MonaiAdapter(NormalizeIntensity(), include=["t1"])
        result = adapter(subject)
        torch.testing.assert_close(result.t2.data, original_t2)

    def test_array_skips_label_maps(self) -> None:
        from monai.transforms import NormalizeIntensity

        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
            seg=tio.LabelMap(torch.randint(0, 3, (1, 8, 8, 8))),
        )
        original_seg = subject.seg.data.clone()
        adapter = tio.MonaiAdapter(NormalizeIntensity())
        result = adapter(subject)
        torch.testing.assert_close(result.seg.data, original_seg)


@pytest.mark.skipif(not HAS_MONAI, reason="MONAI not installed")
class TestMonaiAdapterDict:
    def test_dict_transform(self) -> None:
        from monai.transforms import NormalizeIntensityd

        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 8, 8, 8) + 1),
        )
        adapter = tio.MonaiAdapter(NormalizeIntensityd(keys=["t1"]))
        result = adapter(subject)
        assert abs(result.t1.data.mean().item()) < 0.5

    def test_dict_only_modifies_specified_keys(self) -> None:
        from monai.transforms import NormalizeIntensityd

        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 8, 8, 8) + 5),
            t2=tio.ScalarImage(torch.rand(1, 8, 8, 8) + 5),
        )
        original_t2 = subject.t2.data.clone()
        adapter = tio.MonaiAdapter(NormalizeIntensityd(keys=["t1"]))
        result = adapter(subject)
        torch.testing.assert_close(result.t2.data, original_t2)

    def test_dict_updates_metadata(self) -> None:
        from monai.transforms import MapTransform

        class UpdateMetadata(MapTransform):
            def __init__(self) -> None:
                super().__init__(keys=["t1"])

            def __call__(self, data):
                return {**data, "site": data["site"].lower()}

        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
            site="ABC",
        )

        result = tio.MonaiAdapter(UpdateMetadata())(subject)

        assert result.site == "abc"

    def test_dict_rejects_added_tensor_key(self) -> None:
        from monai.transforms import MapTransform

        class AddTensor(MapTransform):
            def __init__(self) -> None:
                super().__init__(keys=["t1"])

            def __call__(self, data):
                return {**data, "new_image": torch.zeros(1, 8, 8, 8)}

        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)))

        with pytest.raises(ValueError, match=r"new tensor field.*new_image"):
            tio.MonaiAdapter(AddTensor())(subject)


@pytest.mark.skipif(not HAS_MONAI, reason="MONAI not installed")
class TestMonaiAdapterGeneral:
    def test_history_not_recorded(self) -> None:
        from monai.transforms import NormalizeIntensity

        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
        )
        adapter = tio.MonaiAdapter(NormalizeIntensity())
        result = adapter(subject)
        # MonaiAdapter should not record itself in history
        assert len(result.applied_transforms) == 0

    def test_accepts_image(self) -> None:
        from monai.transforms import NormalizeIntensity

        image = tio.ScalarImage(torch.rand(1, 8, 8, 8) + 1)
        result = tio.MonaiAdapter(NormalizeIntensity())(image)
        assert isinstance(result, tio.Image)

    def test_not_callable_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            tio.MonaiAdapter("not a transform")  # type: ignore[arg-type]

    def test_in_compose(self) -> None:
        from monai.transforms import NormalizeIntensity

        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 8, 8, 8) + 1),
        )
        pipeline = tio.Compose([tio.MonaiAdapter(NormalizeIntensity())])
        result = pipeline(subject)
        assert isinstance(result, tio.Subject)

    def test_preserves_prior_history_and_annotations(self) -> None:
        from monai.transforms import NormalizeIntensity

        subject = tio.Gamma(log_gamma=0.2)(
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 8, 8, 8) + 1),
                landmarks=tio.Points(torch.rand(2, 3)),
            )
        )

        result = tio.MonaiAdapter(NormalizeIntensity())(subject)

        assert [trace.name for trace in result.applied_transforms] == ["Gamma"]
        assert set(result.points) == {"landmarks"}

    def test_probability_zero_when_random_draw_is_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from monai.transforms import NormalizeIntensity

        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 8, 8, 8) + 1))
        original = subject.t1.data.clone()
        monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.zeros(1))

        result = tio.MonaiAdapter(NormalizeIntensity(), p=0)(subject)

        torch.testing.assert_close(result.t1.data, original)

    def test_copy_false_allows_in_place_transform(self) -> None:
        class AddInPlace:
            def __call__(self, tensor):
                return tensor.add_(1)

        batch = tio.SubjectsBatch.from_subjects(
            [tio.Subject(t1=tio.ScalarImage(torch.zeros(1, 4, 4, 4)))]
        )

        result = tio.MonaiAdapter(AddInPlace(), copy=False)(batch)

        assert torch.all(batch.t1.data == 1)
        assert torch.all(result.t1.data == 1)
