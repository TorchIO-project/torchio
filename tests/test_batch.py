"""Tests for ImagesBatch and SubjectsBatch."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch

import torchio as tio
from torchio.data.batch import ImagesBatch
from torchio.data.batch import SubjectsBatch


class TestImagesBatch:
    def test_from_images(self) -> None:
        images = [tio.ScalarImage(torch.rand(1, 8, 8, 8)) for _ in range(4)]
        batch = ImagesBatch.from_images(images)
        assert batch.data.shape == (4, 1, 8, 8, 8)

    def test_from_tensor(self) -> None:
        data = torch.rand(3, 1, 4, 5, 6)

        batch = ImagesBatch.from_tensor(data)

        assert batch.data is data
        assert batch.batch_size == 3
        assert all(affine.spacing == (1.0, 1.0, 1.0) for affine in batch.affines)

    def test_from_tensor_default_affines_follow_data_device(self) -> None:
        batch = ImagesBatch.from_tensor(torch.empty(2, 1, 2, 3, 4, device="meta"))

        assert all(affine.device.type == "meta" for affine in batch.affines)

    def test_image_class_properties(self) -> None:
        scalar = ImagesBatch.from_tensor(
            torch.rand(2, 1, 4, 4, 4),
            image_class=tio.ScalarImage,
        )
        label = ImagesBatch.from_tensor(
            torch.zeros(2, 1, 4, 4, 4),
            image_class=tio.LabelMap,
        )

        assert scalar.image_class is tio.ScalarImage
        assert scalar.is_label is False
        assert label.image_class is tio.LabelMap
        assert label.is_label is True

    def test_payload_round_trip(self) -> None:
        images = [
            tio.ScalarImage(
                torch.rand(1, 4, 4, 4),
                protocol=f"protocol-{index}",
                points={"landmarks": tio.Points(torch.rand(index + 1, 3))},
                bounding_boxes={
                    "tumors": tio.BoundingBoxes(
                        torch.rand(index + 1, 6),
                        format=tio.BoundingBoxFormat.IJKIJK,
                    )
                },
            )
            for index in range(2)
        ]

        restored = ImagesBatch.from_images(images).unbatch()

        assert [image.protocol for image in restored] == [
            "protocol-0",
            "protocol-1",
        ]
        assert restored[0].points["landmarks"].num_points == 1
        assert restored[1].bounding_boxes["tumors"].num_boxes == 2

    def test_per_image_history_round_trip(self) -> None:
        images = [
            tio.ScalarImage(torch.rand(1, 4, 4, 4)),
            tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        ]
        images[0].applied_transforms = [tio.AppliedTransform("Flip", {"axes": (0,)})]
        images[1].applied_transforms = [tio.AppliedTransform("Flip", {"axes": (1,)})]

        restored = ImagesBatch.from_images(images).unbatch()

        assert restored[0].applied_transforms[0].params["axes"] == (0,)
        assert restored[1].applied_transforms[0].params["axes"] == (1,)

    def test_custom_image_subclass_round_trip(self) -> None:
        class CustomScalarImage(tio.ScalarImage):
            pass

        images = [
            CustomScalarImage(torch.rand(1, 4, 4, 4), sequence="custom")
            for _ in range(2)
        ]

        restored = ImagesBatch.from_images(images).unbatch()

        assert all(type(image) is CustomScalarImage for image in restored)
        assert all(image.sequence == "custom" for image in restored)

    def test_template_does_not_share_affine(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 4, 4, 4))

        batch = ImagesBatch.from_images([image])

        assert batch._prototypes[0].affine is not image.affine

    def test_batch_size(self) -> None:
        batch = ImagesBatch(
            data=torch.rand(4, 1, 8, 8, 8),
            affines=[tio.AffineMatrix() for _ in range(4)],
        )
        assert batch.batch_size == 4

    def test_to_device(self) -> None:
        batch = ImagesBatch(
            data=torch.rand(2, 1, 4, 4, 4),
            affines=[tio.AffineMatrix() for _ in range(2)],
        )
        result = batch.to(torch.float64)
        assert result.data.dtype == torch.float64

    def test_unbatch(self) -> None:
        images = [tio.ScalarImage(torch.rand(1, 8, 8, 8)) for _ in range(3)]
        batch = ImagesBatch.from_images(images)
        restored = batch.unbatch()
        assert len(restored) == 3
        for img in restored:
            assert isinstance(img, tio.ScalarImage)
            assert img.shape == (1, 8, 8, 8)

    def test_getitem_int(self) -> None:
        batch = ImagesBatch(
            data=torch.rand(4, 1, 8, 8, 8),
            affines=[tio.AffineMatrix() for _ in range(4)],
        )
        img = batch[0]
        assert isinstance(img, tio.ScalarImage)
        assert img.shape == (1, 8, 8, 8)

    def test_per_sample_affines(self) -> None:
        affine_a = tio.AffineMatrix.from_spacing((1.0, 1.0, 1.0))
        affine_b = tio.AffineMatrix.from_spacing((2.0, 2.0, 2.0))
        batch = ImagesBatch(
            data=torch.rand(2, 1, 8, 8, 8),
            affines=[affine_a, affine_b],
        )
        assert batch[0].affine.spacing == (1.0, 1.0, 1.0)
        assert batch[1].affine.spacing == (2.0, 2.0, 2.0)

    def test_repr(self) -> None:
        batch = ImagesBatch(
            data=torch.rand(4, 1, 8, 8, 8),
            affines=[tio.AffineMatrix() for _ in range(4)],
        )
        assert "4" in repr(batch)
        assert "8" in repr(batch)


class TestSubjectsBatch:
    def test_from_subjects(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
                seg=tio.LabelMap(torch.randint(0, 3, (1, 8, 8, 8))),
                age=42 + i,
            )
            for i in range(4)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        assert batch["t1"].data.shape == (4, 1, 8, 8, 8)
        assert batch["seg"].data.shape == (4, 1, 8, 8, 8)

    def test_attribute_access(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
            )
            for _ in range(2)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        assert batch.t1.data.shape == (2, 1, 8, 8, 8)

    def test_batch_size(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
            )
            for _ in range(3)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        assert batch.batch_size == 3

    def test_unbatch(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
                age=42 + i,
            )
            for i in range(3)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        restored = batch.unbatch()
        assert len(restored) == 3
        for i, sub in enumerate(restored):
            assert isinstance(sub, tio.Subject)
            assert sub.t1.shape == (1, 8, 8, 8)
            assert sub.age == 42 + i

    def test_to_device(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
            )
            for _ in range(2)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        result = batch.to(torch.float64)
        assert result.t1.data.dtype == torch.float64

    def test_metadata_preserved(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
                age=42 + i,
                name=f"sub_{i}",
            )
            for i in range(3)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        assert batch.metadata["age"] == [42, 43, 44]
        assert batch.metadata["name"] == ["sub_0", "sub_1", "sub_2"]

    def test_reordered_schema_is_accepted(self) -> None:
        first = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
            seg=tio.LabelMap(torch.zeros(1, 4, 4, 4)),
            age=42,
            name="first",
        )
        second = tio.Subject(
            seg=tio.LabelMap(torch.zeros(1, 4, 4, 4)),
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
            name="second",
            age=43,
        )

        batch = SubjectsBatch.from_subjects([first, second])

        assert list(batch.images) == ["t1", "seg"]
        assert list(batch.metadata) == ["age", "name"]

    def test_loaded_and_lazy_equivalent_dtypes_are_accepted(self) -> None:
        loaded = tio.ScalarImage(torch.zeros(1, 4, 4, 4, dtype=torch.float32))
        lazy = tio.ScalarImage(
            nib.Nifti1Image(
                np.zeros((4, 4, 4), dtype=np.float32),
                np.eye(4),
            )
        )

        batch = SubjectsBatch.from_subjects(
            [tio.Subject(image=loaded), tio.Subject(image=lazy)]
        )

        assert batch.image.data.dtype == torch.float32

    @pytest.mark.parametrize(
        ("first", "second", "message"),
        [
            (
                tio.Subject(t1=tio.ScalarImage(torch.rand(1, 4, 4, 4))),
                tio.Subject(
                    t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
                    t2=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
                ),
                "Subject images.*unexpected.*t2",
            ),
            (
                tio.Subject(
                    t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
                    age=42,
                ),
                tio.Subject(t1=tio.ScalarImage(torch.rand(1, 4, 4, 4))),
                "Subject metadata.*missing.*age",
            ),
            (
                tio.Subject(
                    t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
                    landmarks=tio.Points(torch.rand(2, 3)),
                ),
                tio.Subject(t1=tio.ScalarImage(torch.rand(1, 4, 4, 4))),
                "Subject points.*missing.*landmarks",
            ),
        ],
    )
    def test_incompatible_schema_raises(
        self,
        first: tio.Subject,
        second: tio.Subject,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            SubjectsBatch.from_subjects([first, second])

    def test_metadata_only_batch(self) -> None:
        batch = SubjectsBatch.from_subjects([tio.Subject(age=42), tio.Subject(age=43)])

        assert batch.batch_size == 2
        assert batch.device.type == "cpu"
        assert [subject.age for subject in batch.unbatch()] == [42, 43]

    def test_annotation_only_batch(self) -> None:
        subjects = [
            tio.Subject(landmarks=tio.Points(torch.rand(index + 1, 3)))
            for index in range(2)
        ]

        restored = SubjectsBatch.from_subjects(subjects).unbatch()

        assert restored[0].landmarks.num_points == 1
        assert restored[1].landmarks.num_points == 2

    def test_subject_annotations_round_trip(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
            landmarks=tio.Points(torch.rand(2, 3), metadata={"source": "manual"}),
            tumors=tio.BoundingBoxes(
                torch.rand(3, 6),
                format=tio.BoundingBoxFormat.IJKWHD,
                metadata={"reader": "test"},
            ),
        )

        restored = SubjectsBatch.from_subjects([subject]).unbatch()[0]

        assert restored.landmarks.metadata == {"source": "manual"}
        assert restored.tumors.metadata == {"reader": "test"}
        assert restored.tumors.format == tio.BoundingBoxFormat.IJKWHD

    def test_metadata_remains_mutable(self) -> None:
        batch = SubjectsBatch.from_subjects([tio.Subject(age=42), tio.Subject(age=43)])

        batch.metadata["age"][0] = 50
        batch.metadata["site"] = ["A", "B"]

        assert batch.unbatch()[0].age == 50
        assert batch.unbatch()[1].site == "B"


class TestBatchTransforms:
    def test_flip_images_batch(self) -> None:
        images = [
            tio.ScalarImage(torch.arange(8).reshape(1, 2, 2, 2).float())
            for _ in range(3)
        ]
        batch = ImagesBatch.from_images(images)
        original = batch.data.clone()
        result = tio.Flip(axes=(0,))(batch)
        assert isinstance(result, ImagesBatch)
        assert result.data.shape == (3, 1, 2, 2, 2)
        # Check data was actually flipped
        assert not torch.equal(result.data, original)

    def test_flip_subjects_batch(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
            )
            for _ in range(4)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        result = tio.Flip(axes=(0,))(batch)
        assert isinstance(result, SubjectsBatch)
        assert result.t1.data.shape == (4, 1, 8, 8, 8)

    def test_noise_images_batch(self) -> None:
        images = [tio.ScalarImage(torch.zeros(1, 4, 4, 4)) for _ in range(3)]
        batch = ImagesBatch.from_images(images)
        result = tio.Noise(std=1.0)(batch)
        # Noise should have been added
        assert result.data.abs().sum() > 0

    def test_batch_preserves_affines(self) -> None:
        affine_a = tio.AffineMatrix.from_spacing((1.0, 1.0, 1.0))
        affine_b = tio.AffineMatrix.from_spacing((2.0, 2.0, 2.0))
        images = [
            tio.ScalarImage(torch.rand(1, 8, 8, 8), affine=affine_a),
            tio.ScalarImage(torch.rand(1, 8, 8, 8), affine=affine_b),
        ]
        batch = ImagesBatch.from_images(images)
        result = tio.Flip(axes=(0,))(batch)
        assert result.affines[0].spacing == (1.0, 1.0, 1.0)
        assert result.affines[1].spacing == (2.0, 2.0, 2.0)

    def test_batch_copy_preserves_original(self) -> None:
        images = [tio.ScalarImage(torch.zeros(1, 4, 4, 4)) for _ in range(2)]
        batch = ImagesBatch.from_images(images)
        original = batch.data.clone()
        tio.Noise(std=1.0)(batch)
        # Original should be unchanged (copy=True default)
        torch.testing.assert_close(batch.data, original)

    def test_intensity_transform_preserves_annotations(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(
                torch.rand(1, 4, 4, 4) + 0.1,
                points={"image_landmarks": tio.Points(torch.rand(2, 3))},
            ),
            landmarks=tio.Points(torch.rand(3, 3)),
        )

        result = tio.Gamma(log_gamma=0.2)(subject)

        assert set(result.points) == {"landmarks"}
        assert set(result.t1.points) == {"image_landmarks"}

    @pytest.mark.parametrize(
        "data",
        [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
                landmarks=tio.Points(torch.rand(2, 3)),
            ),
            {
                "t1": torch.rand(1, 4, 4, 4),
                "landmarks": tio.Points(torch.rand(2, 3)),
            },
        ],
    )
    def test_spatial_transform_rejects_annotations(self, data: object) -> None:
        with pytest.raises(NotImplementedError, match="annotations"):
            tio.Flip(axes=(0,))(data)

    @pytest.mark.parametrize(
        "transform",
        [
            tio.Flip(axes=(0,), p=0),
            tio.CropOrPad(2, p=0),
        ],
    )
    def test_skipped_spatial_transform_allows_annotations(
        self,
        transform: tio.Transform,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        subject = tio.Subject(
            image=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
            landmarks=tio.Points(torch.rand(2, 3)),
        )
        monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.zeros(1))

        result = transform(subject)

        assert set(result.points) == {"landmarks"}


# ── Coverage gap tests ───────────────────────────────────────────────


class TestImagesBatchValidation:
    def test_non_5d_raises(self) -> None:
        from torchio.data.batch import ImagesBatch

        with pytest.raises(ValueError, match="5"):
            ImagesBatch(
                data=torch.rand(1, 10, 10),
                affines=[tio.AffineMatrix()],
                image_class=tio.ScalarImage,
            )

    def test_affine_count_mismatch_raises(self) -> None:
        from torchio.data.batch import ImagesBatch

        with pytest.raises(ValueError, match="affines"):
            ImagesBatch(
                data=torch.rand(2, 1, 5, 5, 5),
                affines=[tio.AffineMatrix()],  # only 1 for batch of 2
                image_class=tio.ScalarImage,
            )

    def test_from_images_empty_raises(self) -> None:
        from torchio.data.batch import ImagesBatch

        with pytest.raises(ValueError, match="empty"):
            ImagesBatch.from_images([])

    def test_empty_history_view_is_defensive(self) -> None:
        batch = object.__new__(ImagesBatch)
        batch._data = torch.empty(0, 1, 1, 1, 1)
        batch._histories = []

        assert batch.has_divergent_history is False
        assert batch.applied_transforms == ()

    def test_data_setter_non_5d_raises(self) -> None:
        from torchio.data.batch import ImagesBatch

        batch = ImagesBatch(
            data=torch.rand(1, 1, 5, 5, 5),
            affines=[tio.AffineMatrix()],
            image_class=tio.ScalarImage,
        )
        with pytest.raises(ValueError, match="5"):
            batch.data = torch.rand(1, 5, 5)

    def test_device_property(self) -> None:
        from torchio.data.batch import ImagesBatch

        batch = ImagesBatch(
            data=torch.rand(1, 1, 5, 5, 5),
            affines=[tio.AffineMatrix()],
            image_class=tio.ScalarImage,
        )
        assert batch.device.type == "cpu"

    def test_len(self) -> None:
        from torchio.data.batch import ImagesBatch

        batch = ImagesBatch(
            data=torch.rand(3, 1, 5, 5, 5),
            affines=[tio.AffineMatrix() for _ in range(3)],
            image_class=tio.ScalarImage,
        )
        assert len(batch) == 3


class TestSubjectsBatchEdgeCases:
    def test_from_subjects_empty_raises(self) -> None:
        from torchio.data.batch import SubjectsBatch

        with pytest.raises(ValueError, match="empty"):
            SubjectsBatch.from_subjects([])

    def test_device_property(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 5, 5, 5)))
        from torchio.data.batch import SubjectsBatch

        batch = SubjectsBatch.from_subjects([subject])
        assert batch.device.type == "cpu"

    def test_getattr_invalid_raises(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 5, 5, 5)))
        from torchio.data.batch import SubjectsBatch

        batch = SubjectsBatch.from_subjects([subject])
        with pytest.raises(AttributeError):
            _ = batch.nonexistent_image

    def test_len(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 5, 5, 5)))
        from torchio.data.batch import SubjectsBatch

        batch = SubjectsBatch.from_subjects([subject])
        assert len(batch) == 1

    def test_repr(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 5, 5, 5)))
        from torchio.data.batch import SubjectsBatch

        batch = SubjectsBatch.from_subjects([subject])
        r = repr(batch)
        assert "SubjectsBatch" in r
        assert "t1" in r


class TestPerElementHistory:
    def _batch(self, batch_size: int = 4) -> SubjectsBatch:
        subjects = [
            tio.Subject(t1=tio.ScalarImage(torch.rand(1, 6, 6, 6)))
            for _ in range(batch_size)
        ]
        return SubjectsBatch.from_subjects(subjects)

    def test_restack_preserves_divergent_history(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        branched = tio.OneOf([tio.Flip(axes=(0,)), tio.Flip(axes=(1,))])(batch)
        subjects = branched.unbatch()
        rebuilt = SubjectsBatch.from_subjects(subjects)
        for original, restored in zip(
            branched.unbatch(),
            rebuilt.unbatch(),
            strict=True,
        ):
            assert [t.name for t in restored.applied_transforms] == [
                t.name for t in original.applied_transforms
            ]

    def test_restack_preserves_uniform_history(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        transformed = tio.Gamma(log_gamma=0.3, per_instance=False)(batch)
        subjects = transformed.unbatch()
        rebuilt = SubjectsBatch.from_subjects(subjects)
        assert not rebuilt.has_divergent_history
        for subject in rebuilt.unbatch():
            assert [t.name for t in subject.applied_transforms] == ["Gamma"]

    def test_uniform_applied_transforms_view_is_immutable(self) -> None:
        result = tio.Gamma(log_gamma=0.2, per_instance=False)(self._batch())

        assert isinstance(result.applied_transforms, tuple)
        with pytest.raises(AttributeError):
            result.applied_transforms.append("invalid")  # type: ignore[attr-defined]


class TestMapSubjects:
    def _batch(self) -> SubjectsBatch:
        return SubjectsBatch.from_subjects(
            [
                tio.Subject(
                    t1=tio.ScalarImage(torch.zeros(1, 4, 4, 4)),
                    identifier=f" subject-{index} ",
                    index=index,
                )
                for index in range(2)
            ]
        )

    def test_maps_text_metadata(self) -> None:
        batch = self._batch()

        def strip_identifier(subject: tio.Subject) -> tio.Subject:
            subject.metadata["identifier"] = subject.identifier.strip()
            return subject

        result = batch.map_subjects(strip_identifier)

        assert result.metadata["identifier"] == ["subject-0", "subject-1"]
        assert batch.metadata["identifier"] == [" subject-0 ", " subject-1 "]

    def test_preserves_annotations(self) -> None:
        batch = SubjectsBatch.from_subjects(
            [
                tio.Subject(
                    landmarks=tio.Points(torch.rand(index + 1, 3)),
                    identifier=f"subject-{index}",
                )
                for index in range(2)
            ]
        )

        result = batch.map_subjects(lambda subject: subject)

        assert result.points["landmarks"][0].num_points == 1
        assert result.points["landmarks"][1].num_points == 2

    def test_allows_uniform_schema_change(self) -> None:
        def add_site(subject: tio.Subject) -> tio.Subject:
            subject.metadata["site"] = "A"
            return subject

        result = self._batch().map_subjects(add_site)

        assert result.metadata["site"] == ["A", "A"]

    def test_rejects_divergent_schema_change(self) -> None:
        def add_site_to_first(subject: tio.Subject) -> tio.Subject:
            if subject.index == 0:
                subject.metadata["site"] = "A"
            return subject

        with pytest.raises(ValueError, match=r"metadata.*index 1.*missing.*site"):
            self._batch().map_subjects(add_site_to_first)

    def test_rejects_non_subject_result(self) -> None:
        def return_dict(subject: tio.Subject) -> dict:
            return {"subject": subject}

        with pytest.raises(TypeError, match=r"index 0.*Subject.*dict"):
            self._batch().map_subjects(return_dict)  # type: ignore[arg-type]

    def test_retains_callback_history(self) -> None:
        def flip_by_index(subject: tio.Subject) -> tio.Subject:
            return tio.Flip(axes=(subject.index,))(subject)

        result = self._batch().map_subjects(flip_by_index)

        assert result.has_divergent_history
        assert result.history(0)[0].params["axes"] == (0,)
        assert result.history(1)[0].params["axes"] == (1,)

    def test_in_place_callback_does_not_mutate_input(self) -> None:
        batch = self._batch()

        def add_in_place(subject: tio.Subject) -> tio.Subject:
            subject.t1.data.add_(1)
            return subject

        result = batch.map_subjects(add_in_place)

        assert torch.count_nonzero(batch.t1.data) == 0
        assert torch.all(result.t1.data == 1)

    def test_copy_false_allows_in_place_mutation(self) -> None:
        batch = self._batch()

        def add_in_place(subject: tio.Subject) -> tio.Subject:
            subject.t1.data.add_(1)
            return subject

        result = batch.map_subjects(add_in_place, copy=False)

        assert torch.all(batch.t1.data == 1)
        assert torch.all(result.t1.data == 1)
