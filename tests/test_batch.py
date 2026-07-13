"""Tests for ImagesBatch and SubjectsBatch."""

from __future__ import annotations

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

    def test_payload_and_history_round_trip(self) -> None:
        images = [
            tio.ScalarImage(
                torch.rand(1, 4, 4, 4),
                protocol=f"protocol-{index}",
                points={"landmarks": tio.Points(torch.rand(index + 1, 3))},
            )
            for index in range(2)
        ]
        images[0].applied_transforms = [tio.AppliedTransform("Flip", {"axes": (0,)})]
        images[1].applied_transforms = [tio.AppliedTransform("Flip", {"axes": (1,)})]

        restored = ImagesBatch.from_images(images).unbatch()

        assert [image.protocol for image in restored] == [
            "protocol-0",
            "protocol-1",
        ]
        assert restored[0].points["landmarks"].num_points == 1
        assert restored[1].points["landmarks"].num_points == 2
        assert restored[0].applied_transforms[0].params["axes"] == (0,)
        assert restored[1].applied_transforms[0].params["axes"] == (1,)


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
            t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
            seg=tio.LabelMap(torch.zeros(1, 8, 8, 8)),
            age=42,
            name="first",
        )
        second = tio.Subject(
            seg=tio.LabelMap(torch.zeros(1, 8, 8, 8)),
            t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
            name="second",
            age=43,
        )

        batch = SubjectsBatch.from_subjects([first, second])

        assert list(batch.images) == ["t1", "seg"]
        assert list(batch.metadata) == ["age", "name"]

    @pytest.mark.parametrize(
        ("first", "second", "message"),
        [
            (
                tio.Subject(t1=tio.ScalarImage(torch.rand(1, 4, 4, 4))),
                tio.Subject(
                    t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
                    t2=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
                ),
                "image.*unexpected.*t2",
            ),
            (
                tio.Subject(
                    t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
                    age=42,
                ),
                tio.Subject(t1=tio.ScalarImage(torch.rand(1, 4, 4, 4))),
                "metadata.*missing.*age",
            ),
            (
                tio.Subject(
                    t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
                    landmarks=tio.Points(torch.rand(2, 3)),
                ),
                tio.Subject(t1=tio.ScalarImage(torch.rand(1, 4, 4, 4))),
                "point.*missing.*landmarks",
            ),
            (
                tio.Subject(
                    t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
                    tumors=tio.BoundingBoxes(
                        torch.rand(2, 6),
                        format=tio.BoundingBoxFormat.IJKIJK,
                    ),
                ),
                tio.Subject(t1=tio.ScalarImage(torch.rand(1, 4, 4, 4))),
                "bounding box.*missing.*tumors",
            ),
        ],
    )
    def test_incompatible_subject_schema_raises(
        self,
        first: tio.Subject,
        second: tio.Subject,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            SubjectsBatch.from_subjects([first, second])

    def test_mixed_image_classes_raise(self) -> None:
        subjects = [
            tio.Subject(t1=tio.ScalarImage(torch.rand(1, 4, 4, 4))),
            tio.Subject(t1=tio.LabelMap(torch.zeros(1, 4, 4, 4))),
        ]
        with pytest.raises(ValueError, match=r"index 1.*ScalarImage.*LabelMap"):
            SubjectsBatch.from_subjects(subjects)

    def test_mismatched_image_shapes_raise(self) -> None:
        subjects = [
            tio.Subject(t1=tio.ScalarImage(torch.rand(1, 4, 4, 4))),
            tio.Subject(t1=tio.ScalarImage(torch.rand(1, 5, 4, 4))),
        ]
        with pytest.raises(ValueError, match=r"index 1.*shape"):
            SubjectsBatch.from_subjects(subjects)

    def test_mismatched_image_dtypes_raise(self) -> None:
        subjects = [
            tio.Subject(t1=tio.ScalarImage(torch.rand(1, 4, 4, 4))),
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 4, 4, 4, dtype=torch.float64))
            ),
        ]
        with pytest.raises(ValueError, match=r"index 1.*dtype"):
            SubjectsBatch.from_subjects(subjects)

    def test_metadata_only_batch(self) -> None:
        batch = SubjectsBatch.from_subjects(
            [
                tio.Subject(age=42, name="first"),
                tio.Subject(age=43, name="second"),
            ]
        )

        assert batch.batch_size == 2
        assert batch.device.type == "cpu"
        assert batch.metadata == {
            "age": [42, 43],
            "name": ["first", "second"],
        }
        assert [subject.age for subject in batch.unbatch()] == [42, 43]

    def test_annotation_only_batch(self) -> None:
        subjects = [
            tio.Subject(
                landmarks=tio.Points(torch.rand(index + 1, 3)),
                tumors=tio.BoundingBoxes(
                    torch.rand(index + 1, 6),
                    format=tio.BoundingBoxFormat.IJKIJK,
                ),
            )
            for index in range(2)
        ]

        batch = SubjectsBatch.from_subjects(subjects)
        restored = batch.to(torch.float64).unbatch()

        assert batch.batch_size == 2
        assert restored[0].landmarks.data.dtype == torch.float64
        assert restored[1].tumors.data.dtype == torch.float64
        assert restored[0].landmarks.num_points == 1
        assert restored[1].tumors.num_boxes == 2

    def test_subject_annotations_round_trip(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
            landmarks=tio.Points(
                torch.rand(2, 3),
                metadata={"source": "manual"},
            ),
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

    def test_image_payload_round_trip(self) -> None:
        image = tio.ScalarImage(
            torch.rand(1, 4, 4, 4),
            protocol="MPRAGE",
            points={"landmarks": tio.Points(torch.rand(2, 3))},
            bounding_boxes={
                "tumors": tio.BoundingBoxes(
                    torch.rand(2, 6),
                    format=tio.BoundingBoxFormat.IJKIJK,
                )
            },
        )
        subject = tio.Subject(t1=image)

        restored = SubjectsBatch.from_subjects([subject]).unbatch()[0].t1

        assert restored.metadata == {"protocol": "MPRAGE"}
        assert set(restored.points) == {"landmarks"}
        assert set(restored.bounding_boxes) == {"tumors"}

    def test_image_nested_schema_mismatch_raises(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(
                    torch.rand(1, 4, 4, 4),
                    protocol="MPRAGE",
                )
            ),
            tio.Subject(t1=tio.ScalarImage(torch.rand(1, 4, 4, 4))),
        ]

        with pytest.raises(
            ValueError,
            match=r"image 't1'.*metadata.*missing.*protocol",
        ):
            SubjectsBatch.from_subjects(subjects)

    def test_custom_image_subclass_round_trip(self) -> None:
        class CustomScalarImage(tio.ScalarImage):
            pass

        subjects = [
            tio.Subject(
                t1=CustomScalarImage(
                    torch.rand(1, 4, 4, 4),
                    sequence="custom",
                )
            )
            for _ in range(2)
        ]

        restored = SubjectsBatch.from_subjects(subjects).unbatch()

        assert all(type(subject.t1) is CustomScalarImage for subject in restored)
        assert all(subject.t1.sequence == "custom" for subject in restored)

    def test_shared_history_round_trip(self) -> None:
        trace = tio.AppliedTransform("Flip", {"axes": (0,)})
        subjects = [
            tio.Subject(t1=tio.ScalarImage(torch.rand(1, 4, 4, 4))) for _ in range(2)
        ]
        for subject in subjects:
            subject.applied_transforms = [trace]

        batch = SubjectsBatch.from_subjects(subjects)

        assert batch._per_element_history is None
        assert batch.applied_transforms == [trace]
        assert all(subject.applied_transforms == [trace] for subject in batch.unbatch())

    def test_divergent_history_round_trip(self) -> None:
        subjects = [
            tio.Subject(t1=tio.ScalarImage(torch.rand(1, 4, 4, 4))) for _ in range(2)
        ]
        subjects[0].applied_transforms = [tio.AppliedTransform("Flip", {"axes": (0,)})]
        subjects[1].applied_transforms = [tio.AppliedTransform("Flip", {"axes": (1,)})]

        batch = SubjectsBatch.from_subjects(subjects)
        restored = batch.unbatch()

        assert batch._per_element_history is not None
        assert restored[0].applied_transforms[0].params["axes"] == (0,)
        assert restored[1].applied_transforms[0].params["axes"] == (1,)


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
        "subject",
        [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
                landmarks=tio.Points(torch.rand(2, 3)),
            ),
            tio.Subject(
                t1=tio.ScalarImage(
                    torch.rand(1, 4, 4, 4),
                    bounding_boxes={
                        "tumors": tio.BoundingBoxes(
                            torch.rand(2, 6),
                            format=tio.BoundingBoxFormat.IJKIJK,
                        )
                    },
                )
            ),
        ],
    )
    def test_spatial_transform_rejects_annotations(
        self,
        subject: tio.Subject,
    ) -> None:
        with pytest.raises(NotImplementedError, match="annotations"):
            tio.Flip(axes=(0,))(subject)


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

    def test_adopt_history_preserves_per_element(self) -> None:
        # Simulate the adapter pattern: a per-element batch is unbatched,
        # processed, and re-stacked; history must survive.
        torch.manual_seed(0)
        batch = self._batch()
        branched = tio.OneOf([tio.Flip(axes=(0,)), tio.Flip(axes=(1,))])(batch)
        subjects = branched.unbatch()
        rebuilt = SubjectsBatch.from_subjects(subjects)
        rebuilt.adopt_history(branched, subjects)
        for original, restored in zip(
            branched.unbatch(),
            rebuilt.unbatch(),
            strict=True,
        ):
            assert [t.name for t in restored.applied_transforms] == [
                t.name for t in original.applied_transforms
            ]

    def test_adopt_history_shared_case(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        transformed = tio.Gamma(log_gamma=0.3, per_instance=False)(batch)
        subjects = transformed.unbatch()
        rebuilt = SubjectsBatch.from_subjects(subjects)
        rebuilt.adopt_history(transformed, subjects)
        assert rebuilt._per_element_history is None
        for subject in rebuilt.unbatch():
            assert [t.name for t in subject.applied_transforms] == ["Gamma"]


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

        with pytest.raises(ValueError, match=r"index 1.*metadata.*missing.*site"):
            self._batch().map_subjects(add_site_to_first)

    def test_rejects_non_subject_result(self) -> None:
        def return_dict(subject: tio.Subject) -> dict:
            return {"subject": subject}

        with pytest.raises(TypeError, match=r"index 0.*Subject.*dict"):
            self._batch().map_subjects(return_dict)  # type: ignore[arg-type]

    def test_retains_shared_callback_history(self) -> None:
        def flip(subject: tio.Subject) -> tio.Subject:
            return tio.Flip(axes=(0,))(subject)

        result = self._batch().map_subjects(flip)

        assert result._per_element_history is None
        assert [trace.name for trace in result.applied_transforms] == ["Flip"]

    def test_retains_shared_history_with_tensor_params(self) -> None:
        batch = self._batch()
        trace = tio.AppliedTransform(
            "Custom",
            {"value": torch.tensor([1, 2])},
        )
        batch.applied_transforms = [trace]

        result = batch.map_subjects(lambda subject: subject)

        assert result._per_element_history is None
        assert result.applied_transforms == [trace]

    def test_retains_divergent_callback_history(self) -> None:
        def flip_by_index(subject: tio.Subject) -> tio.Subject:
            return tio.Flip(axes=(subject.index,))(subject)

        result = self._batch().map_subjects(flip_by_index)
        histories = [subject.applied_transforms for subject in result.unbatch()]

        assert result._per_element_history is not None
        assert histories[0][0].params["axes"] == (0,)
        assert histories[1][0].params["axes"] == (1,)

    def test_in_place_callback_does_not_mutate_input(self) -> None:
        batch = self._batch()

        def add_in_place(subject: tio.Subject) -> tio.Subject:
            subject.t1.data.add_(1)
            return subject

        result = batch.map_subjects(add_in_place)

        assert torch.count_nonzero(batch.t1.data) == 0
        assert torch.all(result.t1.data == 1)
