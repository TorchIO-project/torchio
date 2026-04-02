"""Tests for the Reorient transform."""

from __future__ import annotations

import pytest
import torch

import torchio as tio
from torchio.data.affine import Affine


def _make_subject(
    shape: tuple[int, int, int] = (10, 20, 30),
    orientation: str = "RAS",
) -> tio.Subject:
    """Create a subject with a known orientation."""
    import nibabel as nib

    # Build an affine for the desired orientation
    target_ornt = nib.orientations.axcodes2ornt(tuple(orientation))
    # Start from identity and apply the orientation
    base_affine = nib.orientations.inv_ornt_aff(target_ornt, shape)
    affine = Affine(base_affine)
    data = torch.arange(shape[0] * shape[1] * shape[2], dtype=torch.float32)
    data = data.reshape(1, *shape)
    image = tio.ScalarImage.from_tensor(data, affine=affine)
    return tio.Subject(t1=image)


# ---------------------------------------------------------------------------
# Basic reorientation
# ---------------------------------------------------------------------------


class TestReorientBasic:
    def test_no_op_when_already_target(self) -> None:
        subject = _make_subject(orientation="RAS")
        result = tio.Reorient(orientation="RAS")(subject)
        assert result.t1.affine.orientation == ("R", "A", "S")
        torch.testing.assert_close(result.t1.data, subject.t1.data)

    def test_ras_to_las(self) -> None:
        subject = _make_subject(orientation="RAS")
        result = tio.Reorient(orientation="LAS")(subject)
        assert result.t1.affine.orientation == ("L", "A", "S")

    def test_ras_to_pls(self) -> None:
        subject = _make_subject(orientation="RAS")
        result = tio.Reorient(orientation="PLS")(subject)
        assert result.t1.affine.orientation == ("P", "L", "S")

    def test_default_is_ras(self) -> None:
        subject = _make_subject(orientation="LAS")
        result = tio.Reorient()(subject)
        assert result.t1.affine.orientation == ("R", "A", "S")

    def test_shape_changes_with_permutation(self) -> None:
        subject = _make_subject(shape=(10, 20, 30), orientation="RAS")
        result = tio.Reorient(orientation="ASR")(subject)
        assert result.t1.affine.orientation == ("A", "S", "R")
        # Spatial shape should be a permutation of (10, 20, 30)
        assert sorted(result.t1.spatial_shape) == [10, 20, 30]


# ---------------------------------------------------------------------------
# Round-trip: reorient and back preserves data
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_ras_to_las_and_back(self) -> None:
        subject = _make_subject(orientation="RAS")
        original_data = subject.t1.data.clone()
        to_las = tio.Reorient(orientation="LAS")
        to_ras = tio.Reorient(orientation="RAS")
        result = to_ras(to_las(subject))
        torch.testing.assert_close(result.t1.data, original_data)

    def test_ras_to_spl_and_back(self) -> None:
        subject = _make_subject(orientation="RAS")
        original_data = subject.t1.data.clone()
        result = tio.Reorient(orientation="RAS")(
            tio.Reorient(orientation="SPL")(subject)
        )
        torch.testing.assert_close(result.t1.data, original_data)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_length(self) -> None:
        with pytest.raises(ValueError, match="3-letter"):
            tio.Reorient(orientation="RA")

    def test_invalid_characters(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            tio.Reorient(orientation="XYZ")

    def test_missing_axis(self) -> None:
        with pytest.raises(ValueError, match="each axis"):
            tio.Reorient(orientation="RAA")

    def test_case_insensitive(self) -> None:
        transform = tio.Reorient(orientation="ras")
        assert transform.orientation == "RAS"


# ---------------------------------------------------------------------------
# All images in subject
# ---------------------------------------------------------------------------


class TestAllImages:
    def test_reorients_all_images(self) -> None:
        import nibabel as nib

        ornt = nib.orientations.axcodes2ornt(tuple("RAS"))
        affine = Affine(nib.orientations.inv_ornt_aff(ornt, (10, 20, 30)))
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 10, 20, 30), affine=affine),
            seg=tio.LabelMap.from_tensor(
                torch.randint(0, 3, (1, 10, 20, 30)),
                affine=affine,
            ),
        )
        result = tio.Reorient(orientation="LAS")(subject)
        assert result.t1.affine.orientation == ("L", "A", "S")
        assert result.seg.affine.orientation == ("L", "A", "S")


# ---------------------------------------------------------------------------
# Invertibility
# ---------------------------------------------------------------------------


class TestInvertibility:
    def test_invertible(self) -> None:
        assert tio.Reorient().invertible

    def test_inverse_restores_shape(self) -> None:
        subject = _make_subject(shape=(10, 20, 30), orientation="RAS")
        original_data = subject.t1.data.clone()
        transformed = tio.Reorient(orientation="SPL")(subject)
        restored = transformed.apply_inverse_transform()
        assert restored.t1.spatial_shape == (10, 20, 30)
        torch.testing.assert_close(restored.t1.data, original_data)


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


class TestInputTypes:
    def test_accepts_image(self) -> None:
        import nibabel as nib

        ornt = nib.orientations.axcodes2ornt(tuple("RAS"))
        affine = Affine(nib.orientations.inv_ornt_aff(ornt, (10, 20, 30)))
        image = tio.ScalarImage.from_tensor(
            torch.rand(1, 10, 20, 30),
            affine=affine,
        )
        result = tio.Reorient(orientation="LAS")(image)
        assert isinstance(result, tio.Image)
        assert result.affine.orientation == ("L", "A", "S")

    def test_accepts_subject(self) -> None:
        subject = _make_subject(orientation="RAS")
        result = tio.Reorient(orientation="LAS")(subject)
        assert isinstance(result, tio.Subject)


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_reorient(self) -> None:
        import nibabel as nib

        from torchio.data.batch import SubjectsBatch

        ornt = nib.orientations.axcodes2ornt(tuple("RAS"))
        affine = Affine(nib.orientations.inv_ornt_aff(ornt, (10, 20, 30)))
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage.from_tensor(
                    torch.rand(1, 10, 20, 30),
                    affine=affine,
                ),
            )
            for _ in range(3)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        result = tio.Reorient(orientation="LAS")(batch)
        assert result.t1.data.shape == (3, 1, 10, 20, 30)


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------


class TestProbability:
    def test_p_zero_is_no_op(self) -> None:
        subject = _make_subject(orientation="RAS")
        result = tio.Reorient(orientation="LAS", p=0)(subject)
        assert result.t1.affine.orientation == ("R", "A", "S")
