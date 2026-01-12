"""Tests for custom Image subclass support in spatial transforms."""

import torch
import pytest
import torchio as tio


class HistoryScalarImage(tio.ScalarImage):
    """Custom ScalarImage subclass that stores history metadata."""

    def __init__(self, tensor, affine, history, **kwargs):
        super().__init__(tensor=tensor, affine=affine, **kwargs)
        self.history = history

    def new_like(self, tensor, affine=None):
        """Override new_like to preserve history attribute."""
        return type(self)(
            tensor=tensor,
            affine=affine if affine is not None else self.affine,
            history=self.history,  # Preserve custom attribute
            check_nans=self.check_nans,
            reader=self.reader,
        )


class MetadataLabelMap(tio.LabelMap):
    """Custom LabelMap subclass that stores metadata."""

    def __init__(self, tensor, affine, metadata=None, **kwargs):
        super().__init__(tensor=tensor, affine=affine, **kwargs)
        self.metadata = metadata or {}

    def new_like(self, tensor, affine=None):
        """Override new_like to preserve metadata attribute."""
        return type(self)(
            tensor=tensor,
            affine=affine if affine is not None else self.affine,
            metadata=self.metadata,  # Preserve custom attribute
            check_nans=self.check_nans,
            reader=self.reader,
        )


class CustomImageWithoutNewLike(tio.ScalarImage):
    """Custom ScalarImage subclass without new_like override (tests fallback)."""

    def __init__(self, tensor, affine, custom_attr, **kwargs):
        super().__init__(tensor=tensor, affine=affine, **kwargs)
        self.custom_attr = custom_attr


class TestCustomImageSubclassSupport:
    """Test that spatial transforms work with custom Image subclasses."""

    @pytest.fixture
    def history_image(self):
        """Create a HistoryScalarImage for testing."""
        return HistoryScalarImage(
            tensor=torch.rand(1, 10, 10, 10), affine=torch.eye(4), history=['created']
        )

    @pytest.fixture
    def metadata_label(self):
        """Create a MetadataLabelMap for testing."""
        return MetadataLabelMap(
            tensor=torch.randint(0, 3, (1, 8, 8, 8)),
            affine=torch.eye(4),
            metadata={'id': 123, 'name': 'test_label'},
        )

    @pytest.fixture
    def custom_without_new_like(self):
        """Create a custom image without new_like override."""
        return CustomImageWithoutNewLike(
            tensor=torch.rand(1, 6, 6, 6), affine=torch.eye(4), custom_attr='test_value'
        )

    def test_crop_with_custom_subclass(self, history_image):
        """Test Crop transform with custom Image subclass."""
        subject = tio.Subject(image=history_image)
        transform = tio.Crop(cropping=2)

        result = transform(subject)

        # Check type preservation
        assert isinstance(result.image, HistoryScalarImage)
        assert type(result.image) is HistoryScalarImage

        # Check custom attribute preservation
        assert result.image.history == ['created']

        # Check shape change
        assert result.image.shape == (1, 6, 6, 6)

    def test_pad_with_custom_subclass(self, history_image):
        """Test Pad transform with custom Image subclass."""
        subject = tio.Subject(image=history_image)
        transform = tio.Pad(padding=2)

        result = transform(subject)

        # Check type preservation
        assert isinstance(result.image, HistoryScalarImage)
        assert type(result.image) is HistoryScalarImage

        # Check custom attribute preservation
        assert result.image.history == ['created']

        # Check shape change
        assert result.image.shape == (1, 14, 14, 14)

    def test_crop_or_pad_with_custom_subclass(self, history_image):
        """Test CropOrPad transform with custom Image subclass."""
        subject = tio.Subject(image=history_image)
        transform = tio.CropOrPad(target_shape=(8, 8, 8))

        result = transform(subject)

        # Check type preservation
        assert isinstance(result.image, HistoryScalarImage)
        assert type(result.image) is HistoryScalarImage

        # Check custom attribute preservation
        assert result.image.history == ['created']

        # Check shape change
        assert result.image.shape == (1, 8, 8, 8)

    def test_resample_with_custom_subclass(self, history_image):
        """Test Resample transform with custom Image subclass."""
        subject = tio.Subject(image=history_image)
        transform = tio.Resample(target=(2.0, 2.0, 2.0))

        result = transform(subject)

        # Check type preservation
        assert isinstance(result.image, HistoryScalarImage)
        assert type(result.image) is HistoryScalarImage

        # Check custom attribute preservation
        assert result.image.history == ['created']

    def test_to_orientation_with_custom_subclass(self, history_image):
        """Test ToOrientation transform with custom Image subclass."""
        subject = tio.Subject(image=history_image)
        transform = tio.ToOrientation('RAS')

        result = transform(subject)

        # Check type preservation
        assert isinstance(result.image, HistoryScalarImage)
        assert type(result.image) is HistoryScalarImage

        # Check custom attribute preservation
        assert result.image.history == ['created']

    def test_transpose_with_custom_subclass(self, history_image):
        """Test Transpose transform with custom Image subclass."""
        subject = tio.Subject(image=history_image)
        transform = tio.Transpose()

        result = transform(subject)

        # Check type preservation
        assert isinstance(result.image, HistoryScalarImage)
        assert type(result.image) is HistoryScalarImage

        # Check custom attribute preservation
        assert result.image.history == ['created']

    def test_label_map_custom_subclass(self, metadata_label):
        """Test that custom LabelMap subclasses work."""
        subject = tio.Subject(label=metadata_label)
        transform = tio.Crop(cropping=1)

        result = transform(subject)

        # Check type preservation
        assert isinstance(result.label, MetadataLabelMap)
        assert type(result.label) is MetadataLabelMap

        # Check custom attribute preservation
        assert result.label.metadata == {'id': 123, 'name': 'test_label'}

        # Check shape change
        assert result.label.shape == (1, 6, 6, 6)

    def test_fallback_without_new_like_override(self, custom_without_new_like):
        """Test that custom subclasses work even without new_like override (fallback)."""
        subject = tio.Subject(image=custom_without_new_like)
        transform = tio.Crop(cropping=1)

        result = transform(subject)

        # Check type preservation
        assert isinstance(result.image, CustomImageWithoutNewLike)
        assert type(result.image) is CustomImageWithoutNewLike

        # Check custom attribute preservation (via deepcopy fallback)
        assert result.image.custom_attr == 'test_value'

        # Check shape change
        assert result.image.shape == (1, 4, 4, 4)

    def test_new_like_method_directly(self, history_image):
        """Test the new_like method directly."""
        new_tensor = torch.rand(1, 5, 5, 5)
        new_affine = torch.eye(4) * 2

        # Create new image using new_like
        new_image = history_image.new_like(tensor=new_tensor, affine=new_affine)

        # Check type preservation
        assert isinstance(new_image, HistoryScalarImage)
        assert type(new_image) is HistoryScalarImage

        # Check custom attribute preservation
        assert new_image.history == ['created']

        # Check tensor and affine
        assert torch.allclose(new_image.data, new_tensor)
        assert torch.allclose(
            torch.tensor(new_image.affine).float(), new_affine.float()
        )

    def test_new_like_with_default_affine(self, metadata_label):
        """Test new_like method with default affine (None)."""
        new_tensor = torch.rand(1, 8, 8, 8)

        # Create new image using new_like with default affine
        new_image = metadata_label.new_like(tensor=new_tensor)

        # Check that original affine is used
        assert torch.allclose(
            torch.tensor(new_image.affine).float(),
            torch.tensor(metadata_label.affine).float(),
        )

    def test_multiple_custom_images_in_subject(self):
        """Test subject with multiple custom images."""
        history_img = HistoryScalarImage(
            tensor=torch.rand(1, 8, 8, 8), affine=torch.eye(4), history=['img1']
        )
        metadata_label = MetadataLabelMap(
            tensor=torch.randint(0, 2, (1, 8, 8, 8)),
            affine=torch.eye(4),
            metadata={'type': 'segmentation'},
        )

        subject = tio.Subject(image=history_img, label=metadata_label)

        transform = tio.Crop(cropping=1)
        result = transform(subject)

        # Check both images preserved their types and attributes
        assert isinstance(result.image, HistoryScalarImage)
        assert result.image.history == ['img1']

        assert isinstance(result.label, MetadataLabelMap)
        assert result.label.metadata == {'type': 'segmentation'}

    def test_original_github_issue_scenario(self):
        """Test the exact scenario from the GitHub issue."""

        # This is the exact code from the GitHub issue
        class HistoryScalarImageOriginal(tio.ScalarImage):
            def __init__(self, tensor, affine, history, **kwargs):
                super().__init__(tensor=tensor, affine=affine, **kwargs)
                self.history = history

        img = HistoryScalarImageOriginal(
            tensor=torch.rand(1, 10, 10, 10), affine=torch.eye(4), history=[]
        )
        subject = tio.Subject(image=img)
        transform = tio.Crop(cropping=2)

        # This should not raise a TypeError anymore
        result = transform(subject)

        # Check that it worked
        assert isinstance(result.image, HistoryScalarImageOriginal)
        assert hasattr(result.image, 'history')
        assert result.image.history == []
