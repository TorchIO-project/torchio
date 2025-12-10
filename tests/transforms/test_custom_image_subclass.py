"""Tests for custom Image subclasses with transforms."""

import torch
import pytest
import torchio as tio


class HistoryImage(tio.ScalarImage):
    """Test custom Image with required parameter."""
    
    def __init__(self, tensor, affine, history, **kwargs):
        super().__init__(tensor=tensor, affine=affine, **kwargs)
        self.history = history
    
    def new_like(self, tensor, affine=None):
        return type(self)(
            tensor=tensor,
            affine=affine if affine is not None else self.affine,
            history=self.history,
            check_nans=self.check_nans,
            reader=self.reader,
        )


class MetadataImage(tio.ScalarImage):
    """Test custom Image with optional parameter."""
    
    def __init__(self, tensor, affine, metadata=None, **kwargs):
        super().__init__(tensor=tensor, affine=affine, **kwargs)
        self.metadata = metadata or {}
    
    def new_like(self, tensor, affine=None):
        return type(self)(
            tensor=tensor,
            affine=affine if affine is not None else self.affine,
            metadata=self.metadata,
            check_nans=self.check_nans,
            reader=self.reader,
        )


class TestCustomImageSubclass:
    """Test suite for custom Image subclasses with transforms."""
    
    @pytest.fixture
    def history_image(self):
        """Create a HistoryImage for testing."""
        tensor = torch.rand(1, 10, 10, 10)
        affine = torch.eye(4)
        return HistoryImage(tensor=tensor, affine=affine, history=['created'])
    
    @pytest.fixture
    def metadata_image(self):
        """Create a MetadataImage for testing."""
        tensor = torch.rand(1, 12, 12, 12)
        affine = torch.eye(4)
        return MetadataImage(
            tensor=tensor, 
            affine=affine, 
            metadata={'id': 123, 'source': 'test'}
        )
    
    @pytest.fixture
    def history_subject(self, history_image):
        """Create a Subject with HistoryImage."""
        return tio.Subject(image=history_image)
    
    @pytest.fixture
    def metadata_subject(self, metadata_image):
        """Create a Subject with MetadataImage."""
        return tio.Subject(image=metadata_image)
    
    def test_crop_with_history_image(self, history_subject):
        """Test that Crop transform works with custom Image requiring history parameter."""
        transform = tio.Crop(cropping=2)
        result = transform(history_subject)
        
        # Check that the result is still a HistoryImage
        assert isinstance(result.image, HistoryImage)
        
        # Check that custom attribute is preserved
        assert result.image.history == ['created']
        
        # Check that cropping worked correctly
        assert result.image.shape == (1, 6, 6, 6)
    
    def test_crop_with_metadata_image(self, metadata_subject):
        """Test that Crop transform works with custom Image with optional parameters."""
        transform = tio.Crop(cropping=1)
        result = transform(metadata_subject)
        
        # Check that the result is still a MetadataImage
        assert isinstance(result.image, MetadataImage)
        
        # Check that custom attribute is preserved
        assert result.image.metadata == {'id': 123, 'source': 'test'}
        
        # Check that cropping worked correctly
        assert result.image.shape == (1, 10, 10, 10)
    
    def test_chained_transforms_preserve_attributes(self, history_subject):
        """Test that chained transforms preserve custom attributes."""
        # Chain multiple transforms
        transform = tio.Compose([
            tio.Crop(cropping=1),
            tio.Crop(cropping=1),
        ])
        
        result = transform(history_subject)
        
        # Check that the result is still a HistoryImage after multiple transforms
        assert isinstance(result.image, HistoryImage)
        
        # Check that custom attribute is preserved through the chain
        assert result.image.history == ['created']
        
        # Check that both crops were applied
        assert result.image.shape == (1, 6, 6, 6)
    
    def test_backward_compatibility_standard_images(self):
        """Test that standard Images still work with transforms."""
        # Create a standard ScalarImage
        tensor = torch.rand(1, 10, 10, 10)
        affine = torch.eye(4)
        image = tio.ScalarImage(tensor=tensor, affine=affine)
        subject = tio.Subject(image=image)
        
        # Apply transform
        transform = tio.Crop(cropping=2)
        result = transform(subject)
        
        # Check that it still works
        assert isinstance(result.image, tio.ScalarImage)
        assert result.image.shape == (1, 6, 6, 6)
    
    def test_to_reference_space_with_custom_image(self, history_image):
        """Test that ToReferenceSpace works with custom images."""
        # Create a reference image
        reference_tensor = torch.rand(1, 20, 20, 20)
        reference_affine = torch.eye(4)
        reference = tio.ScalarImage(tensor=reference_tensor, affine=reference_affine)
        
        # Create embedding tensor (smaller than reference)
        embedding_tensor = torch.rand(1, 10, 10, 10)
        
        # Use ToReferenceSpace.from_tensor
        result = tio.ToReferenceSpace.from_tensor(embedding_tensor, history_image)
        
        # Check that the result preserves the custom class type
        assert isinstance(result, HistoryImage)
        
        # Check that custom attribute is preserved
        assert result.history == ['created']
    
    def test_new_like_method_directly(self, history_image):
        """Test the new_like method directly."""
        new_tensor = torch.rand(1, 5, 5, 5)
        new_affine = torch.eye(4) * 2
        
        # Create new image using new_like
        new_image = history_image.new_like(tensor=new_tensor, affine=new_affine)
        
        # Check type preservation
        assert isinstance(new_image, HistoryImage)
        
        # Check attribute preservation
        assert new_image.history == ['created']
        
        # Check new data
        assert torch.equal(new_image.data, new_tensor)
        assert torch.allclose(torch.tensor(new_image.affine).float(), new_affine.float())
    
    def test_new_like_with_default_affine(self, metadata_image):
        """Test new_like method with default affine (None)."""
        new_tensor = torch.rand(1, 8, 8, 8)
        
        # Create new image using new_like with default affine
        new_image = metadata_image.new_like(tensor=new_tensor)
        
        # Check that original affine is used
        assert torch.allclose(torch.tensor(new_image.affine), torch.tensor(metadata_image.affine))
        
        # Check attribute preservation
        assert new_image.metadata == {'id': 123, 'source': 'test'}
    
    def test_label_map_subclass(self):
        """Test that custom LabelMap subclasses also work."""
        class CustomLabelMap(tio.LabelMap):
            def __init__(self, tensor, affine, labels_info, **kwargs):
                super().__init__(tensor=tensor, affine=affine, **kwargs)
                self.labels_info = labels_info
            
            def new_like(self, tensor, affine=None):
                return type(self)(
                    tensor=tensor,
                    affine=affine if affine is not None else self.affine,
                    labels_info=self.labels_info,
                    check_nans=self.check_nans,
                    reader=self.reader,
                )
        
        # Create custom label map
        tensor = torch.randint(0, 3, (1, 8, 8, 8))
        affine = torch.eye(4)
        labels_info = {0: 'background', 1: 'tissue1', 2: 'tissue2'}
        
        custom_label = CustomLabelMap(
            tensor=tensor, 
            affine=affine, 
            labels_info=labels_info
        )
        subject = tio.Subject(labels=custom_label)
        
        # Apply transform
        transform = tio.Crop(cropping=1)
        result = transform(subject)
        
        # Check preservation
        assert isinstance(result.labels, CustomLabelMap)
        assert result.labels.labels_info == labels_info
        assert result.labels.shape == (1, 6, 6, 6)