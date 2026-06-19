"""TorchIO transforms."""

from .compose import Compose
from .compose import OneOf
from .compose import SomeOf
from .cornucopia_adapter import CornucopiaAdapter
from .intensity.bias_field import BiasField
from .intensity.blur import Blur
from .intensity.clamp import Clamp
from .intensity.gamma import Gamma
from .intensity.ghosting import Ghosting
from .intensity.histogram_standardization import HistogramStandardization
from .intensity.inversion import IntensityInversion
from .intensity.labels_to_image import LabelsToImage
from .intensity.mask import Mask
from .intensity.motion import Motion
from .intensity.noise import Noise
from .intensity.normalize import Normalize
from .intensity.normalize import RescaleIntensity
from .intensity.pca import PCA
from .intensity.spike import Spike
from .intensity.standardize import Standardize
from .intensity.standardize import ZNormalization
from .intensity.swap import Swap
from .label.contour import Contour
from .label.keep_largest import KeepLargestComponent
from .label.one_hot import OneHot
from .label.remap_labels import RemapLabels
from .label.remove_labels import RemoveLabels
from .label.sequential_labels import SequentialLabels
from .lambda_transform import Lambda
from .monai_adapter import MonaiAdapter
from .parameter_range import Choice
from .spatial.anisotropy import Anisotropy
from .spatial.copy_affine import CopyAffine
from .spatial.crop import Crop
from .spatial.crop_or_pad import CropOrPad
from .spatial.ensure_shape_multiple import EnsureShapeMultiple
from .spatial.flip import Flip
from .spatial.pad import Pad
from .spatial.reorient import Reorient
from .spatial.resize import Resize
from .spatial.spatial import Affine
from .spatial.spatial import ElasticDeformation
from .spatial.spatial import Resample
from .spatial.spatial import Spatial
from .spatial.to_reference_space import ToReferenceSpace
from .spatial.transpose import Transpose
from .to import To
from .transform import AppliedTransform
from .transform import IntensityTransform
from .transform import SpatialTransform
from .transform import Transform

__all__ = [
    "PCA",
    "Affine",
    "Anisotropy",
    "AppliedTransform",
    "BiasField",
    "Blur",
    "Choice",
    "Clamp",
    "Compose",
    "Contour",
    "CopyAffine",
    "CornucopiaAdapter",
    "Crop",
    "CropOrPad",
    "ElasticDeformation",
    "EnsureShapeMultiple",
    "Flip",
    "Gamma",
    "Ghosting",
    "HistogramStandardization",
    "IntensityInversion",
    "IntensityTransform",
    "KeepLargestComponent",
    "LabelsToImage",
    "Lambda",
    "Mask",
    "MonaiAdapter",
    "Motion",
    "Noise",
    "Normalize",
    "OneHot",
    "OneOf",
    "Pad",
    "RemapLabels",
    "RemoveLabels",
    "Reorient",
    "Resample",
    "RescaleIntensity",
    "Resize",
    "SequentialLabels",
    "SomeOf",
    "Spatial",
    "SpatialTransform",
    "Spike",
    "Standardize",
    "Swap",
    "To",
    "ToReferenceSpace",
    "Transform",
    "Transpose",
    "ZNormalization",
]
