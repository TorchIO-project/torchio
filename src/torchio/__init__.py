"""TorchIO: Medical image preprocessing, augmentation, and patch-based training."""

from importlib.metadata import version

from . import datasets
from .data.affine import AffineMatrix
from .data.aggregator import PatchAggregator
from .data.batch import ImagesBatch
from .data.batch import StudiesBatch
from .data.batch import SubjectsBatch
from .data.bboxes import BoundingBoxes
from .data.bboxes import BoundingBoxFormat
from .data.bboxes import Representation
from .data.image import Image
from .data.image import LabelMap
from .data.image import ScalarImage
from .data.patch import PatchLocation
from .data.points import Points
from .data.queue import Queue
from .data.sampler import GridSampler
from .data.sampler import LabelSampler
from .data.sampler import PatchSampler
from .data.sampler import UniformSampler
from .data.sampler import WeightedSampler
from .data.subject import Study
from .data.subject import Subject
from .io import read_matrix
from .io import write_matrix
from .loader import ImagesLoader
from .loader import StudiesLoader
from .loader import SubjectsLoader
from .loader import collate_images
from .loader import collate_studies
from .loader import collate_subjects
from .logging import enable_logging
from .transforms import Affine
from .transforms import AppliedTransform
from .transforms import Compose
from .transforms import Crop
from .transforms import CropOrPad
from .transforms import ElasticDeformation
from .transforms import EnsureShapeMultiple
from .transforms import Flip
from .transforms import IntensityTransform
from .transforms import MonaiAdapter
from .transforms import Noise
from .transforms import Normalize
from .transforms import OneOf
from .transforms import Pad
from .transforms import ParameterRange
from .transforms import Reorient
from .transforms import Resample
from .transforms import RescaleIntensity
from .transforms import SomeOf
from .transforms import Spatial
from .transforms import SpatialTransform
from .transforms import Standardize
from .transforms import To
from .transforms import Transform
from .transforms import ZNormalization
from .transforms.inverse import apply_inverse_transform
from .types import TypeAffineMatrix
from .types import TypeDirection
from .types import TypeImageData
from .types import TypeOrientationCodes
from .types import TypeOrigin
from .types import TypePath
from .types import TypeSpacing
from .types import TypeSpatialShape
from .types import TypeTensorShape
from .types import TypeWorldPoints

__all__ = [
    "Affine",
    "AffineMatrix",
    "AppliedTransform",
    "BoundingBoxFormat",
    "BoundingBoxes",
    "Compose",
    "Crop",
    "CropOrPad",
    "ElasticDeformation",
    "EnsureShapeMultiple",
    "Flip",
    "GridSampler",
    "Image",
    "ImagesBatch",
    "ImagesLoader",
    "IntensityTransform",
    "LabelMap",
    "LabelSampler",
    "MonaiAdapter",
    "Noise",
    "Normalize",
    "OneOf",
    "Pad",
    "ParameterRange",
    "PatchAggregator",
    "PatchLocation",
    "PatchSampler",
    "Points",
    "Queue",
    "Reorient",
    "Representation",
    "Resample",
    "RescaleIntensity",
    "ScalarImage",
    "SomeOf",
    "Spatial",
    "SpatialTransform",
    "Standardize",
    "StudiesBatch",
    "StudiesLoader",
    "Study",
    "Subject",
    "SubjectsBatch",
    "SubjectsLoader",
    "To",
    "Transform",
    "TypeAffineMatrix",
    "TypeDirection",
    "TypeImageData",
    "TypeOrientationCodes",
    "TypeOrigin",
    "TypePath",
    "TypeSpacing",
    "TypeSpatialShape",
    "TypeTensorShape",
    "TypeWorldPoints",
    "UniformSampler",
    "WeightedSampler",
    "ZNormalization",
    "apply_inverse_transform",
    "collate_images",
    "collate_studies",
    "collate_subjects",
    "datasets",
    "enable_logging",
    "read_matrix",
    "write_matrix",
]

__version__ = version(__name__)
