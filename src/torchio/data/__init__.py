"""Data classes for TorchIO."""

from .aggregator import PatchAggregator
from .batch import ImagesBatch
from .batch import StudiesBatch
from .batch import SubjectsBatch
from .bboxes import BoundingBoxes
from .bboxes import BoundingBoxFormat
from .bboxes import Representation
from .image import Image
from .image import LabelMap
from .image import ScalarImage
from .patch import PatchLocation
from .points import Points
from .queue import Queue
from .sampler import GridSampler
from .sampler import LabelSampler
from .sampler import PatchSampler
from .sampler import UniformSampler
from .sampler import WeightedSampler
from .subject import Study
from .subject import Subject

__all__ = [
    "BoundingBoxFormat",
    "BoundingBoxes",
    "GridSampler",
    "Image",
    "ImagesBatch",
    "LabelMap",
    "LabelSampler",
    "PatchAggregator",
    "PatchLocation",
    "PatchSampler",
    "Points",
    "Queue",
    "Representation",
    "ScalarImage",
    "StudiesBatch",
    "Study",
    "Subject",
    "SubjectsBatch",
    "UniformSampler",
    "WeightedSampler",
]
