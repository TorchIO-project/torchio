"""Data classes for TorchIO."""

from .aggregator import PatchAggregator
from .backends import BackendRequest
from .backends import ImageDataBackend
from .backends import LazyReader
from .backends import register_backend
from .backends import resolve_backend
from .backends import unregister_backend
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
    "BackendRequest",
    "BoundingBoxFormat",
    "BoundingBoxes",
    "GridSampler",
    "Image",
    "ImageDataBackend",
    "ImagesBatch",
    "LabelMap",
    "LabelSampler",
    "LazyReader",
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
    "register_backend",
    "resolve_backend",
    "unregister_backend",
]
