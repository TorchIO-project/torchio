"""Data classes for TorchIO."""

from .batch import ImagesBatch
from .batch import StudiesBatch
from .batch import SubjectsBatch
from .bboxes import BoundingBoxes
from .bboxes import BoundingBoxFormat
from .bboxes import Representation
from .image import Image
from .image import LabelMap
from .image import ScalarImage
from .points import Points
from .subject import Study
from .subject import Subject

__all__ = [
    "BoundingBoxFormat",
    "BoundingBoxes",
    "Image",
    "ImagesBatch",
    "LabelMap",
    "Points",
    "Representation",
    "ScalarImage",
    "StudiesBatch",
    "Study",
    "Subject",
    "SubjectsBatch",
]
