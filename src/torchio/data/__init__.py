"""Data classes for TorchIO."""

from .bboxes import BoundingBoxes
from .bboxes import BoundingBoxFormat
from .bboxes import Representation
from .image import Image
from .image import LabelMap
from .image import ScalarImage
from .points import Points
from .subject import Subject

__all__ = [
    "BoundingBoxFormat",
    "BoundingBoxes",
    "Image",
    "LabelMap",
    "Points",
    "Representation",
    "ScalarImage",
    "Subject",
]
