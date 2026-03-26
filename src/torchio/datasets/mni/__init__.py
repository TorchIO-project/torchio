"""MNI atlases."""

from .colin import Colin27
from .icbm import ICBM2009CNonlinearSymmetric
from .mni import SubjectMNI
from .pediatric import Pediatric
from .sheep import Sheep

__all__ = [
    "Colin27",
    "ICBM2009CNonlinearSymmetric",
    "Pediatric",
    "Sheep",
    "SubjectMNI",
]
