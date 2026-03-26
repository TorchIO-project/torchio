"""Built-in demo subjects and datasets."""

from .fpg import FPG
from .itk_snap import T1T2
from .itk_snap import AorticValve
from .itk_snap import BrainTumor
from .mni import Colin27
from .mni import ICBM2009CNonlinearSymmetric
from .mni import Pediatric
from .mni import Sheep
from .slicer import Slicer
from .zone_plate import ZonePlate

__all__ = [
    "FPG",
    "T1T2",
    "AorticValve",
    "BrainTumor",
    "Colin27",
    "ICBM2009CNonlinearSymmetric",
    "Pediatric",
    "Sheep",
    "Slicer",
    "ZonePlate",
]
