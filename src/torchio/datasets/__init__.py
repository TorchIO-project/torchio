"""Built-in demo subjects and datasets."""

from .fpg import FPG
from .itk_snap import T1T2
from .itk_snap import AorticValve
from .itk_snap import BrainTumor
from .ixi import ixi
from .ixi import ixi_tiny
from .medmnist import adrenal_mnist_3d
from .medmnist import fracture_mnist_3d
from .medmnist import nodule_mnist_3d
from .medmnist import organ_mnist_3d
from .medmnist import synapse_mnist_3d
from .medmnist import vessel_mnist_3d
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
    "adrenal_mnist_3d",
    "fracture_mnist_3d",
    "ixi",
    "ixi_tiny",
    "nodule_mnist_3d",
    "organ_mnist_3d",
    "synapse_mnist_3d",
    "vessel_mnist_3d",
]
