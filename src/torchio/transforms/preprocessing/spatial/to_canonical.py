from .to_orientation import ToOrientation


class ToCanonical(ToOrientation):
    """Reorder the data to be closest to canonical (RAS+) orientation.

    This transform reorders the voxels and modifies the affine matrix so that
    the voxel orientations are nearest to:

    1. First voxel axis goes from left to Right
    2. Second voxel axis goes from posterior to Anterior
    3. Third voxel axis goes from inferior to Superior

    See `NiBabel docs about image orientation`_ for more information.

    Args:
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. note:: The reorientation is performed using
        :meth:`~torchio.transforms.preprocessing.to_orientation.ToOrientation`.

    .. _NiBabel docs about image orientation: https://nipy.org/nibabel/image_orientation.html
    """

    def __init__(self, **kwargs):
        super().__init__('RAS', **kwargs)
