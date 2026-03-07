from __future__ import annotations

import copy
import pprint
from collections.abc import Mapping
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import TypeAlias
from typing import overload

import numpy as np

from ..utils import get_subclasses
from .image import Image
from .image import LabelMap
from .image import ScalarImage

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ..transforms import Compose
    from ..transforms import Transform


AppliedTransformParameters: TypeAlias = dict[str, object]
AppliedTransform: TypeAlias = tuple[str, AppliedTransformParameters]


class Subject(dict[str, object]):
    """Class to store information about the images corresponding to a subject.

    Args:
        *args: If provided, a dictionary of items.
        **kwargs: Items that will be added to the subject sample.

    Examples:
        >>> import torchio as tio
        >>> # One way:
        >>> subject = tio.Subject(
        ...     one_image=tio.ScalarImage('path_to_image.nii.gz'),
        ...     a_segmentation=tio.LabelMap('path_to_seg.nii.gz'),
        ...     age=45,
        ...     name='John Doe',
        ...     hospital='Hospital Juan Negrín',
        ... )
        >>> # If you want to create the mapping before, or have spaces in the keys:
        >>> subject_dict = {
        ...     'one image': tio.ScalarImage('path_to_image.nii.gz'),
        ...     'a segmentation': tio.LabelMap('path_to_seg.nii.gz'),
        ...     'age': 45,
        ...     'name': 'John Doe',
        ...     'hospital': 'Hospital Juan Negrín',
        ... }
        >>> subject = tio.Subject(subject_dict)
    """

    def __init__(self, *args: Mapping[str, object], **kwargs: object):
        if args:
            if len(args) == 1 and isinstance(args[0], Mapping):
                kwargs.update(args[0])
            else:
                message = 'Only one dictionary as positional argument is allowed'
                raise ValueError(message)
        super().__init__(**kwargs)
        self._parse_images(self.get_images(intensity_only=False))
        self.update_attributes()  # this allows me to do e.g. subject.t1
        self.applied_transforms: list[AppliedTransform] = []

    def __repr__(self):
        num_images = len(self.get_images(intensity_only=False))
        string = (
            f'{self.__class__.__name__}'
            f'(Keys: {tuple(self.keys())}; images: {num_images})'
        )
        return string

    def _repr_html_(self):
        try:
            from matplotlib.figure import Figure
        except ImportError:
            return self.__repr__()

        fig = self.plot(return_fig=True, show=False)
        assert isinstance(fig, Figure)

        from ..visualization import _figure_to_html

        return _figure_to_html(fig)

    def __len__(self):
        return len(self.get_images(intensity_only=False))

    @overload
    def __getitem__(self, item: str) -> object: ...

    @overload
    def __getitem__(self, item: slice | int | tuple[object, ...]) -> Subject: ...

    def __getitem__(
        self, item: str | slice | int | tuple[object, ...]
    ) -> object | Subject:
        if isinstance(item, (slice, int, tuple)):
            try:
                self.check_consistent_spatial_shape()
            except RuntimeError as e:
                message = (
                    'To use indexing, all images in the subject must have the'
                    ' same spatial shape'
                )
                raise RuntimeError(message) from e
            copied = copy.deepcopy(self)
            for image_name, image in copied.items():
                if isinstance(image, Image):
                    copied[image_name] = image[item]
            return copied
        else:
            return super().__getitem__(item)

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as error:
            raise AttributeError(
                f'{self.__class__.__name__!s} has no attribute {item!r}',
            ) from error

    @staticmethod
    def _parse_images(images: list[Image]) -> None:
        # Check that it's not empty
        if not images:
            raise TypeError('A subject without images cannot be created')

    @property
    def shape(self):
        """Return shape of first image in subject.

        Consistency of shapes across images in the subject is checked first.

        Examples:
            >>> import torchio as tio
            >>> colin = tio.datasets.Colin27()
            >>> colin.shape
            (1, 181, 217, 181)
        """
        self.check_consistent_attribute('shape')
        return self.get_first_image().shape

    @property
    def spatial_shape(self):
        """Return spatial shape of first image in subject.

        Consistency of spatial shapes across images in the subject is checked
        first.

        Examples:
            >>> import torchio as tio
            >>> colin = tio.datasets.Colin27()
            >>> colin.spatial_shape
            (181, 217, 181)
        """
        self.check_consistent_spatial_shape()
        return self.get_first_image().spatial_shape

    @property
    def spacing(self):
        """Return spacing of first image in subject.

        Consistency of spacings across images in the subject is checked first.

        Examples:
            >>> import torchio as tio
            >>> colin = tio.datasets.Slicer()
            >>> colin.spacing
            (1.0, 1.0, 1.2999954223632812)
        """
        self.check_consistent_attribute('spacing')
        return self.get_first_image().spacing

    @property
    def history(self):
        # Kept for backwards compatibility
        return self.get_applied_transforms()

    def is_2d(self):
        return all(i.is_2d() for i in self.get_images(intensity_only=False))

    def get_applied_transforms(
        self,
        ignore_intensity: bool = False,
        image_interpolation: str | None = None,
    ) -> list[Transform]:
        from ..transforms.intensity_transform import IntensityTransform
        from ..transforms.transform import Transform

        name_to_transform = {cls.__name__: cls for cls in get_subclasses(Transform)}
        transforms_list = []
        for transform_name, arguments in self.applied_transforms:
            transform = name_to_transform[transform_name](**arguments)
            if ignore_intensity and isinstance(transform, IntensityTransform):
                continue
            resamples = hasattr(transform, 'image_interpolation')
            if resamples and image_interpolation is not None:
                parsed = transform.parse_interpolation(image_interpolation)
                transform.image_interpolation = parsed
            transforms_list.append(transform)
        return transforms_list

    def get_composed_history(
        self,
        ignore_intensity: bool = False,
        image_interpolation: str | None = None,
    ) -> Compose:
        from ..transforms.augmentation.composition import Compose

        transforms = self.get_applied_transforms(
            ignore_intensity=ignore_intensity,
            image_interpolation=image_interpolation,
        )
        return Compose(transforms)

    def get_inverse_transform(
        self,
        warn: bool = True,
        ignore_intensity: bool = False,
        image_interpolation: str | None = None,
    ) -> Compose:
        """Get a reversed list of the inverses of the applied transforms.

        Args:
            warn: Issue a warning if some transforms are not invertible.
            ignore_intensity: If `True`, all instances of
                [`IntensityTransform`][torchio.transforms.intensity_transform.IntensityTransform]
                will be ignored.
            image_interpolation: Modify interpolation for scalar images inside
                transforms that perform resampling.
        """
        history_transform = self.get_composed_history(
            ignore_intensity=ignore_intensity,
            image_interpolation=image_interpolation,
        )
        inverse_transform = history_transform.inverse(warn=warn)
        return inverse_transform

    def apply_inverse_transform(self, **kwargs) -> Subject:
        """Apply the inverse of all applied transforms, in reverse order.

        Args:
            **kwargs: Keyword arguments passed on to
                [`get_inverse_transform()`][torchio.data.subject.Subject.get_inverse_transform].
        """
        inverse_transform = self.get_inverse_transform(**kwargs)
        transformed = inverse_transform(self)
        transformed.clear_history()
        return transformed

    def clear_history(self) -> None:
        self.applied_transforms = []

    def check_consistent_attribute(
        self,
        attribute: str,
        relative_tolerance: float = 1e-6,
        absolute_tolerance: float = 1e-6,
        message: str | None = None,
    ) -> None:
        r"""Check for consistency of an attribute across all images.

        Args:
            attribute: Name of the image attribute to check
            relative_tolerance: Relative tolerance for `numpy.allclose()`
            absolute_tolerance: Absolute tolerance for `numpy.allclose()`

        Examples:
            >>> import numpy as np
            >>> import torch
            >>> import torchio as tio
            >>> scalars = torch.randn(1, 512, 512, 100)
            >>> mask = torch.tensor(scalars > 0).type(torch.int16)
            >>> af1 = np.eye([0.8, 0.8, 2.50000000000001, 1])
            >>> af2 = np.eye([0.8, 0.8, 2.49999999999999, 1])  # small difference here (e.g. due to different reader)
            >>> subject = tio.Subject(
            ...   image = tio.ScalarImage(tensor=scalars, affine=af1),
            ...   mask = tio.LabelMap(tensor=mask, affine=af2)
            ... )
            >>> subject.check_consistent_attribute('spacing')  # no error as tolerances are > 0

        Note:
            To check that all values for a specific attribute are close
            between all images in the subject, `numpy.allclose()` is used.
            This function returns `True` if
            $|a_i - b_i| \leq t_{abs} + t_{rel} * |b_i|$, where
            $a_i$ and $b_i$ are the $i$-th element of the same
            attribute of two images being compared,
            $t_{abs}$ is the `absolute_tolerance` and
            $t_{rel}$ is the `relative_tolerance`.
        """
        message = (
            f'More than one value for "{attribute}" found in subject images:\n{{}}'
        )

        names_images = self.get_images_dict(intensity_only=False).items()
        try:
            first_attribute = None
            first_image = None

            for image_name, image in names_images:
                if first_attribute is None:
                    first_attribute = getattr(image, attribute)
                    first_image = image_name
                    continue
                current_attribute = getattr(image, attribute)
                all_close = np.allclose(
                    current_attribute,
                    first_attribute,
                    rtol=relative_tolerance,
                    atol=absolute_tolerance,
                )
                if not all_close:
                    message = message.format(
                        pprint.pformat(
                            {
                                first_image: first_attribute,
                                image_name: current_attribute,
                            }
                        ),
                    )
                    raise RuntimeError(message)
        except TypeError:
            # fallback for non-numeric values
            values_dict = {}
            for image_name, image in names_images:
                values_dict[image_name] = getattr(image, attribute)
            num_unique_values = len(set(values_dict.values()))
            if num_unique_values > 1:
                message = message.format(pprint.pformat(values_dict))
                raise RuntimeError(message) from None

    def check_consistent_spatial_shape(self) -> None:
        self.check_consistent_attribute('spatial_shape')

    def check_consistent_orientation(self) -> None:
        self.check_consistent_attribute('orientation')

    def check_consistent_affine(self) -> None:
        self.check_consistent_attribute('affine')

    def check_consistent_space(self) -> None:
        try:
            self.check_consistent_attribute('spacing')
            self.check_consistent_attribute('direction')
            self.check_consistent_attribute('origin')
            self.check_consistent_spatial_shape()
        except RuntimeError as e:
            message = (
                'As described above, some images in the subject are not in the'
                ' same space. You probably can use the transforms ToCanonical'
                ' and Resample to fix this, as explained at'
                ' https://github.com/TorchIO-project/torchio/issues/647#issuecomment-913025695'
            )
            raise RuntimeError(message) from e

    def get_images_names(self) -> list[str]:
        return list(self.get_images_dict(intensity_only=False).keys())

    @overload
    def get_images_dict(
        self,
        intensity_only: Literal[True] = True,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> dict[str, ScalarImage]: ...

    @overload
    def get_images_dict(
        self,
        intensity_only: Literal[False] = False,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> dict[str, Image]: ...

    def get_images_dict(
        self,
        intensity_only: bool = True,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> dict[str, ScalarImage] | dict[str, Image]:
        if intensity_only:
            scalar_images: dict[str, ScalarImage] = {}
            for image_name, image in self.items():
                if not isinstance(image, ScalarImage):
                    continue
                if include is not None and image_name not in include:
                    continue
                if exclude is not None and image_name in exclude:
                    continue
                scalar_images[image_name] = image
            return scalar_images

        all_images: dict[str, Image] = {}
        for image_name, image in self.items():
            if not isinstance(image, Image):
                continue
            if include is not None and image_name not in include:
                continue
            if exclude is not None and image_name in exclude:
                continue
            all_images[image_name] = image
        return all_images

    @overload
    def get_images(
        self,
        intensity_only: Literal[True] = True,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> list[ScalarImage]: ...

    @overload
    def get_images(
        self,
        intensity_only: Literal[False] = False,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> list[Image]: ...

    @overload
    def get_images(
        self,
        intensity_only: bool,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> list[ScalarImage] | list[Image]: ...

    def get_images(
        self,
        intensity_only: bool = True,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> list[ScalarImage] | list[Image]:
        if intensity_only:
            scalar_images = self.get_images_dict(
                intensity_only=True,
                include=include,
                exclude=exclude,
            )
            return list(scalar_images.values())

        all_images = self.get_images_dict(
            intensity_only=False,
            include=include,
            exclude=exclude,
        )
        return list(all_images.values())

    def get_image(self, image_name: str) -> Image:
        """Get a single image by its name."""
        return self.get_images_dict(intensity_only=False)[image_name]

    def get_scalar_image(self, image_name: str) -> ScalarImage:
        image = self.get_image(image_name)
        if not isinstance(image, ScalarImage):
            message = f'Image "{image_name}" is not a scalar image'
            raise TypeError(message)
        return image

    def get_label_map(self, image_name: str) -> LabelMap:
        image = self.get_image(image_name)
        if not isinstance(image, LabelMap):
            message = f'Image "{image_name}" is not a label map'
            raise TypeError(message)
        return image

    def get_first_image(self) -> Image:
        return self.get_images(intensity_only=False)[0]

    def add_transform(
        self,
        transform: Transform,
        parameters_dict: AppliedTransformParameters,
    ) -> None:
        self.applied_transforms.append((transform.name, parameters_dict))

    def load(self) -> None:
        """Load images in subject on RAM."""
        for image in self.get_images(intensity_only=False):
            image.load()

    def unload(self) -> None:
        """Unload images in subject."""
        for image in self.get_images(intensity_only=False):
            image.unload()

    def update_attributes(self) -> None:
        # This allows to get images using attribute notation, e.g. subject.t1
        self.__dict__.update(self)

    @staticmethod
    def _check_image_name(image_name: object) -> str:
        if not isinstance(image_name, str):
            message = (
                f'The image name must be a string, but it has type "{type(image_name)}"'
            )
            raise ValueError(message)
        return image_name

    def add_image(self, image: Image, image_name: str) -> None:
        """Add an image to the subject instance."""
        if not isinstance(image, Image):
            message = (
                'Image must be an instance of torchio.Image,'
                f' but its type is "{type(image)}"'
            )
            raise ValueError(message)
        self._check_image_name(image_name)
        self[image_name] = image
        self.update_attributes()

    def remove_image(self, image_name: str) -> None:
        """Remove an image from the subject instance."""
        self._check_image_name(image_name)
        del self[image_name]
        delattr(self, image_name)

    def plot(self, return_fig: bool = False, **kwargs) -> None | Figure:
        """Plot images using matplotlib.

        Args:
            return_fig: If ``True``, return the figure instead of showing it.
            **kwargs: Keyword arguments that will be passed on to
                [`plot()`][torchio.Image.plot].
        """
        from ..visualization import plot_subject  # avoid circular import

        figure = plot_subject(self, **kwargs)
        if return_fig:
            return figure
        return None
