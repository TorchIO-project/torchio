from collections.abc import Mapping
from typing import TypeVar
from typing import cast

from ..data.image import ScalarImage
from ..data.subject import Subject
from .transform import Transform

ValueT = TypeVar('ValueT')


class IntensityTransform(Transform):
    """Transform that modifies voxel intensities only."""

    def get_images_dict(self, subject: Subject) -> dict[str, ScalarImage]:
        return subject.get_images_dict(
            intensity_only=True,
            include=self.include,
            exclude=self.exclude,
        )

    def get_images(self, subject: Subject) -> list[ScalarImage]:
        return subject.get_images(
            intensity_only=True,
            include=self.include,
            exclude=self.exclude,
        )

    @staticmethod
    def get_parameter(value: ValueT | Mapping[str, ValueT], name: str) -> ValueT:
        if isinstance(value, Mapping):
            mapping = cast(Mapping[str, ValueT], value)
            return mapping[name]
        return value

    def arguments_are_dict(self) -> bool:
        """Check if main arguments are dict.

        Return `True` if the type of all attributes specified in the
        `args_names` have `dict` type.
        """
        args = list(self._get_named_arguments().values())
        are_dict = [isinstance(arg, dict) for arg in args]
        if all(are_dict):
            return True
        elif not any(are_dict):
            return False
        else:
            message = 'Either all or none of the arguments must be dicts'
            raise ValueError(message)
