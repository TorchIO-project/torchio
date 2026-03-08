from __future__ import annotations

import copy
import warnings
from abc import ABC
from abc import abstractmethod
from collections.abc import MutableMapping
from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeGuard
from typing import TypeVar
from typing import cast
from typing import overload

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

from ..data.image import Image
from ..data.image import LabelMap
from ..data.io import nib_to_sitk
from ..data.io import sitk_to_nib
from ..data.subject import Subject
from ..types import TypeCallable
from ..types import TypeData
from ..types import TypeKeys
from ..types import TypeNumber
from ..types import TypeTripletInt
from ..utils import is_iterable
from .data_parser import DataParser
from .data_parser import TypeTransformInput
from .interpolation import Interpolation
from .interpolation import get_sitk_interpolator

TypeSixBounds = tuple[int, int, int, int, int, int]
TypeBounds = int | Sequence[int] | None
TypeMaskingMethod = str | TypeCallable | TypeBounds | None
ANATOMICAL_AXES = (
    'Left',
    'Right',
    'Posterior',
    'Anterior',
    'Inferior',
    'Superior',
)

ArgumentsDictT = TypeVar('ArgumentsDictT', bound=MutableMapping[str, object])
ImageT = TypeVar('ImageT', bound=Image)

__all__ = ['Transform', 'TypeBounds', 'TypeMaskingMethod', 'TypeTripletInt']


class Transform(ABC):
    """Abstract class for all TorchIO transforms.

    When called, the input can be an instance of
    [`torchio.Subject`][torchio.Subject],
    [`torchio.Image`][torchio.Image],
    [`numpy.ndarray`][numpy.ndarray],
    [`torch.Tensor`][torch.Tensor],
    [SimpleITK.Image][SimpleITK.Image],
    or [`dict`][dict] containing 4D tensors as values.

    All subclasses must overwrite
    [`apply_transform()`][torchio.transforms.Transform.apply_transform],
    which takes an instance of [`Subject`][torchio.Subject],
    modifies it and returns the result.

    Args:
        p: Probability that this transform will be applied.
        copy: Make a deep copy of the input before applying the transform.
        include: Sequence of strings with the names of the only images to which
            the transform will be applied.
            Mandatory if the input is a [`dict`][dict].
        exclude: Sequence of strings with the names of the images to which the
            the transform will not be applied, apart from the ones that are
            excluded because of the transform type.
            For example, if a subject includes an MRI, a CT and a label map,
            and the CT is added to the list of exclusions of an intensity
            transform such as [`RandomBlur`][torchio.transforms.RandomBlur],
            the transform will be only applied to the MRI, as the label map is
            excluded by default by spatial transforms.
        keep: Dictionary with the names of the input images that will be kept
            in the output and their new names. For example:
            `{'t1': 't1_original'}`. This might be useful for autoencoders
            or registration tasks.
        parse_input: If `True`, the input will be converted to an instance of
            [`Subject`][torchio.Subject]. This is used internally by some special
            transforms like
            [`Compose`][torchio.transforms.augmentation.composition.Compose].
        label_keys: If the input is a dictionary, names of images that
            correspond to label maps.
    """

    if TYPE_CHECKING:
        invert_transform: bool

    def __init__(
        self,
        p: float = 1,
        copy: bool = True,
        include: TypeKeys = None,
        exclude: TypeKeys = None,
        keys: TypeKeys = None,
        keep: dict[str, str] | None = None,
        parse_input: bool = True,
        label_keys: TypeKeys = None,
    ):
        self.probability = self.parse_probability(p)
        self.copy = copy
        if keys is not None:
            message = (
                'The "keys" argument is deprecated and will be removed in the'
                ' future. Use "include" instead'
            )
            warnings.warn(message, FutureWarning, stacklevel=2)
            include = keys
        self.include, self.exclude = self.parse_include_and_exclude_keys(
            include,
            exclude,
            label_keys,
        )
        self.keep = keep
        self.parse_input = parse_input
        self.label_keys = label_keys
        # args_names is the sequence of parameters from self that need to be
        # passed to a non-random version of a random transform. They are also
        # used to invert invertible transforms
        self.args_names: list[str] = []

    @overload
    def __call__(self, data: Subject) -> Subject: ...

    @overload
    def __call__(self, data: ImageT) -> ImageT: ...

    @overload
    def __call__(self, data: torch.Tensor) -> torch.Tensor: ...

    @overload
    def __call__(self, data: np.ndarray) -> np.ndarray: ...

    @overload
    def __call__(self, data: sitk.Image) -> sitk.Image: ...

    @overload
    def __call__(self, data: dict[str, object]) -> dict[str, object]: ...

    @overload
    def __call__(self, data: nib.Nifti1Image) -> nib.Nifti1Image: ...

    def __call__(self, data: TypeTransformInput) -> TypeTransformInput:
        """Transform data and return a result of the same type.

        Args:
            data: Instance of [`torchio.Subject`][torchio.Subject], 4D
                [`torch.Tensor`][torch.Tensor] or [`numpy.ndarray`][numpy.ndarray] with dimensions
                $(C, W, H, D)$, where $C$ is the number of channels
                and $W, H, D$ are the spatial dimensions. If the input is
                a tensor, the affine matrix will be set to identity. Other
                valid input types are a SimpleITK image, a
                [`torchio.Image`][torchio.Image], a NiBabel Nifti1 image or a
                [`dict`][dict]. The output type is the same as the input type.
        """
        if torch.rand(1).item() > self.probability:
            return data

        # Some transforms such as Compose should not modify the input data
        if self.parse_input:
            data_parser = DataParser(
                data,
                keys=self.include,
                label_keys=self.label_keys,
            )
            subject = data_parser.get_subject()
        else:
            subject = cast(Subject, data)

        if self.keep is not None:
            images_to_keep: dict[str, Image] = {}
            for name, new_name in self.keep.items():
                images_to_keep[new_name] = copy.deepcopy(subject.get_image(name))
        if self.copy:
            subject = copy.deepcopy(subject)
        with np.errstate(all='raise', under='ignore'):
            transformed = self.apply_transform(subject)
        if self.keep is not None:
            for name, image in images_to_keep.items():
                transformed.add_image(image, name)

        if self.parse_input:
            self.add_transform_to_subject_history(transformed)
            for image in transformed.get_images(intensity_only=False):
                ndim = image.data.ndim
                assert ndim == 4, f'Output of {self.name} is {ndim}D'
            output = data_parser.get_output(transformed)
        else:
            output = transformed

        return output

    def __repr__(self):
        if hasattr(self, 'args_names'):
            named_args = self._get_named_arguments()
            args_strings = [f'{arg}={value}' for arg, value in named_args.items()]
            if hasattr(self, 'invert_transform') and self.invert_transform:
                args_strings.append('invert=True')
            args_string = ', '.join(args_strings)
            return f'{self.name}({args_string})'
        else:
            return super().__repr__()

    def get_base_args(self) -> dict[str, object]:
        r"""Provides easy access to the arguments used to instantiate the base class
        ([`Transform`][torchio.transforms.transform.Transform]) of any transform.

        This method is particularly useful when a new transform can be represented as a variant
        of an existing transform (e.g. all random transforms), allowing for seamless instantiation
        of the existing transform with the same arguments as the new transform during `apply_transform`.

        Note:
            The `p` argument (probability of applying the transform) is excluded to avoid
            multiplying the probability of both existing and new transform.
        """
        return {
            'copy': self.copy,
            'include': self.include,
            'exclude': self.exclude,
            'keep': self.keep,
            'parse_input': self.parse_input,
            'label_keys': self.label_keys,
        }

    def add_base_args(
        self,
        arguments: ArgumentsDictT,
        overwrite_on_existing: bool = False,
    ) -> ArgumentsDictT:
        """Add the init args to existing arguments"""
        for key, value in self.get_base_args().items():
            if key in arguments and not overwrite_on_existing:
                continue
            arguments[key] = value
        return arguments

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def apply_transform(self, subject: Subject) -> Subject:
        raise NotImplementedError

    def add_transform_to_subject_history(self, subject):
        from . import Compose
        from . import CropOrPad
        from . import EnsureShapeMultiple
        from . import OneOf
        from .augmentation import RandomTransform
        from .preprocessing import Resize
        from .preprocessing import SequentialLabels

        call_others = (
            RandomTransform,
            Compose,
            OneOf,
            CropOrPad,
            EnsureShapeMultiple,
            SequentialLabels,
            Resize,
        )
        if not isinstance(self, call_others):
            subject.add_transform(self, self._get_reproducing_arguments())

    @staticmethod
    def to_range(n: TypeNumber, around: float | None) -> tuple[float, float]:
        if around is None:
            return 0.0, float(n)
        else:
            return float(around - n), float(around + n)

    @overload
    def parse_params(
        self,
        params: TypeNumber | Sequence[TypeNumber],
        around: float | None,
        name: str,
        make_ranges: Literal[True] = True,
        min_constraint: TypeNumber | None = None,
        max_constraint: TypeNumber | None = None,
        type_constraint: type[int] | type[float] | None = None,
    ) -> tuple[float, float, float, float, float, float]: ...

    @overload
    def parse_params(
        self,
        params: TypeNumber | Sequence[TypeNumber],
        around: float | None,
        name: str,
        make_ranges: Literal[False],
        min_constraint: TypeNumber | None = None,
        max_constraint: TypeNumber | None = None,
        type_constraint: type[int] | type[float] | None = None,
    ) -> tuple[float, ...]: ...

    def parse_params(
        self,
        params: TypeNumber | Sequence[TypeNumber],
        around: float | None,
        name: str,
        make_ranges: bool = True,
        min_constraint: TypeNumber | None = None,
        max_constraint: TypeNumber | None = None,
        type_constraint: type[int] | type[float] | None = None,
    ) -> tuple[float, float, float, float, float, float] | tuple[float, ...]:
        if isinstance(params, torch.Tensor):
            params_tuple = tuple(float(param) for param in params.reshape(-1))
        elif isinstance(params, np.ndarray):
            params_tuple = tuple(float(param) for param in np.ravel(params))
        elif isinstance(params, Sequence):
            params_sequence = cast(Sequence[TypeNumber], params)
            params_tuple = tuple(float(param) for param in params_sequence)
        else:
            params_tuple = (float(params),)
        # d or (a, b)
        if len(params_tuple) == 1 or (len(params_tuple) == 2 and make_ranges):
            params_tuple *= 3  # (d, d, d) or (a, b, a, b, a, b)
        if len(params_tuple) == 3 and make_ranges:  # (a, b, c)
            items = [self.to_range(n, around) for n in params_tuple]
            # (-a, a, -b, b, -c, c) or (1-a, 1+a, 1-b, 1+b, 1-c, 1+c)
            params_tuple = tuple(n for prange in items for n in prange)
        if make_ranges:
            if len(params_tuple) != 6:
                message = (
                    f'If "{name}" is a sequence, it must have length 2, 3 or'
                    f' 6, not {len(params_tuple)}'
                )
                raise ValueError(message)
            for param_range in zip(
                params_tuple[::2],
                params_tuple[1::2],
                strict=True,
            ):
                self._parse_range(
                    cast(tuple[float, float], param_range),
                    name,
                    min_constraint=min_constraint,
                    max_constraint=max_constraint,
                    type_constraint=type_constraint,
                )
            a, b, c, d, e, f = params_tuple
            return a, b, c, d, e, f
        return params_tuple

    @overload
    @staticmethod
    def _parse_range(
        nums_range: int | tuple[int, int],
        name: str,
        min_constraint: int | None = None,
        max_constraint: int | None = None,
        type_constraint: type[int] = int,
    ) -> tuple[int, int]: ...

    @overload
    @staticmethod
    def _parse_range(
        nums_range: float | tuple[float, float],
        name: str,
        min_constraint: float | None = None,
        max_constraint: float | None = None,
        type_constraint: type[float] | None = None,
    ) -> tuple[float, float]: ...

    @staticmethod
    def _parse_range(
        nums_range: TypeNumber | tuple[TypeNumber, TypeNumber],
        name: str,
        min_constraint: TypeNumber | None = None,
        max_constraint: TypeNumber | None = None,
        type_constraint: type[int] | type[float] | None = None,
    ) -> tuple[TypeNumber, TypeNumber]:
        r"""Adapted from [torchvision.transforms.RandomRotation][torchvision.transforms.RandomRotation].

        Args:
            nums_range: Tuple of two numbers $(n_{min}, n_{max})$,
                where $n_{min} \leq n_{max}$.
                If a single positive number $n$ is provided,
                $n_{min} = -n$ and $n_{max} = n$.
            name: Name of the parameter, so that an informative error message
                can be printed.
            min_constraint: Minimal value that $n_{min}$ can take,
                default is None, i.e. there is no minimal value.
            max_constraint: Maximal value that $n_{max}$ can take,
                default is None, i.e. there is no maximal value.
            type_constraint: Precise type that $n_{max}$ and
                $n_{min}$ must take.

        Returns:
            A tuple of two numbers $(n_{min}, n_{max})$.

        Raises:
            ValueError: if `nums_range` is negative
            ValueError: if $n_{max}$ or $n_{min}$ is not a number
            ValueError: if $n_{max} \lt n_{min}$
            ValueError: if `min_constraint` is not None and
                $n_{min}$ is smaller than `min_constraint`
            ValueError: if `max_constraint` is not None and
                $n_{max}$ is greater than `max_constraint`
            ValueError: if `type_constraint` is not None and
                $n_{max}$ and $n_{max}$ are not of type
                `type_constraint`.
        """
        if isinstance(nums_range, (int, float)):  # single number given
            if nums_range < 0:
                raise ValueError(
                    f'If {name} is a single number,'
                    f' it must be positive, not {nums_range}',
                )
            if min_constraint is not None and nums_range < min_constraint:
                raise ValueError(
                    f'If {name} is a single number, it must be greater'
                    f' than {min_constraint}, not {nums_range}',
                )
            if max_constraint is not None and nums_range > max_constraint:
                raise ValueError(
                    f'If {name} is a single number, it must be smaller'
                    f' than {max_constraint}, not {nums_range}',
                )
            if type_constraint is not None:
                if not isinstance(nums_range, type_constraint):
                    raise ValueError(
                        f'If {name} is a single number, it must be of'
                        f' type {type_constraint}, not {nums_range}',
                    )
            min_range = -nums_range if min_constraint is None else min_constraint
            return (min_range, nums_range)

        try:
            values = tuple(nums_range)
        except TypeError as err:
            message = (
                f'If {name} is not a single number, it must be'
                f' a sequence of len 2, not {nums_range}'
            )
            raise ValueError(message) from err
        if len(values) != 2:
            message = (
                f'If {name} is not a single number, it must be'
                f' a sequence of len 2, not {nums_range}'
            )
            raise ValueError(message)
        min_value, max_value = values

        min_is_number = isinstance(min_value, (int, float))
        max_is_number = isinstance(max_value, (int, float))
        if not min_is_number or not max_is_number:
            message = f'{name} values must be numbers, not {nums_range}'
            raise ValueError(message)

        if min_value > max_value:
            raise ValueError(
                f'If {name} is a sequence, the second value must be'
                f' equal or greater than the first, but it is {nums_range}',
            )

        if min_constraint is not None and min_value < min_constraint:
            raise ValueError(
                f'If {name} is a sequence, the first value must be greater'
                f' than {min_constraint}, but it is {min_value}',
            )

        if max_constraint is not None and max_value > max_constraint:
            raise ValueError(
                f'If {name} is a sequence, the second value must be'
                f' smaller than {max_constraint}, but it is {max_value}',
            )

        if type_constraint is not None:
            min_type_ok = isinstance(min_value, type_constraint)
            max_type_ok = isinstance(max_value, type_constraint)
            if not min_type_ok or not max_type_ok:
                raise ValueError(
                    f'If "{name}" is a sequence, its values must be of'
                    f' type "{type_constraint}", not "{type(nums_range)}"',
                )
        return min_value, max_value

    @staticmethod
    def parse_interpolation(interpolation: str) -> str:
        if not isinstance(interpolation, str):
            itype = type(interpolation)
            raise TypeError(f'Interpolation must be a string, not {itype}')
        interpolation = interpolation.lower()
        is_string = isinstance(interpolation, str)
        supported_values = [key.name.lower() for key in Interpolation]
        is_supported = interpolation.lower() in supported_values
        if is_string and is_supported:
            return interpolation
        message = (
            f'Interpolation "{interpolation}" of type {type(interpolation)}'
            f' must be a string among the supported values: {supported_values}'
        )
        raise ValueError(message)

    @staticmethod
    def parse_probability(probability: float) -> float:
        is_number = isinstance(probability, (int, float))
        if not (is_number and 0 <= probability <= 1):
            message = f'Probability must be a number in [0, 1], not {probability}'
            raise ValueError(message)
        return probability

    @staticmethod
    def parse_include_and_exclude_keys(
        include: TypeKeys,
        exclude: TypeKeys,
        label_keys: TypeKeys,
    ) -> tuple[TypeKeys, TypeKeys]:
        if include is not None and exclude is not None:
            raise ValueError('Include and exclude cannot both be specified')
        Transform.validate_keys_sequence(include, 'include')
        Transform.validate_keys_sequence(exclude, 'exclude')
        Transform.validate_keys_sequence(label_keys, 'label_keys')
        return include, exclude

    @staticmethod
    def validate_keys_sequence(keys: TypeKeys, name: str) -> None:
        """Ensure that the input is not a string but a sequence of strings."""
        if keys is None:
            return
        if isinstance(keys, str):
            message = f'"{name}" must be a sequence of strings, not a string "{keys}"'
            raise ValueError(message)
        if not is_iterable(keys):
            message = f'"{name}" must be a sequence of strings, not {type(keys)}'
            raise ValueError(message)

    @staticmethod
    def nib_to_sitk(data: TypeData, affine: TypeData) -> sitk.Image:
        return nib_to_sitk(data, affine)

    @staticmethod
    def sitk_to_nib(image: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
        return sitk_to_nib(image)

    def _get_reproducing_arguments(self):
        """Return a dictionary with the arguments that would be necessary to
        reproduce the transform exactly."""
        reproducing_arguments = {
            'include': self.include,
            'exclude': self.exclude,
            'copy': self.copy,
        }
        reproducing_arguments.update(self._get_named_arguments())
        return reproducing_arguments

    def _get_named_arguments(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.args_names}

    def is_invertible(self):
        return hasattr(self, 'invert_transform')

    def inverse(self):
        if not self.is_invertible():
            raise RuntimeError(f'{self.name} is not invertible')
        new = copy.deepcopy(self)
        new.invert_transform = not self.invert_transform
        return new

    @staticmethod
    @contextmanager
    def _use_seed(seed):
        """Perform an operation using a specific seed for the PyTorch RNG."""
        torch_rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        yield
        torch.random.set_rng_state(torch_rng_state)

    @staticmethod
    def get_sitk_interpolator(interpolation: str) -> int:
        return get_sitk_interpolator(interpolation)

    @staticmethod
    def parse_bounds(bounds_parameters: TypeBounds) -> TypeSixBounds | None:
        if bounds_parameters is None:
            return None
        if isinstance(bounds_parameters, int):
            values: tuple[int, ...] = (bounds_parameters,)
        else:
            values = tuple(bounds_parameters)

        # Check that numbers are integers
        for number in values:
            if not isinstance(number, (int, np.integer)) or number < 0:
                message = (
                    'Bounds values must be integers greater or equal to zero,'
                    f' not "{bounds_parameters}" of type {type(number)}'
                )
                raise ValueError(message)
        bounds_parameters_tuple = tuple(int(n) for n in values)
        bounds_parameters_length = len(bounds_parameters_tuple)
        if bounds_parameters_length == 6:
            i0, i1, j0, j1, k0, k1 = bounds_parameters_tuple
            return i0, i1, j0, j1, k0, k1
        if bounds_parameters_length == 1:
            (value,) = bounds_parameters_tuple
            return value, value, value, value, value, value
        if bounds_parameters_length == 3:
            i, j, k = bounds_parameters_tuple
            return i, i, j, j, k, k
        message = (
            'Bounds parameter must be an integer or a tuple of'
            f' 3 or 6 integers, not {bounds_parameters_tuple}'
        )
        raise ValueError(message)

    @staticmethod
    def _is_mask_callable(
        masking_method: object,
    ) -> TypeGuard[TypeCallable]:
        return callable(masking_method)

    @staticmethod
    def ones(tensor: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(tensor, dtype=torch.bool)

    @staticmethod
    def mean(tensor: torch.Tensor) -> torch.Tensor:
        mask = tensor > tensor.float().mean()
        return mask

    def get_mask_from_masking_method(
        self,
        masking_method: TypeMaskingMethod,
        subject: Subject,
        tensor: torch.Tensor,
        labels: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if masking_method is None:
            return self.ones(tensor)
        elif self._is_mask_callable(masking_method):
            return masking_method(tensor)
        elif type(masking_method) is str:
            in_subject = masking_method in subject
            if in_subject:
                label_map = subject[masking_method]
            else:
                label_map = None
            if isinstance(label_map, LabelMap):
                if labels is None:
                    return label_map.data.bool()
                else:
                    mask_data = label_map.data
                    volumes = [mask_data == label for label in labels]
                    return torch.stack(volumes).sum(0).bool()
            possible_axis = masking_method.capitalize()
            if possible_axis in ANATOMICAL_AXES:
                return self.get_mask_from_anatomical_label(
                    possible_axis,
                    tensor,
                )
        elif isinstance(masking_method, int):
            return self.get_mask_from_bounds(masking_method, tensor)
        elif isinstance(masking_method, (tuple, list)):
            if all(isinstance(number, (int, np.integer)) for number in masking_method):
                bounds_list: list[int] = []
                for number in masking_method:
                    assert isinstance(number, (int, np.integer))
                    bounds_list.append(int(number))
                return self.get_mask_from_bounds(bounds_list, tensor)
        first_anat_axes = tuple(s[0] for s in ANATOMICAL_AXES)
        message = (
            'Masking method must be one of:\n'
            ' 1) A callable object, such as a function\n'
            ' 2) The name of a label map in the subject'
            f' ({subject.get_images_names()})\n'
            f' 3) An anatomical label {ANATOMICAL_AXES + first_anat_axes}\n'
            ' 4) A bounds parameter'
            ' (int, tuple of 3 ints, or tuple of 6 ints)\n'
            f' The passed value, "{masking_method}",'
            f' of type "{type(masking_method)}", is not valid'
        )
        raise ValueError(message)

    @staticmethod
    def get_mask_from_anatomical_label(
        anatomical_label: str,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        # Assume the image is in RAS orientation
        anatomical_label = anatomical_label.capitalize()
        if anatomical_label not in ANATOMICAL_AXES:
            message = (
                f'Anatomical label must be one of {ANATOMICAL_AXES}'
                f' not {anatomical_label}'
            )
            raise ValueError(message)
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        _, width, height, depth = tensor.shape
        if anatomical_label == 'Right':
            mask[:, width // 2 :] = True
        elif anatomical_label == 'Left':
            mask[:, : width // 2] = True
        elif anatomical_label == 'Anterior':
            mask[:, :, height // 2 :] = True
        elif anatomical_label == 'Posterior':
            mask[:, :, : height // 2] = True
        elif anatomical_label == 'Superior':
            mask[:, :, :, depth // 2 :] = True
        elif anatomical_label == 'Inferior':
            mask[:, :, :, : depth // 2] = True
        return mask

    def get_mask_from_bounds(
        self,
        bounds_parameters: TypeBounds,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        bounds_parameters = self.parse_bounds(bounds_parameters)
        assert bounds_parameters is not None
        low = bounds_parameters[::2]
        high = bounds_parameters[1::2]
        i0, j0, k0 = low
        i1, j1, k1 = np.array(tensor.shape[1:]) - high
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        mask[:, i0:i1, j0:j1, k0:k1] = True
        return mask

    def _get_name_with_module(self) -> str:
        """Return the name of the transform including its module."""
        return f'{self.__class__.__module__}.{self.__class__.__name__}'

    @staticmethod
    def _tuples_to_lists(obj):
        if isinstance(obj, (tuple, list)):
            return [Transform._tuples_to_lists(x) for x in obj]
        if isinstance(obj, dict):
            return {k: Transform._tuples_to_lists(v) for k, v in obj.items()}
        return obj

    def to_hydra_config(self) -> dict:
        """Return a dictionary representation of the transform for Hydra instantiation."""
        target = self._get_name_with_module()
        transform_dict = {'_target_': target}
        transform_dict.update(self._get_reproducing_arguments())
        return self._tuples_to_lists(transform_dict)
