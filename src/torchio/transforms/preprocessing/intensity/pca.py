from __future__ import annotations

from typing import Any

import numpy as np
from einops import rearrange

from ....data.image import ScalarImage
from ....data.subject import Subject
from ....external.imports import get_sklearn
from ...intensity_transform import IntensityTransform


class PCA(IntensityTransform):
    """Compute principal component analysis (PCA) of an image.

    PCA can be useful to visualize embeddings generated by a neural network.
    See for example Figure 8 in `Cluster and Predict Latent Patches for
    Improved Masked Image Modeling <https://arxiv.org/abs/2502.08769>`_.

    Args:
        num_components: Number of components to compute.
        whiten: If ``True``, the components are normalized to have unit variance.
        normalize: If ``True``, all components are divided by the standard
            deviation of the first component.
        make_skewness_positive: If ``True``, the skewness of each component is
            made positive by multiplying the component by -1 if its skewness is
            negative.
        values_range: If not ``None``, these values are linearly mappped to
            :math:`[0, 1]`.
        clip: If ``True``, the output values are clipped to :math:`[0, 1]`.
        pca_kwargs: Additional keyword arguments to pass to
            :class:`sklearn.decomposition.PCA`.

    Example:

    >>> import torchio as tio
    >>> from torchio.visualization import build_image_from_reference
    >>> ct = my_preprocessed_ct_image  # Assume this is a preprocessed CT image
    >>> ct
    ScalarImage(shape: (1, 240, 480, 480); spacing: (1.50, 0.75, 0.75); orientation: SLP+; dtype: torch.FloatTensor; memory: 210.9 MiB)
    >>> embedding_tensor = model(ct.data[None])[0]  # `model` is some pre-trained neural network
    >>> embedding_image = ToReferenceSpace(ct)(embedding_tensor)
    >>> embedding_image
    ScalarImage(shape: (512, 24, 24, 24); spacing: (15.00, 15.00, 15.00); orientation: SLP+; dtype: torch.FloatTensor; memory: 27.0 MiB)
    >>> pca = tio.PCA()(embedding_image)
    >>> pca
    ScalarImage(shape: (3, 24, 24, 24); spacing: (15.00, 15.00, 15.00); orientation: SLP+; dtype: torch.FloatTensor; memory: 162.0 KiB)
    """

    def __init__(
        self,
        num_components: int = 3,
        *,
        whiten: bool = True,
        normalize: bool = True,
        make_skewness_positive: bool = True,
        values_range: tuple[float, float] | None = (-2.3, 2.3),
        clip: bool = True,
        pca_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_components = num_components
        self.whiten = whiten
        self.normalize = normalize
        self.make_skewness_positive = make_skewness_positive
        self.values_range = values_range
        self.clip = clip
        self.pca_kwargs = pca_kwargs
        self.args_names = [
            'num_components',
            'whiten',
            'normalize',
            'make_skewness_positive',
            'values_range',
            'clip',
            'pca_kwargs',
        ]

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            kwargs = {} if self.pca_kwargs is None else self.pca_kwargs
            pca_image = _compute_pca(
                image,
                num_components=self.num_components,
                whiten=self.whiten,
                normalize=self.normalize,
                make_skewness_positive=self.make_skewness_positive,
                values_range=self.values_range,
                clip=self.clip,
                **kwargs,
            )
            image.set_data(pca_image.data)
        return subject


def _compute_pca(
    embeddings: ScalarImage,
    num_components: int,
    whiten: bool,
    normalize: bool,
    make_skewness_positive: bool,
    values_range: tuple[float, float] | None,
    clip: bool,
    **pca_kwargs,
) -> ScalarImage:
    # Adapted from https://github.com/facebookresearch/capi/blob/main/eval_visualizations.py
    # 2.3 is roughly 2σ for a standard-normal variable, 99% of values map inside [0,1].
    sklearn = get_sklearn()
    PCA = sklearn.decomposition.PCA

    data = embeddings.numpy()
    _, size_x, size_y, size_z = data.shape
    X = rearrange(data, 'c x y z -> (x y z) c')
    pca = PCA(n_components=num_components, whiten=whiten, **pca_kwargs)
    projected: np.ndarray = pca.fit_transform(X).T
    if normalize:
        projected /= projected[0].std()
    if make_skewness_positive:
        for component in projected:
            third_cumulant = np.mean(component**3)
            second_cumulant = np.mean(component**2)
            skewness = third_cumulant / second_cumulant ** (3 / 2)
            if skewness < 0:
                component *= -1
    grid: np.ndarray = rearrange(
        projected,
        'c (x y z) -> c x y z',
        x=size_x,
        y=size_y,
        z=size_z,
    )
    if values_range is not None:
        vmin, vmax = values_range
    else:
        vmin, vmax = grid.min(), grid.max()
    grid = (grid - vmin) / (vmax - vmin)
    if clip:
        grid = np.clip(grid, 0, 1)
    return ScalarImage(tensor=grid, affine=embeddings.affine)
