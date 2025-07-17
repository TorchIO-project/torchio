import numpy as np
import torch
from einops import rearrange

from ....data.image import Image
from ....data.image import ScalarImage
from ....external.imports import get_sklearn
from ....visualization import build_image_from_input_and_output


def _pca(
    data: torch.Tensor,
    num_components: int = 6,
    whiten: bool = True,
    clip_range: tuple[float, float] | None = (-2.3, 2.3),
    normalize: bool = True,
    make_skewness_positive: bool = True,
    **pca_kwargs,
) -> torch.Tensor:
    # Adapted from https://github.com/facebookresearch/capi/blob/main/eval_visualizations.py
    # 2.3 is roughly 2σ for a standard-normal variable, 99% of values map inside [0,1].
    sklearn = get_sklearn()
    PCA = sklearn.decomposition.PCA

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
        projected.T,
        '(x y z) c -> c x y z',
        x=size_x,
        y=size_y,
        z=size_z,
    )
    if clip_range is not None:
        vmin, vmax = clip_range
    else:
        vmin, vmax = grid.min(), grid.max()
    grid = (grid - vmin) / (vmax - vmin)
    return torch.from_numpy(grid.clip(0, 1))


def build_pca_image(
    embeddings: torch.Tensor, image: Image, keep_components: int | None = 3
) -> ScalarImage:
    pca = _pca(embeddings)
    if keep_components is not None:
        pca = pca[:keep_components]
    pca_image = build_image_from_input_and_output(pca, image)
    return pca_image
