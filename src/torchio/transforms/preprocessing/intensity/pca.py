import numpy as np
import torch
from einops import rearrange

from ....external.imports import get_sklearn


def _pca(
    data: torch.Tensor,
    num_components: int = 6,
    whiten: bool = True,
    vmin: float = -2.3,
    vmax: float = 2.3,
) -> torch.Tensor:
    # Adapted from https://github.com/facebookresearch/capi/blob/main/eval_visualizations.py

    sklearn = get_sklearn()
    PCA = sklearn.decomposition.PCA

    _, size_x, size_y, size_z = data.shape
    X = rearrange(data, 'c x y z -> (x y z) c')
    pca = PCA(n_components=num_components, whiten=whiten)
    projected: np.ndarray = pca.fit_transform(X)
    projected /= projected[:, 0].std()
    for i in range(num_components):
        numerator = np.mean(np.power(projected[:, i], 3))
        denominator = np.power(np.mean(np.power(projected[:, i], 2)), 1.5)
        skew = numerator / denominator
        if skew < 0:
            projected[:, i] *= -1
    grid: np.ndarray = rearrange(
        projected,
        '(x y z) c -> c x y z',
        x=size_x,
        y=size_y,
        z=size_z,
    )
    grid = (grid - vmin) / (vmax - vmin)
    return torch.from_numpy(grid.clip(0, 1))
